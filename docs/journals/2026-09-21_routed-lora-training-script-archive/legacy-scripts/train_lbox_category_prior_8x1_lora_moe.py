#!/usr/bin/env python
"""Train an 8-category top-1 routed LoRA-MoE on LBox."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from moe.routed_lora_ffn import RoutedLoraConfig, attach_routed_lora_ffns, load_expert_adapters, trainable_parameter_groups
from scripts.train_sni_17x2_lora_moe import CompletionRouteCollator, RoutedLoraTrainer, load_balancing_loss, router_logits, router_z_loss, supervised_router_loss
from train_sft import stringify_completion


LOGGER = logging.getLogger(__name__)
CATEGORIES = (
    "civil_property_obligation", "civil_family_inheritance", "criminal_property", "criminal_non_property",
    "admin_traffic", "admin_labor", "admin_other", "family_patent_special",
)


class RowDataset:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rows(source_jsonl: Path, tags_jsonl: Path, *, seed: int, eval_ratio: float):
    source = {str(row["id"]): row for row in read_jsonl(source_jsonl)}
    category_to_index = {name: i for i, name in enumerate(CATEGORIES)}
    rows = []
    for tag in read_jsonl(tags_jsonl):
        category = str(tag.get("primary_category", ""))
        raw = source.get(str(tag["id"]))
        if raw is None:
            raise ValueError(f"Category tag has no source row: {tag['id']}")
        if category not in category_to_index:
            raise ValueError(f"Unsupported LBox category: {category!r}")
        rows.append({
            "id": str(raw["id"]),
            "system": "You are a precise Korean legal classification assistant. Return only the requested answer in the required format without explanation.",
            "user": str(raw["instruction"]),
            "target": stringify_completion(raw["ground_truth"]),
            "route_targets": [category_to_index[category]],
            "category": category,
        })
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:category-prior:{row['id']}".encode()).hexdigest())
    eval_size = max(1, int(len(rows) * eval_ratio))
    LOGGER.info("LBox category-prior rows: train=%d eval=%d", len(rows) - eval_size, eval_size)
    return rows[eval_size:], rows[:eval_size]


def verify_gradients(model, batch, args):
    model.train()
    batch = {name: value.to(model.device) for name, value in batch.items()}
    target_mask = batch.pop("route_target_mask")
    outputs = model(**batch, use_cache=False)
    logits = router_logits(model)
    route = supervised_router_loss(logits, target_mask, batch["attention_mask"])
    balance = load_balancing_loss(logits, batch["attention_mask"], top_k=args.top_k)
    z_loss = router_z_loss(logits, batch["attention_mask"])
    (outputs.loss + args.route_loss_coef * route + args.load_balance_coef * balance + args.router_z_loss_coef * z_loss).backward()
    groups = trainable_parameter_groups(model)
    nonzero = {group: sum(p.grad is not None and bool(torch.count_nonzero(p.grad).item()) for _, p in values) for group, values in groups.items()}
    if groups["unexpected"] or nonzero["router"] == 0 or nonzero["expert_lora"] == 0:
        raise RuntimeError(f"Gradient verification failed: nonzero={nonzero}")
    model.zero_grad(set_to_none=True)
    LOGGER.info("Gradient verification passed: nonzero=%s", nonzero)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-jsonl", required=True); p.add_argument("--tags-jsonl", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--hub-model-id")
    p.add_argument("--epochs", type=float, default=1.0); p.add_argument("--eval-ratio", type=float, default=.05); p.add_argument("--eval-steps", type=int, default=500); p.add_argument("--save-steps", type=int, default=500); p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--router-warmup-steps", type=int, default=500); p.add_argument("--router-lr", type=float, default=1e-4); p.add_argument("--lora-lr", type=float, default=5e-6); p.add_argument("--route-loss-coef", type=float, default=1.0); p.add_argument("--load-balance-coef", type=float, default=.01); p.add_argument("--router-z-loss-coef", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=1); p.add_argument("--max-length", type=int, default=8192); p.add_argument("--gradient-accumulation-steps", type=int, default=8); p.add_argument("--seed", type=int, default=42); p.add_argument("--run-name", default="lbox_category_prior_8x1_lora_moe")
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_k != 1: raise ValueError("This category-prior run is intentionally top-1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    train_rows, eval_rows = build_rows(Path(args.source_jsonl), Path(args.tags_jsonl), seed=args.seed, eval_ratio=args.eval_ratio)
    config = RoutedLoraConfig("meta-llama/Llama-3.1-8B-Instruct", CATEGORIES, rank=16, alpha=32, dropout=.05, top_k=1)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1")); device_map = None
    if local_rank >= 0: torch.cuda.set_device(local_rank); device_map = {"": local_rank}
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path); tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map=device_map)
    model.requires_grad_(False); attach_routed_lora_ffns(model, config)
    root = Path(args.source_jsonl).parents[2]
    adapter_dirs = [root / f"checkpoints/sft_llama31_8b_lbox_{category}_ffn_only_5ep" for category in CATEGORIES]
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing: raise FileNotFoundError(f"Missing category-prior adapters: {missing}")
    load_expert_adapters(model, adapter_dirs, config); model.enable_input_require_grads()
    collator = CompletionRouteCollator(tokenizer, args.max_length, config.num_experts)
    training_args = TrainingArguments(output_dir=args.output_dir, run_name=args.run_name, num_train_epochs=args.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, learning_rate=args.router_lr, warmup_ratio=.03, lr_scheduler_type="cosine", bf16=True, tf32=True, logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False, label_names=["labels"], ddp_find_unused_parameters=True, report_to=["wandb"], push_to_hub=bool(args.hub_model_id), hub_model_id=args.hub_model_id, hub_strategy="every_save", seed=args.seed)
    trainer = RoutedLoraTrainer(model=model, args=training_args, train_dataset=RowDataset(train_rows), eval_dataset=RowDataset(eval_rows), data_collator=collator, processing_class=tokenizer, routed_config=config, route_loss_coef=args.route_loss_coef, load_balance_coef=args.load_balance_coef, router_z_loss_coef=args.router_z_loss_coef, router_lr=args.router_lr, lora_lr=args.lora_lr, router_warmup_steps=args.router_warmup_steps)
    verify_gradients(model, collator([train_rows[0], train_rows[-1]]), args)
    trainer.train(); trainer.save_model(); trainer.save_state()
    if args.hub_model_id: trainer.push_to_hub(commit_message="End of training: best 8-way top-1 routed LBox category-prior LoRA-MoE")
    LOGGER.info("Complete; best LBox category-prior MoE saved and pushed: %s", args.hub_model_id)


if __name__ == "__main__": main()
