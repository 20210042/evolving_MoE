#!/usr/bin/env python
"""Train a 10-specialist + 1-generalist token-routed LoRA-MoE on LBox."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from moe.routed_lora_ffn import (
    RoutedLoraConfig,
    attach_routed_lora_ffns,
    load_expert_adapters,
    trainable_parameter_groups,
)
from scripts.train_sni_17x2_lora_moe import (
    CompletionRouteCollator,
    RoutedLoraTrainer,
    load_balancing_loss,
    router_logits,
    router_z_loss,
    supervised_router_loss,
)
from train_sft import stringify_completion


LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_rows(package_dir: Path, source_jsonl: Path, *, seed: int, eval_ratio: float):
    mapping = json.loads((package_dir / "agent_mapping.json").read_text(encoding="utf-8"))
    expert_ids = list(mapping)
    if len(expert_ids) != 10:
        raise ValueError(f"Expected 10 LBox experts, found {len(expert_ids)}")
    source = {str(row["id"]): row for row in read_jsonl(source_jsonl)}
    labels = read_jsonl(package_dir / "binning_labels.jsonl")
    rows = []
    excluded = {"all_fail": 0, "unmatched": 0}
    for label in labels:
        row = source.get(str(label["id"]))
        if row is None:
            excluded["unmatched"] += 1
            continue
        n_solved = int(label.get("n_solved", 0))
        if n_solved >= 8:
            targets = [10]
        elif 1 <= n_solved <= 7:
            targets = [i for i, expert_id in enumerate(expert_ids) if int((label.get("per_expert") or {}).get(expert_id, 0)) == 1]
        else:
            targets = []
            excluded["all_fail"] += 1
        if not targets:
            continue
        rows.append({
            "id": str(row["id"]),
            "system": "You are a precise Korean legal classification assistant. Return only the requested answer in the required format without explanation.",
            "user": str(row["instruction"]),
            "target": stringify_completion(row["ground_truth"]),
            "route_targets": targets,
            "task_type": row.get("task_type"),
            "domain": row.get("domain"),
        })
    # Deterministic held-out split; preserve all route labels in the training pool.
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:lbox:{row['id']}".encode()).hexdigest())
    eval_size = max(1, int(len(rows) * eval_ratio))
    eval_rows, train_rows = rows[:eval_size], rows[eval_size:]
    LOGGER.info("LBox routed rows: train=%d eval=%d excluded=%s", len(train_rows), len(eval_rows), excluded)
    return train_rows, eval_rows, expert_ids, mapping


def verify_gradients(model, batch, args):
    model.train()
    batch = {name: value.to(model.device) for name, value in batch.items()}
    target_mask = batch.pop("route_target_mask")
    outputs = model(**batch, use_cache=False)
    logits = router_logits(model)
    route = supervised_router_loss(logits, target_mask, batch["attention_mask"])
    balance = load_balancing_loss(logits, batch["attention_mask"], top_k=args.top_k)
    z_loss = router_z_loss(logits, batch["attention_mask"])
    loss = outputs.loss + args.route_loss_coef * route + args.load_balance_coef * balance + args.router_z_loss_coef * z_loss
    loss.backward()
    groups = trainable_parameter_groups(model)
    nonzero = {group: sum(p.grad is not None and bool(torch.count_nonzero(p.grad).item()) for _, p in values) for group, values in groups.items()}
    if groups["unexpected"] or nonzero["router"] == 0 or nonzero["expert_lora"] == 0:
        raise RuntimeError(f"Gradient verification failed: nonzero={nonzero}")
    model.zero_grad(set_to_none=True)
    LOGGER.info("Gradient verification passed: nonzero=%s", nonzero)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--package-dir", required=True)
    p.add_argument("--source-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hub-model-id")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--router-warmup-steps", type=int, default=500)
    p.add_argument("--router-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=5e-6)
    p.add_argument("--route-loss-coef", type=float, default=1.0)
    p.add_argument("--load-balance-coef", type=float, default=0.01)
    p.add_argument("--router-z-loss-coef", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="lbox_11x2_lora_moe")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    package_dir = Path(args.package_dir)
    train_rows, eval_rows, expert_ids, mapping = build_rows(package_dir, Path(args.source_jsonl), seed=args.seed, eval_ratio=args.eval_ratio)
    expert_names = tuple(expert_ids + ["all-pass-generalist"])
    config = RoutedLoraConfig("meta-llama/Llama-3.1-8B-Instruct", expert_names, rank=16, alpha=32, dropout=0.05, top_k=args.top_k)
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "-1"))
    device_map = None
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map=device_map)
    model.requires_grad_(False)
    attach_routed_lora_ffns(model, config)
    # LBox package lives directly under ``results/`` (unlike the nested SNI
    # package), so the repository root is parents[1].
    checkpoint_root = package_dir.parents[1] / "checkpoints"
    adapter_dirs = [checkpoint_root / f"sft_llama31_8b_lbox_{slug(mapping[expert_id]['name'])}_ffn_only_5ep_eval500" for expert_id in expert_ids]
    adapter_dirs.append(checkpoint_root / "sft_llama31_8b_lbox_generalist_ffn_only_5ep_eval500")
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LBox adapters: {missing}")
    load_expert_adapters(model, adapter_dirs, config)
    model.enable_input_require_grads()
    collator = CompletionRouteCollator(tokenizer, args.max_length, config.num_experts)
    training_args = TrainingArguments(
        output_dir=args.output_dir, run_name=args.run_name, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, learning_rate=args.router_lr,
        warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True, tf32=True,
        logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, label_names=["labels"], ddp_find_unused_parameters=True,
        report_to=["wandb"], push_to_hub=bool(args.hub_model_id), hub_model_id=args.hub_model_id,
        hub_strategy="every_save", seed=args.seed,
    )
    trainer = RoutedLoraTrainer(
        model=model, args=training_args, train_dataset=train_rows, eval_dataset=eval_rows,
        data_collator=collator, processing_class=tokenizer, routed_config=config,
        route_loss_coef=args.route_loss_coef, load_balance_coef=args.load_balance_coef,
        router_z_loss_coef=args.router_z_loss_coef, router_lr=args.router_lr, lora_lr=args.lora_lr,
        router_warmup_steps=args.router_warmup_steps,
    )
    # Verify one specialist-targeted and one generalist-targeted example.
    specialist = next(row for row in train_rows if 10 not in row["route_targets"])
    generalist = next(row for row in train_rows if 10 in row["route_targets"])
    verify_gradients(model, collator([specialist, generalist]), args)
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    if args.hub_model_id:
        trainer.push_to_hub(commit_message="End of training: best 10+1 top-2 routed LBox LoRA-MoE")
    LOGGER.info("Complete; best LBox routed LoRA-MoE saved and pushed: %s", args.hub_model_id)


if __name__ == "__main__":
    main()
