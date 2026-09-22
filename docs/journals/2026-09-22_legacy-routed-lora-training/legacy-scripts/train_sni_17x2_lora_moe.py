#!/usr/bin/env python
"""Train a 16-specialist + 1-generalist token-routed LoRA-MoE on SNI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from moe.routed_lora_ffn import (
    RoutedLoraConfig,
    attach_routed_lora_ffns,
    load_expert_adapters,
    trainable_parameter_groups,
)
from moe.routed_lora_training import (
    CompletionRouteCollator,
    RoutedLoraTrainer,
    verify_gradients,
)


LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()


def build_roster_rows(
    package_dir: Path,
    *,
    seed: int,
    eval_per_specialist: int,
    eval_generalist: int,
) -> tuple[list[dict], list[dict], list[str], list[Path]]:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    routed_count = int(manifest.get("n_experts_routed", 16))
    experts = sorted(
        [item for item in manifest["experts"] if int(item["index"]) < routed_count],
        key=lambda item: int(item["index"]),
    )
    if [int(item["index"]) for item in experts] != list(range(16)):
        raise ValueError("Expected exactly 16 contiguous routed experts")

    expert_names = [str(item["id"]) for item in experts] + ["all-pass-generalist"]
    adapter_dirs: list[Path] = []
    all_rows: list[dict] = []
    targets_by_id: dict[str, set[int]] = {}
    canonical_by_id: dict[str, dict] = {}

    for expert in experts:
        index = int(expert["index"])
        expert_id = str(expert["id"])
        paths = list((package_dir / "train").glob(f"expert_{index:02d}_{expert_id}.jsonl"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected one data file for expert {index} {expert_id}, got {paths}")
        for row in read_jsonl(paths[0]):
            row_id = str(row["id"])
            targets_by_id.setdefault(row_id, set()).add(index)
            canonical_by_id.setdefault(row_id, row)
            all_rows.append(row)

    generalist_index = len(experts)
    for row in read_jsonl(package_dir / "train" / "shared.jsonl"):
        row_id = str(row["id"])
        targets_by_id.setdefault(row_id, set()).add(generalist_index)
        canonical_by_id.setdefault(row_id, row)
        all_rows.append(row)

    eval_ids: set[str] = set()
    quotas = [eval_per_specialist] * len(experts) + [eval_generalist]
    for expert_index, quota in enumerate(quotas):
        candidates = [row_id for row_id, targets in targets_by_id.items() if expert_index in targets]
        candidates.sort(key=lambda row_id: _hash(seed, f"eval-{expert_index}", row_id))
        chosen = [row_id for row_id in candidates if row_id not in eval_ids][:quota]
        if len(chosen) != quota:
            raise ValueError(f"Expert {expert_index}: requested {quota} eval IDs, found {len(chosen)}")
        eval_ids.update(chosen)

    def with_targets(row: dict) -> dict:
        return {**row, "route_targets": sorted(targets_by_id[str(row["id"])])}

    train_rows = [with_targets(row) for row in all_rows if str(row["id"]) not in eval_ids]
    eval_rows = [with_targets(canonical_by_id[row_id]) for row_id in eval_ids]
    train_rows.sort(key=lambda row: _hash(seed, "train", str(row["id"])))
    eval_rows.sort(key=lambda row: _hash(seed, "eval", str(row["id"])))

    checkpoint_root = package_dir.parents[2] / "checkpoints"
    for expert_id in expert_names[:-1]:
        adapter_dirs.append(
            checkpoint_root / f"sft_llama31_8b_sni_ours-{expert_id}_ffn_only_5ep"
        )
    adapter_dirs.append(checkpoint_root / "sft_llama31_8b_sni_all-pass-generalist_ffn_only_5ep")
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expert adapters: {missing}")
    return train_rows, eval_rows, expert_names, adapter_dirs


class RosterDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume model, optimizer, scheduler, RNG, and Trainer state from this checkpoint.",
    )
    parser.add_argument("--hub-model-id")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-per-specialist", type=int, default=8)
    parser.add_argument("--eval-generalist", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--router-warmup-steps", type=int, default=500)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=5e-6)
    parser.add_argument("--route-loss-coef", type=float, default=1.0)
    parser.add_argument("--load-balance-coef", type=float, default=0.01)
    parser.add_argument("--router-z-loss-coef", type=float, default=1e-3)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="sni_17x2_lora_moe")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    package_dir = Path(args.package_dir)
    train_rows, eval_rows, expert_names, adapter_dirs = build_roster_rows(
        package_dir,
        seed=args.seed,
        eval_per_specialist=args.eval_per_specialist,
        eval_generalist=args.eval_generalist,
    )
    LOGGER.info("Dataset prepared: train_rows=%d eval_unique_ids=%d experts=%d", len(train_rows), len(eval_rows), len(expert_names))

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    device_map = None
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.requires_grad_(False)
    routed_config = RoutedLoraConfig(
        base_model_name_or_path=args.base_model,
        expert_names=tuple(expert_names),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        top_k=args.top_k,
    )
    attach_routed_lora_ffns(model, routed_config)
    init_stats = load_expert_adapters(model, adapter_dirs, routed_config)
    model.enable_input_require_grads()
    LOGGER.info("Loaded routed experts: %s", init_stats)
    groups = trainable_parameter_groups(model)
    if groups["unexpected"]:
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in groups['unexpected'][:10]]}")
    LOGGER.info(
        "Trainable parameters: router=%d lora=%d total=%d",
        sum(p.numel() for _, p in groups["router"]),
        sum(p.numel() for _, p in groups["expert_lora"]),
        sum(p.numel() for _, p in groups["router"] + groups["expert_lora"]),
    )

    collator = CompletionRouteCollator(tokenizer, args.max_length, routed_config.num_experts)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.router_lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        # Keep label detection explicit.  Router logits are consumed through
        # the routed-MLP side channel, so no forward wrapper is needed here.
        label_names=["labels"],
        # Top-k dispatch leaves non-selected expert LoRA tensors unused on a
        # given rank/batch; DDP must discover and synchronize those sparse
        # parameter gradients instead of waiting for a missing all-reduce.
        ddp_find_unused_parameters=True,
        report_to=["wandb"],
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save",
        seed=args.seed,
    )
    trainer = RoutedLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=RosterDataset(train_rows),
        eval_dataset=RosterDataset(eval_rows),
        data_collator=collator,
        processing_class=tokenizer,
        routed_config=routed_config,
        route_loss_coef=args.route_loss_coef,
        load_balance_coef=args.load_balance_coef,
        router_z_loss_coef=args.router_z_loss_coef,
        router_lr=args.router_lr,
        lora_lr=args.lora_lr,
        router_warmup_steps=args.router_warmup_steps,
    )
    verification_rows = [
        next(row for row in train_rows if 16 not in row["route_targets"]),
        next(row for row in train_rows if 16 in row["route_targets"]),
    ]
    verify_gradients(model, CompletionRouteCollator(tokenizer, 512, 17)(verification_rows), args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    if args.hub_model_id:
        trainer.push_to_hub(commit_message="End of training: best 16+1 top-2 routed LoRA-MoE")


if __name__ == "__main__":
    main()
