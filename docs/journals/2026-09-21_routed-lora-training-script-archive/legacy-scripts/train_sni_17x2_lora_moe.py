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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer import TRAINING_ARGS_NAME

from moe.routed_lora_ffn import (
    RoutedLoraConfig,
    attach_routed_lora_ffns,
    load_expert_adapters,
    load_routed_lora_state,
    router_logits,
    save_routed_lora,
    trainable_parameter_groups,
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


class CompletionRouteCollator:
    def __init__(self, tokenizer, max_length: int, num_experts: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_experts = num_experts

    def _tokenize(self, row: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt = [
            {"role": "system", "content": str(row["system"])},
            {"role": "user", "content": str(row["user"])},
        ]
        full = prompt + [{"role": "assistant", "content": str(row["target"])}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        full_ids = self.tokenizer.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"Chat template prefix mismatch for row {row['id']}")
        completion_ids = full_ids[len(prompt_ids) :]
        if not completion_ids:
            raise RuntimeError(f"Empty completion for row {row['id']}")
        available_prompt = self.max_length - len(completion_ids)
        if available_prompt < 2:
            raise RuntimeError(f"Completion exceeds max_length for row {row['id']}")
        if len(prompt_ids) > available_prompt:
            head = available_prompt // 2
            prompt_ids = prompt_ids[:head] + prompt_ids[-(available_prompt - head) :]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return torch.tensor(input_ids), torch.tensor(labels), len(prompt_ids)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        tokenized = [self._tokenize(row) for row in features]
        input_ids = pad_sequence(
            [item[0] for item in tokenized], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence([item[1] for item in tokenized], batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(
            [torch.ones_like(item[0]) for item in tokenized], batch_first=True, padding_value=0
        )
        route_target_mask = torch.zeros(len(features), self.num_experts, dtype=torch.bool)
        for row_index, row in enumerate(features):
            route_target_mask[row_index, row["route_targets"]] = True
        if not all(torch.all(labels[i, :prompt_length] == -100) for i, (*_, prompt_length) in enumerate(tokenized)):
            raise RuntimeError("Completion-only invariant failed: prompt token has a label")
        if not all(torch.any(labels[i] != -100) for i in range(len(features))):
            raise RuntimeError("Completion-only invariant failed: empty supervised completion")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "route_target_mask": route_target_mask,
        }


def supervised_router_loss(
    layer_logits: tuple[torch.Tensor, ...],
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = attention_mask.shape
    token_mask = attention_mask.reshape(-1).bool()
    losses = []
    for logits in layer_logits:
        log_probs = F.log_softmax(logits.float(), dim=-1).reshape(batch_size, sequence_length, -1)
        expanded_targets = target_mask[:, None, :].expand(-1, sequence_length, -1)
        target_log_mass = torch.logsumexp(log_probs.masked_fill(~expanded_targets, -torch.inf), dim=-1)
        losses.append((-target_log_mass.reshape(-1)[token_mask]).mean())
    return torch.stack(losses).mean()


def load_balancing_loss(
    layer_logits: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    batch_size, sequence_length = attention_mask.shape
    per_layer = []
    for logits in layer_logits:
        probabilities = F.softmax(logits.float(), dim=-1).reshape(batch_size, sequence_length, -1)
        selected = torch.topk(probabilities, top_k, dim=-1).indices
        assignments = F.one_hot(selected, probabilities.shape[-1]).float().sum(dim=-2) / top_k
        mask = attention_mask[..., None].to(probabilities.dtype)
        denominator = mask.sum().clamp_min(1.0)
        load = (assignments * mask).sum(dim=(0, 1)) / denominator
        importance = (probabilities * mask).sum(dim=(0, 1)) / denominator
        per_layer.append(probabilities.shape[-1] * torch.sum(load * importance))
    return torch.stack(per_layer).mean()


def router_z_loss(layer_logits: tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    token_mask = attention_mask.reshape(-1).bool()
    return torch.stack(
        [torch.logsumexp(logits.float(), dim=-1).square()[token_mask].mean() for logits in layer_logits]
    ).mean()


class RoutedLoraTrainer(Trainer):
    def __init__(
        self,
        *args,
        routed_config: RoutedLoraConfig,
        route_loss_coef: float,
        load_balance_coef: float,
        router_z_loss_coef: float,
        router_lr: float,
        lora_lr: float,
        router_warmup_steps: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.routed_config = routed_config
        self.route_loss_coef = route_loss_coef
        self.load_balance_coef = load_balance_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router_lr = router_lr
        self.lora_lr = lora_lr
        self.router_warmup_steps = router_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_mask = inputs.pop("route_target_mask")
        outputs = model(**inputs, use_cache=False, num_items_in_batch=num_items_in_batch)
        # The routed MLPs retain their per-layer router logits as a side
        # channel during the forward pass.  Do not rely on attaching a
        # ``router_logits`` attribute to Transformers' ModelOutput here:
        # under DDP/Trainer the returned object is a plain
        # CausalLMOutputWithPast, so that attribute is not guaranteed to be
        # present (the previous implementation failed on the first step).
        logits = router_logits(model)
        route = supervised_router_loss(logits, target_mask, inputs["attention_mask"])
        balance = load_balancing_loss(logits, inputs["attention_mask"], top_k=self.routed_config.top_k)
        z_loss = router_z_loss(logits, inputs["attention_mask"])
        loss = (
            outputs.loss
            + self.route_loss_coef * route
            + self.load_balance_coef * balance
            + self.router_z_loss_coef * z_loss
        )
        self._latest_losses = {
            "router_supervised_loss": route.detach(),
            "router_load_balance_loss": balance.detach(),
            "router_z_loss": z_loss.detach(),
        }
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            groups = trainable_parameter_groups(self.model)
            if groups["unexpected"]:
                raise RuntimeError(f"Unexpected trainable parameters: {[n for n, _ in groups['unexpected'][:10]]}")
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [p for _, p in groups["router"]], "lr": self.router_lr, "weight_decay": 0.0},
                    {"params": [p for _, p in groups["expert_lora"]], "lr": self.lora_lr, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=torch.cuda.is_available(),
            )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step < self.router_warmup_steps:
            for _, parameter in trainable_parameter_groups(model)["expert_lora"]:
                parameter.grad = None
        return loss

    def log(self, logs, *args, **kwargs):
        for name, value in getattr(self, "_latest_losses", {}).items():
            logs[name] = float(value.cpu())
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        save_routed_lora(self.model, output_dir, self.routed_config)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_routed_lora_state(model or self.model, resume_from_checkpoint)

    def _load_best_model(self):
        """Restore the custom routed-LoRA state selected by eval_loss."""
        if not self.state.best_model_checkpoint:
            return
        LOGGER.info("Loading best routed-LoRA checkpoint from %s", self.state.best_model_checkpoint)
        load_routed_lora_state(self.model, self.state.best_model_checkpoint)


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
    nonzero = {
        group: sum(p.grad is not None and bool(torch.count_nonzero(p.grad).item()) for _, p in values)
        for group, values in groups.items()
    }
    if groups["unexpected"] or nonzero["router"] == 0 or nonzero["expert_lora"] == 0:
        raise RuntimeError(f"Gradient verification failed: nonzero={nonzero}")
    if any(parameter.grad is not None for name, parameter in model.named_parameters() if not parameter.requires_grad):
        raise RuntimeError("Frozen base parameter received a gradient")
    LOGGER.info(
        "Gradient verification passed: loss=%.6f lm=%.6f route=%.6f balance=%.6f z=%.6f nonzero=%s",
        float(loss.detach()), float(outputs.loss.detach()), float(route.detach()), float(balance.detach()),
        float(z_loss.detach()), nonzero,
    )
    model.zero_grad(set_to_none=True)


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
