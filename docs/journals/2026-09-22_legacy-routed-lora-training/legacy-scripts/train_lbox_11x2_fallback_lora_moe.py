#!/usr/bin/env python
"""Fallback-aware 10-specialist + 1-generalist top-2 LBox LoRA-MoE."""
from __future__ import annotations
import argparse, hashlib, json, logging, os, re
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from moe.routed_lora_ffn import RoutedLoraConfig, attach_routed_lora_ffns, load_expert_adapters, router_logits, trainable_parameter_groups
from moe.routed_lora_training import load_balancing_loss, router_z_loss
from moe.routed_lora_training import CompletionRouteCollator, RoutedLoraTrainer
from train_sft import stringify_completion

LOGGER = logging.getLogger(__name__)

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def build_rows(
    package_dir: Path,
    assignment_package_dir: Path,
    source_jsonl: Path,
    *,
    seed: int,
    eval_ratio: float,
):
    mapping = json.loads((package_dir / "agent_mapping.json").read_text(encoding="utf-8"))
    expert_ids = list(mapping)
    if len(expert_ids) != 10:
        raise ValueError(f"Expected 10 LBox experts, found {len(expert_ids)}")
    source = {str(row["id"]): row for row in read_jsonl(source_jsonl)}
    labels = {str(row["id"]): row for row in read_jsonl(package_dir / "binning_labels.jsonl")}

    # The materialized fallback package is the authoritative routing contract:
    #   n_solved=0   -> exactly one centroid-assigned specialist
    #   n_solved=1-8 -> every specialist that solved the problem
    #   n_solved=9-10 -> shared generalist
    assignments: dict[str, list[int]] = {}
    assignment_kinds: dict[str, str] = {}
    for expert_index, expert_id in enumerate(expert_ids):
        candidates = list((assignment_package_dir / "train").glob(f"expert_{expert_index:02d}_{expert_id}.jsonl"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one assignment file for expert {expert_index}:{expert_id}, found {candidates}"
            )
        for item in read_jsonl(candidates[0]):
            item_id = str(item["id"])
            assignments.setdefault(item_id, []).append(expert_index)
            assignment_kinds[item_id] = str(item["kind"])
    for item in read_jsonl(assignment_package_dir / "train" / "shared.jsonl"):
        item_id = str(item["id"])
        assignments.setdefault(item_id, []).append(10)
        assignment_kinds[item_id] = str(item["kind"])

    if set(assignments) != set(source) or set(labels) != set(source):
        raise RuntimeError(
            "LBox source/label/assignment ID mismatch: "
            f"source={len(source)} labels={len(labels)} assignments={len(assignments)} "
            f"missing_assignments={len(set(source) - set(assignments))} "
            f"extra_assignments={len(set(assignments) - set(source))}"
        )

    rows, counts = [], {"all_fail_centroid": 0, "specialist": 0, "shared": 0}
    for item_id, row in source.items():
        label = labels[item_id]
        n_solved = int(label.get("n_solved", 0))
        targets = assignments[item_id]
        if n_solved == 0:
            expected = len(targets) == 1 and targets[0] < 10 and assignment_kinds[item_id] == "all_fail"
            mode = "all-fail-centroid-specialist"
            counts["all_fail_centroid"] += 1
        elif 1 <= n_solved <= 8:
            solved_targets = [
                index
                for index, expert_id in enumerate(expert_ids)
                if int((label.get("per_expert") or {}).get(expert_id, 0)) == 1
            ]
            expected = targets == solved_targets and assignment_kinds[item_id] == "indiv"
            mode = "solving-specialists"
            counts["specialist"] += 1
        else:
            expected = n_solved in (9, 10) and targets == [10] and assignment_kinds[item_id] == "shared"
            mode = "shared-high-consensus"
            counts["shared"] += 1
        if not expected:
            raise RuntimeError(
                f"Invalid materialized assignment for id={item_id} n_solved={n_solved} "
                f"targets={targets} kind={assignment_kinds[item_id]}"
            )
        rows.append({"id": item_id, "system": "You are a precise Korean legal classification assistant. Return only the requested answer in the required format without explanation.", "user": str(row["instruction"]), "target": stringify_completion(row["ground_truth"]), "route_targets": targets, "route_mode": mode, "n_solved": n_solved, "task_type": row.get("task_type"), "domain": row.get("domain")})
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:lbox-fallback:{r['id']}".encode()).hexdigest())
    eval_size = max(1, int(len(rows) * eval_ratio))
    eval_rows, train_rows = rows[:eval_size], rows[eval_size:]
    LOGGER.info(
        "Clustered fallback rows: train=%d eval=%d counts=%s; package assignments preserved",
        len(train_rows), len(eval_rows), counts,
    )
    return train_rows, eval_rows, expert_ids, mapping

class CompletionOnlyRouteCollator(CompletionRouteCollator):
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        batch["route_loss_mask"] = batch["labels"].ne(-100)
        return batch

def improved_router_loss(layer_logits, target_mask, route_loss_mask, *, top_k: int, slot_coef: float):
    batch_size, sequence_length = route_loss_mask.shape
    token_mask = route_loss_mask.reshape(-1).bool()
    expanded = target_mask[:, None, :].expand(-1, sequence_length, -1).reshape(-1, target_mask.shape[-1])
    targets, valid_count = expanded[token_mask], expanded[token_mask].sum(dim=-1)
    total, masses, slots = [], [], []
    for logits in layer_logits:
        probs = F.softmax(logits.float(), dim=-1).reshape(batch_size * sequence_length, -1)[token_mask]
        mass_loss = -torch.log((probs * targets.to(probs.dtype)).sum(dim=-1).clamp_min(1e-8)).mean()
        selected_p, selected_i = torch.topk(probs, top_k, dim=-1)
        selected_valid = targets.gather(1, selected_i)
        multi = valid_count >= top_k
        if torch.any(multi):
            p, valid = selected_p[multi].clamp(1e-6, 1 - 1e-6), selected_valid[multi]
            slot_loss = torch.where(valid, -torch.log(p), -torch.log1p(-p)).mean()
        else:
            slot_loss = probs.new_zeros(())
        total.append(mass_loss + slot_coef * slot_loss); masses.append(mass_loss); slots.append(slot_loss)
    return torch.stack(total).mean(), torch.stack(masses).mean(), torch.stack(slots).mean()

class FallbackAwareRoutedLoraTrainer(RoutedLoraTrainer):
    def __init__(self, *args, slot_loss_coef: float, **kwargs):
        super().__init__(*args, **kwargs); self.slot_loss_coef = slot_loss_coef
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_mask, route_loss_mask = inputs.pop("route_target_mask"), inputs.pop("route_loss_mask")
        outputs = model(**inputs, use_cache=False, num_items_in_batch=num_items_in_batch); logits = router_logits(model)
        route, mass, slot = improved_router_loss(logits, target_mask, route_loss_mask, top_k=self.routed_config.top_k, slot_coef=self.slot_loss_coef)
        balance, z_loss = load_balancing_loss(logits, route_loss_mask, top_k=self.routed_config.top_k), router_z_loss(logits, route_loss_mask)
        loss = outputs.loss + self.route_loss_coef * route + self.load_balance_coef * balance + self.router_z_loss_coef * z_loss
        self._latest_losses = {"router_supervised_loss": route.detach(), "router_valid_mass_loss": mass.detach(), "router_slot_loss": slot.detach(), "router_load_balance_loss": balance.detach(), "router_z_loss": z_loss.detach()}
        return (loss, outputs) if return_outputs else loss

def verify_gradients(model, batch, args):
    model.train(); batch = {name: value.to(model.device) for name, value in batch.items()}
    target_mask, route_loss_mask = batch.pop("route_target_mask"), batch.pop("route_loss_mask")
    outputs = model(**batch, use_cache=False); logits = router_logits(model)
    route, _, _ = improved_router_loss(logits, target_mask, route_loss_mask, top_k=args.top_k, slot_coef=args.slot_loss_coef)
    loss = outputs.loss + args.route_loss_coef * route + args.load_balance_coef * load_balancing_loss(logits, route_loss_mask, top_k=args.top_k) + args.router_z_loss_coef * router_z_loss(logits, route_loss_mask)
    loss.backward(); groups = trainable_parameter_groups(model)
    nonzero = {group: sum(p.grad is not None and bool(torch.count_nonzero(p.grad).item()) for _, p in values) for group, values in groups.items()}
    if groups["unexpected"] or nonzero["router"] == 0 or nonzero["expert_lora"] == 0: raise RuntimeError(f"Gradient verification failed: nonzero={nonzero}")
    model.zero_grad(set_to_none=True); LOGGER.info("Gradient verification passed: nonzero=%s", nonzero)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--package-dir", required=True); p.add_argument("--assignment-package-dir", required=True); p.add_argument("--source-jsonl", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--hub-model-id")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--epochs", type=float, default=5.0); p.add_argument("--eval-ratio", type=float, default=0.05); p.add_argument("--eval-steps", type=int, default=500); p.add_argument("--save-steps", type=int, default=500); p.add_argument("--logging-steps", type=int, default=10); p.add_argument("--router-warmup-steps", type=int, default=500)
    p.add_argument("--router-lr", type=float, default=1e-4); p.add_argument("--lora-lr", type=float, default=5e-6); p.add_argument("--route-loss-coef", type=float, default=1.0); p.add_argument("--slot-loss-coef", type=float, default=0.5); p.add_argument("--load-balance-coef", type=float, default=0.01); p.add_argument("--router-z-loss-coef", type=float, default=1e-3); p.add_argument("--top-k", type=int, default=2); p.add_argument("--max-length", type=int, default=8192); p.add_argument("--gradient-accumulation-steps", type=int, default=8); p.add_argument("--seed", type=int, default=42); p.add_argument("--run-name", default="lbox_11x2_fallback_lora_moe_5ep")
    return p.parse_args()

def main():
    args = parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"); set_seed(args.seed)
    package_dir = Path(args.package_dir); train_rows, eval_rows, expert_ids, mapping = build_rows(package_dir, Path(args.assignment_package_dir), Path(args.source_jsonl), seed=args.seed, eval_ratio=args.eval_ratio)
    expert_names = tuple(expert_ids + ["fallback-generalist"]); config = RoutedLoraConfig("meta-llama/Llama-3.1-8B-Instruct", expert_names, rank=16, alpha=32, dropout=0.05, top_k=args.top_k)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1")); device_map = None
    if local_rank >= 0: torch.cuda.set_device(local_rank); device_map = {"": local_rank}
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, revision=args.model_revision); tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, revision=args.model_revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map=device_map); model.requires_grad_(False); attach_routed_lora_ffns(model, config)
    checkpoint_root = package_dir.parents[1] / "checkpoints"
    adapter_dirs = [checkpoint_root / f"sft_llama31_8b_lbox_{slug(mapping[expert_id]['name'])}_fallback_ffn_only_5ep_eval500" for expert_id in expert_ids] + [checkpoint_root / "sft_llama31_8b_lbox_shared-generalist_fallback_ffn_only_5ep_eval500"]
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing: raise FileNotFoundError(f"Missing fallback LBox adapters: {missing}")
    load_expert_adapters(model, adapter_dirs, config); model.enable_input_require_grads(); collator = CompletionOnlyRouteCollator(tokenizer, args.max_length, config.num_experts)
    training_args = TrainingArguments(output_dir=args.output_dir, run_name=args.run_name, num_train_epochs=args.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, learning_rate=args.router_lr, warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True, tf32=True, logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False, label_names=["labels"], ddp_find_unused_parameters=True, report_to=["wandb"], push_to_hub=bool(args.hub_model_id), hub_model_id=args.hub_model_id, hub_strategy="every_save", seed=args.seed)
    trainer = FallbackAwareRoutedLoraTrainer(model=model, args=training_args, train_dataset=train_rows, eval_dataset=eval_rows, data_collator=collator, processing_class=tokenizer, routed_config=config, route_loss_coef=args.route_loss_coef, load_balance_coef=args.load_balance_coef, router_z_loss_coef=args.router_z_loss_coef, router_lr=args.router_lr, lora_lr=args.lora_lr, router_warmup_steps=args.router_warmup_steps, slot_loss_coef=args.slot_loss_coef)
    specialist = next(row for row in train_rows if 10 not in row["route_targets"]); generalist = next(row for row in train_rows if row["route_targets"] == [10]); verify_gradients(model, collator([specialist, generalist]), args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint); trainer.save_model(); trainer.save_state()
    if args.hub_model_id: trainer.push_to_hub(commit_message="End of training: fallback-aware 10+1 top-2 routed LBox LoRA-MoE")
    LOGGER.info("Complete; best fallback-aware LBox routed LoRA-MoE pushed: %s", args.hub_model_id)

if __name__ == "__main__": main()
