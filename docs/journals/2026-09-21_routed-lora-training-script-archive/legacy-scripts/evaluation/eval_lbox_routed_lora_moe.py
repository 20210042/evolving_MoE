#!/usr/bin/env python
"""Evaluate a routed-LoRA LBox model on the official local test split."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluation.scorer import score_lbox_item
from moe.routed_lora_ffn import (
    RoutedLoraConfig,
    attach_routed_lora_ffns,
    iter_routed_layers,
    load_expert_adapters,
    load_routed_lora_state,
    router_logits,
)

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_config(checkpoint: Path) -> RoutedLoraConfig:
    data = json.loads((checkpoint / "routed_lora_config.json").read_text(encoding="utf-8"))
    return RoutedLoraConfig(
        base_model_name_or_path=data["base_model_name_or_path"],
        expert_names=tuple(data["expert_names"]),
        rank=int(data["rank"]), alpha=int(data["alpha"]),
        dropout=float(data["dropout"]), top_k=int(data["top_k"]),
    )


def resolve_adapter_dirs(config: RoutedLoraConfig, package_dir: Path, repo_root: Path) -> list[Path]:
    """Resolve the frozen FFN-only adapters used to initialize each routed expert."""
    checkpoint_root = repo_root / "checkpoints"
    names = list(config.expert_names)
    if names and names[-1] in {"all-pass-generalist", "fallback-generalist"}:
        mapping = json.loads((package_dir / "agent_mapping.json").read_text(encoding="utf-8"))
        specialist_suffix = "_fallback" if names[-1] == "fallback-generalist" else ""
        dirs = [
            checkpoint_root
            / f"sft_llama31_8b_lbox_{slug(mapping[name]['name'])}{specialist_suffix}_ffn_only_5ep_eval500"
            for name in names[:-1]
        ]
        generalist_dir = (
            "sft_llama31_8b_lbox_shared-generalist_fallback_ffn_only_5ep_eval500"
            if names[-1] == "fallback-generalist"
            else "sft_llama31_8b_lbox_generalist_ffn_only_5ep_eval500"
        )
        dirs.append(checkpoint_root / generalist_dir)
        return dirs

    # Category/task-prior runs use stable machine-readable names.  The task
    # adapters predate the eval500 suffix, while category adapters do not.
    candidates = []
    for name in names:
        base = checkpoint_root / f"sft_llama31_8b_lbox_{name}_ffn_only_5ep"
        eval500 = checkpoint_root / f"sft_llama31_8b_lbox_{name}_ffn_only_5ep_eval500"
        candidates.append(eval500 if (eval500 / "adapter_model.safetensors").is_file() else base)
    return candidates


def prompt_ids(tokenizer, row: dict, max_input_length: int) -> tuple[list[int], bool]:
    messages = [
        {"role": "system", "content": "You are a precise Korean legal classification assistant. Return only the requested answer in the required format without explanation."},
        {"role": "user", "content": str(row["instruction"])},
    ]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    truncated = len(ids) > max_input_length
    if truncated:
        head = max_input_length // 2
        ids = ids[:head] + ids[-(max_input_length - head):]
    return ids, truncated


def pad_left(tokenizer, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(item) for item in sequences)
    input_ids, masks = [], []
    for item in sequences:
        padding = width - len(item)
        input_ids.append([tokenizer.pad_token_id] * padding + item)
        masks.append([0] * padding + [1] * len(item))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def generated_mask(ids: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(ids, dtype=torch.long)
    for row_index, row in enumerate(ids.tolist()):
        valid = len(row)
        for token_index, token_id in enumerate(row):
            if token_id in eos_ids:
                valid = token_index + 1
                break
        mask[row_index, :valid] = 1
    return mask


@torch.inference_mode()
def route_counts(model, sequences: torch.Tensor, full_mask: torch.Tensor, completion_mask: torch.Tensor):
    model.model(input_ids=sequences, attention_mask=full_mask, use_cache=False, return_dict=True)
    result = []
    for logits in router_logits(model):
        selected = torch.topk(logits, k=model.config.routed_lora_top_k, dim=-1).indices
        layer_result = []
        for mask in (full_mask, completion_mask):
            valid = mask.reshape(-1).bool()
            layer_result.append(torch.bincount(selected[valid].reshape(-1), minlength=logits.shape[-1]).cpu().tolist())
        result.append(layer_result)
    return result


def add_counts(dst, src):
    if dst is None:
        return [list(values) for values in src]
    for dst_values, src_values in zip(dst, src):
        for i, value in enumerate(src_values):
            dst_values[i] += int(value)
    return dst


def summarize_counts(counts, collapse_threshold: float) -> dict:
    if not counts:
        return {"counts": [], "shares": [], "layers": []}
    layers, total_counts = [], [0] * len(counts[0])
    for layer, values in enumerate(counts):
        total = sum(values)
        shares = [value / total if total else 0.0 for value in values]
        entropy = -sum(p * math.log(p) for p in shares if p > 0)
        layers.append({
            "layer": layer, "counts": values, "shares": shares,
            "max_share": max(shares, default=0.0),
            "normalized_entropy": entropy / math.log(len(shares)) if len(shares) > 1 and entropy else 0.0,
            "collapsed": max(shares, default=0.0) >= collapse_threshold,
        })
        for index, value in enumerate(values):
            total_counts[index] += int(value)
    total = sum(total_counts)
    return {
        "counts": total_counts,
        "shares": [value / total if total else 0.0 for value in total_counts],
        "collapse_threshold": collapse_threshold,
        "collapsed_layer_count": sum(item["collapsed"] for item in layers),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    config = load_config(checkpoint)
    package_dir, test_path, output_dir = Path(args.package_dir), Path(args.test_jsonl), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(test_path)
    repo_root = test_path.parents[2]
    adapter_dirs = resolve_adapter_dirs(config, package_dir, repo_root)
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing initial FFN-only adapters: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, revision=args.model_revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, revision=args.model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.requires_grad_(False)
    attach_routed_lora_ffns(model, config)
    load_expert_adapters(model, adapter_dirs, config)
    load_routed_lora_state(model, checkpoint)
    model.eval(); model.config.use_cache = True
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = {int(item) for item in (eos if isinstance(eos, (list, tuple)) else [eos])}
    predictions_path = output_dir / "predictions.jsonl"
    completed = {}
    if predictions_path.is_file():
        completed = {str(row["id"]): row for row in read_jsonl(predictions_path)}
    full_counts = completion_counts = None
    started = time.time()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            if str(row["id"]) in completed:
                continue
            prompt, truncated = prompt_ids(tokenizer, row, args.max_input_length)
            input_ids, prompt_mask = pad_left(tokenizer, [prompt])
            input_ids, prompt_mask = input_ids.to(device), prompt_mask.to(device)
            sequences = model.generate(input_ids=input_ids, attention_mask=prompt_mask, do_sample=False,
                                       max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=list(eos_ids), use_cache=True)
            generated = sequences[:, input_ids.shape[1]:]
            prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            gen_mask = generated_mask(generated, eos_ids).to(device)
            full_mask = torch.cat([prompt_mask, gen_mask], dim=1)
            completion_mask = torch.cat([torch.zeros_like(prompt_mask), gen_mask], dim=1)
            counts = route_counts(model, sequences, full_mask, completion_mask)
            score = score_lbox_item(row, prediction)
            result = {"id": str(row["id"]), "task_type": row.get("task_type"), "prediction": prediction,
                      "ground_truth": row.get("ground_truth"), "score": score, "passed": score >= 100.0,
                      "prompt_truncated": truncated, "route_counts_full_and_completion": counts}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            completed[result["id"]] = result
            full_counts = add_counts(full_counts, [item[0] for item in counts])
            completion_counts = add_counts(completion_counts, [item[1] for item in counts])
            if index % args.log_every == 0 or index == len(rows):
                LOGGER.info("progress=%d/%d rate=%.3f examples/s", index, len(rows), index / max(time.time() - started, 1e-6))

    scores = [float(item["score"]) for item in completed.values()]
    task_summary = {}
    for task in sorted({str(row.get("task_type", "")) for row in rows}):
        task_ids = {str(row["id"]) for row in rows if str(row.get("task_type", "")) == task}
        values = [float(completed[item]["score"]) for item in task_ids if item in completed]
        task_summary[task] = {"examples": len(values), "pass_rate": 100.0 * sum(v >= 100.0 for v in values) / len(values) if values else 0.0}
    summary = {"checkpoint": str(checkpoint), "experts": list(config.expert_names), "top_k": config.top_k,
               "decoding": "greedy", "test": {"examples": len(scores), "pass_rate": 100.0 * sum(v >= 100.0 for v in scores) / len(scores),
               "prompt_truncated": sum(bool(item.get("prompt_truncated")) for item in completed.values()), "by_task_type": task_summary,
               "routing_full": summarize_counts(full_counts, args.collapse_threshold),
               "routing_completion": summarize_counts(completion_counts, args.collapse_threshold)}}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("FINAL_SUMMARY=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
