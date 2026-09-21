#!/usr/bin/env python
"""Evaluate the trained 16-specialist + 1-generalist routed LoRA-MoE on SNI."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluation.scorer import score_sni_exact_match, score_sni_rouge_l
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


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def prompt_ids(tokenizer, row: dict, max_input_length: int) -> tuple[list[int], bool]:
    messages = [
        {"role": "system", "content": str(row["system"])},
        {"role": "user", "content": str(row["user"])},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    truncated = len(ids) > max_input_length
    if truncated:
        head = max_input_length // 2
        ids = ids[:head] + ids[-(max_input_length - head) :]
    return ids, truncated


def pad_left(tokenizer, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in sequences)
    input_ids, attention_mask = [], []
    for ids in sequences:
        padding = width - len(ids)
        input_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_mask.append([0] * padding + [1] * len(ids))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def generated_attention_mask(generated_ids: torch.Tensor, eos_token_ids: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(generated_ids, dtype=torch.long)
    for row_index, row in enumerate(generated_ids.tolist()):
        valid = len(row)
        for token_index, token_id in enumerate(row):
            if token_id in eos_token_ids:
                valid = token_index + 1
                break
        mask[row_index, :valid] = 1
    return mask


@torch.inference_mode()
def routed_counts(model, input_ids: torch.Tensor, masks: list[torch.Tensor]) -> list[list[list[int]]]:
    """Return counts[layer][mask_index][expert] after one routed forward."""
    core = model.module if hasattr(model, "module") else model
    core.model(
        input_ids=input_ids,
        attention_mask=masks[0],
        use_cache=False,
        return_dict=True,
    )
    logits = router_logits(model)
    result: list[list[list[int]]] = []
    for layer_logits in logits:
        # Count both assignments made by the model's top-2 router.  The
        # resulting shares sum to one over expert assignments (two per token).
        selected = torch.topk(layer_logits, k=2, dim=-1).indices
        layer_result = []
        for mask in masks:
            valid = mask.reshape(-1).bool()
            layer_result.append(
                torch.bincount(selected[valid].reshape(-1), minlength=layer_logits.shape[-1]).cpu().tolist()
            )
        result.append(layer_result)
    return result


def add_counts(destination, source):
    if destination is None:
        return [list(counts) for counts in source]
    if len(destination) != len(source):
        raise ValueError("Layer count mismatch")
    for dst_counts, src_counts in zip(destination, source):
        if len(dst_counts) != len(src_counts):
            raise ValueError("Expert count mismatch")
        for index, value in enumerate(src_counts):
            dst_counts[index] += int(value)
    return destination


def routing_summary(counts, collapse_threshold: float) -> dict:
    if not counts:
        return {"counts": [], "shares": [], "layers": []}
    layers = []
    total_counts = [0] * len(counts[0])
    for layer_index, layer_counts in enumerate(counts):
        total = sum(layer_counts)
        shares = [count / total if total else 0.0 for count in layer_counts]
        entropy = -sum(share * math.log(share) for share in shares if share > 0)
        normalized = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
        max_share = max(shares, default=0.0)
        layers.append({
            "layer": layer_index,
            "counts": layer_counts,
            "shares": shares,
            "max_share": max_share,
            "normalized_entropy": normalized,
            "active_experts_ge_1pct": sum(share >= 0.01 for share in shares),
            "collapsed": max_share >= collapse_threshold,
        })
        for index, count in enumerate(layer_counts):
            total_counts[index] += count
    total = sum(total_counts)
    return {
        "counts": total_counts,
        "shares": [count / total if total else 0.0 for count in total_counts],
        "collapse_threshold": collapse_threshold,
        "collapsed_layer_count": sum(layer["collapsed"] for layer in layers),
        "layers": layers,
    }


def score_row(row: dict, prediction: str) -> tuple[float, float, bool]:
    item = {"ground_truth": row.get("targets") or [row["target"]]}
    exact = score_sni_exact_match(item, prediction)
    rouge = score_sni_rouge_l(item, prediction)
    return exact, rouge, exact == 100.0 or rouge > 70.0


def evaluate_shard(args, model, tokenizer, rows: list[dict], rank: int, world_size: int) -> None:
    output_path = Path(args.output_dir) / f"test_rank{rank:02d}.jsonl"
    completed = load_jsonl_by_id(output_path)
    shard = rows[rank::world_size]
    LOGGER.info("rank=%d shard=%d already_done=%d", rank, len(shard), len(completed))
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = {int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos])}
    started = time.time()
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(shard), args.batch_size):
            batch = [r for r in shard[offset : offset + args.batch_size] if str(r["id"]) not in completed]
            if not batch:
                continue
            if len(batch) != 1:
                raise ValueError("This evaluator currently requires --batch-size 1 for exact route accounting")
            encoded = [prompt_ids(tokenizer, batch[0], args.max_input_length)]
            prompt_ids_tensor, prompt_mask = pad_left(tokenizer, [encoded[0][0]])
            prompt_ids_tensor, prompt_mask = prompt_ids_tensor.to(model.device), prompt_mask.to(model.device)
            sequences = model.generate(
                input_ids=prompt_ids_tensor,
                attention_mask=prompt_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(eos_ids),
                use_cache=True,
            )
            generated = sequences[:, prompt_ids_tensor.shape[1] :]
            prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            generated_mask = generated_attention_mask(generated, eos_ids).to(model.device)
            full_mask = torch.cat([prompt_mask, generated_mask], dim=1)
            counts = routed_counts(model, sequences, [full_mask, torch.cat([torch.zeros_like(prompt_mask), generated_mask], dim=1)])
            exact, rouge, passed = score_row(batch[0], prediction)
            result = {
                "id": str(batch[0]["id"]),
                "task_name": batch[0].get("task_name"),
                "category": batch[0].get("category"),
                "sni_domain": batch[0].get("sni_domain"),
                "prediction": prediction,
                "targets": batch[0].get("targets") or [batch[0]["target"]],
                "exact_match": exact,
                "rouge_l": rouge,
                "passed": passed,
                "prompt_truncated": encoded[0][1],
                "route_counts_full_and_completion": counts,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed[result["id"]] = result
            done = min(offset + 1, len(shard))
            if done % args.log_every == 0 or done == len(shard):
                elapsed = max(time.time() - started, 1e-6)
                LOGGER.info("rank=%d progress=%d/%d rate=%.3f examples/s", rank, done, len(shard), done / elapsed)


def aggregate(args, rows: list[dict], config: RoutedLoraConfig, world_size: int) -> dict:
    results = {}
    for rank in range(world_size):
        results.update(load_jsonl_by_id(Path(args.output_dir) / f"test_rank{rank:02d}.jsonl"))
    if len(results) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} test results, found {len(results)}")
    full_counts = completion_counts = None
    for result in results.values():
        full, completion = result["route_counts_full_and_completion"], result["route_counts_full_and_completion"]
        full_counts = add_counts(full_counts, [layer[0] for layer in full])
        completion_counts = add_counts(completion_counts, [layer[1] for layer in completion])
    exact = [float(r["exact_match"]) for r in results.values()]
    rouge = [float(r["rouge_l"]) for r in results.values()]
    passes = [bool(r["passed"]) for r in results.values()]
    summary = {
        "model": config.base_model_name_or_path,
        "checkpoint": args.checkpoint,
        "experts": list(config.expert_names),
        "top_k": config.top_k,
        "decoding": "greedy",
        "test": {
            "examples": len(results),
            "exact_match": sum(exact) / len(exact),
            "rouge_l": sum(rouge) / len(rouge),
            "pass_rate": 100.0 * sum(passes) / len(passes),
            "pass_rule": "exact_match == 100 or rouge_l > 70",
            "prompt_truncated": sum(bool(r["prompt_truncated"]) for r in results.values()),
            "routing_full": routing_summary(full_counts, args.collapse_threshold),
            "routing_completion": routing_summary(completion_counts, args.collapse_threshold),
        },
    }
    path = Path(args.output_dir) / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    config_data = json.loads((checkpoint / "routed_lora_config.json").read_text(encoding="utf-8"))
    config = RoutedLoraConfig(
        base_model_name_or_path=config_data["base_model_name_or_path"],
        expert_names=tuple(config_data["expert_names"]),
        rank=int(config_data["rank"]), alpha=int(config_data["alpha"]),
        dropout=float(config_data["dropout"]), top_k=int(config_data["top_k"]),
    )
    if config.num_experts != 17 or config.top_k != 2:
        raise ValueError(f"Expected 17 experts/top-2, got {config.num_experts}/{config.top_k}")
    package_dir = Path(args.package_dir)
    rows = read_jsonl(package_dir / "test.jsonl")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": rank},
    )
    model.requires_grad_(False)
    attach_routed_lora_ffns(model, config)
    checkpoint_root = package_dir.parents[2] / "checkpoints"
    adapter_dirs = [
        checkpoint_root / f"sft_llama31_8b_sni_ours-{name}_ffn_only_5ep"
        for name in config.expert_names[:-1]
    ]
    adapter_dirs.append(
        checkpoint_root / "sft_llama31_8b_sni_all-pass-generalist_ffn_only_5ep"
    )
    missing = [str(path) for path in adapter_dirs if not (path / "adapter_model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing initial expert adapters: {missing}")
    load_expert_adapters(model, adapter_dirs, config)
    load_routed_lora_state(model, checkpoint)
    model.eval()
    model.config.use_cache = True
    LOGGER.info("Loaded checkpoint=%s experts=%d top_k=%d rank=%d/%d", checkpoint, config.num_experts, config.top_k, rank, world_size)
    evaluate_shard(args, model, tokenizer, rows, rank, world_size)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        summary = aggregate(args, rows, config, world_size)
        LOGGER.info("FINAL_SUMMARY=%s", json.dumps(summary, ensure_ascii=False))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
