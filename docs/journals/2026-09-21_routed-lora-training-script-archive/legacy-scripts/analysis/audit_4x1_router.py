#!/usr/bin/env python
"""Collect per-example, per-layer routing counts from a 4-expert top-1 MoE.

The token sequence is reconstructed from an existing evaluation JSONL. This
keeps raw and SFT models on exactly the same prompt and generated completion,
so differences in routing are not confounded by different decoded text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["id"]) for line in handle if line.strip()}


def render_sequence(tokenizer, row: dict, max_length: int) -> tuple[list[int], int, bool]:
    messages = row["input"]
    prediction = str(row.get("prediction") or "")
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": prediction}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"Chat-template prefix mismatch for {row['id']}")

    completion_ids = full_ids[len(prompt_ids) :]
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1]
    available_prompt = max_length - len(completion_ids)
    truncated = len(prompt_ids) > available_prompt
    if truncated:
        head = available_prompt // 2
        prompt_ids = prompt_ids[:head] + prompt_ids[-(available_prompt - head) :]
    return prompt_ids + completion_ids, len(prompt_ids), truncated


@torch.inference_mode()
def route_counts(model, input_ids: torch.Tensor, prompt_length: int) -> tuple[list[list[int]], list[list[int]]]:
    core = model.get_base_model() if isinstance(model, PeftModel) else model
    attention_mask = torch.ones_like(input_ids)
    outputs = core.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_router_logits=True,
        return_dict=True,
    )
    if not outputs.router_logits:
        raise RuntimeError("Model did not return router logits")
    sequence_length = input_ids.shape[1]
    prompt_counts = []
    completion_counts = []
    for layer_logits in outputs.router_logits:
        if layer_logits.shape[0] != sequence_length:
            raise RuntimeError(
                f"Router shape {tuple(layer_logits.shape)} does not match sequence length {sequence_length}"
            )
        selected = layer_logits.argmax(dim=-1)
        num_experts = layer_logits.shape[-1]
        prompt_counts.append(
            torch.bincount(selected[:prompt_length], minlength=num_experts).cpu().tolist()
        )
        completion_counts.append(
            torch.bincount(selected[prompt_length:], minlength=num_experts).cpu().tolist()
        )
    return prompt_counts, completion_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("raw", "sft"), required=True)
    parser.add_argument("--dataset", choices=("lbox", "sni"), required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.variant}_rank{rank:02d}.jsonl"
    completed = load_completed(output_path)
    rows = read_jsonl(args.predictions)
    shard = rows[rank::world_size]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": rank},
    )
    if args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    core = model.get_base_model() if isinstance(model, PeftModel) else model
    if int(core.config.num_local_experts) != 4 or int(core.config.num_experts_per_tok) != 1:
        raise RuntimeError(
            f"Expected 4x1 MoE, got {core.config.num_local_experts}x{core.config.num_experts_per_tok}"
        )
    LOGGER.info(
        "loaded dataset=%s variant=%s rank=%d/%d examples=%d adapter=%s",
        args.dataset,
        args.variant,
        rank,
        world_size,
        len(shard),
        args.adapter,
    )

    started = time.time()
    with output_path.open("a", encoding="utf-8") as handle:
        for offset, row in enumerate(shard, start=1):
            row_id = str(row["id"])
            if row_id in completed:
                continue
            ids, prompt_length, truncated = render_sequence(tokenizer, row, args.max_length)
            input_ids = torch.tensor([ids], dtype=torch.long, device=core.device)
            prompt_counts, completion_counts = route_counts(model, input_ids, prompt_length)
            result = {
                "id": row_id,
                "dataset": args.dataset,
                "variant": args.variant,
                "sequence_source": "saved_sft_prediction",
                "prompt_tokens": prompt_length,
                "completion_tokens": len(ids) - prompt_length,
                "truncated": truncated,
                "pass_score": row.get("pass_score"),
                "em_score": row.get("em_score"),
                "rouge_l_score": row.get("rouge_l_score"),
                "prompt_route_counts": prompt_counts,
                "completion_route_counts": completion_counts,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(row_id)
            if offset % args.log_every == 0 or offset == len(shard):
                elapsed = max(time.time() - started, 1e-6)
                LOGGER.info(
                    "progress rank=%d variant=%s %d/%d rate=%.3f examples/s",
                    rank,
                    args.variant,
                    offset,
                    len(shard),
                    offset / elapsed,
                )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
