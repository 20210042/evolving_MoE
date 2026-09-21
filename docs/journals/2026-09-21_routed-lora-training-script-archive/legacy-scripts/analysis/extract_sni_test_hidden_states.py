#!/usr/bin/env python3
"""Extract dense Llama input embeddings for the SNI test prompts.

The rendering is read from the saved evaluation predictions so it exactly matches
the prompts used by the routing audit.  The output layout mirrors the existing
LBox cache in ``results/embed_viz_test``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/recent_sft_best_test/sni_sni_moe_best_checkpoint-4000_baseline_235806.jsonl"),
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/embed_viz_test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    ids = [str(row["id"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()

    mean_vectors: list[np.ndarray] = []
    last_vectors: list[np.ndarray] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                row["input"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to("cuda")
        with torch.inference_mode():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[-1]
        mask = encoded.attention_mask.unsqueeze(-1)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        mean_vectors.append(mean.float().cpu().numpy())
        last_vectors.append(hidden[:, -1].float().cpu().numpy())
        if start == 0 or (start // args.batch_size) % 20 == 0:
            print(f"{min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    np.save(args.output_dir / "sni_test_hs_mean.npy", np.concatenate(mean_vectors))
    np.save(args.output_dir / "sni_test_hs_last.npy", np.concatenate(last_vectors))
    (args.output_dir / "sni_test_hs_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {len(ids)} vectors to {args.output_dir}")


if __name__ == "__main__":
    main()
