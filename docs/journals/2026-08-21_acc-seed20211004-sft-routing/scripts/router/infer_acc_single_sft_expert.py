#!/usr/bin/env python3
"""Evaluate one trained ACC LoRA expert over the complete held-out test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
from evaluation.scorer import score_one  # noqa: E402
from prompts.coding import build_baseline_prompt, build_expert_prompt  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--agent-mapping", default="export/acc_binning_seed20211004_persona/agent_mapping.json")
    ap.add_argument("--specialist-root", default="checkpoints/expert_sft/acc_seed20211004/cap8_core200")
    ap.add_argument("--luca-root", default="checkpoints/expert_sft/acc_seed20211004/luca_allpass1000")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output-dir", default="results/acc/seed20211004/sft_oracle/parts")
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wandb-project", default="acc-seed20211004-sft-oracle")
    ap.add_argument("--wandb-entity", default="jongbin-kr-skiml_moe")
    a = ap.parse_args()

    rows = load_jsonl(Path(a.test_jsonl))[: a.limit or None]
    mapping = json.loads(Path(a.agent_mapping).read_text())
    adapter = Path(a.luca_root) if a.expert == "luca" else Path(a.specialist_root) / a.expert
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing adapter: {adapter}")
    if a.expert != "luca" and a.expert not in mapping:
        raise KeyError(f"missing persona mapping for {a.expert}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()

    out = Path(a.output_dir) / f"{a.expert}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(out) if out.is_file() else []
    done = {str(r["id"]) for r in existing}

    import wandb

    run = wandb.init(
        project=a.wandb_project,
        entity=a.wandb_entity,
        name=f"acc_sft_oracle_{a.expert}",
    )
    newly_processed = 0
    with out.open("a", encoding="utf-8") as f:
        for item in rows:
            if str(item["id"]) in done:
                continue
            if a.expert == "luca":
                messages = build_baseline_prompt(
                    item["instruction"], dataset="acc", model_name=a.base_model,
                    starter_code=item.get("starter_code"), domain=item.get("domain"),
                )
            else:
                messages = build_expert_prompt(
                    item["instruction"], mapping[a.expert]["system_prompt"], dataset="acc",
                    model_name=a.base_model, starter_code=item.get("starter_code"),
                    domain=item.get("domain"),
                )
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_length).to("cuda")
            with torch.inference_mode():
                seq = model.generate(
                    **enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            prediction = tok.decode(seq[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()
            score = float(score_one(item, prediction))
            f.write(json.dumps({
                "id": item["id"], "expert": a.expert, "prediction": prediction,
                "pass_score": score,
            }, ensure_ascii=False) + "\n")
            f.flush()
            newly_processed += 1
            if newly_processed % 10 == 0:
                current = load_jsonl(out)
                accuracy = float(np.mean([float(r["pass_score"]) > 0 for r in current]))
                print(f"expert={a.expert} processed={len(current)}/{len(rows)} pass@1={accuracy:.4f}", flush=True)
                wandb.log({"processed": len(current), "running/pass_at_1": accuracy})

    result = load_jsonl(out)
    if len(result) != len(rows):
        raise RuntimeError(f"incomplete output for {a.expert}: {len(result)}/{len(rows)}")
    accuracy = float(np.mean([float(r["pass_score"]) > 0 for r in result]))
    summary = {"expert": a.expert, "num_examples": len(result), "pass_at_1": accuracy}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.summary.update(summary)
    run.finish()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
