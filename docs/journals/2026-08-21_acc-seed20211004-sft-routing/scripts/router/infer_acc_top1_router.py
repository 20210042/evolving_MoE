#!/usr/bin/env python3
"""Run ACC inference with one exclusively selected LoRA expert per problem."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
ROUTER_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROUTER_SCRIPTS))
from evaluation.scorer import score_one  # noqa: E402
from prompts.coding import build_baseline_prompt, build_expert_prompt  # noqa: E402
from train_acc_soft_router import SoftRouter  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def adapter_path(expert: str, specialist_root: Path, luca_root: Path) -> Path:
    return luca_root if expert == "luca" else specialist_root / expert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-dir", default="checkpoints/router/acc_seed20211004_top1_set")
    ap.add_argument("--embedding-cache-dir", default="checkpoints/router/acc_seed20211004_soft12")
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--test-labels", default="results/acc/seed20211004/binning_test_full.binned.jsonl")
    ap.add_argument("--agent-mapping", default="export/acc_binning_seed20211004_persona/agent_mapping.json")
    ap.add_argument("--specialist-root", default="checkpoints/expert_sft/acc_seed20211004/cap8_core200")
    ap.add_argument("--luca-root", default="checkpoints/expert_sft/acc_seed20211004/luca_allpass1000")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output", default="results/acc/seed20211004/router_top1/set_router_test.jsonl")
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wandb-project", default="acc-seed20211004-top1-router")
    ap.add_argument("--wandb-entity", default="jongbin-kr-skiml_moe")
    a = ap.parse_args()

    router_dir = Path(a.router_dir)
    cfg = json.loads((router_dir / "router_config.json").read_text())
    experts = cfg["experts"]
    rows = load_jsonl(Path(a.test_jsonl))[: a.limit or None]
    cache = Path(a.embedding_cache_dir)
    emb = np.load(cache / "test_embeddings.npy")[: len(rows)]
    emb_ids = json.loads((cache / "test_embeddings.ids.json").read_text())[: len(rows)]
    if emb_ids != [str(r["id"]) for r in rows]:
        raise ValueError("cached test embedding order does not match source")
    x = ((emb - np.asarray(cfg["mean"], np.float32)) / np.asarray(cfg["std"], np.float32)).astype(np.float32)
    router = SoftRouter(cfg["input_dim"], cfg["hidden_dim"], len(experts), cfg["dropout"])
    router.load_state_dict(torch.load(router_dir / "router_state.pt", map_location="cpu", weights_only=True))
    router.eval()
    with torch.no_grad():
        probs = router(torch.from_numpy(x)).softmax(-1).numpy()
    picks = probs.argmax(1)

    labels = load_jsonl(Path(a.test_labels))[: len(rows)]
    label_by_id = {str(r["id"]): r for r in labels}
    if set(label_by_id) != {str(r["id"]) for r in rows}:
        raise ValueError("test source/label ID sets do not match")
    solver_hit = np.asarray(
        [label_by_id[str(row["id"])]["per_expert"][experts[int(pick)]] for row, pick in zip(rows, picks)],
        dtype=np.float32,
    )
    n_solved = np.asarray(
        [sum(label_by_id[str(row["id"])]["per_expert"].values()) for row in rows], dtype=np.int64
    )

    specialist_root, luca_root = Path(a.specialist_root), Path(a.luca_root)
    paths = {e: adapter_path(e, specialist_root, luca_root) for e in experts}
    missing = [str(p) for p in paths.values() if not (p / "adapter_config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing adapter(s): {missing}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model = PeftModel.from_pretrained(base, str(paths[experts[0]]), adapter_name=experts[0])
    for expert in experts[1:]:
        model.load_adapter(str(paths[expert]), adapter_name=expert)
    model.eval()
    mapping = json.loads(Path(a.agent_mapping).read_text())

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {str(r["id"]) for r in load_jsonl(out)} if out.is_file() else set()
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, pick in enumerate(picks):
        if str(rows[idx]["id"]) not in done:
            grouped[experts[int(pick)]].append(idx)

    import wandb

    run = wandb.init(
        project=a.wandb_project,
        entity=a.wandb_entity,
        name="acc_seed20211004_top1_exclusive_inference",
    )
    processed = 0
    with out.open("a", encoding="utf-8") as f:
        for expert in experts:
            indices = grouped.get(expert, [])
            if not indices:
                continue
            model.set_adapter(expert)
            print(f"expert={expert} assigned={len(indices)}", flush=True)
            for idx in indices:
                item, p = rows[idx], probs[idx]
                if expert == "luca":
                    messages = build_baseline_prompt(
                        item["instruction"], dataset="acc", model_name=a.base_model,
                        starter_code=item.get("starter_code"), domain=item.get("domain"),
                    )
                else:
                    messages = build_expert_prompt(
                        item["instruction"], mapping[expert]["system_prompt"], dataset="acc",
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
                record = {
                    "id": item["id"],
                    "selected_expert": expert,
                    "router_confidence": float(p.max()),
                    "router_probabilities": {e: float(v) for e, v in zip(experts, p)},
                    "roster_solver_hit": int(solver_hit[idx]),
                    "n_solved": int(n_solved[idx]),
                    "prediction": prediction,
                    "pass_score": score,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                processed += 1
                if processed % 10 == 0:
                    current = load_jsonl(out)
                    acc = float(np.mean([float(r["pass_score"]) > 0 for r in current]))
                    print(f"new_processed={processed}/{sum(map(len, grouped.values()))} total={len(current)} pass@1={acc:.4f}", flush=True)
                    wandb.log({"processed": len(current), "running/pass_at_1": acc})

    result = load_jsonl(out)
    accuracy = float(np.mean([float(r["pass_score"]) > 0 for r in result]))
    route_counts = {e: sum(r["selected_expert"] == e for r in result) for e in experts}
    contested = (n_solved > 0) & (n_solved < len(experts))
    summary = {
        "pass_at_1": accuracy,
        "num_examples": len(result),
        "roster_top1_hit_all": float(solver_hit.mean()),
        "roster_top1_hit_contested": float(solver_hit[contested].mean()),
        "route_counts": route_counts,
        "parameter_merge": False,
        "exclusive_top1": True,
    }
    run.summary.update(summary)
    run.finish()
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
