#!/usr/bin/env python3
"""Route ACC problems to a full 12-expert probability mixture and merge LoRA parameters."""
from __future__ import annotations

import argparse
import json
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-dir", default="checkpoints/router/acc_seed20211004_soft12")
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--agent-mapping", default="export/acc_binning_seed20211004_persona/agent_mapping.json")
    ap.add_argument("--specialist-root", default="checkpoints/expert_sft/acc_seed20211004/cap8_core200")
    ap.add_argument("--luca-root", default="checkpoints/expert_sft/acc_seed20211004/luca_allpass1000")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output", default="results/acc/seed20211004/router_merge/soft12_test.jsonl")
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wandb-project", default="acc-seed20211004-soft-router")
    ap.add_argument("--wandb-entity", default="jongbin-kr-skiml_moe")
    a = ap.parse_args()

    router_dir = Path(a.router_dir)
    cfg = json.loads((router_dir / "router_config.json").read_text())
    experts = cfg["experts"]
    rows = load_jsonl(Path(a.test_jsonl)); rows = rows[:a.limit or None]
    emb = np.load(router_dir / "test_embeddings.npy")[:len(rows)]
    emb_ids = json.loads((router_dir / "test_embeddings.ids.json").read_text())[:len(rows)]
    if emb_ids != [str(r["id"]) for r in rows]:
        raise ValueError("cached test embedding order does not match source")
    x = ((emb - np.asarray(cfg["mean"], np.float32)) / np.asarray(cfg["std"], np.float32)).astype(np.float32)
    router = SoftRouter(cfg["input_dim"], cfg["hidden_dim"], len(experts), cfg["dropout"])
    router.load_state_dict(torch.load(router_dir / "router_state.pt", map_location="cpu", weights_only=True))
    router.eval()
    with torch.no_grad(): weights = router(torch.from_numpy(x)).softmax(-1).numpy()

    specialist_root, luca_root = Path(a.specialist_root), Path(a.luca_root)
    paths = {e: adapter_path(e, specialist_root, luca_root) for e in experts}
    missing = [str(p) for p in paths.values() if not (p / "adapter_config.json").is_file()]
    if missing: raise FileNotFoundError(f"missing adapter(s): {missing}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model = PeftModel.from_pretrained(base, str(paths[experts[0]]), adapter_name=experts[0])
    for e in experts[1:]: model.load_adapter(str(paths[e]), adapter_name=e)
    model.eval()
    mapping = json.loads(Path(a.agent_mapping).read_text())

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = {str(r["id"]) for r in load_jsonl(out)} if out.is_file() else set()
    import wandb
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name="acc_seed20211004_soft12_parameter_merge")
    passed, processed = 0, 0
    with out.open("a", encoding="utf-8") as f:
        for idx, (item, w) in enumerate(zip(rows, weights)):
            if str(item["id"]) in done: continue
            # ``cat`` represents sum_i w_i * B_i A_i exactly.  ``linear`` first
            # combines A/B separately and introduces destructive cross-adapter terms.
            temp = "__router_mixture__"
            model.add_weighted_adapter(experts, [float(v) for v in w], adapter_name=temp, combination_type="cat")
            model.set_adapter(temp)
            lead = experts[int(w.argmax())]
            if lead == "luca":
                messages = build_baseline_prompt(item["instruction"], dataset="acc", model_name=a.base_model,
                                                 starter_code=item.get("starter_code"), domain=item.get("domain"))
            else:
                messages = build_expert_prompt(item["instruction"], mapping[lead]["system_prompt"], dataset="acc",
                                               model_name=a.base_model, starter_code=item.get("starter_code"),
                                               domain=item.get("domain"))
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_length).to("cuda")
            with torch.inference_mode():
                seq = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            pred = tok.decode(seq[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            score = float(score_one(item, pred)); passed += score > 0; processed += 1
            record = {"id": item["id"], "router_weights": {e: float(v) for e, v in zip(experts, w)},
                      "lead_expert": lead, "prediction": pred, "pass_score": score}
            f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush()
            model.delete_adapter(temp)
            if processed % 10 == 0:
                acc = passed / processed
                print(f"processed={processed}/{len(rows)-len(done)} pass@1={acc:.4f}", flush=True)
                wandb.log({"processed": processed, "running/pass_at_1": acc})
    all_rows = load_jsonl(out); acc = np.mean([float(r["pass_score"]) > 0 for r in all_rows])
    run.summary.update({"pass_at_1": float(acc), "num_examples": len(all_rows)}); run.finish()
    print(f"done: {len(all_rows)} examples, pass@1={acc:.4f}, output={out}")


if __name__ == "__main__":
    main()
