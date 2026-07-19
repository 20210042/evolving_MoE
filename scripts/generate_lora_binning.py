#!/usr/bin/env python3
"""Phase 1: 우리 아키텍처를 llama backbone에서 실측.

13개 llama LoRA 어댑터를 base에 붙여 QASC 문제를 각자 '실제 생성'하고, 같은
generate 호출에서 첫 스텝 answer-logit(A-H)도 잡아 real confidence를 동시 산출.
- 출력① inference_<split>_lora13.jsonl : {id,dataset,ground_truth,scoring_kind,instruction,
        expert_outputs:{13 expert: 생성텍스트}}  (score_binning으로 채점)
- 출력② qasc_<split>_lora_conf.npz : conf/pred (N×13, real)

extract_expert_confidence.py의 base+13어댑터 로드·set_adapter 골격을 생성 모드로 확장.
Usage: python scripts/generate_lora_binning.py --split validation [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
sys.path.insert(0, str(REPO / "src"))
from prompts import baseline_prompts as bp  # noqa: E402

OUTNPZ = REPO / "results" / "embed_viz_test"
RESDIR = REPO / "results" / "qasc" / "seed20210211"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
CKPT = REPO / "checkpoints/expert_sft/qasc_seed20210211_cap10"
SRC = {"train": "export/qasc/qasc_train.jsonl", "validation": "export/qasc/qasc_validation.jsonl"}
LETTERS = list("ABCDEFGH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="스모크용: 앞 N문제만")
    ap.add_argument("--ckpt", default=str(CKPT), help="어댑터 로스터 디렉토리")
    ap.add_argument("--tag", default="lora13", help="출력 파일 태그 (inference_<split>_<tag>.jsonl)")
    a = ap.parse_args()
    OUTNPZ.mkdir(parents=True, exist_ok=True)
    RESDIR.mkdir(parents=True, exist_ok=True)

    CKPT_DIR = Path(a.ckpt)
    experts = sorted([p.name for p in CKPT_DIR.iterdir()
                      if p.is_dir() and (p / "adapter_config.json").is_file()])
    print("experts:", experts, flush=True)
    rows = [json.loads(l) for l in open(REPO / SRC[a.split], encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    ids = [str(r["id"]) for r in rows]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / experts[0]), adapter_name=experts[0])
    for e in experts[1:]:
        model.load_adapter(str(CKPT_DIR / e), adapter_name=e)
    model.eval()

    letter_ids = []
    for ch in LETTERS:
        cand = set()
        for s in (ch, " " + ch):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                cand.add(t[0])
        letter_ids.append(sorted(cand))

    def prompt(r):
        msgs = [{"role": "system", "content": bp.QASC_GEN_SYSTEM},
                {"role": "user", "content": bp.QASC_GEN_USER.format(instruction=r["instruction"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    texts = [prompt(r) for r in rows]
    n, E = len(rows), len(experts)
    conf = np.zeros((n, E), np.float32)
    pred = np.zeros((n, E), np.int64)
    gen = [[None] * E for _ in range(n)]   # gen[i][ei] = 생성 텍스트

    for ei, e in enumerate(experts):
        model.set_adapter(e)
        for i in range(0, n, a.batch):
            enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=tok.pad_token_id)
            # 첫 생성 토큰 로짓 → A-H confidence
            lg = out.scores[0].float()                              # (B,V)
            ll = torch.stack([lg[:, ids_].max(1).values for ids_ in letter_ids], dim=1)
            pr = torch.softmax(ll, dim=1)
            conf[i:i + a.batch, ei] = pr.max(1).values.cpu().numpy()
            pred[i:i + a.batch, ei] = pr.argmax(1).cpu().numpy()
            # 생성 텍스트(프롬프트 이후만)
            new_tok = out.sequences[:, enc.input_ids.shape[1]:]
            dec = tok.batch_decode(new_tok, skip_special_tokens=True)
            for j, t in enumerate(dec):
                gen[i + j][ei] = t.strip()
        print(f"  {e} done", flush=True)

    # 출력① binning jsonl (score_binning 호환)
    outjsonl = RESDIR / f"inference_{a.split}_{a.tag}.jsonl"
    with open(outjsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            rec = {"id": ids[i], "dataset": "qasc",
                   "ground_truth": r.get("ground_truth"), "scoring_kind": r.get("scoring_kind", "qasc"),
                   "instruction": r.get("instruction"),
                   "expert_outputs": {e: gen[i][ei] for ei, e in enumerate(experts)}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 출력② real conf npz
    np.savez(OUTNPZ / f"qasc_{a.split}_{a.tag}_conf.npz",
             conf=conf, pred=pred, ids=np.array(ids), experts=np.array(experts))
    print(f"done -> {outjsonl}  &  qasc_{a.split}_{a.tag}_conf.npz  ({n}×{E})")


if __name__ == "__main__":
    main()
