#!/usr/bin/env python3
"""라우터 입력용 base LLM hidden-state 추출 (어댑터 off, forward-only).

base llama3.1-8b에 문제를 baseline GEN 프롬프트로 넣고 마지막 레이어 hidden-state를
pooling → 문제당 벡터. last-token / mean 둘 다 저장.
결과: results/embed_viz_test/<ds>_<split>_hs_{last,mean}.npy + _ids.json

Usage: python scripts/extract_hidden_states.py --dataset qasc --split train|validation
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
OUT = rc.FEAT_DIR


def main():
    ap = argparse.ArgumentParser()
    rc.add_dataset_arg(ap)
    ap.add_argument("--split", default=None, help="기본값은 데이터셋의 train split")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    a = ap.parse_args()
    sp = rc.spec(a.dataset)
    split = a.split or sp.train_split
    MODEL = sp.base_model
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(REPO / sp.src[split], encoding="utf-8")]
    ids = [str(r["id"]) for r in rows]
    sys_p, user_t = sp.gen

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()

    def prompt(r):
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user_t.format(instruction=r["instruction"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    last_all, mean_all = [], []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts = [prompt(r) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_len).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]  # (B,T,H)
        mask = enc.attention_mask.unsqueeze(-1)                              # (B,T,1)
        last = hs[:, -1, :]                                                  # left-pad → 마지막=진짜 마지막 토큰
        mean = (hs * mask).sum(1) / mask.sum(1)
        last_all.append(last.float().cpu().numpy())
        mean_all.append(mean.float().cpu().numpy())
        if (i // a.batch) % 20 == 0:
            print(f"{i+len(chunk)}/{len(rows)}", flush=True)

    np.save(rc.feat_path(sp, split, "hs_last"), np.concatenate(last_all))
    np.save(rc.feat_path(sp, split, "hs_mean"), np.concatenate(mean_all))
    json.dump(ids, open(rc.feat_path(sp, split, "hs_ids"), "w"))
    print(f"done -> {OUT}/{sp.name}_{split}_hs_(last|mean).npy  "
          f"shape {np.concatenate(last_all).shape}")


if __name__ == "__main__":
    main()
