#!/usr/bin/env python
"""로스터 프로파일(각 expert의 system prompt) → 벡터.

라우터 입력의 expert 쪽 축. 문제 벡터와 같은 모델·같은 pooling으로 뽑아야
같은 공간에 놓인다(extract_hidden_states.py와 동일 경로).

결과: results/embed_viz_test/<ds>_experts_hs_{last,mean}.npy + _experts_ids.json
      (ids 순서 = 벡터 행 순서 = expert id)

Usage: python3 scripts/extract_expert_profiles.py --dataset sni \
           --roster results/sni/seed20212003/roster_final.json --max_len 4096
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    rc.add_dataset_arg(ap)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    a = ap.parse_args()
    sp = rc.spec(a.dataset)

    roster = json.load(open(a.roster, encoding="utf-8"))
    ids = [p["id"] for p in roster]
    prompts = [p["system_prompt"] for p in roster]
    print(f"expert {len(ids)}명: " + ", ".join(p.get("name", p["id"]) for p in roster))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sp.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        sp.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()

    # 문제 쪽과 같은 형식으로 감싼다 — system 턴에 프로파일만 넣고 user는 비운다.
    texts = [tok.apply_chat_template([{"role": "system", "content": t}],
                                     tokenize=False, add_generation_prompt=True)
             for t in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=a.max_len).to("cuda")
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True,
                   logits_to_keep=1).hidden_states[-1]
    mask = enc.attention_mask.unsqueeze(-1)
    last = hs[:, -1, :].float().cpu().numpy()
    mean = ((hs * mask).sum(1) / mask.sum(1)).float().cpu().numpy()

    out = rc.FEAT_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{sp.name}_experts_hs_last.npy", last)
    np.save(out / f"{sp.name}_experts_hs_mean.npy", mean)
    json.dump(ids, open(out / f"{sp.name}_experts_ids.json", "w"))
    print(f"done -> {out}/{sp.name}_experts_hs_(last|mean).npy  shape {last.shape}")


if __name__ == "__main__":
    main()
