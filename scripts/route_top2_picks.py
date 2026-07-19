#!/usr/bin/env python3
"""Phase 2: top-2 confidence 라우팅 (무학습).

Phase1의 real conf npz에서 문제별 self-confidence 상위 2 expert를 골라 픽 export.
per-expert 정규화(z-score)로 expert 간 확신 스케일 통일. Phase1 real solve 매트릭스로
top-2 union 커버리지·best-single 대비 보고 = llama에서 우리 라우팅 회수율.

Usage: python scripts/route_top2_picks.py
"""
import json
from pathlib import Path

import numpy as np

R = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
CONF = R / "results/embed_viz_test/qasc_validation_lora_conf.npz"
BINNED = R / "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl"
OUT = R / "results/qasc/seed20210211/qasc_val_top2_picks.json"


def main():
    d = np.load(CONF, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    experts = [str(x) for x in d["experts"]]
    conf = d["conf"].astype(np.float32)                       # (N, E)
    # confidence 직접: raw self-confidence 상위 2 (real llama에서 z-norm보다 우수)
    top2 = conf.argsort(1)[:, -2:][:, ::-1]                   # (N,2) 내림차순

    picks = {ids[i]: [experts[top2[i, 0]], experts[top2[i, 1]]] for i in range(len(ids))}
    OUT.write_text(json.dumps(picks, ensure_ascii=False))
    print(f"picks -> {OUT}  ({len(picks)}문제)")

    # real solve 매트릭스로 커버리지 평가
    binned = {str(json.loads(l)["id"]): json.loads(l) for l in open(BINNED, encoding="utf-8")}
    S = np.array([[binned[i]["per_expert"].get(e, 0) for e in experts] for i in ids], np.float32)
    best = 100 * S.mean(0).max()
    oracle = 100 * (S.sum(1) > 0).mean()
    t2u = 100 * np.mean([S[i, top2[i]].max() for i in range(len(ids))])
    beste = experts[int(S.mean(0).argmax())]
    print(f"\n=== QASC val {len(ids)} (llama LoRA real) ===")
    print(f"  best-single      : {best:.1f}%  ({beste})")
    print(f"  top-2 union(라우팅): {t2u:.1f}%  (confidence z-norm 상위2)")
    print(f"  oracle union(13) : {oracle:.1f}%")
    print(f"  → top-2 − best   : {t2u-best:+.1f}pp")


if __name__ == "__main__":
    main()
