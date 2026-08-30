#!/usr/bin/env python3
"""evo_repro_exclusive128 원본(128문제 × 11명 × 5회)을 binned 포맷으로 변환.

interaction_lowrank_test.py가 읽는 형식(`per_expert`: expert -> 0/1)으로 맞춘다.
라벨은 다수결(5회 중 3회 이상 통과 = 1). K=5라 단일 드로우보다 라벨 노이즈가 훨씬 작다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/acc/evo_repro_exclusive128.raw.jsonl"
OUT = ROOT / "results/acc/evo_repro128_majority.binned.jsonl"

cnt: dict = defaultdict(lambda: defaultdict(list))
for line in open(SRC, encoding="utf-8"):
    r = json.loads(line)
    if r.get("arm") != "persona":
        continue
    cnt[r["pid"]][r["cid"]].append(int(r["pass"]))

experts = sorted({c for p in cnt for c in cnt[p]})
n_pos = 0
with open(OUT, "w", encoding="utf-8") as f:
    for pid in sorted(cnt):
        per = {e: int(sum(cnt[pid].get(e, [])) * 2 >= len(cnt[pid].get(e, [1])))
               for e in experts}
        n_pos += sum(per.values())
        f.write(json.dumps({"id": pid, "dataset": "acc",
                            "n_solved": sum(per.values()),
                            "per_expert": per}, ensure_ascii=False) + "\n")
print(f"{len(cnt)}문제 × {len(experts)}명 · 양성 셀 {n_pos} -> {OUT}")
