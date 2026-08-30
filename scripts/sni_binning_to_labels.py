#!/usr/bin/env python
"""binning raw 로그 → 라우터 학습용 `binning_labels.jsonl`.

`router_common.load_binning()` / `align()`이 읽는 형식에 맞춘다:
    {"id", "dataset", "solved_by", "n_solved", "per_expert": {cid: 값}}

`per_expert` 값은 **K회 중 통과 비율 p̂** 를 그대로 넣는다(소프트 타깃, 사용자 결정).
K=3이면 {0, 1/3, 2/3, 1}. `BCEWithLogitsLoss`는 소프트 타깃을 그대로 받는다.
soft WAR `(E−n)/(E−1)`가 쓰는 정보와 같은 층이다.

`solved_by` / `n_solved`는 하드 라벨이 필요한 스크립트를 위한 보조 필드이고
**majority(p̂ > 0.5)** 규칙으로 만든다 — `evo_multisample_pilot.py`의 라벨 규칙 중 하나.
어떤 규칙을 썼는지가 수치를 바꾸므로 `rule` 필드에 같이 적는다.

Usage:
  python3 scripts/sni_binning_to_labels.py \
      --raw results/sni/binning_seed20212003/train_raw.jsonl \
      --out export/sni_binning_seed20212003/binning_labels.jsonl
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="sni")
    ap.add_argument("--drop", nargs="*", default=[], help="제외할 expert id")
    a = ap.parse_args()

    cell = defaultdict(lambda: defaultdict(list))
    order = []
    for line in open(a.raw, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:      # 실행 중이면 마지막 줄이 잘려 있을 수 있다
            continue
        if r["cid"] in a.drop:
            continue
        if r["pid"] not in cell:
            order.append(r["pid"])
        cell[r["pid"]][r["cid"]].append(int(r["pass"]))

    experts = sorted({c for v in cell.values() for c in v})
    K = max(len(v) for pv in cell.values() for v in pv.values())
    n_part = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for pid in order:
            pv = cell[pid]
            if len(pv) != len(experts) or any(len(v) != K for v in pv.values()):
                n_part += 1          # 미완결 문제는 버린다(라벨이 expert마다 다른 K가 된다)
                continue
            p = {c: sum(pv[c]) / K for c in experts}
            solved = [c for c in experts if p[c] > 0.5]
            f.write(json.dumps({
                "id": pid, "dataset": a.dataset, "k": K, "rule": "majority",
                "solved_by": solved, "n_solved": len(solved),
                "per_expert": p,
            }, ensure_ascii=False) + "\n")
    print(f"{a.out}: 문제 {len(order)-n_part:,} · expert {len(experts)} · K={K} "
          f"· 미완결 제외 {n_part:,}")


if __name__ == "__main__":
    main()
