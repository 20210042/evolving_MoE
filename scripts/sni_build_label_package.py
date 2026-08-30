#!/usr/bin/env python3
"""라벨링 raw(evo_multisample_pilot 산출)에서 **정본 라벨 패키지**를 만든다.

**정본 = rep 0 (K=3 중 첫 번째 생성).** 사용자 결정(2026-08-28). 다운스트림(per-expert SFT
분할 등)은 전부 이 라벨을 쓴다.

`--rule majority`는 **부차 라벨**이다(사용자 지시: "subsidiary로 봐"). K=3 중 2회 이상
통과를 solved로 본다. 정본을 대체하지 않고, 라벨 규칙이 union·분포를 얼마나 흔드는지
대조용으로만 쓴다 — acc에서 규칙만 바꿔 WAR 총합이 15→4로 무너진 전례가 있다.

판정은 raw의 `pass` 필드를 그대로 쓴다 = `score_sni_item` = **EM==100 또는 ROUGE-L>70**.
진화가 로스터를 만들 때 쓴 것과 같은 기준이라 로스터와 라벨이 같은 잣대 위에 놓인다.

출력 형식은 `scripts/score_binning.py`와 동일하게 맞춘다(다운스트림 재사용):
  <out>.binned.jsonl        — {"id", "dataset", "solved_by": [...], "n_solved", "per_expert"}
  <out>.binned.summary.json — per-expert pass@1, union UB, coverage 히스토그램

Usage:
  python scripts/sni_build_label_package.py \
      --raw results/sni/binning_seed20212001/train_raw.jsonl \
      --roster results/sni/seed20212001/roster_final.json \
      --out results/sni/binning_seed20212001/train
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

CANONICAL_REP = 0
ARM = "persona"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--out", required=True, help="접두사. .binned.jsonl / .binned.summary.json")
    ap.add_argument("--dataset", default="sni")
    ap.add_argument("--rep", type=int, default=CANONICAL_REP,
                    help="--rule rep일 때 정본으로 쓸 생성 회차")
    ap.add_argument("--rule", default="rep", choices=["rep", "majority", "any", "strict"],
                    help="rep=정본(기본) · 나머지는 부차 대조용")
    a = ap.parse_args()

    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    expert_ids = [p["id"] for p in roster]
    names = {p["id"]: (p.get("name") or p.get("prompt_name") or p["id"]) for p in roster}

    # rule=rep이면 해당 회차만, 아니면 전 회차를 모아 (통과수, 시도수)로 접는다.
    acc: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    n_rows = skipped = 0
    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        if d.get("arm", ARM) != ARM:
            skipped += 1
            continue
        if a.rule == "rep" and int(d.get("rep", 0)) != a.rep:
            skipped += 1
            continue
        cell = acc[d["pid"]][d["cid"]]
        cell[0] += int(d["pass"]); cell[1] += 1
        n_rows += 1

    def fold(hit: int, tries: int) -> int:
        if a.rule in ("rep", "any"):
            return int(hit > 0)
        if a.rule == "majority":
            return int(hit * 2 > tries)
        return int(hit == tries and tries > 0)          # strict

    per = {pid: {eid: fold(v[0], v[1]) for eid, v in cells.items()}
           for pid, cells in acc.items()}

    per_expert_pass = collections.Counter()
    coverage = collections.Counter()
    out_path = Path(a.out + ".binned.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = incomplete = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in sorted(per):
            cells = per[pid]
            if len(cells) != len(expert_ids):
                incomplete += 1          # 생성 누락 — 라벨에서 제외(부분 셀은 쓰지 않는다)
                continue
            solved_by = [e for e in expert_ids if cells.get(e)]
            for e in solved_by:
                per_expert_pass[e] += 1
            coverage[len(solved_by)] += 1
            total += 1
            f.write(json.dumps({
                "id": pid,
                "dataset": a.dataset,
                "solved_by": solved_by,
                "n_solved": len(solved_by),
                "per_expert": {e: int(bool(cells.get(e))) for e in expert_ids},
            }, ensure_ascii=False) + "\n")

    union = sum(c for n, c in coverage.items() if n > 0)
    summary = {
        "raw": a.raw,
        "rule": a.rule,
        "canonical": a.rule == "rep",
        "canonical_rep": a.rep if a.rule == "rep" else None,
        "label_rule": ("score_sni_item: EM==100 or ROUGE-L>70 (진화와 동일)"
                       f" · 집계={a.rule}" + (f"(rep {a.rep})" if a.rule == "rep" else "")),
        "experts": expert_ids,
        "expert_names": names,
        "total_problems": total,
        "incomplete_dropped": incomplete,
        "rows_used": n_rows,
        "rows_skipped_other_rep_or_arm": skipped,
        "per_expert_pass_at_1": {e: (per_expert_pass[e] / total * 100.0 if total else 0.0)
                                 for e in expert_ids},
        "per_expert_solved": dict(per_expert_pass),
        "union_ub_pct": (union / total * 100.0 if total else 0.0),
        "coverage_n_solved": {str(k): coverage[k] for k in sorted(coverage)},
    }
    Path(a.out + ".binned.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{a.dataset}/{a.rule}{'=정본' if a.rule == 'rep' else '=부차'}] {out_path} · "
          f"문제 {total:,} (누락 제외 {incomplete:,}) · "
          f"union {summary['union_ub_pct']:.2f}%")
    for e in expert_ids:
        print(f"   {summary['per_expert_pass_at_1'][e]:6.2f}%  {e:10s} {names[e]}")
    print("   n_solved 분포:", summary["coverage_n_solved"])


if __name__ == "__main__":
    main()
