#!/usr/bin/env python3
"""두 트랙(12명씩)의 어퍼바운드 비교.

오라클 union을 최고단일과 비교하는 건 "분할이 이론상 얼마나 남았나"만 말해준다.
정작 물어야 할 건 **같은 인원수(12명)로 자른 두 축 중 어느 쪽 천장이 높나**이다.

트랙:
  태스크 축 12명 / 도메인 축 12명 / 무작위 12명(24명 풀에서 추출, 대조군) / luca 단독

무작위 대조군이 필요한 이유: QASC 4조건에서 Random 분할이 Evolved 분할을 이겼다.
"의미 있는 축"이 무작위보다 나은지가 분할의 최소 조건이다.

Usage: python scripts/sni_track_ub.py --out docs/REPORT_track_ub.md
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluation.scorer import score_sni_item_partial  # noqa: E402


def stratum(it: dict) -> str:
    if it.get("task_closed"):
        return "객관식(닫힘)"
    g = it.get("gold_len_median") or 0
    return ("열림 1-3단어" if g < 3 else "열림 3-8단어" if g < 8 else
            "열림 8-15단어" if g < 15 else "서술형 15-40단어" if g < 40 else "서술형 40단어+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--n_random", type=int, default=20, help="무작위 12명 추출 횟수")
    ap.add_argument("--out", default="docs/REPORT_track_ub.md")
    a = ap.parse_args()

    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        full[d["id"]] = d
    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    cat = [p["id"] for p in roster if p["id"].startswith("cat_")]
    dom = [p["id"] for p in roster if p["id"].startswith("dom_")]
    pool = cat + dom

    val = collections.defaultdict(lambda: [0.0, 0])
    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        it = full.get(d["pid"])
        if it is None:
            continue
        v = (float(d["pass"]) if it.get("task_closed")
             else score_sni_item_partial(it, d.get("code") or "") / 100.0)
        c = val[(d["cid"], d["pid"])]
        c[0] += v; c[1] += 1
    mean = {k: v[0] / v[1] for k, v in val.items() if v[1]}
    pids = sorted({p for p in full if ("luca", p) in mean})
    strat = {p: stratum(full[p]) for p in pids}

    def track(experts, ps):
        per = {c: sum(mean[(c, p)] for p in ps) / len(ps) for c in experts}
        best = max(per.values())
        uni = sum(max(mean[(c, p)] for c in experts) for p in ps) / len(ps)
        return best, uni, max(per, key=lambda c: per[c])

    rng = random.Random(0)
    L = ["# 두 트랙의 어퍼바운드 — 같은 12명으로 잘랐을 때", "",
         "오라클 union은 “어느 전문가가 맞힐지 미리 안다”고 가정한 **천장**이다. 실현치가 아니다.",
         "여기서 보는 건 **같은 인원수로 자른 두 축 중 어느 쪽 천장이 높나**이다.",
         "무작위 12명은 대조군 — QASC에서 Random 분할이 Evolved를 이긴 전례가 있어,",
         "“의미 있는 축”이 무작위보다 나은지가 분할의 최소 조건이다.", ""]

    for setname in ("전체", "만장일치 제외"):
        ps = pids
        if setname == "만장일치 제외":
            ps = [p for p in pids
                  if max(mean[(c, p)] for c in pool) - min(mean[(c, p)] for c in pool) > 1e-12]
        luca = sum(mean[("luca", p)] for p in ps) / len(ps)
        cb, cu, cbest = track(cat, ps)
        db, du, dbest = track(dom, ps)
        rus, rbs = [], []
        for _ in range(a.n_random):
            sel = rng.sample(pool, 12)
            b, u, _ = track(sel, ps)
            rus.append(u); rbs.append(b)
        L += [f"## {setname} (n = {len(ps):,})", "",
              "| 트랙 | 최고 단일 | **오라클 union** | union − 최고단일 | luca 대비 |",
              "|---|---:|---:|---:|---:|",
              f"| 태스크 축 12명 | {100*cb:.1f}% ({cbest}) | **{100*cu:.1f}%** | +{100*(cu-cb):.1f}%p | +{100*(cu-luca):.1f}%p |",
              f"| 도메인 축 12명 | {100*db:.1f}% ({dbest}) | **{100*du:.1f}%** | +{100*(du-db):.1f}%p | +{100*(du-luca):.1f}%p |",
              f"| 무작위 12명 ({a.n_random}회) | {100*statistics.mean(rbs):.1f}% | **{100*statistics.mean(rus):.1f}%** "
              f"± {100*statistics.pstdev(rus):.2f} | +{100*(statistics.mean(rus)-statistics.mean(rbs)):.1f}%p | "
              f"+{100*(statistics.mean(rus)-luca):.1f}%p |",
              f"| luca 단독 | {100*luca:.1f}% | — | — | — |", ""]

    # 층별 (만장일치 제외)
    ps_all = [p for p in pids
              if max(mean[(c, p)] for c in pool) - min(mean[(c, p)] for c in pool) > 1e-12]
    L += ["## 층별 union — 만장일치 제외", "",
          "| 층 | n | 태스크 축 | 도메인 축 | 무작위 12 | 차이(태스크−도메인) |",
          "|---|---:|---:|---:|---:|---:|"]
    for st in ["객관식(닫힘)", "열림 1-3단어", "열림 3-8단어", "열림 8-15단어",
               "서술형 15-40단어", "서술형 40단어+"]:
        ps = [p for p in ps_all if strat[p] == st]
        if len(ps) < 20:
            continue
        _, cu, _ = track(cat, ps)
        _, du, _ = track(dom, ps)
        ru = statistics.mean(track(rng.sample(pool, 12), ps)[1] for _ in range(a.n_random))
        L.append(f"| {st} | {len(ps):,} | {100*cu:.1f}% | {100*du:.1f}% | {100*ru:.1f}% | "
                 f"{100*(cu-du):+.1f}%p |")
    L.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"완료 -> {a.out}")


if __name__ == "__main__":
    main()
