#!/usr/bin/env python3
"""한 배치에 유효 hard error가 몇 개나 나올지 예측한다.

진화에서 hard error = **로스터 전원이 못 푼 문제**(orchestrator.run_batch의 `not any_solved`).
스카우트는 이것만 본다. 너무 많으면 스카우트가 무엇을 고를지 알 수 없고, 너무 적으면
새 전문가를 만들 근거가 없다.

로스터는 진화 중에 1명(LUCA)에서 시작해 자란다. 그래서 로스터 크기별로 전부 낸다.
그리고 **판정 기준**에 따라 결과가 완전히 달라진다 —
  EM(현행)        : 열린 태스크는 구조적으로 전원 0점 → 거의 전부 hard error가 된다
  ROUGE-L 임계    : 부분점수로 판정 (soft_partial 계열)
둘 다 내서 배치 구성 결정에 쓴다.

Usage: python scripts/sni_hard_error_rate.py --out docs/REPORT_hard_error_rate.md
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


ORDER = ["객관식(닫힘)", "열림 1-3단어", "열림 3-8단어", "열림 8-15단어",
         "서술형 15-40단어", "서술형 40단어+"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--rouge_thr", type=float, default=0.5)
    ap.add_argument("--draws", type=int, default=30, help="로스터 무작위 구성 반복")
    ap.add_argument("--out", default="docs/REPORT_hard_error_rate.md")
    a = ap.parse_args()

    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        full[d["id"]] = d
    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    pool = [p["id"] for p in roster if p["id"] != "luca"]

    em = collections.defaultdict(lambda: [0, 0])
    rg = collections.defaultdict(lambda: [0.0, 0])
    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        it = full.get(d["pid"])
        if it is None:
            continue
        k = (d["cid"], d["pid"])
        e = em[k]; e[0] += int(d["pass"]); e[1] += 1
        r = rg[k]; r[0] += score_sni_item_partial(it, d.get("code") or "") / 100.0; r[1] += 1
    solved_em = {k: (v[0] / v[1]) > 0.5 for k, v in em.items() if v[1]}
    solved_rg = {k: (v[0] / v[1]) > a.rouge_thr for k, v in rg.items() if v[1]}
    pids = sorted({k[1] for k in solved_em})
    strat = {p: stratum(full[p]) for p in pids}

    def hard_frac(experts, table, ps):
        n = sum(1 for p in ps if not any(table.get((c, p), False) for c in experts))
        return n / len(ps)

    rng = random.Random(0)
    L = ["# 한 배치의 유효 hard error 예측", "",
         "hard error = **로스터 전원이 못 푼 문제**. 스카우트는 이것만 본다.",
         "너무 많으면 무엇을 고를지 알 수 없고, 너무 적으면 새 전문가를 만들 근거가 없다.",
         f"판정: EM(현행) / ROUGE-L > {a.rouge_thr}", ""]

    L += ["## 로스터 크기별 hard error 비율 (전수 87,028 기준)", "",
          "| 로스터 | EM 기준 | ROUGE 기준 | batch 50 | batch 100 | batch 200 |",
          "|---|---:|---:|---:|---:|---:|"]
    rows = [("luca 단독 (진화 시작점)", ["luca"])]
    for k in (2, 4, 8, 12):
        rows.append((f"무작위 {k}명 ({a.draws}회 평균)", k))
    rows.append(("태스크 축 12명", [p for p in pool if p.startswith("cat_")]))
    rows.append(("도메인 축 12명", [p for p in pool if p.startswith("dom_")]))
    rows.append(("전체 25명", [p["id"] for p in roster]))
    for label, spec in rows:
        if isinstance(spec, int):
            fe = statistics.mean(hard_frac(rng.sample(pool, spec), solved_em, pids)
                                 for _ in range(a.draws))
            fr = statistics.mean(hard_frac(rng.sample(pool, spec), solved_rg, pids)
                                 for _ in range(a.draws))
        else:
            fe = hard_frac(spec, solved_em, pids)
            fr = hard_frac(spec, solved_rg, pids)
        L.append(f"| {label} | {100*fe:.1f}% | {100*fr:.1f}% | {50*fe:.0f} / {50*fr:.0f} | "
                 f"{100*fe:.0f} / {100*fr:.0f} | {200*fe:.0f} / {200*fr:.0f} |")
    L += ["", "> batch 열은 `EM / ROUGE` 기준 예상 hard error 개수.", ""]

    L += ["## 층별 — 어떤 문제가 스카우트에게 흘러가나 (전체 25명)", "",
          "| 층 | 문제수 | 코퍼스 비중 | EM hard% | ROUGE hard% | EM hard 중 이 층 비중 |",
          "|---|---:|---:|---:|---:|---:|"]
    allx = [p["id"] for p in roster]
    tot_hard_em = sum(1 for p in pids if not any(solved_em.get((c, p), False) for c in allx))
    for st in ORDER:
        ps = [p for p in pids if strat[p] == st]
        if not ps:
            continue
        he = sum(1 for p in ps if not any(solved_em.get((c, p), False) for c in allx))
        hr = sum(1 for p in ps if not any(solved_rg.get((c, p), False) for c in allx))
        L.append(f"| {st} | {len(ps):,} | {100*len(ps)/len(pids):.1f}% | {100*he/len(ps):.1f}% | "
                 f"{100*hr/len(ps):.1f}% | **{100*he/max(1,tot_hard_em):.1f}%** |")
    L += ["", "> 마지막 열이 핵심 — 스카우트가 보는 hard error가 어느 층에서 오는가.",
          "> 한 층이 과반이면 진화는 그 층만 보고 분화한다.", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"완료 -> {a.out}")


if __name__ == "__main__":
    main()
