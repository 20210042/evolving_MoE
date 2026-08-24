#!/usr/bin/env python3
"""SNI 프로브 v2 — 두 축 로스터의 반복측정 이원 ANOVA.

요인 = 문제 × expert, 반복 = K회 재생성. K가 있으므로 **오차(재생성 노이즈)를 따로 떼어**
상호작용과 구분할 수 있다.

  SS_문제      : 문제마다 난이도가 다른가
  SS_expert    : 어떤 expert가 전반적으로 잘하는가
  SS_상호작용  : 문제에 따라 잘하는 expert가 갈리는가   ← 로스터가 존재할 이유
  SS_오차      : 같은 (expert,문제)를 다시 돌렸을 때의 흔들림

로스터별로 따로 돌린다(category 12 / domain 12 / 전체 25). 그리고 두 문제집합에서 각각:
  전체            : 87,028문제 전부
  만장일치 제외   : 전원 정답 / 전원 오답인 문제를 뺀 것 — 이 문제들은 어느 로스터에서도
                    아무 차이를 못 만들므로 남겨두면 상호작용 비중이 희석된다.

점수는 닫힘 EM(0/1) / 열림 ROUGE-L(0~1). 만장일치 판정은 닫힘=EM 동일, 열림=분산 0.

Usage:
  python scripts/sni_anova.py --out docs/REPORT_axis_anova.md
"""
from __future__ import annotations

import argparse
import collections
import json
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


def anova(cells: dict, experts: list, pids: list) -> dict:
    """cells[(cid,pid)] = (합, 제곱합, n). 반복측정 이원 ANOVA."""
    E, N = len(experts), len(pids)
    if E < 2 or N < 2:
        return {}
    # 셀 평균 · 셀 내 제곱합(오차)
    mean, ss_err, n_tot = {}, 0.0, 0
    for c in experts:
        for p in pids:
            v = cells.get((c, p))
            if v is None or v[2] == 0:
                return {}
            s, sq, n = v
            m = s / n
            mean[(c, p)] = m
            ss_err += sq - n * m * m          # Σ(x-x̄)²
            n_tot += n
    K = n_tot / (E * N)
    grand = sum(mean.values()) / (E * N)
    pm = {p: sum(mean[(c, p)] for c in experts) / E for p in pids}
    em = {c: sum(mean[(c, p)] for p in pids) / N for c in experts}
    ss_p = K * E * sum((pm[p] - grand) ** 2 for p in pids)
    ss_e = K * N * sum((em[c] - grand) ** 2 for c in experts)
    ss_i = K * sum((mean[(c, p)] - pm[p] - em[c] + grand) ** 2
                   for c in experts for p in pids)
    tot = ss_p + ss_e + ss_i + ss_err
    if tot <= 0:
        return {}
    # F: 상호작용 / 오차 (df: (E-1)(N-1) vs EN(K-1))
    df_i, df_err = (E - 1) * (N - 1), E * N * max(1e-9, K - 1)
    f_i = (ss_i / df_i) / (ss_err / df_err) if ss_err > 0 else float("inf")
    f_e = (ss_e / (E - 1)) / (ss_i / df_i) if ss_i > 0 else float("inf")
    return {"N": N, "E": E, "K": K, "grand": grand,
            "p": 100 * ss_p / tot, "e": 100 * ss_e / tot,
            "i": 100 * ss_i / tot, "err": 100 * ss_err / tot,
            "F_i": f_i, "F_e": f_e}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--out", default="docs/REPORT_axis_anova.md")
    a = ap.parse_args()

    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        full[d["id"]] = d
    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    cat = [p["id"] for p in roster if p["id"].startswith("cat_")]
    dom = [p["id"] for p in roster if p["id"].startswith("dom_")]
    allx = [p["id"] for p in roster]

    cells = collections.defaultdict(lambda: [0.0, 0.0, 0])
    em_pass = collections.defaultdict(lambda: [0, 0])
    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        it = full.get(d["pid"])
        if it is None:
            continue
        v = (float(d["pass"]) if it.get("task_closed")
             else score_sni_item_partial(it, d.get("code") or "") / 100.0)
        c = cells[(d["cid"], d["pid"])]
        c[0] += v; c[1] += v * v; c[2] += 1
        e = em_pass[(d["cid"], d["pid"])]
        e[0] += int(d["pass"]); e[1] += 1

    pids = sorted({k[1] for k in cells})
    strat = {p: stratum(full[p]) for p in pids}

    def contested(experts: list) -> list:
        """만장일치 제외: 그 로스터 안에서 아무 차이도 안 나는 문제를 뺀다."""
        out = []
        for p in pids:
            vals = [cells[(c, p)][0] / cells[(c, p)][2] for c in experts]
            if max(vals) - min(vals) > 1e-12:
                out.append(p)
        return out

    L = ["# SNI 프로브 v2 — 두 축 로스터의 반복측정 이원 ANOVA", "",
         "요인 = 문제 × expert, 반복 = K=3회 재생성. K가 있어 **오차(재생성 흔들림)를 따로 떼어**",
         "상호작용과 구분한다. 점수 = 닫힘 EM / 열림 ROUGE-L.", "",
         "- **상호작용**이 로스터가 존재할 이유다 — 문제에 따라 잘하는 expert가 갈리는 몫.",
         "- **만장일치 제외** = 그 로스터 안에서 전원이 똑같은 점수를 낸 문제를 뺀 것.", ""]

    for label, experts in (("category 축 (12명)", cat), ("domain 축 (12명)", dom),
                           ("전체 (25명)", allx)):
        L += [f"## {label}", "",
              "| 문제집합 | n(문제) | 평균점수 | 문제 | expert | **상호작용** | 오차(재생성) | F(상호작용/오차) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for setname, ps in (("전체", pids), ("만장일치 제외", contested(experts))):
            r = anova(cells, experts, ps)
            if not r:
                continue
            L.append(f"| {setname} | {r['N']:,} | {100*r['grand']:.1f}% | {r['p']:.1f}% | "
                     f"{r['e']:.2f}% | **{r['i']:.1f}%** | {r['err']:.1f}% | {r['F_i']:.1f} |")
        L.append("")

    # 층별 (전체 25명, 만장일치 제외)
    L += ["## 층별 — 전체 25명 · 만장일치 제외", "",
          "| 층 | n(문제) | 평균점수 | 문제 | expert | **상호작용** | 오차 | F |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    con = set(contested(allx))
    for st in ["객관식(닫힘)", "열림 1-3단어", "열림 3-8단어", "열림 8-15단어",
               "서술형 15-40단어", "서술형 40단어+"]:
        ps = [p for p in pids if strat[p] == st and p in con]
        r = anova(cells, allx, ps)
        if not r:
            continue
        L.append(f"| {st} | {r['N']:,} | {100*r['grand']:.1f}% | {r['p']:.1f}% | {r['e']:.2f}% | "
                 f"**{r['i']:.1f}%** | {r['err']:.1f}% | {r['F_i']:.1f} |")
    L += ["", "> F(상호작용/오차) > 1 이면 상호작용이 재생성 노이즈보다 크다 = 실재한다.", ""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"ANOVA 완료 -> {a.out}")


if __name__ == "__main__":
    main()
