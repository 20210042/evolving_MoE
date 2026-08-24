#!/usr/bin/env python3
"""상호작용 중 **태스크 축이 설명하는 몫**과 **오라클 헤드룸**.

ANOVA는 "문제마다 잘하는 전문가가 다르다"(상호작용)가 크다고 말한다. 하지만 그게 곧
"태스크 라벨이 그걸 설명한다"는 뜻은 아니다. 라벨이 안 붙은 재현 가능한 구조일 수도 있다.
학습(가중치 변경)으로 그 몫을 가져올 수 있느냐는 **축 정렬 비율**에 달려 있다.

  상호작용 잔차 r(c,p) = mean(c,p) − 문제평균(p) − expert평균(c) + 전체평균
  이 r을 "c가 p의 구역 담당인가" 지시변수(문제·expert별로 중심화)에 사영해
  SS_축정렬 / SS_상호작용 을 낸다.

동시에 헤드룸도 낸다 — 오라클 union(누구든 맞히면 정답) vs 최고 단일 전문가 vs luca.
분할·학습이 노릴 수 있는 최대치가 union이고, 넘어야 할 선이 best-single이다.

Usage: python scripts/sni_interaction_share.py --out docs/REPORT_interaction_share.md
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--out", default="docs/REPORT_interaction_share.md")
    a = ap.parse_args()

    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        full[d["id"]] = d
    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    axes = {"태스크": [p["id"] for p in roster if p["id"].startswith("cat_")],
            "도메인": [p["id"] for p in roster if p["id"].startswith("dom_")]}
    owner = {"태스크": {}, "도메인": {}}
    for p in roster:
        if p["id"] == "luca":
            continue
        k = "태스크" if p["id"].startswith("cat_") else "도메인"
        owner[k][p["strengths"]] = p["id"]

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
    pids = sorted({k[1] for k in mean})

    L = ["# 상호작용 중 축이 설명하는 몫 · 오라클 헤드룸", "",
         "ANOVA의 상호작용(“문제마다 잘하는 전문가가 다르다”)이 크다는 것과, "
         "**그것을 태스크 라벨로 설명할 수 있다**는 것은 다른 주장이다.",
         "학습으로 그 몫을 가져오려면 축이 상호작용과 정렬돼 있어야 한다.", ""]

    L += ["## 축 정렬 비율", "",
          "| 축 | 문제집합 | n | 상호작용 SS | 축정렬 SS | **정렬 비율** |",
          "|---|---|---:|---:|---:|---:|"]
    for axname, experts in axes.items():
        idx = "category" if axname == "태스크" else "sni_domain"
        for setname in ("전체", "만장일치 제외"):
            ps = []
            for p in pids:
                vals = [mean.get((c, p)) for c in experts]
                if any(v is None for v in vals):
                    continue
                if setname == "만장일치 제외" and max(vals) - min(vals) <= 1e-12:
                    continue
                ps.append(p)
            if len(ps) < 2:
                continue
            E, N = len(experts), len(ps)
            g = sum(mean[(c, p)] for c in experts for p in ps) / (E * N)
            pm = {p: sum(mean[(c, p)] for c in experts) / E for p in ps}
            em = {c: sum(mean[(c, p)] for p in ps) / N for c in experts}
            # 상호작용 잔차와 중심화된 지시변수
            ind = {}
            for p in ps:
                own = owner[axname].get(full[p][idx])
                for c in experts:
                    ind[(c, p)] = 1.0 if c == own else 0.0
            ipm = {p: sum(ind[(c, p)] for c in experts) / E for p in ps}
            iem = {c: sum(ind[(c, p)] for p in ps) / N for c in experts}
            ig = sum(ind.values()) / (E * N)
            num = den = ss_i = 0.0
            for c in experts:
                for p in ps:
                    r = mean[(c, p)] - pm[p] - em[c] + g
                    x = ind[(c, p)] - ipm[p] - iem[c] + ig
                    ss_i += r * r; num += r * x; den += x * x
            ss_axis = (num * num / den) if den > 0 else 0.0
            L.append(f"| {axname} | {setname} | {N:,} | {ss_i:.1f} | {ss_axis:.1f} | "
                     f"**{100*ss_axis/ss_i:.2f}%** |")
    L += ["", "> 정렬 비율 = 상호작용 중 “자기 구역이라서” 생긴 몫. "
          "나머지는 라벨이 안 붙은 구조이고, **그 축으로 전문가를 학습시켜도 가져올 수 없다.**", ""]

    # 헤드룸 — 25명 전체
    allx = [p["id"] for p in roster]
    L += ["## 헤드룸 — 분할·학습이 노릴 수 있는 최대치", "",
          "| 문제집합 | n | luca 단독 | 최고 단일 전문가 | 오라클 union | union − best |",
          "|---|---:|---:|---:|---:|---:|"]
    for setname in ("전체", "만장일치 제외"):
        ps = []
        for p in pids:
            vals = [mean.get((c, p)) for c in allx]
            if any(v is None for v in vals):
                continue
            if setname == "만장일치 제외" and max(vals) - min(vals) <= 1e-12:
                continue
            ps.append(p)
        if not ps:
            continue
        per = {c: sum(mean[(c, p)] for p in ps) / len(ps) for c in allx}
        best = max(per, key=lambda c: per[c])
        uni = sum(max(mean[(c, p)] for c in allx) for p in ps) / len(ps)
        L.append(f"| {setname} | {len(ps):,} | {100*per['luca']:.1f}% | "
                 f"{100*per[best]:.1f}% ({best}) | {100*uni:.1f}% | "
                 f"**+{100*(uni-per[best]):.1f}%p** |")
    L += ["", "> union은 **오라클**이다(어느 전문가가 맞힐지 미리 안다고 가정). 실현 가능한 상한이 아니라 "
          "천장이고, 라우팅이 완벽할 때만 닿는다.", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"완료 -> {a.out}")


if __name__ == "__main__":
    main()
