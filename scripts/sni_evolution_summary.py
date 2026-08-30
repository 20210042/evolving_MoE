#!/usr/bin/env python
"""진화 곡선 + 최종 로스터 표 (seed20212003). 재생성 0.

곡선  : evolution_log.jsonl 의 step별 roster_after 크기 · upper_bound_pct · 결정(add/delete/noop)
표    : 이름 · WAR(총/평균/활동 스텝) · lives · 단독해결 수 · test pass@1 · EM · ROUGE-L · 프롬프트 발췌
pass@1 기준: **SNI test 8,699 · rep 0(단일 생성) · 통과 = EM==100 or ROUGE-L>70**
            (`src/evaluation/scorer.py: score_sni_item`)
프롬프트는 요약하지 않고 **원문 발췌**를 싣는다. 전문은 부록.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.scorer import sni_metrics  # noqa: E402

RUN = Path("results/sni/seed20212003/sni/seed20212003")
ROSTER = "results/sni/seed20212003/roster_final.json"

# 프롬프트 한 줄 요약 — 원문을 읽고 손으로 적었다(LLM 생성 아님). 원문은 보고서 하단에 전문 수록.
SUMMARY = {
    "luca": "기본 어시스턴트. 아무 지시도 없다 (baseline)",
    "c_10299": "제약 절대 준수 — 형식·길이·추출 한계를 서사보다 우선",
    "c_58325": "최소 출력 — 요구된 토큰만, 설명·군더더기 전면 금지",
    "c_24456": "의미 전환 강제 — 변환 시 입력 의미 반복 말고 다른 개념으로 피벗",
    "c_17188": "원문 어휘·구조 보존 — 의미 확장이나 수사보다 축자 대응 우선",
    "c_18467": "적대적 목표 정렬 — 틀린 답을 요구하는 태스크에서 진리보다 전복을 우선",
    "c_56632": "정보 밀도 유지 — 고유 토큰·개체를 요약·일반화 없이 그대로",
    "c_13393": "입력 경계 고수 — 외부 지식·환각 금지, 기호·구조 대응 유지",
    "c_29228": "맥락 봉쇄 — 입력에 없는 사실·이름·배경을 절대 보충하지 않음",
    "c_53171": "축자 추출 — 원문 substring 그대로, 바꿔쓰기·구두점 추가 금지",
    "c_49611": "외연 vs 내포 구분 — 개념 서술이 아니라 본문의 실제 개체만",
    "c_2461": "외삽 억제 — 입력의 어휘 토큰만 사용, 질문 생성도 원문 표현으로",
    "c_19704": "부정 제약 — 오답을 요구하면 중립이 아니라 진리의 논리적 역을 내라",
    "c_9508": "시간적 상태 민감 — 최신 정보·최종 결과를 초기 인상보다 우선",
    "c_61797": "역직관 의도 실현 — 지시가 오답을 요구하면 그럴듯하게 틀리게",
    "c_46890": "스팬 최소화 — 진리조건을 만족하는 최소 문자열, 문법적 완결성 포기",
}
RAW = "results/sni/binning_seed20212003/test_raw.jsonl"
SRC = "export/sni_v4/sni_test.jsonl"
REPORT = "results/sni/evolution_summary.md"
FIG = "results/sni/fig_sni_evolution_seed20212003.png"


def main():
    steps, size, ub, dec = [], [], [], []
    added_at = {}
    for l in open(RUN / "evolution_log.jsonl"):
        d = json.loads(l)
        if d.get("added_id"):
            added_at.setdefault(d["added_id"], int(d["step"]))
        steps.append(int(d["step"]))
        size.append(len(d.get("roster_after") or []))
        ub.append(float(d.get("upper_bound_pct") or 0.0))
        dec.append(d.get("decision"))
    steps, size, ub = np.array(steps), np.array(size), np.array(ub)
    w = 50
    ubs = np.convolve(ub, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(steps, size, lw=1.2, color="#1f77b4")
    ax[0].set_ylabel("roster size")
    ax[0].grid(alpha=.3)
    ax[0].set_title(f"SNI seed20212003 - {len(steps):,} steps, final roster {size[-1]}")
    ax[1].plot(steps, ub, lw=.4, alpha=.25, color="#888")
    ax[1].plot(steps[w - 1:], ubs, lw=1.6, color="#d62728", label=f"moving avg ({w} steps)")
    ax[1].set_ylabel("batch upper bound (%)")
    ax[1].set_xlabel("step")
    ax[1].grid(alpha=.3)
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    Path(FIG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)

    # --- test pass@1 (rep 0) ---
    src = {json.loads(l)["id"]: json.loads(l) for l in open(SRC)}
    roster = json.load(open(ROSTER))
    ex = [p["id"] for p in roster]
    agg = {c: [0, 0, 0.0, 0.0] for c in ex}     # n, pass, em, rouge
    cache = {}
    for l in open(RAW):
        d = json.loads(l)
        if int(d["rep"]) != 0 or d["cid"] not in agg or d["pid"] not in src:
            continue
        key = (d["pid"], d.get("code") or "")
        m = cache.get(key) or cache.setdefault(key, sni_metrics(src[d["pid"]], key[1]))
        a = agg[d["cid"]]
        a[0] += 1
        a[1] += int(m["exact_match"] >= 100.0 or m["rougeL"] > 70.0)
        a[2] += m["exact_match"]
        a[3] += m["rougeL"]

    L = ["# SNI 진화 요약 — seed20212003 (재생성 0)", "",
         f"- 진화 스텝 {len(steps):,} · 최종 로스터 **{size[-1]}명** · "
         f"배치 upper bound 평균 {ub.mean():.1f}% (마지막 200스텝 {ub[-200:].mean():.1f}%)",
         f"- 결정 분포: add {dec.count('add'):,} · delete {dec.count('delete'):,} · "
         f"noop {dec.count('noop'):,} · swap {dec.count('swap'):,}",
         f"- 곡선: `{FIG}`", "",
         f"![evolution](../{FIG})", "",
         "## 최종 로스터", "",
         "**pass@1 기준** — SNI test 8,699 · **rep 0 단일 생성** · 통과 = `EM==100 or ROUGE-L>70` "
         "(`score_sni_item`). EM·ROUGE-L은 임계 없는 공식 원값이다.", "",
         "| # | 이름 | 추가 step | WAR 총 | WAR 평균 | pass@1 | EM | ROUGE-L | 프롬프트 요약 |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    rows = []
    for i, p in enumerate(roster):
        a = agg[p["id"]]
        n = max(1, a[0])
        rows.append((i, p, 100 * a[1] / n, a[2] / n, a[3] / n))
    for i, p, pk, em, rl in sorted(rows, key=lambda t: -t[2]):
        L.append(f"| {i} | {p['name']} | {added_at.get(p['id'], 0)} | {p['total_war']:,.0f} | "
                 f"{p['average_war']:.2f} | **{pk:.2f}** | {em:.2f} | {rl:.2f} | "
                 f"{SUMMARY.get(p['id'], '—')} |")

    L += ["", "⚠️ `exclusive_solves`는 최근 10건만 남기는 링버퍼라 단독해결 **횟수가 아니다** — 표에 싣지 않았다.",
          "", "## system prompt (원문)", "",
          "요약하지 않았다. 축 판단은 이름이 아니라 프롬프트 본문을 읽고 해야 한다.", ""]
    for i, p, pk, em, rl in sorted(rows, key=lambda t: -t[2]):
        sp = (p.get("system_prompt") or "").strip()
        L += [f"### {i}. {p['name']} — pass@1 {pk:.2f} · WAR평균 {p['average_war']:.2f} · "
              f"step {added_at.get(p['id'], 0)}에 추가", "", "```", sp, "```", ""]
        if p.get("fixes"):
            L += [f"*무엇을 고치려고 만들어졌나*: {p['fixes']}", ""]
    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n[saved] {REPORT} · {FIG}")


if __name__ == "__main__":
    main()
