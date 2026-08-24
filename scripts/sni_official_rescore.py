#!/usr/bin/env python3
"""공식 채점으로 프로브 raw 전량 재채점 + 우리 이식본이 공식과 일치하는지 대조.

지금까지 보고한 SNI 수치는 전부 내가 만든 채점 기준(task_closed 분기 · 자체 LCS ·
_sni_extract 형식제거) 위에 있었다. 공식은 그런 분기가 없고 모든 인스턴스에 EM·ROUGE-L을
둘 다 계산한다(eval/automatic/evaluation.py). 그 차이를 먼저 숫자로 낸다.

1단계: 공식 evaluation.py를 직접 로드해 우리 sni_metrics와 케이스별 대조 (불일치면 즉시 중단)
2단계: results/sni/probe_v2_raw.jsonl 전량을 공식 EM·ROUGE로 재채점, 층별 집계

Usage: python scripts/sni_official_rescore.py --out docs/REPORT_official_rescore.md
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OFFICIAL = "/data5/jaehoonjeong/datasets/natural-instructions/eval/automatic/evaluation.py"


def load_official():
    """공식 evaluation.py는 `from rouge import rouge_scorer`를 쓴다 — README가 svn export로
    벤더링하라는 사본이다. 우리가 설치한 rouge_score가 같은 라이브러리이므로 별칭을 건다."""
    import rouge_score
    sys.modules.setdefault("rouge", rouge_score)
    spec = importlib.util.spec_from_file_location("official_eval", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stratum(it: dict) -> str:
    g = it.get("gold_len_median") or 0
    if it.get("task_closed"):
        return "닫힌 라벨집합"
    return ("gold 1-3단어" if g < 3 else "gold 3-8단어" if g < 8 else
            "gold 8-15단어" if g < 15 else "gold 15-40단어" if g < 40 else "gold 40단어+")


ORDER = ["닫힌 라벨집합", "gold 1-3단어", "gold 3-8단어", "gold 8-15단어",
         "gold 15-40단어", "gold 40단어+"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--out", default="docs/REPORT_official_rescore.md")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 상한(0=전량)")
    a = ap.parse_args()

    off = load_official()
    from evaluation.scorer import sni_metrics

    # --- 1단계: 공식과 대조 -------------------------------------------------
    cases = [
        (["Many people are killed by cars . "], "Many people were killed in car accidents."),
        (["Parmesan Broccoli Balls"], "Broccoli Stuffing Balls"),
        (["User Choice/Control"], "(3) User Choice/Control"),
        (["b"], "b ) 42"),
        (["[17, -18, 12, -18, -3, 1]"], "[17, -17, 12, -17, -2, 1]"),
        (["Far away"], "When they were far away from him."),
        (["Happy", "Not happy"], "Happy"),
        ([""], ""),
    ]
    L = ["# 공식 채점 이식 대조 · 프로브 전량 재채점", "",
         f"공식 구현: `{OFFICIAL}`", "",
         "## 1. 우리 이식본이 공식과 같은가", "",
         "| gold | prediction | 공식 EM | 우리 EM | 공식 ROUGE-L | 우리 ROUGE-L |",
         "|---|---|---:|---:|---:|---:|"]
    all_ok = True
    for gts, pred in cases:
        oem = 100.0 * off.metric_max_over_ground_truths(
            off.exact_match, prediction=pred, ground_truths=gts)
        orl = 100.0 * off.metric_max_over_ground_truths(
            off.rouge, prediction=pred, ground_truths=gts)
        m = sni_metrics({"ground_truth": gts}, pred)
        ok = abs(oem - m["exact_match"]) < 1e-6 and abs(orl - m["rougeL"]) < 1e-4
        all_ok &= ok
        L.append(f"| `{gts[0][:30]}` | `{pred[:30]}` | {oem:.1f} | {m['exact_match']:.1f} | "
                 f"{orl:.2f} | {m['rougeL']:.2f} |{'' if ok else ' ⚠️불일치'}")
    L += ["", f"**공식과 완전 일치: {all_ok}**", ""]
    if not all_ok:
        L.append("> ⚠️ 불일치가 있으므로 아래 재채점 결과를 쓰면 안 된다.")
        Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        raise SystemExit("공식과 불일치 — 중단")

    # --- 2단계: 전량 재채점 -------------------------------------------------
    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        full[d["id"]] = d

    em_sum = collections.Counter(); rl_sum = collections.Counter(); cnt = collections.Counter()
    old_pass = collections.Counter()
    n = 0
    with open(a.raw, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            it = full.get(d["pid"])
            if it is None:
                continue
            st = stratum(it)
            m = sni_metrics(it, d.get("code") or "")
            em_sum[st] += m["exact_match"]; rl_sum[st] += m["rougeL"]; cnt[st] += 1
            old_pass[st] += 100.0 * int(d["pass"])     # 이전 채점(내 EM)의 통과 여부
            n += 1
            if a.limit and n >= a.limit:
                break

    L += ["## 2. 프로브 생성물 전량 재채점", "",
          f"대상 `{a.raw}` {n:,}건 (재생성 없음)", "",
          "| 층 | n | 이전(내 기준) 통과율 | **공식 EM** | **공식 ROUGE-L** | EM 차이 |",
          "|---|---:|---:|---:|---:|---:|"]
    tot_c = tot_em = tot_rl = tot_old = 0.0
    for st in ORDER:
        c = cnt[st]
        if not c:
            continue
        old = old_pass[st] / c; em = em_sum[st] / c; rl = rl_sum[st] / c
        tot_c += c; tot_em += em_sum[st]; tot_rl += rl_sum[st]; tot_old += old_pass[st]
        L.append(f"| {st} | {c:,} | {old:.1f}% | **{em:.1f}%** | **{rl:.1f}** | {em-old:+.1f}%p |")
    if tot_c:
        L.append(f"| **전체** | {int(tot_c):,} | {tot_old/tot_c:.1f}% | "
                 f"**{tot_em/tot_c:.1f}%** | **{tot_rl/tot_c:.1f}** | "
                 f"{(tot_em-tot_old)/tot_c:+.1f}%p |")
    L += ["", "> 층 구분은 **판독용**이다. 공식 채점은 태스크를 분기하지 않는다.",
          "> EM 차이가 크면 지금까지의 SNI 결론을 공식 기준으로 다시 내야 한다.", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"완료 (일치={all_ok}, {n:,}건) -> {a.out}")


if __name__ == "__main__":
    main()
