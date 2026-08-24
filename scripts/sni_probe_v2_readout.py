#!/usr/bin/env python3
"""SNI 프로브 v2 판독 — docs/PLAN_sni_probe_v2.md §5의 순서를 코드로 고정한다.

두 계열을 분리해 잰다.
  문체(레퍼런스 무관) : 출력 토큰수 · 문장수 · type-token ratio · 불릿/번호 유무 · 유니크 출력률
  점수                : 닫힘 EM / 열림 ROUGE-L (raw의 score를 그대로 씀)

해석 순서(이 순서로만):
  1. 로스터가 luca 대비 문체지표를 바꾸는가          → 아니면 manipulation 실패, 점수 null 인용 금지
  2. between-expert 분산 vs within-luca(K회) 분산     → 같으면 노이즈
  3. 통과했을 때만 점수를 층별로 읽는다
  4. category 축 vs domain 축을 같은 N에서 비교

Usage:
  python scripts/sni_probe_v2_readout.py \
      --raw results/sni/probe_v2_raw.jsonl --data export/sni_v2/sni_all.jsonl \
      --roster configs/roster_sni_probe_v2.json --out docs/REPORT_regime_selection.md
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluation.scorer import score_sni_item_partial  # noqa: E402

BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
SENT = re.compile(r"[.!?]+(?:\s|$)")
WORD = re.compile(r"\w+", re.UNICODE)

# 층: 다른 데이터셋에도 옮겨지는 성질로만 잡는다(SNI 고유 라벨은 보조 축).
def stratum(it: dict) -> str:
    if it.get("task_closed"):
        return "객관식(닫힘)"
    g = it.get("gold_len_median") or 0
    if g < 3:
        return "열림 1-3단어"
    if g < 8:
        return "열림 3-8단어"
    if g < 15:
        return "열림 8-15단어"
    if g < 40:
        return "서술형 15-40단어"
    return "서술형 40단어+"


def style(text: str) -> tuple:
    w = WORD.findall(text or "")
    n = len(w)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    sents = max(1, len(SENT.findall(text)))
    ttr = len({x.lower() for x in w}) / n
    return (float(n), float(sents), ttr, 1.0 if BULLET.search(text or "") else 0.0)


METRICS = ("길이(토큰)", "문장수", "TTR", "불릿비율")


def var_decomp(cell: dict, experts: list) -> tuple:
    """cell[(cid,pid)] = 값. (문제분산, expert분산, 잔차) 비중을 돌려준다."""
    vals = collections.defaultdict(dict)
    for (cid, pid), v in cell.items():
        vals[pid][cid] = v
    pids = [p for p, d in vals.items() if len(d) == len(experts)]
    if not pids:
        return (0.0, 0.0, 0.0, 0.0)
    N, E = len(pids), len(experts)
    grand = sum(sum(vals[p].values()) for p in pids) / (N * E)
    tot = sum((vals[p][c] - grand) ** 2 for p in pids for c in experts) / (N * E)
    if tot <= 0:
        return (0.0, 0.0, 0.0, grand)
    pm = {p: sum(vals[p].values()) / E for p in pids}
    em = {c: sum(vals[p][c] for p in pids) / N for c in experts}
    vp = sum((pm[p] - grand) ** 2 for p in pids) / N
    ve = sum((em[c] - grand) ** 2 for c in experts) / E
    return (100 * vp / tot, 100 * ve / tot, 100 * (tot - vp - ve) / tot, grand)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--out", default="docs/REPORT_regime_selection.md")
    a = ap.parse_args()

    items = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        items[d["id"]] = d
    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    experts = [p["id"] for p in roster]
    axis = {p["id"]: ("luca" if p["id"] == "luca"
                      else ("category" if p["id"].startswith("cat_") else "domain"))
            for p in roster}

    # (stratum, cid, pid) -> 누적. 문체는 rep 전체 평균, 점수는 pass 횟수.
    st_sum = collections.defaultdict(lambda: [0.0] * len(METRICS))
    st_n = collections.Counter()
    passes = collections.Counter()
    reps = collections.Counter()
    outs_by_pid = collections.defaultdict(set)     # 유니크 출력률(rep 0만)
    luca_rep = collections.defaultdict(list)       # within-luca: rep별 길이
    rouge = collections.Counter()
    rouge_n = collections.Counter()
    len_by_prob = collections.defaultdict(dict)

    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        it = items.get(d["pid"])
        if it is None:
            continue
        key = (d["cid"], d["pid"])
        s = style(d.get("code") or "")
        acc = st_sum[key]
        for i in range(len(METRICS)):
            acc[i] += s[i]
        st_n[key] += 1
        passes[key] += int(d["pass"])
        reps[key] += 1
        if d["rep"] == 0:
            # 원문이 아니라 해시를 담는다 — rep0만 해도 2.18M 문자열이라 장문 층에서 GB급이 된다.
            outs_by_pid[d["pid"]].add(hash((d.get("code") or "").strip().lower()))
        if d["cid"] == "luca":
            luca_rep[d["pid"]].append(s[0])
        # 열린 태스크는 EM이 구조적으로 0에 가깝다(raw의 score는 EM). 사전등록 §4대로
        # ROUGE-L을 여기서 따로 계산한다 — 안 하면 서술형 층이 판독 불가.
        if not it.get("task_closed"):
            rouge[key] += score_sni_item_partial(it, d.get("code") or "")
            rouge_n[key] += 1
        # 문제별 expert 길이 (게이트 2 교정용)
        len_by_prob[d["pid"]][d["cid"]] = len_by_prob[d["pid"]].get(d["cid"], 0.0) + s[0]

    strat = {pid: stratum(items[pid]) for pid in outs_by_pid}
    L = ["# SNI 프로브 v2 판독 — 레짐 선택표", "",
         f"- raw `{a.raw}` · 로스터 {len(experts)}명 · 사전등록 [PLAN_sni_probe_v2.md](PLAN_sni_probe_v2.md) §5",
         "- 해석 순서를 지킨다: **manipulation → 노이즈 바닥 → 점수 → 축 비교**", ""]

    # ---- 게이트 1: 로스터가 luca 대비 문체를 바꾸는가 -------------------------
    L += ["## 게이트 1 — 로스터가 luca 대비 문체를 바꾸는가", "",
          "출력 길이(토큰) 기준. luca=1.00으로 정규화한 상대값.", "",
          "| expert | 축 | 상대 길이 | 상대 문장수 | TTR 차 | 불릿비율 |",
          "|---|---|---:|---:|---:|---:|"]
    per_exp = collections.defaultdict(lambda: [0.0] * len(METRICS))
    per_exp_n = collections.Counter()
    for (cid, pid), acc in st_sum.items():
        n = st_n[(cid, pid)]
        e = per_exp[cid]
        for i in range(len(METRICS)):
            e[i] += acc[i] / n
        per_exp_n[cid] += 1
    means = {c: [per_exp[c][i] / per_exp_n[c] for i in range(len(METRICS))] for c in experts}
    base = means["luca"]
    for c in sorted(experts, key=lambda x: -means[x][0]):
        m = means[c]
        L.append(f"| {c} | {axis[c]} | {m[0]/max(1e-9,base[0]):.3f} | "
                 f"{m[1]/max(1e-9,base[1]):.3f} | {m[2]-base[2]:+.4f} | {m[3]:.3f} |")
    spread = max(means[c][0] for c in experts) / max(1e-9, min(means[c][0] for c in experts))
    L += ["", f"- 최장/최단 출력길이 비 **{spread:.2f}×**", ""]

    # ---- 게이트 2: 노이즈 바닥 ------------------------------------------------
    # ⚠️ 둘 다 **문제 단위**로 재야 비교가 성립한다. expert 평균(87k문제 평균)의 분산과
    # 문제별 재생성 분산을 나누면 축척이 달라 무의미하다(초판의 결함).
    win, btw = [], []
    for pid, per_c in len_by_prob.items():
        v = luca_rep.get(pid) or []
        if len(v) > 1:
            m = sum(v) / len(v)
            win.append(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        vals = [per_c[c] / max(1, st_n[(c, pid)]) for c in experts if c in per_c]
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            btw.append(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    within = sum(win) / max(1, len(win))
    between = sum(btw) / max(1, len(btw))
    L += ["## 게이트 2 — 노이즈 바닥 (문제 단위)", "",
          "같은 문제 안에서 잰다. **across-expert(25명 사이 흩어짐)** vs "
          "**within-luca(luca가 K회 재생성했을 때 흩어짐)**.", "",
          f"- within-luca 분산(출력길이) **{within:.1f}** · across-expert 분산 **{between:.1f}**",
          f"- 비 **{between/max(1e-9,within):.2f}×** — 1에 가까우면 관측된 문체차는 재생성 노이즈와 구분 안 됨", ""]

    # ---- 게이트 3: 층별 문체 갈림 × 점수 갈림 ---------------------------------
    L += ["## 게이트 3 — 층별: 문체는 갈리나, 그게 점수로 가나", "",
          "| 층 | 문제수 | 문체 분산 중 expert | EM 분산 중 expert | EM 상호작용 | 평균 EM | 평균 ROUGE-L | ROUGE 분산 중 expert | 유니크출력률 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = ["객관식(닫힘)", "열림 1-3단어", "열림 3-8단어", "열림 8-15단어",
             "서술형 15-40단어", "서술형 40단어+"]
    rows_out = {}
    for st in order:
        pids = [p for p, s in strat.items() if s == st]
        if not pids:
            continue
        ps = set(pids)
        cell_style = {k: st_sum[k][0] / st_n[k] for k in st_sum if k[1] in ps}
        cell_score = {k: passes[k] / max(1, reps[k]) for k in passes if k[1] in ps}
        cell_rouge = {k: rouge[k] / rouge_n[k] for k in rouge if k[1] in ps and rouge_n[k]}
        sp, se, sr, _ = var_decomp(cell_style, experts)
        pp, pe, pr, gm = var_decomp(cell_score, experts)
        _, re_, rr_, rgm = var_decomp(cell_rouge, experts) if cell_rouge else (0, 0.0, 0.0, 0.0)
        uniq = sum(len(outs_by_pid[p]) for p in pids) / (len(pids) * len(experts))
        rows_out[st] = (len(pids), se, pe, pr, gm, uniq)
        L.append(f"| {st} | {len(pids):,} | {se:.1f}% | {pe:.1f}% | {pr:.1f}% | "
                 f"{100*gm:.1f}% | {rgm:.1f} | {re_:.1f}% | {uniq:.3f} |")
    L += ["", "> 문체 expert 비중이 크고 점수 expert 비중이 0에 가까우면 = **문체는 갈리는데 채점에 안 들어간다**.",
          "> 두 값이 함께 큰 층이 있으면 그 층이 다음 진화 레짐 후보다.", ""]

    # ---- 게이트 4: 축 비교 ----------------------------------------------------
    L += ["## 게이트 4 — category 축 vs domain 축 (같은 N=12)", "",
          "| 축 | 평균 점수 | 점수 표준편차 | 평균 출력길이 | 길이 표준편차 |", "|---|---:|---:|---:|---:|"]
    for ax in ("category", "domain", "luca"):
        cs = [c for c in experts if axis[c] == ax]
        sc = [sum(passes[(c, p)] / max(1, reps[(c, p)]) for p in outs_by_pid) / len(outs_by_pid)
              for c in cs]
        ln = [means[c][0] for c in cs]
        mu = sum(sc) / len(sc)
        sd = math.sqrt(sum((x - mu) ** 2 for x in sc) / len(sc))
        lm = sum(ln) / len(ln)
        lsd = math.sqrt(sum((x - lm) ** 2 for x in ln) / len(ln))
        L.append(f"| {ax} ({len(cs)}명) | {100*mu:.2f}% | {100*sd:.2f}%p | {lm:.1f} | {lsd:.1f} |")
    L += ["", "> 표준편차가 큰 축이 = 그 축으로 자른 로스터가 더 많은 변화를 만든다.", ""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"판독 완료 -> {a.out}")


if __name__ == "__main__":
    main()
