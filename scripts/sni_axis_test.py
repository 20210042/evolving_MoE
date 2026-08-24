#!/usr/bin/env python3
"""SNI 프로브 v2 — **분류기준(축) 단위** 검정.

개인 25명의 WAR을 세는 건 "이 분류기준이 분화를 만드는가"를 재지 못한다.
검정 단위는 축이어야 하고, 문제 난이도가 전체 분산의 89%라 그것부터 상쇄해야 한다.

그래서 **문제 내 짝지은 비교**를 쓴다. 문제 i의 라벨이 c(i)/d(i)일 때, 같은 축 안에서만 비교:

    Δ_cat(i) = score(c(i) 담당 cat expert, i) − mean(score(다른 cat experts, i))

같은 문제 안에서 빼므로 난이도가 소거되고, 남는 것은 "자기 구역이라는 사실"의 효과뿐이다.
domain 축도 동일. 축마다 Δ의 평균·부호검정을 층별로 낸다.

⚠️ 두 축은 얽혀 있다(Code·Mathematics·Sociology는 사실상 category와 동의어).
그래서 **두 축의 매치가 서로 다른 문제만** 따로 떼어 한 번 더 본다.

Usage:
  python scripts/sni_axis_test.py --raw results/sni/probe_v2_raw.jsonl \
      --data export/sni_v2/sni_all.jsonl --roster configs/roster_sni_probe_v2.json \
      --out docs/REPORT_axis_test.md
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

WORD = re.compile(r"\w+", re.UNICODE)


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


ORDER = ["객관식(닫힘)", "열림 1-3단어", "열림 3-8단어", "열림 8-15단어",
         "서술형 15-40단어", "서술형 40단어+"]


def paired(deltas: list) -> tuple:
    """(n, 평균, 표준오차, z, 부호검정 승률). 귀무 = 평균 0."""
    n = len(deltas)
    if n < 2:
        return (n, 0.0, 0.0, 0.0, 0.0)
    mu = sum(deltas) / n
    var = sum((x - mu) ** 2 for x in deltas) / (n - 1)
    se = math.sqrt(var / n)
    pos = sum(1 for x in deltas if x > 0)
    nz = sum(1 for x in deltas if x != 0)
    return (n, mu, se, (mu / se if se > 0 else 0.0), (pos / nz if nz else 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_v2_raw.jsonl")
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--out", default="docs/REPORT_axis_test.md")
    a = ap.parse_args()

    items = {}
    full = {}
    for line in open(a.data, encoding="utf-8"):
        d = json.loads(line)
        items[d["id"]] = (d["category"], d["sni_domain"], stratum(d))
        full[d["id"]] = d

    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    # 구역 → 담당 expert (축별)
    owner = {"category": {}, "domain": {}}
    members = {"category": [], "domain": []}
    for p in roster:
        if p["id"] == "luca":
            continue
        ax = "category" if p["id"].startswith("cat_") else "domain"
        owner[ax][p["strengths"]] = p["id"]
        members[ax].append(p["id"])

    # (cid,pid) -> [pass], [len]
    sc = collections.defaultdict(lambda: [0.0, 0])     # [합, 횟수]
    ln = collections.defaultdict(lambda: [0.0, 0])
    for line in open(a.raw, encoding="utf-8"):
        d = json.loads(line)
        it = full.get(d["pid"])
        if it is None:
            continue
        k = (d["cid"], d["pid"])
        # 닫힘 = EM(0/1). 열림 = ROUGE-L(0~1) — 열림에서 EM은 전원 0이라 뺄 게 없다
        # (서술형 40단어+에서 Δ가 정확히 0.00%p·부호승률 0.0%로 나온 게 그 바닥이다).
        v = (int(d["pass"]) if it.get("task_closed")
             else score_sni_item_partial(it, d.get("code") or "") / 100.0)
        s = sc[k]; s[0] += v; s[1] += 1
        t = ln[k]; t[0] += len(WORD.findall(d.get("code") or "")); t[1] += 1

    def phat(c, p):
        v = sc.get((c, p))
        return v[0] / v[1] if v and v[1] else None

    def plen(c, p):
        v = ln.get((c, p))
        return v[0] / v[1] if v and v[1] else None

    pids = sorted({k[1] for k in sc})
    L = ["# SNI 프로브 v2 — 분류기준(축) 단위 검정", "",
         f"- raw `{a.raw}` · 문제 {len(pids):,} · category 축 {len(members['category'])}명 / "
         f"domain 축 {len(members['domain'])}명",
         "- **문제 내 짝지은 비교**: Δ = (자기 구역 담당 expert) − (같은 축 나머지 평균). "
         "같은 문제 안에서 빼므로 난이도(전체 분산의 89%)가 소거된다.",
         "- 귀무가설: Δ 평균 = 0 (분류기준이 아무 분화도 만들지 않는다)",
         "- **점수 = 닫힘 EM / 열림 ROUGE-L**(사전등록 §4). 열림에 EM을 쓰면 전원 0점이라 측정이 성립하지 않는다.",
         ""]

    def run(scope_name: str, keep) -> None:
        L.append(f"## {scope_name}")
        L.append("")
        L.append("| 축 | 층 | n(문제) | Δ | z(문제) | 부호승률 | n(태스크) | Δ(태스크) | **z(태스크)** | 부호승률 | Δ 길이 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        per_task = collections.defaultdict(list)
        for ax in ("category", "domain"):
            idx = 0 if ax == "category" else 1
            per_st = collections.defaultdict(list)
            per_st_len = collections.defaultdict(list)
            for p in pids:
                meta = items.get(p)
                if meta is None or not keep(p, meta):
                    continue
                own = owner[ax].get(meta[idx])
                if own is None:
                    continue                      # 로스터가 안 덮는 구역
                oth = [c for c in members[ax] if c != own]
                me, others = phat(own, p), [phat(c, p) for c in oth]
                others = [x for x in others if x is not None]
                if me is None or not others:
                    continue
                per_st[meta[2]].append(me - sum(others) / len(others))
                per_task[(ax, meta[2], full[p]["task_name"])].append(me - sum(others) / len(others))
                ml, ol = plen(own, p), [plen(c, p) for c in oth]
                ol = [x for x in ol if x is not None]
                if ml is not None and ol:
                    per_st_len[meta[2]].append(ml - sum(ol) / len(ol))
            allv = [x for st in ORDER for x in per_st.get(st, [])]
            alll = [x for st in ORDER for x in per_st_len.get(st, [])]
            for st in ORDER + ["**전체**"]:
                v = allv if st.startswith("**") else per_st.get(st, [])
                vl = alll if st.startswith("**") else per_st_len.get(st, [])
                if len(v) < 2:
                    continue
                n, mu, se, z, sg = paired(v)
                _, lmu, _, _, _ = paired(vl) if len(vl) >= 2 else (0, 0.0, 0, 0, 0)
                # ⚠️ category/domain은 태스크 단위 라벨이라 한 태스크의 100문제가 담당자를
                # 공유한다 → 문제를 독립으로 세면 z가 부푼다. 태스크 평균으로 묶어 다시 낸다.
                if st.startswith("**"):
                    tv = [sum(x)/len(x) for k, x in per_task.items() if k[0] == ax]
                else:
                    tv = [sum(x)/len(x) for k, x in per_task.items()
                          if k[0] == ax and k[1] == st]
                tn, tmu, _, tz, tsg = paired(tv) if len(tv) >= 2 else (len(tv), 0., 0., 0., 0.)
                L.append(f"| {ax} | {st} | {n:,} | {100*mu:+.2f}%p | {z:+.1f} | "
                         f"{100*sg:.1f}% | {tn} | {100*tmu:+.2f}%p | {tz:+.1f} | {100*tsg:.1f}% | {lmu:+.1f} |")
        L.append("")

    run("전체 (로스터가 덮는 구역의 문제 전부)", lambda p, m: True)
    # 두 축이 서로 다른 것을 가리키는 문제만 — 축 얽힘 통제
    def disjoint(p, m):
        c = owner["category"].get(m[0])
        d = owner["domain"].get(m[1])
        return c is not None and d is not None
    run("두 축이 모두 덮는 문제만 (축 얽힘 통제 — 같은 문제에서 두 축을 직접 비교)", disjoint)

    L += ["> **z(태스크)가 진짜 검정이다** — 문제 단위 z는 같은 태스크 100문제를 독립으로 세어 부풀려져 있다.",
          "> |z|>2면 우연 이상. 부호승률 50%는 동전던지기.",
          "> **Δ 점수 ≈ 0인데 Δ 출력길이 ≠ 0이면 = 분류기준이 문체는 바꾸지만 채점에는 안 들어간다.**", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"축 검정 완료 -> {a.out}")


if __name__ == "__main__":
    main()
