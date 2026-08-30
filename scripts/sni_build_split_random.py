#!/usr/bin/env python
"""대조군 분할 — Random split (크기 맞춤). 재생성 0.

`sni_build_split.py`가 만든 우리 분할과 **모든 것을 같게 두고 '누구에게 가느냐'만 무작위로** 바꾼다.
같게 두는 것:
  · 어느 문제가 indiv / shared / all_fail 인지 (n_solved 기준이므로 조건과 무관한 사실)
  · 문제당 배정 인원 수 (indiv는 n_solved명, all_fail은 1명)
  · expert별 총 학습량 (우리 분할의 분포를 그대로 맞춘다)
바꾸는 것:
  · 그 자리에 **어떤 expert가 들어가는가** — 차수 보존 무작위 재배치

그래서 이 조건과 Ours의 차이는 오직 **분할 축**이다. 크기·중복도·shared 구조가 전부 같다.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SRC = "export/sni_split_seed20212003/split.jsonl"
SHARED = "__shared__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="export/sni_split_random/split.jsonl")
    ap.add_argument("--report", default="results/sni/split_build_random.md")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(SRC)]
    ex = sorted({c for r in rows for c in r["experts"] if c != SHARED})
    E = len(ex)

    # 목표 정원 = 우리 분할의 expert별 총량 (kind별로 따로 맞춘다)
    quota = {k: Counter() for k in ("indiv", "all_fail")}
    for r in rows:
        if r["kind"] == SHARED or r["kind"] == "shared":
            continue
        for c in r["experts"]:
            quota[r["kind"]][c] += 1

    def refill(kind):
        """정원만큼 expert id를 담은 자루를 만들어 섞는다."""
        bag = [c for c in ex for _ in range(quota[kind][c])]
        rng.shuffle(bag)
        return bag

    # 슬롯 채우기: 각 kind마다 정원만큼의 expert 자루를 만들어 한 번만 섞고 순서대로 소비한다.
    # 한 문제 안에서 중복이 나오면 자루 뒤쪽의 쓸 수 있는 값과 자리를 바꾼다(정원 보존).
    out, cnt = [], {c: [0, 0] for c in ex + [SHARED]}
    bag = {k: refill(k) for k in quota}
    pos = {k: 0 for k in quota}
    for r in rows:
        if r["kind"] == "shared":
            out.append({**r, "experts": [SHARED]})
            cnt[SHARED][0] += 1
            continue
        k, b = r["kind"], bag[r["kind"]]
        need = len(r["experts"])
        picked = []
        for _ in range(need):
            i = pos[k]
            if b[i] in picked:
                for j in range(i + 1, len(b)):
                    if b[j] not in picked:
                        b[i], b[j] = b[j], b[i]
                        break
            picked.append(b[i])
            pos[k] += 1
        out.append({**r, "experts": picked})
        col = 1 if k == "all_fail" else 0
        for c in picked:
            cnt[c][col] += 1

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 우리 분할 대비 검증
    ours = {c: [0, 0] for c in ex + [SHARED]}
    for r in rows:
        col = 1 if r["kind"] == "all_fail" else 0
        for c in r["experts"]:
            ours[c][col] += 1
    import itertools
    fin = {c: set() for c in ex}
    for i, r in enumerate(out):
        for c in r["experts"]:
            if c in fin:
                fin[c].add(i)
    J = [len(fin[x] & fin[y]) / max(1, len(fin[x] | fin[y]))
         for x, y in itertools.combinations(ex, 2)]

    L = [f"# 대조군 분할 — Random split (크기 맞춤, seed={a.seed})", "",
         "우리 분할과 **문제 구간·문제당 인원 수·expert별 총량**을 전부 맞추고 "
         "**누가 맡느냐만** 무작위로 재배치했다. 차이는 오직 분할 축이다.", "",
         f"- 평균 쌍 Jaccard **{sum(J)/len(J):.3f}** (우리 분할 0.163)", "",
         "| expert | Random 합계 | Ours 합계 | 차 |", "|---|---:|---:|---:|"]
    for c in ex:
        rr, oo = sum(cnt[c]), sum(ours[c])
        L.append(f"| {c} | {rr:,} | {oo:,} | {rr-oo:+,} |")
    L.append(f"| **shared** | {cnt[SHARED][0]:,} | {ours[SHARED][0]:,} | "
             f"{cnt[SHARED][0]-ours[SHARED][0]:+,} |")
    L += ["", f"산출: `{a.out}`"]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    open(a.report, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
