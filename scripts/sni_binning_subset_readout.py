#!/usr/bin/env python
"""binning raw 로그에서 **로스터 부분집합**의 per-expert / union 수치를 재계산한다.

생성은 다시 하지 않는다 — `test_raw.jsonl`이 (pid, cid, rep, pass)를 행마다 남기므로
어떤 부분집합의 union이든 사후에 나온다.

정의는 `evo_multisample_pilot.py:analyze()`와 동일하게 맞췄다:
  p̂(expert, 문제) = K회 중 pass 비율
  per-expert pass@1 = p̂의 문제 평균
  union(any)      = 한 번이라도 푼 사람이 있는 문제 수 (p̂ > 0)
  union(majority) = p̂ > 0.5 인 사람이 있는 문제 수
  union(strict)   = p̂ = 1 인 사람이 있는 문제 수
  union(soft)     = Σ_i [1 − Π_j (1 − p̂_ji)]   (확률 기대값)

Usage:
  python scripts/sni_binning_subset_readout.py results/sni/binning_seed20212003/test_raw.jsonl \
      --drop c_61797 c_19704 c_18467
"""
import argparse
import json
import math
from collections import defaultdict


def load(path):
    """(pid,cid) -> [pass...]"""
    cell = defaultdict(list)
    n_rows = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:      # 실행 중이면 마지막 줄이 잘려 있을 수 있다
            continue
        n_rows += 1
        cell[(r["pid"], r["cid"])].append(int(r["pass"]))
    return cell, n_rows


def readout(cell, experts, pids):
    P = {c: [] for c in experts}
    for i in pids:
        for c in experts:
            v = cell.get((i, c))
            P[c].append(sum(v) / len(v) if v else 0.0)
    N = len(pids)
    per = {c: sum(P[c]) / N for c in experts}
    out = {"E": len(experts), "N": N, "per_expert": per}
    for name, thr in (("any", 0.0), ("majority", 0.5), ("strict", 1.0 - 1e-9)):
        out[f"union_{name}"] = sum(
            1 for k in range(N) if any(P[c][k] > thr for c in experts)
        )
    out["union_soft"] = sum(
        1 - math.prod(1 - P[c][k] for c in experts) for k in range(N)
    )
    # 문제별 몇 명이 푸는가(majority 라벨) — 신호 밀도
    hist = [0] * (len(experts) + 1)
    for k in range(N):
        hist[sum(1 for c in experts if P[c][k] > 0.5)] += 1
    out["hist"] = hist
    out["contested"] = sum(hist[1:-1])
    return out


def fmt(tag, r, names):
    L = [f"### {tag} — {r['E']}명 / {r['N']}문제", ""]
    L.append("| pass@1 | expert |")
    L.append("|---:|---|")
    for c, v in sorted(r["per_expert"].items(), key=lambda x: -x[1]):
        L.append(f"| {v*100:.2f} | {names.get(c, c)} |")
    L += ["", "| union 규칙 | 푼 문제 | 비율 |", "|---|---:|---:|"]
    for name in ("any", "majority", "strict"):
        u = r[f"union_{name}"]
        L.append(f"| {name} | {u} | {u/r['N']*100:.2f}% |")
    L.append(f"| soft | {r['union_soft']:.1f} | {r['union_soft']/r['N']*100:.2f}% |")
    best = max(r["per_expert"].values()) * 100
    L += ["", f"- best-single: {best:.2f}%  ·  헤드룸(union any − best-single): "
              f"{r['union_any']/r['N']*100 - best:+.2f}pp",
          f"- contested(0<푼사람<전원, majority 기준): {r['contested']} "
          f"({r['contested']/r['N']*100:.1f}%)  ·  아무도 못 푼 문제: {r['hist'][0]}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--roster", default="results/sni/seed20212003/roster_final.json")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = {p["id"]: p["name"] for p in json.load(open(a.roster))}
    cell, n_rows = load(a.raw)
    experts = sorted({c for _, c in cell})
    pids = sorted({i for i, _ in cell})
    # 미완료 문제(전원×K가 안 찬 것)는 제외 — 잡이 도는 중에도 돌릴 수 있게
    K = max(len(v) for v in cell.values())
    full = [i for i in pids
            if all(len(cell.get((i, c), [])) == K for c in experts)]

    keep = [c for c in experts if c not in a.drop]
    body = [f"# binning 부분집합 판독 — `{a.raw}`", "",
            f"- 원본 행 {n_rows:,} · expert {len(experts)}명 · K={K}",
            f"- 완결 문제 {len(full):,} / 등장 문제 {len(pids):,} (미완결은 제외)",
            f"- 제외: {', '.join(names.get(c, c) for c in a.drop) or '없음'}", ""]
    body.append(fmt("제외 후", readout(cell, keep, full), names))
    body.append("")
    body.append(fmt("전원(대조)", readout(cell, experts, full), names))
    text = "\n".join(body)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
