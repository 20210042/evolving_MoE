#!/usr/bin/env python
"""정답을 모르는 채로 여러 답 중 하나를 고를 수 있나 — 다수결(self-consistency). 재생성 0.

union은 오라클이라 배포 수치가 아니다. 여기서는 **정답 없이** 고른다:
문제마다 k명의 실제 출력 문자열을 모아 **공식 normalize 후 최빈 문자열**을 채택하고,
그 답이 맞았는지로 정답률을 낸다. 출력·채점결과는 binning raw에 이미 있다(`code`/`pass`).

대조가 핵심이다 — 이득이 '여러 명'에서 오는지 '여러 번'에서 오는지:
  · k명 × 1회 다수결   (사람 다양성)
  · 1명 × k회 다수결   (같은 사람 재샘플링)
같은 생성 예산에서 비교한다.

로스터 선택(어느 k명인가)은 **train에서** 하고 test는 한 번만 본다.

Usage: python3 scripts/sni_selfconsistency.py --out results/sni/selfconsistency.md
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.scorer import _sni_normalize  # noqa: E402

D = "results/sni/binning_seed20212003"


def load(path):
    """pid -> cid -> [(정규화출력, pass), ...]  (rep 순서 유지)"""
    cell = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cell[r["pid"]][r["cid"]].append((_sni_normalize(r.get("code") or ""), int(r["pass"])))
    return cell


def vote(cands, rng):
    """cands = [(정규화답, pass)...] → 최빈 답을 고르고 그 정답 여부를 돌려준다. 동점은 무작위."""
    if not cands:
        return 0, False
    c = Counter(a for a, _ in cands)
    top = max(c.values())
    best = [a for a, n in c.items() if n == top]
    pick = rng.choice(best)
    ok = next(p for a, p in cands if a == pick)
    return ok, len(best) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sni/selfconsistency.md")
    ap.add_argument("--draws", type=int, default=50)
    a = ap.parse_args()
    rng = random.Random(0)

    te = load(f"{D}/test_raw.jsonl")
    tr = load(f"{D}/train_raw.jsonl")
    pids = sorted(te)
    ex = sorted(te[pids[0]])
    N = len(pids)

    # train에서 고르는 것들
    tr_pids = sorted(tr)
    mean_tr = {c: sum(p for pid in tr_pids for _, p in tr[pid][c]) / (len(tr_pids) * 3) for c in ex}
    best_c = max(ex, key=lambda c: mean_tr[c])

    def acc(sel):        # sel(pid, rng) -> (맞음, 동점여부)
        hit = ties = 0
        for pid in pids:
            ok, t = sel(pid)
            hit += ok; ties += t
        return 100 * hit / N, 100 * ties / N

    L = ["# 다수결로 답을 고를 수 있나 (진화 16명, 재생성 0)", "",
         f"- test {N:,}문제 · expert {len(ex)}명 × K=3 · 정규화는 공식 `_sni_normalize`",
         f"- 로스터·기준선은 train {len(tr_pids):,}문제에서 선택, test는 한 번만",
         "- 동점은 무작위 채택(동점률 병기)", "",
         "| 예산 | 방식 | test 정답률 | 동점률 |", "|---:|---|---:|---:|"]

    # B=1
    L.append(f"| 1 | best-single ({best_c}, train 선택) | "
             f"{100*sum(te[p][best_c][0][1] for p in pids)/N:.2f} | — |")

    for B in (3, 5):
        # 같은 사람 B회 (K=3이므로 B=3까지만 가능) — B=5는 건너뛴다
        if B <= 3:
            v, t = acc(lambda pid: vote(te[pid][best_c][:B], rng))
            L.append(f"| {B} | 같은 1명({best_c}) × {B}회 다수결 | {v:.2f} | {t:.1f}% |")
        # 무작위 B명 × 1회
        vs, ts = [], []
        for _ in range(a.draws):
            grp = rng.sample(ex, B)
            v, t = acc(lambda pid, g=grp: vote([te[pid][c][0] for c in g], rng))
            vs.append(v); ts.append(t)
        L.append(f"| {B} | 무작위 {B}명 × 1회 다수결 | "
                 f"{sum(vs)/len(vs):.2f} ± {(max(vs)-min(vs))/2:.2f} | {sum(ts)/len(ts):.1f}% |")
        # train에서 고른 최적 B명 (탐욕: train union 최대화)
        cur, grp = set(), []
        for _ in range(B):
            c = max([x for x in ex if x not in grp],
                    key=lambda x: len(cur | {p for p in tr_pids if tr[p][x][0][1]}))
            grp.append(c); cur |= {p for p in tr_pids if tr[p][c][0][1]}
        v, t = acc(lambda pid, g=grp: vote([te[pid][c][0] for c in g], rng))
        ov = 100 * sum(1 for p in pids if any(te[p][c][0][1] for c in grp)) / N
        L.append(f"| {B} | train 선택 {B}명 × 1회 다수결 | {v:.2f} | {t:.1f}% |")
        L.append(f"| {B} | └ 같은 {B}명의 union(오라클, 참고) | {ov:.2f} | — |")

    # 전원
    v, t = acc(lambda pid: vote([te[pid][c][0] for c in ex], rng))
    L.append(f"| 16 | 전원 × 1회 다수결 | {v:.2f} | {t:.1f}% |")
    v, t = acc(lambda pid: vote([x for c in ex for x in te[pid][c]], rng))
    L.append(f"| 48 | 전원 × 3회 다수결 | {v:.2f} | {t:.1f}% |")
    L.append(f"| 16 | union(오라클, 참고) | "
             f"{100*sum(1 for p in pids if any(te[p][c][0][1] for c in ex))/N:.2f} | — |")

    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
