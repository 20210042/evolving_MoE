#!/usr/bin/env python3
"""확인 3에서 확인된 expert×문제 효과를 '쓸 수 있는가' 검사 — 살릴 구석 찾기.

확인 3(정답률 대비 +3.4%p, p=0.032)으로 문제별 expert 효과가 실재함이 확인됐다.
그렇다면 그것을 (a) 진화의 점수로, (b) 라우팅 신호로 쓸 수 있는지가 다음 질문이다.
임베딩 예측(확인 5)이 실패했다고 해서 구조 자체가 없는 것은 아니므로,
**특징(feature)에 의존하지 않는 방식**으로 직접 잰다 — 같은 문제를 여러 번 풀린 기록을
반으로 갈라, 앞 절반이 뒤 절반을 맞히는지 본다.

  A. expert 순위 재현성   : 앞 절반으로 매긴 expert 점수가 뒤 절반에서도 같은 순위인가
  B. 문제별 최적 expert   : 앞 절반의 1등이 뒤 절반에서도 잘하는가 (= probe-then-route 가능성)
  C. 효과의 분포          : 확인 3의 +3.4%p가 특정 expert/문제에 몰려 있는가

데이터: results/acc/evo_repro_exclusive128.raw.jsonl (128문제 × 11명 × 5회, 개별 시행 보존)
scipy 미사용.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acc/salvage_signal.md"
RNG = np.random.default_rng(0)
NSPLIT = 200
L: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    L.append(s)


def load() -> tuple[np.ndarray, list[str], list[str]]:
    raw = ROOT / "results/acc/evo_repro_exclusive128.raw.jsonl"
    cnt: dict = defaultdict(lambda: defaultdict(list))
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != "persona":
            continue
        cnt[r["pid"]][r["cid"]].append(int(r["pass"]))
    pids = sorted(cnt)
    experts = sorted({c for p in pids for c in cnt[p]})
    K = max(len(v) for p in pids for v in cnt[p].values())
    D = np.full((len(pids), len(experts), K), np.nan, np.float32)
    for i, p in enumerate(pids):
        for j, e in enumerate(experts):
            v = cnt[p].get(e, [])
            D[i, j, :len(v)] = v
    return D, pids, experts


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    D, pids, experts = load()
    N, E, K = D.shape
    say("# 확인 3의 효과를 쓸 수 있는가 — 절반 나누기 검사")
    say()
    say(f"- 데이터: 문제 {N} × expert {E} × 시행 {K}회 (개별 시행 보존, arm=persona)")
    say(f"- 방식: {K}회를 앞 {K//2}회 / 뒤 {K-K//2}회로 무작위 분할 × {NSPLIT}회 반복. "
        "앞 절반이 뒤 절반을 맞히는지 본다 — 특징(임베딩) 없이 재는 방식이다.")
    say()

    kA = K // 2
    resA_skill, resA_war, resB, resB_null, resB_best = [], [], [], [], []
    for _ in range(NSPLIT):
        perm = RNG.permutation(K)
        A = np.nanmean(D[:, :, perm[:kA]], 2)      # (N,E) 앞 절반 정답률
        B = np.nanmean(D[:, :, perm[kA:]], 2)      # (N,E) 뒤 절반

        # --- A. expert 순위 재현성
        resA_skill.append(pearson(np.nanmean(A, 0), np.nanmean(B, 0)))
        warA = np.array([((A > 0).sum(1) == 1) & (A[:, j] > 0) for j in range(E)]).sum(1)
        warB = np.array([((B > 0).sum(1) == 1) & (B[:, j] > 0) for j in range(E)]).sum(1)
        resA_war.append(pearson(warA.astype(float), warB.astype(float)))

        # --- B. 문제별 최적 expert가 뒤 절반에서도 나은가
        gains, nulls, bests = [], [], []
        for i in range(N):
            a, b = A[i], B[i]
            if np.nanstd(a) < 1e-12:               # 앞 절반이 전원 동률이면 고를 근거 없음
                continue
            top = np.flatnonzero(a == np.nanmax(a))
            pick = int(RNG.choice(top))
            rnd = int(RNG.integers(E))
            gains.append(b[pick] - np.nanmean(b))
            nulls.append(b[rnd] - np.nanmean(b))
            bests.append(np.nanmax(b) - np.nanmean(b))
        resB.append(np.mean(gains))
        resB_null.append(np.mean(nulls))
        resB_best.append(np.mean(bests))

    say("## A. expert 순위가 재현되는가 (앞 절반 점수 vs 뒤 절반 점수, 상관)")
    say()
    say("| 점수 방식 | 상관 (평균 ± 표준편차) | 해석 |")
    say("|---|---:|---|")
    sk = np.array(resA_skill)
    wr = np.array([x for x in resA_war if np.isfinite(x)])
    say(f"| 전체 정답률 (실력) | {sk.mean():+.3f} ± {sk.std():.3f} | "
        f"{'재현됨' if sk.mean() > 0.3 else '재현 안 됨'} |")
    say(f"| 나만 푼 문제 수 (현행 WAR) | {wr.mean():+.3f} ± {wr.std():.3f} | "
        f"{'재현됨' if wr.mean() > 0.3 else '재현 안 됨'} |")
    say()

    say("## B. 문제별로 '앞 절반 1등'이 뒤 절반에서도 나은가 (probe-then-route)")
    say()
    g, nu, be = np.array(resB), np.array(resB_null), np.array(resB_best)
    say("문제마다 뒤 절반 정답률에서 **11명 평균 대비 얼마나 이득인지**로 잰다.")
    say()
    say("| 누구를 고르나 | 평균 대비 이득 |")
    say("|---|---:|")
    say(f"| 무작위 1명 (기준) | {100*nu.mean():+.2f}%p |")
    say(f"| **앞 절반 1등** | **{100*g.mean():+.2f}%p** (±{100*g.std():.2f}) |")
    say(f"| 뒤 절반 정답을 보고 고르기 (상한) | {100*be.mean():+.2f}%p |")
    say()
    if be.mean() > 1e-9:
        say(f"- 실현율: 앞 절반 1등이 상한의 **{100*g.mean()/be.mean():.1f}%**를 가져온다.")
    say(f"- 무작위 대비 z = **{(g.mean()-nu.mean())/(g.std()/math.sqrt(NSPLIT)+1e-12):+.1f}** "
        "(분할 반복 기준 — 참고값)")
    say()

    # --- C. 효과의 분포: 원단독해결자 대비를 expert별로
    prior = json.loads((ROOT / "results/acc/exclusive128.problem_ids.json")
                       .read_text(encoding="utf-8"))["prior_exclusive_solver"]
    P = np.nanmean(D, 2)
    gmean = np.nanmean(P, 0)
    say("## C. 확인 3의 +3.4%p는 누구에게서 나오는가")
    say()
    say("| expert | 원단독해결 문제수 | 그 문제들에서 본인 | 같은 문제 나머지 10명 | 차이(실력 보정) |")
    say("|---|---:|---:|---:|---:|")
    rows = []
    for j, e in enumerate(experts):
        idx = [i for i, p in enumerate(pids) if prior.get(p) == e]
        if not idx:
            continue
        own = float(np.mean([P[i, j] for i in idx]))
        oth = float(np.mean([(P[i].sum() - P[i, j]) / (E - 1) for i in idx]))
        adj = float(np.mean([(P[i, j] - gmean[j])
                             - np.mean([P[i, k] - gmean[k] for k in range(E) if k != j])
                             for i in idx]))
        rows.append((len(idx), e, own, oth, adj))
    for n, e, own, oth, adj in sorted(rows, key=lambda r: -r[4]):
        say(f"| {e} | {n} | {100*own:.1f}% | {100*oth:.1f}% | **{100*adj:+.1f}%p** |")
    pos = sum(1 for r in rows if r[4] > 0)
    say()
    say(f"- 11명 중 **{pos}명**이 양(+)의 차이. 특정 소수가 아니라 전반적인 경향인지 확인용.")
    say()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
