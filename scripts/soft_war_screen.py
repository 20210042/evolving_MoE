#!/usr/bin/env python3
"""soft-WAR 후보 사전 검증 — 진화를 돌리기 전에 기존 행렬로 답할 수 있는 두 질문.

배경: 현행 WAR는 "정확히 한 명만 푼 문제 수"라 로스터가 커질수록 그 사건이 구조적으로
희박해진다(확인 1: 배치 50당 1.0개). exclusivity를 1/0이 아니라 **몇 명이 같이 풀었는지에
따라 나눠 갖는 연속값**으로 바꾸자는 제안을 사전 검증한다.

  (A) 로스터 크기 효과 — expert를 부분표집해 로스터를 E'명으로 줄였을 때 단독해결 밀도가
      어떻게 변하는지. "커지면 사라진다"를 실측으로 보인다.
  (B) 적합도 변형별 재현성 — 문제를 반으로 갈라 각 half에서 11명 점수를 매기고, 두 점수가
      일치하는지(전문가 11명 상관). **그리고 진화가 실제로 쓰는 결정 = "최하위 1명"이
      두 half에서 같은 사람인지**의 일치율.

데이터: results/acc/seed20210111/binning_train_full.binned.jsonl (10,000문제 × 11명, 0/1)
scipy 미사용.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acc/soft_war_screen.md"
RNG = np.random.default_rng(0)
NSPLIT = 200
L: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    L.append(s)


# ---------------------------------------------------------------- 적합도 변형
def fitness(S: np.ndarray, kind: str) -> np.ndarray:
    """S: (N, E) 0/1 → expert별 점수 (E,)"""
    N, E = S.shape
    n = S.sum(1, keepdims=True)                       # 문제별 푼 사람 수
    if kind == "hard_war":                            # 현행: 나만 푼 문제 수
        return (S * (n == 1)).sum(0)
    if kind == "soft_inv":                            # 1/n 배분 (대칭 게임 Shapley와 동치)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(n > 0, 1.0 / np.maximum(n, 1), 0.0)
        return (S * w).sum(0)
    if kind == "soft_linear":                         # 같이 푼 사람이 많을수록 선형 감점
        w = np.where(n > 0, (E - n) / (E - 1), 0.0)
        return (S * w).sum(0)
    if kind == "contrast":                            # 문제별 (본인 − 나머지 평균)의 합
        others = (S.sum(1, keepdims=True) - S) / (E - 1)
        return (S - others).sum(0)
    if kind == "acc":                                 # 그냥 정답률
        return S.mean(0)
    raise ValueError(kind)


KINDS = [
    ("hard_war", "현행 WAR — 나만 푼 문제 수"),
    ("soft_inv", "soft 1/n — 같이 푼 n명이 1점을 나눠 가짐"),
    ("soft_linear", "soft 선형 — (E−n)/(E−1) 만큼 배점"),
    ("contrast", "대비형 — 문제별 (본인 − 나머지 평균)"),
    ("acc", "정답률 (참고: 분업이 아니라 실력순)"),
]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    rows = [json.loads(l) for l in open(
        ROOT / "results/acc/seed20210111/binning_train_full.binned.jsonl", encoding="utf-8")]
    experts = sorted(rows[0]["per_expert"])
    S = np.array([[r["per_expert"].get(e, 0) for e in experts] for r in rows], np.float32)
    N, E = S.shape
    say("# soft-WAR 사전 검증 (진화 실행 전, 기존 행렬 재사용)")
    say()
    say(f"- 데이터: {N:,}문제 × expert {E}명 · 단일 드로우 0/1 · 평균 정답률 {100*S.mean():.1f}%")
    say()

    # ---------------- (A) 로스터 크기 효과
    say("## A. 로스터가 커질수록 '정확히 한 명'이 사라지는가")
    say()
    say("11명 중 E′명을 무작위로 뽑아 그 로스터에서 단독해결 밀도를 잰다(조합 100개 평균).")
    say()
    say("| 로스터 크기 E′ | 단독해결 비율 | 배치 50개당 | 갈리는 문제 비율 |")
    say("|---:|---:|---:|---:|")
    for Ep in range(2, E + 1):
        combos = list(combinations(range(E), Ep))
        pick = [combos[i] for i in RNG.choice(len(combos), min(100, len(combos)), replace=False)]
        ex, mixed = [], []
        for c in pick:
            sub = S[:, list(c)]
            ns = sub.sum(1)
            ex.append((ns == 1).mean())
            mixed.append(((ns > 0) & (ns < Ep)).mean())
        say(f"| {Ep} | {100*np.mean(ex):.2f}% | {50*np.mean(ex):.2f}개 | {100*np.mean(mixed):.1f}% |")
    say()
    say("**갈리는 문제는 로스터가 커져도 남지만, '정확히 한 명'만 골라내는 현행 방식은 계속 줄어든다.**")
    say("soft-WAR의 요지가 이 격차 — 갈리는 문제를 버리지 않고 점수로 쓰는 것 — 이다.")
    say()

    # ---------------- (B) 적합도 변형별 재현성
    say("## B. 적합도 변형별 재현성 (문제를 반으로 갈라 같은 순위가 나오는가)")
    say()
    say(f"{N:,}문제를 무작위 절반씩 나눠 각각 11명 점수를 매기고 두 점수를 비교한다 "
        f"({NSPLIT}회 반복). **최하위 일치율**은 두 half에서 꼴찌로 지목된 사람이 같은 비율 — "
        "진화가 실제로 쓰는 결정이 이것이다.")
    say()
    res: dict = {k: {"r": [], "last": []} for k, _ in KINDS}
    for _ in range(NSPLIT):
        idx = RNG.permutation(N)
        h1, h2 = idx[: N // 2], idx[N // 2:]
        for k, _lab in KINDS:
            a, b = fitness(S[h1], k), fitness(S[h2], k)
            res[k]["r"].append(pearson(a, b))
            res[k]["last"].append(int(np.argmin(a) == np.argmin(b)))
    say("| 적합도 | 두 half 점수 일치도 | **최하위 일치율** |")
    say("|---|---:|---:|")
    for k, lab in KINDS:
        r = np.array([x for x in res[k]["r"] if np.isfinite(x)])
        say(f"| {lab} | {r.mean():+.3f} ± {r.std():.3f} | **{100*np.mean(res[k]['last']):.0f}%** |")
    say()
    say("- 최하위 일치율 9% = 무작위(1/11)와 같음. 100%면 완전 재현.")
    say()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
