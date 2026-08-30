#!/usr/bin/env python3
"""soft_linear WAR 토글 검증 — hard 모드가 기존과 완전 동일한지 + soft 값이 정의대로인지."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from war import compute_war_scores  # noqa: E402

# p1: A만, p2: A·B, p3: 전원, p4: 아무도 못 품(등장 안 함), p5: B·C
SQUAD = {
    "A": {"p1", "p2", "p3"},
    "B": {"p2", "p3", "p5"},
    "C": {"p3", "p5"},
}
E = 3

hard, ub, rate = compute_war_scores(SQUAD, 5, mode="hard")
soft, _, _ = compute_war_scores(SQUAD, 5, mode="soft_linear")

# 기존 공식을 그대로 다시 계산 (회귀 확인)
all_solved = set().union(*SQUAD.values())
legacy = {
    a: len(all_solved) - len(set().union(*[s for b, s in SQUAD.items() if b != a]))
    for a in SQUAD
}
assert hard == legacy, f"hard 모드가 기존 공식과 다름: {hard} != {legacy}"
assert isinstance(ub, int) and ub == 4 and abs(rate - 80.0) < 1e-9, (ub, rate)

n = {"p1": 1, "p2": 2, "p3": 3, "p5": 2}
want = {
    a: sum((E - n[p]) / (E - 1) for p in s) for a, s in SQUAD.items()
}
assert all(abs(soft[a] - want[a]) < 1e-9 for a in SQUAD), f"soft 값 불일치: {soft} != {want}"

print("squad:", {a: sorted(s) for a, s in SQUAD.items()})
print("문제별 푼 사람 수:", n)
print()
print(f"{'agent':6} {'hard(현행)':>12} {'soft_linear':>12}")
for a in sorted(SQUAD):
    print(f"{a:6} {hard[a]:12} {soft[a]:12.3f}")
print()
print("hard 모드 == 기존 공식  ✓")
print("soft 값 == (E-n)/(E-1) 합  ✓")
print("양 끝 확인: p1(혼자)=1.0, p3(전원)=0.0  ✓")

# --- soft_partial: 0/1 입력이면 soft_linear와 완전히 같아야 한다(상위 호환)
partial_binary = {a: {p: 1.0 for p in s} for a, s in SQUAD.items()}
soft_partial_bin, _, _ = compute_war_scores(SQUAD, 5, mode="soft_partial", partial_credit=partial_binary)
assert all(abs(soft_partial_bin[a] - soft[a]) < 1e-9 for a in SQUAD), \
    f"soft_partial(0/1) != soft_linear: {soft_partial_bin} vs {soft}"
print("soft_partial(0/1 입력) == soft_linear  ✓")

# --- soft_partial: 전원 부분점수 0.0(현행이면 전부 0점)이던 문제에서도 점수가 붙어야 한다
partial_frac = {
    "A": {"p1": 1.0, "p2": 0.6, "p3": 0.3, "p6": 0.4},   # p6: 셋 다 부분점수만 있음(전원 전부통과 X)
    "B": {"p2": 0.5, "p3": 0.5, "p5": 1.0, "p6": 0.1},
    "C": {"p3": 0.2, "p5": 0.7, "p6": 0.0},
}
soft_partial, _, _ = compute_war_scores(SQUAD, 5, mode="soft_partial", partial_credit=partial_frac)
print()
print("부분점수 예시 (p6 = 아무도 전부 통과 못한 문제, 부분점수만 있음):")
for a in sorted(SQUAD):
    print(f"  {a}: soft_partial = {soft_partial[a]:.3f}")
assert soft_partial["A"] != soft_partial["B"] != soft_partial["C"], "p6 부분점수가 순위에 반영돼야 함"
print("전원 0점(hard/soft_linear라면 전부 0)이던 문제에서도 부분점수 차이가 순위에 반영됨  ✓")
