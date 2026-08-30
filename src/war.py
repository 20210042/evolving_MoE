"""Wins-above-replacement style marginal contribution on a batch."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Set


def compute_war_scores(
    squad_results: Dict[str, Set[str]],
    total_batch_size: int,
    *,
    tiebreak: str = "random",
    rng: random.Random | None = None,
    mode: str = "hard",
    partial_credit: Dict[str, Dict[str, float]] | None = None,
    score_ids: Set[str] | None = None,
) -> tuple[Dict[str, float], int, float]:
    """
    Returns (war_scores dict, upper_bound_count, upper_bound_rate).

    score_ids: 점수 계산에 쓸 문제 id 집합. None(기본)이면 기존 동작과 바이트 동일.
        지정하면 **적합도만** 이 집합으로 제한하고 upper_bound는 배치 전체로 낸다 —
        UB는 이전 시드와 비교하는 보고용 수치라 정의를 바꾸면 안 되기 때문이다.

    mode="hard" (default, 기존 동작 그대로):
        war = |all_solved| - |union_{others} solved|  (non-redundant solves for each agent)
        "정확히 나만 푼" 문제만 1점. 로스터가 커질수록 그 사건이 구조적으로 희박해진다
        (11명 기준 배치 50당 1.0개; 갈리는 문제 26.8% 중 92.5%를 버린다).

    mode="soft_linear":
        내가 푼 문제 i마다 (E - n_i) / (E - 1) 점.  n_i = 그 문제를 푼 사람 수, E = 로스터 크기.
        = "나 말고 나머지 중 이 문제를 못 푼 사람의 비율" — 대체 불가능성을 전부/전무가 아니라
        정도로 잰다. 양 끝값은 hard와 동일(혼자 풀면 1.0, 전원이 풀면 0.0)하고 가운데만 채운다.

    mode="soft_partial" (partial_credit 필수):
        squad_results(0/1, "전부 통과했나")는 upper_bound 계산에만 쓰고, 점수는 연속 부분점수
        c_{i,j} in [0,1](= 통과 테스트 비율, score_acc_item_partial)로 낸다.
          n_i = Σ_j c_{i,j}                 (문제 i의 "실효 해결자 수" — soft_linear의 정수 카운트를
                                              연속으로 일반화한 것. c가 전부 0/1이면 n_i가 그대로
                                              soft_linear의 n_i와 같다)
          score_j = Σ_i  c_{i,j} · (E − n_i) / (E − 1)
        → c가 0/1이면 soft_linear와 완전히 같은 값이 나온다(상위 호환). 전원이 0점(부분점수도 0)인
        문제였던 곳에서도 이제 c>0인 사람에게 점수가 붙는다 — soft_linear가 못 건드리던 부분.
    """
    if not squad_results:
        return {}, 0, 0.0

    all_solved: Set[str] = set().union(*squad_results.values())
    upper_bound_count = len(all_solved)
    upper_bound_rate = (upper_bound_count / total_batch_size) * 100 if total_batch_size > 0 else 0.0

    logging.info(
        "🏆 [STATIC UPPER BOUND] Results: %s/%s (%.2f%%) solved by at least one expert.",
        upper_bound_count,
        total_batch_size,
        upper_bound_rate,
    )

    # 적합도 전용 뷰. score_ids가 None이면 원본 그대로라 기존 경로와 완전히 동일하다.
    if score_ids is None:
        eff_results = squad_results
        eff_partial = partial_credit
    else:
        eff_results = {a: (s & score_ids) for a, s in squad_results.items()}
        eff_partial = (
            None if partial_credit is None
            else {a: {p: c for p, c in cmap.items() if p in score_ids}
                  for a, cmap in partial_credit.items()}
        )

    war_scores: Dict[str, float] = {}
    if mode == "soft_partial":
        if eff_partial is None:
            raise ValueError("mode='soft_partial' requires partial_credit")
        partial_credit = eff_partial
        E = len(eff_results)
        pids = set()
        for cmap in partial_credit.values():
            pids.update(cmap)
        n_eff: Dict[str, float] = {
            pid: sum(partial_credit.get(a, {}).get(pid, 0.0) for a in eff_results)
            for pid in pids
        }
        for agent_id in eff_results:
            cmap = partial_credit.get(agent_id, {})
            if E <= 1:
                war_scores[agent_id] = sum(cmap.values())
            else:
                war_scores[agent_id] = sum(
                    c * (E - n_eff[pid]) / (E - 1) for pid, c in cmap.items() if c > 0
                )
    elif mode == "soft_linear":
        n_solvers: Dict[str, int] = {}
        for solved_set in eff_results.values():
            for pid in solved_set:
                n_solvers[pid] = n_solvers.get(pid, 0) + 1
        E = len(eff_results)
        for agent_id, solved_set in eff_results.items():
            if E <= 1:
                war_scores[agent_id] = float(len(solved_set))
            else:
                war_scores[agent_id] = sum(
                    (E - n_solvers[pid]) / (E - 1) for pid in solved_set
                )
    else:
        eff_all_solved: Set[str] = set().union(*eff_results.values()) if eff_results else set()
        for agent_id, solved_set in eff_results.items():
            others_solved = set().union(*[res for a, res in eff_results.items() if a != agent_id])
            war_scores[agent_id] = len(eff_all_solved) - len(others_solved)

    return war_scores, upper_bound_count, upper_bound_rate


def pick_worst_agent(
    war_scores: Dict[str, int],
    roster: List[Dict[str, Any]],
    *,
    tiebreak: str = "random",
    rng: random.Random | None = None,
    unique_rate_map: Dict[str, float] | None = None,
) -> str | None:
    rng = rng or random.Random(0)
    active_ids = set(war_scores.keys())

    # Candidates for eviction: only agents whose lives are <= 0
    candidates = [
        p for p in roster
        if p.get("id") in active_ids and p.get("lives", 3) <= 0
    ]

    if not candidates:
        return None

    def sort_key(p: Dict[str, Any]):
        # Window mode unifies the worst-pick on the same accumulated unique-rate the
        # lives/delete gate use; legacy mode keeps the running average_war ordering.
        if unique_rate_map is not None:
            avg_war = unique_rate_map.get(p.get("id", ""), 0.0)
        else:
            avg_war = p.get("average_war", 0.0)
        active_steps = p.get("active_steps", 0)
        pid = p.get("id", "")

        if tiebreak == "alphabetical":
            return (avg_war, -active_steps, pid)
        if tiebreak == "random":
            return (avg_war, -active_steps, rng.random())
        # size-based: fewer unique solves first (more redundant members); add noise
        squad_size = len(p.get("strengths", ""))
        return (avg_war, -active_steps, squad_size, rng.random())

    chosen_agent = min(candidates, key=sort_key)
    return chosen_agent.get("id", "")
