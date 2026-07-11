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
) -> tuple[Dict[str, int], int, float]:
    """
    Returns (war_scores dict, upper_bound_count, upper_bound_rate).

    war = |all_solved| - |union_{others} solved|  (non-redundant solves for each agent)
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

    war_scores: Dict[str, int] = {}
    for agent_id, solved_set in squad_results.items():
        others_solved = set().union(*[res for a, res in squad_results.items() if a != agent_id])
        war_scores[agent_id] = len(all_solved) - len(others_solved)

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
