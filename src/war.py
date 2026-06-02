"""Wins-above-replacement style marginal contribution on a batch."""

from __future__ import annotations

import logging
import random
from typing import Dict, Set


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
    squad_results: Dict[str, Set[str]],
    *,
    tiebreak: str = "random",
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random(0)
    items = list(war_scores.keys())

    def sort_key(k: str):
        w = war_scores[k]
        squad_size = len(squad_results.get(k, ()))
        if tiebreak == "alphabetical":
            return (w, k)
        if tiebreak == "random":
            return (w, rng.random())
        # size-based: fewer unique solves first (more redundant members); add noise
        return (w, squad_size, rng.random())

    return min(items, key=sort_key)
