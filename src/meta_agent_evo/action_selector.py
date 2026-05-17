"""3-action roster gate: keep (A1), add (A2), swap (A3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

Action = Literal["noop", "add", "swap"]


@dataclass
class ActionGateConfig:
    alpha_stability: float = 1.0
    lambda_size: float = 0.01
    epsilon_floor: float = 0.05
    swap_max_gain: Optional[float] = None  # if set: u_swap <= this → swap, else → add
    use_wilson_ci: bool = True
    wilson_confidence: float = 0.95


@dataclass
class ActionDecision:
    action: Action
    utility: Dict[str, float]
    marginal_hard_gain_add: float
    marginal_hard_gain_swap_extra: float
    mcl_worst: float


def wilson_lower_bound(k: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return 0.0
    if confidence >= 0.99:
        z = 2.576
    else:
        z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin)


def _covered_by_roster(
    q: str,
    roster_ids: List[str],
    squad_results: Dict[str, Set[str]],
) -> bool:
    return any(q in squad_results.get(r, set()) for r in roster_ids)


def solve_rate_union(
    roster_ids: List[str],
    squad_results: Dict[str, Set[str]],
    probe: List[str],
    extra_pass: Set[str],
) -> float:
    if not probe:
        return 0.0
    n = 0
    for q in probe:
        n += int(_covered_by_roster(q, roster_ids, squad_results) or (q in extra_pass))
    return n / len(probe)


def select_action(
    *,
    roster_ids: List[str],
    worst_id: str,
    squad_results: Dict[str, Set[str]],
    probe_hard: List[str],
    probe_stability: List[str],
    new_pass_ids: Set[str],
    cfg: ActionGateConfig,
) -> ActionDecision:
    """
    ``new_pass_ids``: question ids in the probe union where the **candidate** persona reached pass@1.
    """
    r_ids = list(roster_ids)
    r_minus = [x for x in r_ids if x != worst_id]

    def sr_add(p: List[str]) -> float:
        return solve_rate_union(r_ids, squad_results, p, new_pass_ids)

    def sr_swap(p: List[str]) -> float:
        return solve_rate_union(r_minus, squad_results, p, new_pass_ids)

    def sr_baseline(p: List[str]) -> float:
        return solve_rate_union(r_ids, squad_results, p, set())

    def sr_baseline_minus(p: List[str]) -> float:
        return solve_rate_union(r_minus, squad_results, p, set())

    gh_add = sr_add(probe_hard) - sr_baseline(probe_hard)
    gs_add = sr_add(probe_stability) - sr_baseline(probe_stability)
    gh_swap = sr_swap(probe_hard) - sr_baseline(probe_hard)
    gs_swap = sr_swap(probe_stability) - sr_baseline(probe_stability)

    u_add = gh_add + cfg.alpha_stability * min(0.0, gs_add) - cfg.lambda_size
    u_swap = gh_swap + cfg.alpha_stability * min(0.0, gs_swap)

    mcl = sr_baseline(probe_hard + probe_stability) - sr_baseline_minus(probe_hard + probe_stability)
    mcg_swap_extra = sr_swap(probe_hard) - sr_add(probe_hard)

    # Wilson CI on additional hard passes
    extra_hard = [q for q in probe_hard if (not _covered_by_roster(q, r_ids, squad_results)) and q in new_pass_ids]
    k_new = len(extra_hard)
    n_h = len(probe_hard)

    ci_ok_add = True
    ci_ok_swap = True
    if cfg.use_wilson_ci and n_h > 0:
        ci_ok_add = wilson_lower_bound(k_new, n_h, cfg.wilson_confidence) > 1e-6 or k_new == 0
        extra_hard_swap = [
            q
            for q in probe_hard
            if (not _covered_by_roster(q, r_minus, squad_results)) and q in new_pass_ids
        ]
        k_sw = len(extra_hard_swap)
        ci_ok_swap = wilson_lower_bound(k_sw, n_h, cfg.wilson_confidence) > 1e-6 or k_sw == 0

    utility = {"A1": 0.0, "A2": u_add, "A3": u_swap}

    max_u = max(u_add, u_swap)
    if max_u <= cfg.epsilon_floor:
        return ActionDecision("noop", utility, gh_add, mcg_swap_extra, mcl)

    # Pure argmax: swap wins unless swap_max_gain threshold routes to add
    if u_swap >= u_add:
        if cfg.swap_max_gain is not None and u_swap > cfg.swap_max_gain and ci_ok_add:
            # High-gain candidate: add rather than swap
            return ActionDecision("add", utility, gh_add, mcg_swap_extra, mcl)
        if ci_ok_swap:
            return ActionDecision("swap", utility, gh_add, mcg_swap_extra, mcl)

    if u_add > cfg.epsilon_floor and ci_ok_add:
        return ActionDecision("add", utility, gh_add, mcg_swap_extra, mcl)

    if u_swap > cfg.epsilon_floor and ci_ok_swap:
        return ActionDecision("swap", utility, gh_add, mcg_swap_extra, mcl)

    return ActionDecision("noop", utility, gh_add, mcg_swap_extra, mcl)
