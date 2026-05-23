"""Phase 1 (Add) & Phase 2 (Delete) independent Action Gate with Non-linear Penalty."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

Action = Literal["noop", "add", "swap", "delete"]


@dataclass
class ActionGateConfig:
    lambda_size: float = 0.005  # Base coefficient for exponential penalty


@dataclass
class ActionDecision:
    action: Action
    utility: Dict[str, float]
    marginal_hard_gain_add: float
    mcl_worst: float


def select_action(
    *,
    roster_ids: List[str],
    worst_id: str,
    squad_results: Dict[str, Set[str]],
    hard_errors: List[str],
    new_pass_ids: Set[str],
    batch_size: int,
    cfg: ActionGateConfig,
) -> ActionDecision:
    """
    Independent Phase 1 (Add or Stay) and Phase 2 (Delete or Stay) Roster Decisions.
    
    Uses an exponential size penalty P(N) = 0.5 * (exp(N * lambda_size) - 1):
    - Marginal cost to add (N -> N+1): 0.5 * exp(N * lambda_size) * (exp(lambda_size) - 1.0)
    - Marginal savings to delete (N -> N-1): 0.5 * exp((N-1) * lambda_size) * (exp(lambda_size) - 1.0)
    """
    r_ids = list(roster_ids)
    N = len(r_ids)
    
    # 1. Non-linear marginal penalty/savings coefficients (Exponential)
    factor = 0.5 * (math.exp(cfg.lambda_size) - 1.0)
    lambda_add = math.exp(N * cfg.lambda_size) * factor
    lambda_del = math.exp((N - 1) * cfg.lambda_size) * factor

    # 2. Phase 1 (Add) Utility
    # Marginal hard gain is scaled against the entire batch_size (overall solve rate delta)
    gh_add = 0.0
    if batch_size > 0:
        solved_hard = set(hard_errors) & new_pass_ids
        gh_add = len(solved_hard) / batch_size
        
    u_add = gh_add - lambda_add
    phase1_add = u_add > 0.0

    # 3. Phase 2 (Delete) Utility
    # mcl is the fraction of total batch problems uniquely solved by the worst agent
    worst_solves = squad_results.get(worst_id, set())
    other_solves = set()
    for rid, solves in squad_results.items():
        if rid != worst_id:
            other_solves.update(solves)
            
    unique_worst_solves = worst_solves - other_solves
    mcl = len(unique_worst_solves) / batch_size if batch_size > 0 else 0.0
    
    u_delete = lambda_del - mcl
    phase2_delete = False
    if N > 1 and u_delete > 0.0:
        phase2_delete = True

    # 4. Combine Decisions
    if phase1_add and phase2_delete:
        final_action = "swap"
    elif phase1_add and not phase2_delete:
        final_action = "add"
    elif not phase1_add and phase2_delete:
        final_action = "delete"
    else:
        final_action = "noop"

    utility = {
        "u_add": u_add,
        "u_delete": u_delete
    }

    return ActionDecision(
        action=final_action,
        utility=utility,
        marginal_hard_gain_add=gh_add,
        mcl_worst=mcl
    )
