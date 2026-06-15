"""Phase 1 (Add) & Phase 2 (Delete) independent Action Gate with Non-linear Penalty."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

Action = Literal["noop", "add", "swap", "delete"]


@dataclass
class ActionGateConfig:
    lambda_size: float = 0.005  # Base coefficient for exponential penalty
    scale: float = 0.5          # Scale factor for exponential penalty
    batch_size_ref: int = 50    # Reference batch size; lambda is scaled by (ref/actual) to keep sensitivity invariant


@dataclass
class ActionDecision:
    action: Action
    utility: Dict[str, float]
    marginal_hard_gain_add: float
    mcl_worst: float
    worst_unique_count: int = 0
    recovered_count: int = 0
    demoted_swap: bool = False  # swap conditions fired but newface failed to recover worst's niche -> add


def select_action(
    *,
    roster_ids: List[str],
    worst_id: str | None,
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
    # batch_norm keeps action gate sensitivity invariant across different batch sizes
    batch_norm = cfg.batch_size_ref / batch_size if batch_size > 0 else 1.0
    factor = cfg.scale * (math.exp(cfg.lambda_size) - 1.0)
    lambda_add = math.exp(N * cfg.lambda_size) * factor * batch_norm
    lambda_del = math.exp((N - 1) * cfg.lambda_size) * factor * batch_norm

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
    phase2_delete = False
    mcl = 0.0
    u_delete = 0.0
    unique_worst_solves: Set[str] = set()

    if worst_id and worst_id in r_ids:
        worst_solves = squad_results.get(worst_id, set())
        other_solves = set()
        for rid, solves in squad_results.items():
            if rid != worst_id:
                other_solves.update(solves)

        unique_worst_solves = worst_solves - other_solves
        mcl = len(unique_worst_solves) / batch_size if batch_size > 0 else 0.0

        u_delete = lambda_del - mcl
        if N > 1 and u_delete > 0.0:
            phase2_delete = True

    # 3b. Hole-aware swap guard (prof feedback #3): on a swap, the worst's niche
    # (unique_worst_solves) becomes a hole. Only swap if the newface recovers that
    # whole niche; otherwise demote to ADD so the niche is never silently lost.
    # NOTE: new_pass_ids spans hard_errors ∪ worst_unique (orchestrator probes both),
    # but gh_add above intersects only hard_errors -> phase-1 add utility unchanged.
    recovered = new_pass_ids & unique_worst_solves
    niche_recovered = recovered == unique_worst_solves  # trivially true when niche empty

    # 4. Combine Decisions
    demoted_swap = False
    if phase1_add and phase2_delete:
        if niche_recovered:
            final_action = "swap"
        else:
            final_action = "add"  # keep worst, just add newface
            demoted_swap = True
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
        mcl_worst=mcl,
        worst_unique_count=len(unique_worst_solves),
        recovered_count=len(recovered),
        demoted_swap=demoted_swap,
    )
