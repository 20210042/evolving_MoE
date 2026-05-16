"""Backward-compatible import: ensure ``src/`` is on path, then re-export orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meta_agent_evo.orchestrator import GMEvolutionOrchestrator

__all__ = ["GMEvolutionOrchestrator"]
