"""Shim → ``meta_agent_evo.pipelines.routing_inference.GMRoutingPipeline``."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meta_agent_evo.pipelines.routing_inference import GMRoutingPipeline

__all__ = ["GMRoutingPipeline"]
