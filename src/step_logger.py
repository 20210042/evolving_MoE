"""Per-step JSONL logging for evolution runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepLogContext:
    run_id: str
    step: int
    epoch: int
    batch_idx: int
    dataset: str
    seed: int
    wallclock_start: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.wallclock_start


class StepLogger:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def log_path(self, run_id: str) -> str:
        os.makedirs(os.path.join(self.results_dir, run_id), exist_ok=True)
        return os.path.join(self.results_dir, run_id, "evolution_log.jsonl")

    def append(
        self,
        ctx: StepLogContext,
        record: Dict[str, Any],
    ) -> None:
        record = {
            **record,
            "step": ctx.step,
            "epoch": ctx.epoch,
            "batch_idx": ctx.batch_idx,
            "dataset": ctx.dataset,
            "seed": ctx.seed,
            "run_id": ctx.run_id,
            "wallclock_sec": ctx.elapsed(),
        }
        path = self.log_path(ctx.run_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_roster_snapshot(self, run_id: str, step: int, roster: List[Dict[str, Any]]) -> str:
        d = os.path.join(self.results_dir, run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"roster_step_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roster, f, indent=4)
        return path
