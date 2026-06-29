"""Base runner contract for execution backends.

Defines the can_run/run interface expected by the dispatcher and a shared unsupported-result helper
for languages or evaluation modes this pipeline does not execute."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExecutionRunner(ABC):
    @abstractmethod
    def can_run(self, problem: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        problem: dict[str, Any],
        candidate_code: str,
        language: str = "python",
        solution_id: str = "candidate",
    ) -> dict[str, Any]:
        raise NotImplementedError


def unsupported_result(problem: dict[str, Any], solution_id: str, status: str = "unsupported_eval_mode") -> dict[str, Any]:
    return {
        "problem_id": problem.get("problem_id"),
        "solution_id": solution_id,
        "passed": False,
        "status": status,
        "num_tests_passed": 0,
        "num_tests_total": len(problem.get("test_cases") or []),
        "failed_tests": [],
        "runtime_seconds": None,
        "memory_mb": None,
        "stderr": None,
    }
