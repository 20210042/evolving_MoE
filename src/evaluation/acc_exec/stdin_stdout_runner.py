from __future__ import annotations

from typing import Any

from .base import ExecutionRunner, unsupported_result
from .comparator import compare_outputs
from .sandbox import run_python_code


class StdinStdoutRunner(ExecutionRunner):
    def can_run(self, problem: dict[str, Any]) -> bool:
        return (problem.get("eval_spec") or {}).get("eval_mode") == "stdin_stdout"

    def run(
        self,
        problem: dict[str, Any],
        candidate_code: str,
        language: str = "python",
        solution_id: str = "candidate",
    ) -> dict[str, Any]:
        if language != "python":
            return unsupported_result(problem, solution_id, "unsupported_language")
        if not self.can_run(problem):
            return unsupported_result(problem, solution_id)
        eval_spec = problem.get("eval_spec") or {}
        comparison = eval_spec.get("comparison") or {}
        passed = 0
        failed_tests: list[dict[str, Any]] = []
        total_runtime = 0.0
        stderr_parts: list[str] = []
        status = "accepted"
        for case in problem.get("test_cases") or []:
            result = run_python_code(
                candidate_code,
                stdin=str((case.get("input") or {}).get("value") or ""),
                timeout_seconds=float(eval_spec.get("timeout_seconds", 5)),
                memory_limit_mb=eval_spec.get("memory_limit_mb", 512),
            )
            total_runtime += result.runtime_seconds
            if result.stderr:
                stderr_parts.append(result.stderr)
            if result.timed_out:
                status = "timeout"
                failed_tests.append(failed_case(case, {"kind": "stdout", "value": result.stdout}, "timeout"))
                continue
            if result.returncode != 0:
                status = "runtime_error"
                failed_tests.append(failed_case(case, {"kind": "stdout", "value": result.stdout}, result.stderr))
                continue
            expected = (case.get("expected_output") or {}).get("value")
            if compare_outputs(result.stdout, expected, comparison):
                passed += 1
            else:
                if status == "accepted":
                    status = "wrong_answer"
                failed_tests.append(failed_case(case, {"kind": "stdout", "value": result.stdout}, None))
        total = len(problem.get("test_cases") or [])
        return {
            "problem_id": problem.get("problem_id"),
            "solution_id": solution_id,
            "passed": passed == total and total > 0,
            "status": "accepted" if passed == total and total > 0 else status,
            "num_tests_passed": passed,
            "num_tests_total": total,
            "failed_tests": failed_tests,
            "runtime_seconds": total_runtime,
            "memory_mb": None,
            "stderr": "\n".join(stderr_parts) if stderr_parts else None,
        }


def failed_case(case: dict[str, Any], actual: dict[str, Any], error: str | None) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "input": case.get("input"),
        "expected_output": case.get("expected_output"),
        "actual_output": actual,
        "error_message": error,
    }
