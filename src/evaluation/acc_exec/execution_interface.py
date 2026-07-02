from __future__ import annotations

from typing import Any

from .base import ExecutionRunner, unsupported_result
from .function_call_runner import FunctionCallRunner
from .stdin_stdout_runner import StdinStdoutRunner


class SpecialJudgeRunner(ExecutionRunner):
    def can_run(self, problem: dict[str, Any]) -> bool:
        return False

    def run(
        self,
        problem: dict[str, Any],
        candidate_code: str,
        language: str = "python",
        solution_id: str = "candidate",
    ) -> dict[str, Any]:
        return unsupported_result(problem, solution_id, "unsupported_eval_mode")


class ExecutionInterface:
    def __init__(self) -> None:
        self.runners: list[ExecutionRunner] = [
            StdinStdoutRunner(),
            FunctionCallRunner(),
            SpecialJudgeRunner(),
        ]

    def can_run(self, problem: dict[str, Any]) -> bool:
        return any(runner.can_run(problem) for runner in self.runners)

    def run(
        self,
        problem: dict[str, Any],
        candidate_code: str,
        language: str = "python",
        solution_id: str = "candidate",
    ) -> dict[str, Any]:
        for runner in self.runners:
            if runner.can_run(problem):
                return runner.run(problem, candidate_code, language=language, solution_id=solution_id)
        return unsupported_result(problem, solution_id, "unsupported_eval_mode")


def prepare_execution_records(
    records: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or {}
    supported_languages = set(config.get("supported_languages", ["python"]))
    default_timeout = config.get("default_timeout_seconds", 5)
    default_memory = config.get("default_memory_limit_mb", 512)
    keep_unsupported = (config.get("unsupported") or {}).get("keep_records", True)
    mark_manual = (config.get("unsupported") or {}).get("mark_requires_manual_review", True)
    output: list[dict[str, Any]] = []
    by_eval_mode: dict[str, int] = {
        "stdin_stdout": 0,
        "function_call": 0,
        "sql_execution": 0,
        "special_judge": 0,
        "unsupported": 0,
        "self_repair": 0,
    }
    supported_count = 0
    unsupported_count = 0
    manual_count = 0
    for record in records:
        spec = record.setdefault("eval_spec", {})
        flags = record.setdefault("quality_flags", {})
        spec.setdefault("language", config.get("default_language", "python"))
        spec.setdefault("timeout_seconds", default_timeout)
        spec.setdefault("memory_limit_mb", default_memory)
        spec.setdefault("comparison", (config.get("comparison") or {}))
        if spec.get("language") not in supported_languages:
            mark_unsupported(record, "unsupported_language", mark_manual)
        elif not valid_cases_for_mode(record):
            mark_unsupported(record, "missing_executable_tests", mark_manual)
        elif spec.get("eval_mode") == "special_judge":
            mark_unsupported(record, "special_judge_not_implemented", mark_manual)
        elif spec.get("eval_mode") == "self_repair":
            mark_unsupported(record, "self_repair_requires_repaired_candidate", mark_manual)
        elif spec.get("eval_mode") == "sql_execution":
            mark_unsupported(record, "sql_execution_runner_not_implemented", mark_manual)
        elif spec.get("eval_mode") not in {"stdin_stdout", "function_call"}:
            mark_unsupported(record, "unsupported_eval_mode", mark_manual)
        else:
            flags["is_supported_for_execution"] = True
            flags.setdefault("requires_manual_review", False)
        mode = spec.get("eval_mode", "unsupported")
        by_eval_mode[mode] = by_eval_mode.get(mode, 0) + 1
        if flags.get("is_supported_for_execution"):
            supported_count += 1
        else:
            unsupported_count += 1
        if flags.get("requires_manual_review"):
            manual_count += 1
        if flags.get("is_supported_for_execution") or keep_unsupported:
            output.append(record)
    report = {
        "input_count": len(records),
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "by_eval_mode": by_eval_mode,
        "requires_manual_review": manual_count,
    }
    return output, report


def valid_cases_for_mode(record: dict[str, Any]) -> bool:
    mode = (record.get("eval_spec") or {}).get("eval_mode")
    for case in record.get("test_cases") or []:
        inp = case.get("input") or {}
        out = case.get("expected_output") or {}
        if mode == "function_call" and inp.get("kind") == "function_args" and out.get("kind") == "return_value":
            return True
        if mode == "sql_execution" and inp.get("kind") in {"sql_query", "database_context", "raw"} and out.get("kind") in {"sql", "raw"}:
            return True
        if mode in {"stdin_stdout", "special_judge", "self_repair"} and inp.get("kind") == "stdin" and out.get("kind") == "stdout":
            return True
    return False


def mark_unsupported(record: dict[str, Any], reason: str, mark_manual: bool) -> None:
    spec = record.setdefault("eval_spec", {})
    original_mode = spec.get("eval_mode")
    spec["eval_mode"] = "unsupported"
    spec["unsupported_reason"] = reason
    flags = record.setdefault("quality_flags", {})
    flags["is_supported_for_execution"] = False
    flags["requires_manual_review"] = mark_manual
    warnings = flags.setdefault("warnings", [])
    if reason not in warnings:
        warnings.append(reason)
    if original_mode and original_mode != "unsupported":
        record.setdefault("native_metadata", {}).setdefault("raw_fields", {})[
            "original_eval_mode_before_prepare"
        ] = original_mode
