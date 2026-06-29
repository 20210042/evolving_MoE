"""Algorithmic-task filter for the normalized corpus.

Removes output-prediction, non-algorithmic, underspecified, unsupported, or non-executable records
so later stages operate only on code-generation and self-repair problems with usable tests."""

from __future__ import annotations

from collections import Counter
from typing import Any

DEFAULT_FILTER_CONFIG: dict[str, Any] = {
    "allowed_task_families": [
        "algorithmic_code_generation",
        "self_repair",
    ],
    "excluded_task_families": [
        "pure_code_execution",
        "test_output_prediction",
        "non_algorithmic_code_task",
    ],
    "require_problem_statement": True,
    "require_test_cases": True,
    "require_supported_eval_mode": True,
    "min_statement_chars": 30,
}


def removal_reason(record: dict[str, Any], config: dict[str, Any]) -> str | None:
    task_family = record.get("task_family", "unknown")
    if task_family in set(config.get("excluded_task_families", [])):
        if task_family == "pure_code_execution":
            return "pure_code_execution_task"
        if task_family == "test_output_prediction":
            return "test_output_prediction_task"
        return "non_algorithmic_code_task"
    if task_family not in set(config.get("allowed_task_families", [])):
        return "unsupported_task_family"
    statement = ((record.get("problem_statement") or {}).get("raw") or "").strip()
    if config.get("require_problem_statement", True) and not statement:
        return "missing_statement"
    if len(statement) < int(config.get("min_statement_chars", 0)):
        return "statement_too_short"
    if config.get("require_test_cases", True) and not executable_test_cases(record):
        return "missing_tests"
    eval_mode = (record.get("eval_spec") or {}).get("eval_mode")
    if config.get("require_supported_eval_mode", True) and eval_mode == "unsupported":
        return "unsupported_eval_mode"
    if task_family == "self_repair" and not has_full_candidate_solution(record):
        return "self_repair_missing_candidate_solution"
    return None


def executable_test_cases(record: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    task_family = record.get("task_family")
    for case in record.get("test_cases") or []:
        if not isinstance(case, dict):
            continue
        inp = case.get("input") or {}
        out = case.get("expected_output") or {}
        if inp.get("kind") in {"stdin", "function_args"} and out.get("kind") in {
            "stdout",
            "return_value",
        }:
            cases.append(case)
    return cases


def has_full_candidate_solution(record: dict[str, Any]) -> bool:
    if record.get("starter_code"):
        return True
    raw_fields = ((record.get("native_metadata") or {}).get("raw_fields") or {})
    keys = {
        "candidate_solution",
        "buggy_solution",
        "incorrect_solution",
        "base_code",
        "code",
    }
    for key in keys:
        value = raw_fields.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def filter_records(
    records: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = {**DEFAULT_FILTER_CONFIG, **(config or {})}
    kept: list[dict[str, Any]] = []
    report_items: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        reason = removal_reason(record, config)
        if reason is None:
            kept.append(record)
            report_items.append(
                {
                    "problem_id": record.get("problem_id"),
                    "source": record.get("source"),
                    "removed": False,
                    "reason": None,
                }
            )
        else:
            reasons[reason] += 1
            report_items.append(
                {
                    "problem_id": record.get("problem_id"),
                    "source": record.get("source"),
                    "removed": True,
                    "reason": reason,
                }
            )
    report = {
        "input_count": len(records),
        "kept_count": len(kept),
        "removed_count": len(records) - len(kept),
        "removed_by_reason": dict(reasons),
        "items": report_items,
    }
    return kept, report
