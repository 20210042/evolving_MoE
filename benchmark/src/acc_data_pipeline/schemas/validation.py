from __future__ import annotations

from typing import Any

from .problem import NormalizedProblem, apply_problem_defaults, validate_problem_dict


def validate_problem(record: dict[str, Any]) -> tuple[bool, list[str]]:
    record = apply_problem_defaults(record)
    errors = validate_problem_dict(record)
    if errors:
        return False, errors
    try:
        NormalizedProblem.model_validate(record)
    except Exception as exc:  # pydantic and fallback both raise here
        return False, [str(exc)]
    return True, []


def validate_required_final_fields(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "problem_id",
        "source",
        "task_family",
        "problem_statement",
        "test_cases",
        "eval_spec",
        "native_metadata",
        "quality_flags",
        "dedup",
    ):
        if field not in record:
            errors.append(f"missing final field: {field}")
    statement = record.get("problem_statement") or {}
    if not str(statement.get("raw") or "").strip():
        errors.append("empty problem_statement.raw")
    if not record.get("test_cases"):
        errors.append("empty test_cases")
    return errors
