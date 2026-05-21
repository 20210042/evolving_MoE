from __future__ import annotations

from typing import Any

from acc_data_pipeline.schemas.problem import apply_problem_defaults, dump_problem


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return dump_problem(apply_problem_defaults(record))
