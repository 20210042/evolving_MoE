from __future__ import annotations

import math
from typing import Any


def compare_outputs(actual: Any, expected: Any, comparison: dict[str, Any] | None = None) -> bool:
    comparison = comparison or {}
    comparison_type = comparison.get("type", "exact_or_token_match")
    if comparison_type == "special_judge_unresolved":
        return False
    if comparison_type == "numeric_tolerance":
        tolerance = comparison.get("numeric_tolerance") or {}
        return numeric_match(actual, expected, tolerance)
    if comparison_type == "case_insensitive":
        return normalize_text(actual, False, comparison) == normalize_text(expected, False, comparison)
    if comparison_type == "exact":
        return normalize_text(actual, comparison.get("case_sensitive", True), comparison) == normalize_text(
            expected, comparison.get("case_sensitive", True), comparison
        )
    return exact_or_token_match(actual, expected, comparison)


def normalize_text(value: Any, case_sensitive: bool = True, comparison: dict[str, Any] | None = None) -> str:
    comparison = comparison or {}
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if comparison.get("strip_trailing_whitespace", True):
        text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    if not case_sensitive:
        text = text.lower()
    return text


def exact_or_token_match(actual: Any, expected: Any, comparison: dict[str, Any]) -> bool:
    case_sensitive = comparison.get("case_sensitive", True)
    left = normalize_text(actual, case_sensitive, comparison)
    right = normalize_text(expected, case_sensitive, comparison)
    if left == right:
        return True
    return left.split() == right.split()


def numeric_match(actual: Any, expected: Any, tolerance: dict[str, Any]) -> bool:
    left_tokens = str(actual).split()
    right_tokens = str(expected).split()
    if len(left_tokens) != len(right_tokens):
        return False
    abs_tol = float(tolerance.get("abs_tol", tolerance.get("absolute", 1e-6)))
    rel_tol = float(tolerance.get("rel_tol", tolerance.get("relative", 1e-6)))
    for left, right in zip(left_tokens, right_tokens):
        try:
            if not math.isclose(float(left), float(right), abs_tol=abs_tol, rel_tol=rel_tol):
                return False
        except ValueError:
            if left != right:
                return False
    return True
