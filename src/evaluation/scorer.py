"""Unified scoring entry point for evolution / probes."""

from __future__ import annotations

from typing import Any, Dict

from evaluation.code_exec import evaluate_code_score, extract_helper_code
from evaluation.lcb_score import score_lcb_item
from evaluation.metrics import math_verify_score
from utils.helpers import extract_math_answer


def score_one(
    item: Dict[str, Any],
    prediction_code: str,
    *,
    lcb_timeout: int = 10,
    code_timeout: float = 3.0,
    lcb_release_version: str = "release_v5",
) -> float:
    """
    Return 0.0 or 100.0 (pass@1) consistent with evolution thresholds.

    Uses ``item["scoring_kind"]``: ``lcb`` | ``humaneval_check`` | ``asserts``
    Falls back to ``item["dataset"]`` when scoring_kind missing.
    """
    kind = (item.get("scoring_kind") or "").lower()
    if not kind:
        ds = (item.get("dataset") or "mbpp").lower()
        if ds == "livecodebench":
            kind = "lcb"
        elif ds == "humaneval":
            kind = "humaneval_check"
        else:
            kind = "asserts"

    if kind == "lcb":
        return score_lcb_item(
            str(item["id"]),
            prediction_code,
            timeout=lcb_timeout,
            release_version=lcb_release_version,
        )

    domain = item.get("domain", "coding")
    ground_truth = item.get("ground_truth")

    if domain == "math":
        extracted = extract_math_answer(prediction_code)
        return 100.0 if math_verify_score(extracted, ground_truth) else 0.0

    test_code = item.get("test_code") or item.get("test") or ""
    if isinstance(item.get("test_list"), list):
        test_code = "\n".join(item["test_list"])
    entry_point = item.get("entry_point")

    helper_code = extract_helper_code(ground_truth) if domain == "coding" else ""
    if domain == "coding" and "instruction" in item:
        instr_helper = extract_helper_code(item["instruction"])
        if instr_helper:
            helper_code = instr_helper + "\n" + helper_code

    return evaluate_code_score(
        prediction_code,
        test_code,
        entry_point,
        None,
        helper_code,
        timeout=code_timeout,
    )


def pass_at_threshold(score: float, threshold: float = 100.0) -> bool:
    return score >= threshold - 1e-6
