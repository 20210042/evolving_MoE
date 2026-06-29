"""ACC algorithm scoring dispatcher.

Routes each example to the right scorer based on `scoring_kind`: LiveCodeBench scoring, HumanEval
checks, stdin/stdout ACC algorithm execution, or the original assertion-based code scorer. The
ACC path is intentionally isolated here so the original `scorer.py` remains untouched."""

from __future__ import annotations

from typing import Any, Dict

from evaluation.code_exec_algorithm import (
    evaluate_code_score,
    evaluate_stdin_stdout_score,
    extract_helper_code,
)
from evaluation.lcb_score import score_lcb_item


def score_one(
    item: Dict[str, Any],
    prediction_code: str,
    *,
    lcb_timeout: int = 10,
    code_timeout: float = 3.0,
    lcb_release_version: str = "release_v5",
) -> float:
    kind = (item.get("scoring_kind") or "").lower()
    if not kind:
        ds = (item.get("dataset") or "mbpp").lower()
        if ds == "livecodebench":
            kind = "lcb"
        elif ds == "humaneval":
            kind = "humaneval_check"
        elif ds == "acc_algorithm":
            kind = "stdin_stdout"
        else:
            kind = "asserts"

    if kind == "lcb":
        return score_lcb_item(
            str(item["id"]),
            prediction_code,
            timeout=lcb_timeout,
            release_version=lcb_release_version,
        )

    if kind == "stdin_stdout":
        return evaluate_stdin_stdout_score(
            prediction_code,
            item.get("test_cases") or [],
            item.get("eval_spec") or {},
            timeout=code_timeout,
        )

    domain = item.get("domain", "coding")
    ground_truth = item.get("ground_truth")
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
