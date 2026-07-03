"""Unified scoring entry point for evolution / probes."""

from __future__ import annotations

import re
from typing import Any, Dict

from evaluation.code_exec import evaluate_code_score, extract_helper_code
from evaluation.lcb_score import score_lcb_item
from evaluation.metrics import math_verify_score
from utils.helpers import extract_math_answer


# --- Multiple-choice aware matching (NuminaMath) ---------------------------
# NuminaMath MC items have inconsistent gold format (option letter "B" vs the
# value "a+b = 3"), so a model that boxes one format fails the ~half whose gold
# uses the other — a scoring artifact (~6pp), not a model error. This helper is
# ADDITIVE: it only turns some MC failures into passes when the boxed answer
# matches the *correct* option by letter OR by value. Non-MC items are untouched.
_MC_OPT_RE = re.compile(r"(?:^|\n|\s)\(?([A-D])[\):.]\s*(.+?)(?=(?:\n|\s)\(?[A-D][\):.]|\Z)", re.S)


def _parse_mc_options(instruction: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for m in _MC_OPT_RE.finditer(instruction or ""):
        opts.setdefault(m.group(1), m.group(2).strip()[:80])
    return opts


def _mc_correct_letter(gold: str, opts: Dict[str, str]) -> str | None:
    g = (gold or "").strip()
    for pat in (r"^\(?([A-D])[\):.\s]", r"^\\?text\{?\s*\(?([A-D])", r"^([A-D])$"):
        m = re.match(pat, g)
        if m:
            return m.group(1)
    for L, v in opts.items():  # gold is a value → find the option whose value matches
        try:
            if math_verify_score(v, gold):
                return L
        except Exception:
            pass
    return None


def _mc_letter_value_match(pred: str, gold: str, instruction: str) -> bool:
    opts = _parse_mc_options(instruction)
    if len(opts) < 3:  # not a multiple-choice item → no change
        return False
    cl = _mc_correct_letter(str(gold or ""), opts)
    if not cl:
        return False
    p = (pred or "").strip()
    if re.fullmatch(r"\(?([A-D])\)?", p) and re.sub(r"[^A-D]", "", p) == cl:
        return True  # model boxed the correct option letter
    try:
        if math_verify_score(p, opts.get(cl, "")):
            return True  # model boxed the correct option's value
    except Exception:
        pass
    return False


def _json(x):
    import json
    return json.loads(x) if isinstance(x, str) else x


def score_acc_item(item: Dict[str, Any], code: str) -> float:
    """QuantCat/TACO coding: exec candidate via acc_exec (stdin / function_call / gfg)."""
    from evaluation.acc_exec import ExecutionInterface
    problem = dict(item)
    problem["eval_spec"] = _json(item.get("eval_spec")) or {}
    problem["test_cases"] = _json(item.get("test_cases")) or []
    out = ExecutionInterface().run(problem, code, solution_id="candidate")
    return 100.0 if out.get("passed") else 0.0


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

    # QuantCat/TACO coding: eval_spec-driven execution (stdin / function_call / gfg).
    if kind == "acc" or item.get("eval_spec"):
        return score_acc_item(item, prediction_code)

    domain = item.get("domain", "coding")
    ground_truth = item.get("ground_truth")

    if domain == "math":
        extracted = extract_math_answer(prediction_code)
        if math_verify_score(extracted, ground_truth):
            return 100.0
        # MC-aware: accept letter↔value equivalence for multiple-choice items
        if _mc_letter_value_match(extracted, ground_truth, item.get("instruction", "")):
            return 100.0
        return 0.0

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
