"""Unified scoring entry point for evolution / probes."""

from __future__ import annotations

import re
from typing import Any, Dict

from evaluation.code_exec import evaluate_code_score, extract_helper_code
from evaluation.lcb_score import score_lcb_item
from evaluation.metrics import math_verify_score
from utils.helpers import extract_math_answer, strip_thinking_channels


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
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return x


def score_acc_item(item: Dict[str, Any], code: str) -> float:
    """QuantCat/TACO coding: exec candidate via acc_exec (stdin / function_call / gfg)."""
    from evaluation.acc_exec import ExecutionInterface
    problem = dict(item)
    problem["eval_spec"] = _json(item.get("eval_spec")) or {}
    problem["test_cases"] = _json(item.get("test_cases")) or []
    out = ExecutionInterface().run(problem, code, solution_id="candidate")
    return 100.0 if out.get("passed") else 0.0


def score_acc_item_partial(item: Dict[str, Any], code: str) -> float:
    """통과 테스트 비율(0~100). score_acc_item과 별개 함수 — 그 0/100 계약은 binning·배포·
    평가 스크립트가 그대로 소비하므로 절대 바꾸지 않는다. 이 함수는 진화의 WAR 계산 전용
    (라벨링 실험, seed20211002~)이며, 실행기가 이미 반환하는 num_tests_passed/num_tests_total을
    그대로 쓴다(acc_exec/*_runner.py). eval_spec 등으로 total이 없는 문제는 기존 0/100로 폴백한다."""
    from evaluation.acc_exec import ExecutionInterface
    problem = dict(item)
    problem["eval_spec"] = _json(item.get("eval_spec")) or {}
    problem["test_cases"] = _json(item.get("test_cases")) or []
    out = ExecutionInterface().run(problem, code, solution_id="candidate")
    total = out.get("num_tests_total") or 0
    if total <= 0:
        return 100.0 if out.get("passed") else 0.0
    passed = out.get("num_tests_passed") or 0
    return 100.0 * passed / total


_QASC_OPT_RE = re.compile(
    r"(?:^|\n|\s)\(?([A-H])[\):.]\s*(.+?)(?=(?:\n|\s)\(?[A-H][\):.]|\Z)",
    re.S,
)


def _norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _parse_qasc_options(instruction: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for m in _QASC_OPT_RE.finditer(instruction or ""):
        opts.setdefault(m.group(1).upper(), _norm_text(m.group(2))[:160])
    return opts


def _extract_qasc_letter(prediction: str) -> str | None:
    text = strip_thinking_channels(prediction or "")
    # 우선순위 순서다. 앞선 패턴이 하나라도 맞으면 거기서 끝내야 한다 —
    # 마지막 "(X)" 패턴은 보기 나열 "(A) ... (H) ..."을 전부 잡으므로,
    # 매치를 한 리스트에 모아 마지막을 취하면 답을 명시한 문장이 있어도
    # 나열된 마지막 보기 글자가 이겨버린다.
    patterns = [
        # 답 선언은 같은 줄에서만 인정한다. \s* 로 두면 "각 option:\n(A) ..." 처럼
        # 산문 뒤 줄바꿈 너머의 첫 보기를 답으로 오인한다.
        r"(?i)(?:final\s+answer|answer|choice|option)[ \t]*(?:is|:)?[ \t]*\(?[ \t]*([A-H])[ \t]*\)?",
        r"(?m)^\s*\(?\s*([A-H])\s*\)?\s*(?:[.)])?\s*$",
        r"\(([A-H])\)",
    ]
    for pat in patterns:
        matches = [m.group(1).upper() for m in re.finditer(pat, text)]
        if matches:
            return matches[-1]
    return None


def score_qasc_item(item: Dict[str, Any], prediction: str) -> float:
    """QASC science MC: exact match on answer letter A-H, with option-text fallback."""
    gold = str(item.get("ground_truth", "")).strip().upper()
    if gold not in set("ABCDEFGH"):
        return 0.0
    pred_letter = _extract_qasc_letter(prediction)
    if pred_letter:
        return 100.0 if pred_letter == gold else 0.0

    opts = _parse_qasc_options(item.get("instruction", ""))
    gold_text = opts.get(gold)
    pred_norm = _norm_text(strip_thinking_channels(prediction or ""))
    if gold_text and (pred_norm == gold_text or gold_text in pred_norm):
        return 100.0
    return 0.0


def _norm_lbox(s: Any) -> str:
    text = str(s or "")
    text = re.sub(r"[\"'`“”‘’]", "", text)
    text = re.sub(r"[\s·ㆍ・\-\(\)\[\]{}.,:;，。]", "", text)
    return text.lower()


def _extract_lbox_casename(prediction: str) -> str:
    text = strip_thinking_channels(prediction or "").strip()
    matches = re.findall(r"(?:죄명|범죄명|정답|답)\s*[:：]\s*(.+)", text)
    candidate = matches[-1] if matches else text.splitlines()[-1] if text.splitlines() else text
    candidate = re.sub(r"^(?:정답은|답은)\s*", "", candidate.strip())
    candidate = re.sub(r"(?:입니다|이다)[.。]?$", "", candidate.strip())
    return candidate.strip(" \t\r\n\"'`.,:;")


_STATUTE_RE = re.compile(
    r"([가-힣A-Za-z0-9·ㆍ()]+법)\s*제\s*\d+(?:조|항|호)(?:의\s*\d+)?(?:\s*제\s*\d+(?:항|호))*"
)


def _as_list(v: Any) -> list[str]:
    v = _json(v)
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if v is None:
        return []
    return [str(v)]


def _parse_lbox_statutes(prediction: str, gold_items: list[str]) -> set[str]:
    text = strip_thinking_channels(prediction or "")
    found = {_norm_lbox(m.group(0)) for m in _STATUTE_RE.finditer(text)}
    pred_norm = _norm_lbox(text)
    for gold in gold_items:
        ng = _norm_lbox(gold)
        if ng and ng in pred_norm:
            found.add(ng)
    return {x for x in found if x}


def score_lbox_item(item: Dict[str, Any], prediction: str) -> float:
    """LBox Open Phase 1/2 EM scorer; task_type chooses the parser."""
    task_type = str(item.get("task_type", "")).lower()
    gold = _json(item.get("ground_truth"))

    if task_type == "casename":
        pred = _extract_lbox_casename(prediction)
        return 100.0 if _norm_lbox(pred) == _norm_lbox(gold) else 0.0

    if task_type == "statute":
        gold_items = _as_list(gold)
        gold_set = {_norm_lbox(x) for x in gold_items if _norm_lbox(x)}
        pred_set = _parse_lbox_statutes(prediction, gold_items)
        return 100.0 if gold_set and pred_set == gold_set else 0.0

    pred_norm = _norm_lbox(strip_thinking_channels(prediction or ""))
    gold_norm = _norm_lbox(gold)
    return 100.0 if gold_norm and pred_norm == gold_norm else 0.0


# --- Super-NaturalInstructions (niv2) --------------------------------------
# ⚠️ 공식 채점을 그대로 쓴다. 레포의 eval/automatic/evaluation.py를 이식한 것이다
#    (/data5/jaehoonjeong/datasets/natural-instructions/eval/automatic/evaluation.py).
#
# 공식이 하는 일:
#   - **태스크를 분기하지 않는다.** 모든 인스턴스에 EM과 ROUGE-L을 둘 다 계산한다.
#   - 둘 다 복수 레퍼런스에 대해 max를 취한다(metric_max_over_ground_truths).
#   - 최종 보고는 {"exact_match": ..., "rougeL": ...} 두 숫자를 병기한다.
#
# 이전 구현(직접 만든 LCS·CJK 문자단위 분기·task_closed 채점 분기·_sni_extract 형식제거)은
# 전부 공식에 없는 자체 발명이었고 산술 태스크가 ROUGE로 새는 등 어긋났다. 폐기했다.
# 정규화 차이도 크다: 공식은 구두점을 **삭제**한다("choice/control" → "choicecontrol").
# ROUGE는 rouge_score 라이브러리 + use_stemmer=True (killed/kill을 같게 본다).
#
# 트랙: 우리 export는 default 트랙(전량 English)이라 xlingual GPT2 토크나이저는 쓰지 않는다.
import string as _string

_ROUGE_SCORER = None


def _rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer  # 공식과 동일 라이브러리
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


_SNI_PUNC = set(_string.punctuation)


def _sni_normalize(s: Any) -> str:
    """공식 normalize_answer: 소문자 + 구두점 제거 + 공백 정리. **관사는 지우지 않는다.**"""
    text = strip_thinking_channels(str(s or "")).lower()
    text = "".join(ch for ch in text if ch not in _SNI_PUNC)
    return " ".join(text.split())


def _sni_refs(item: Dict[str, Any]) -> list:
    gt = item.get("ground_truth")
    if isinstance(gt, list):
        return [str(g) for g in gt]
    return [str(gt)] if gt is not None else []


def sni_metrics(item: Dict[str, Any], prediction: str) -> Dict[str, float]:
    """공식 두 지표를 한 번에. 각각 레퍼런스 최대값(0~100)."""
    pred = strip_thinking_channels(str(prediction or ""))
    refs = _sni_refs(item)
    if not refs:
        return {"exact_match": 0.0, "rougeL": 0.0}
    npred = _sni_normalize(pred)
    em = 100.0 if any(npred == _sni_normalize(r) for r in refs) else 0.0
    sc = _rouge_scorer()
    rl = max(sc.score(prediction=pred, target=r)["rougeL"].fmeasure for r in refs)
    return {"exact_match": em, "rougeL": 100.0 * rl}


def score_sni_item(item: Dict[str, Any], prediction: str) -> float:
    """0/100 이진 계약(진화 WAR·binning이 소비). 공식 EM 그대로.

    ⚠️ 공식은 임계를 두지 않고 EM·ROUGE 연속값 두 개를 병기한다. "풀었다/못 풀었다"라는
    이진 판정은 공식에 없는 **우리 요구사항**이고, 여기서 EM을 쓰는 것이 그 선택이다.
    ROUGE 임계로 판정하려면 war_mode='soft_partial'의 partial_credit 경로를 쓴다.
    """
    return sni_metrics(item, prediction)["exact_match"]


def score_sni_item_partial(item: Dict[str, Any], prediction: str) -> float:
    """공식 ROUGE-L(0~100). 진화 WAR(war_mode='soft_partial')의 부분점수."""
    return sni_metrics(item, prediction)["rougeL"]


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
        elif ds == "acc":
            kind = "acc"
        elif ds == "qasc":
            kind = "qasc"
        elif ds == "lbox":
            kind = "lbox"
        elif ds == "sni":
            kind = "sni"
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

    if kind == "qasc":
        return score_qasc_item(item, prediction_code)

    if kind == "lbox":
        return score_lbox_item(item, prediction_code)

    if kind == "sni":
        return score_sni_item(item, prediction_code)

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
