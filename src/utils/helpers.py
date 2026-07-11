import json
import logging
import random
import re

import torch

from utils.domains import is_text_generation_task

logger = logging.getLogger(__name__)


def strip_thinking_channels(text: str) -> str:
    """Remove Gemma 4 thought channel blocks before parsing JSON or code."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    # Bare "thought\n" prefix without closing tag (decoded Gemma 4 output):
    # thinking lines are indented bullets ("    *   "), so the clean answer
    # starts at the first \n\n followed by a non-indented character.
    text = re.sub(r"^thought\n.*?\n\n(?=[^ \t])", "", text, count=1, flags=re.DOTALL)
    return text.strip()


def extract_json_object(text: str) -> dict | None:
    """Extract the last balanced JSON object from model output (thinking-safe)."""
    text = strip_thinking_channels(text or "")
    if not text:
        return None
    start = text.rfind("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_code_block(text: str) -> str:
    """
    Robustly extract code from model output, handling markdown,
    custom tags ([BEGIN], <code>), and common suffixes.
    """
    if not text:
        return ""

    # Strip Gemma4 thinking blocks before code extraction
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    code = text
    if "[BEGIN]" in code and "[DONE]" in code:
        code = code.split("[BEGIN]")[1].split("[DONE]")[0]

    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1]
        elif len(parts) == 2:
            p0, p1 = parts[0].strip(), parts[1].strip()
            if p0 and not p1:
                code = p0
            elif p1 and not p0:
                code = p1
            else:
                code = parts[0] if len(parts[0]) > len(parts[1]) else parts[1]

    if "<code>" in code:
        code = code.split("<code>")[-1]
    if "</code>" in code:
        code = code.split("</code>")[0]

    return code.strip()


def finalize_generation_output(
    text: str,
    *,
    dataset: str | None = None,
    domain: str | None = None,
) -> str:
    """Return raw natural-language answers for EM/math tasks, code for coding tasks."""
    cleaned = strip_thinking_channels(text or "")
    if is_text_generation_task(dataset=dataset, domain=domain):
        return cleaned
    return extract_code_block(cleaned) or cleaned


def check_stop_condition(feedback: str) -> bool:
    feedback_lower = feedback.lower()

    score_match = re.search(r"(?:Score|Grade|Rating):\s*(\d+)", feedback, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            if score <= 10:
                return score >= 9
            if score <= 100:
                return score >= 90
        except ValueError:
            pass

    strong_signals = [
        "perfect",
        "no issues",
        "no bugs",
        "excellent",
        "flawless",
        "no errors",
        "completely correct",
        "works correctly",
    ]

    if "incorrect" in feedback_lower or "not correct" in feedback_lower or "bug" in feedback_lower:
        return False

    matched = sum(1 for sig in strong_signals if sig in feedback_lower)
    return matched >= 2


def extract_math_answer(text: str) -> str:
    """모델 출력에서 최종 답을 추출한다.

    1순위: 텍스트 내 마지막 "Final Answer: ..." 라인.
    2순위: 텍스트 내 마지막 \\boxed{...} (중첩 괄호 지원).
    3순위: 원본 텍스트 전체.
    """
    text = strip_thinking_channels(text or "")
    if not text:
        return ""
    
    ## boxed match (1순위)
    boxed: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            boxed.append(text[start : j - 1])
        i = idx + 1
    if boxed:
        return boxed[-1].strip()

    ## final answer match (2순위)
    final_answer_mathces = re.findall(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if final_answer_mathces:
        candidate = re.sub(r"^\*+|\*+$", "", final_answer_mathces[-1]).strip()
        if candidate:
            return candidate

    ## fallback to original text
    return text.strip()


def set_all_seeds(seed: int):
    """CPU와 CUDA 모든 시드를 고정한다."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    # Do not initialize CUDA here. vLLM starts worker processes after this call;
    # touching CUDA in the parent can make those workers fail with
    # "Cannot re-initialize CUDA in forked subprocess".
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"모든 시드 고정: {seed} (CPU + CUDA deterministic)")
