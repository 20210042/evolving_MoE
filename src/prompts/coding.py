"""Dataset-aware coding prompts for generation / refinement."""

from __future__ import annotations

from typing import Dict, List, Union

from prompts import baseline_prompts
from prompts import qwen3_lcb

Message = Union[str, List[Dict[str, str]]]


def is_qwen3_model(model_name: str) -> bool:
    return "qwen3" in (model_name or "").lower()


def build_baseline_prompt(
    instruction: str,
    *,
    dataset: str,
    model_name: str,
    starter_code: str | None = None,
) -> Message:
    """Return chat messages or a raw Chat string for baseline code generation."""
    if starter_code:
        instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

    ds = (dataset or "mbpp").lower()
    qwen = is_qwen3_model(model_name)

    if ds == "livecodebench":
        if qwen:
            return (
                f"<|im_start|>system\n{qwen3_lcb.QWEN3_LCB_SYSTEM}\n"
                f"<|im_end|>\n<|im_start|>user\n"
                f"{qwen3_lcb.QWEN3_LCB_USER_TEMPLATE.format(instruction=instruction)}\n"
                f"<|im_end|>\n<|im_start|>assistant\n```python\n"
            )
        return [
            {"role": "system", "content": baseline_prompts.LCB_GEN_SYSTEM},
            {"role": "user", "content": baseline_prompts.LCB_GEN_USER.format(instruction=instruction)},
        ]

    # MBPP, HumanEval, generic coding
    if qwen:
        return (
            f"<|im_start|>system\n{baseline_prompts.CODING_GEN_SYSTEM}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"{baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )
    return [
        {"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM},
        {"role": "user", "content": baseline_prompts.CODING_GEN_USER.format(instruction=instruction)},
    ]


def build_refine_prompt(
    instruction: str,
    feedback: str,
    current_code: str,
    *,
    dataset: str,
    model_name: str,
) -> Message:
    ds = (dataset or "mbpp").lower()
    qwen = is_qwen3_model(model_name)

    if ds == "livecodebench" and qwen:
        return (
            f"<|im_start|>system\n{qwen3_lcb.QWEN3_LCB_SYSTEM}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"Refine the code based on the feedback.\n\n"
            f"[Problem Description]\n{instruction}\n\n"
            f"[CRITICAL RULES]\n"
            f"1. Output ONLY valid, strictly formatted Python code.\n"
            f"2. DO NOT use side-by-side formatting, diffs, tables, or 2-column layouts.\n"
            f"3. Return the COMPLETE, fully executable code.\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Code:\n{current_code}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )

    return [
        {"role": "system", "content": baseline_prompts.CODING_REVISION_SYSTEM},
        {
            "role": "user",
            "content": baseline_prompts.CODING_REVISION_USER.format(
                instruction=instruction,
                code=current_code,
                feedback=feedback,
            ),
        },
    ]


def build_critic_prompt(
    instruction: str,
    current_code: str,
    sys_prompt: str,
    *,
    dataset: str,
    model_name: str,
) -> Message:
    qwen = is_qwen3_model(model_name)
    if qwen:
        return (
            f"<|im_start|>system\n{sys_prompt}\n"
            f"<|im_end|>\n<|im_start|>user\nReview the following code.\n\n"
            f"Problem:\n{instruction}\n\n"
            f"Code:\n{current_code}\n\n"
            "1. Identify any syntax errors or bugs.\n"
            "2. Check if it solves the problem correctly.\n"
            "3. Assess the efficiency.\n"
            "Feedback: [Your detailed feedback]\n"
            f"<|im_end|>\n<|im_start|>assistant\nFeedback: "
        )
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": baseline_prompts.CODING_CRITIC_USER.format(
                instruction=instruction,
                code=current_code,
            ),
        },
    ]
