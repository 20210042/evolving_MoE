"""Dataset-aware coding prompts for generation / refinement."""

from __future__ import annotations

from typing import List, Union

from meta_agent_evo.prompts import baseline_prompts
from meta_agent_evo.prompts import qwen3_lcb

Message = Union[str, List[dict]]

_LLAMA_CODE_HINT = (
    " Output ONLY a single fenced Markdown ```python ... ``` block with no text outside it."
)


def get_prompt_family(model_name: str) -> str:
    m = (model_name or "").lower()
    if "qwen3" in m:
        return "qwen3"
    if "llama" in m or "meta-llama" in m or "gemma" in m:
        return "llama31"
    return "generic"


def build_baseline_prompt(
    instruction: str,
    *,
    dataset: str,
    model_name: str,
    starter_code: str | None = None,
) -> Message:
    if starter_code:
        instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

    ds = (dataset or "mbpp").lower()
    family = get_prompt_family(model_name)

    if family == "qwen3":
        if ds == "livecodebench":
            return (
                f"<|im_start|>system\n{qwen3_lcb.QWEN3_LCB_SYSTEM}\n"
                f"<|im_end|>\n<|im_start|>user\n"
                f"{qwen3_lcb.QWEN3_LCB_USER_TEMPLATE.format(instruction=instruction)}\n"
                f"<|im_end|>\n<|im_start|>assistant\n```python\n"
            )
        return (
            f"<|im_start|>system\n{baseline_prompts.CODING_GEN_SYSTEM}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"{baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )

    if ds == "livecodebench":
        system = baseline_prompts.LCB_GEN_SYSTEM
        user = baseline_prompts.LCB_GEN_USER.format(instruction=instruction)
    else:
        system = baseline_prompts.CODING_GEN_SYSTEM
        user = baseline_prompts.CODING_GEN_USER.format(instruction=instruction)

    if family == "llama31":
        system = system + _LLAMA_CODE_HINT

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_expert_prompt(
    instruction: str,
    system_prompt: str,
    *,
    dataset: str,
    model_name: str,
    starter_code: str | None = None,
) -> Message:
    """One-shot code generation under an expert persona (system_prompt)."""
    if starter_code:
        instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

    ds = (dataset or "mbpp").lower()
    family = get_prompt_family(model_name)
    persona_sys = system_prompt or "You are an expert programmer."

    # qwen3 ChatML prefill — dormant for gemma/llama31 backbones
    if family == "qwen3":
        if ds == "livecodebench":
            return (
                f"<|im_start|>system\n{persona_sys}\n"
                f"<|im_end|>\n<|im_start|>user\n"
                f"{qwen3_lcb.QWEN3_LCB_USER_TEMPLATE.format(instruction=instruction)}\n"
                f"<|im_end|>\n<|im_start|>assistant\n```python\n"
            )
        return (
            f"<|im_start|>system\n{persona_sys}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"{baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )

    if ds == "livecodebench":
        user = baseline_prompts.LCB_GEN_USER.format(instruction=instruction)
    else:
        user = baseline_prompts.CODING_GEN_USER.format(instruction=instruction)

    if family == "llama31":
        persona_sys = persona_sys + _LLAMA_CODE_HINT

    return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]


def build_refine_prompt(
    instruction: str,
    feedback: str,
    current_code: str,
    *,
    dataset: str,
    model_name: str,
) -> Message:
    ds = (dataset or "mbpp").lower()
    family = get_prompt_family(model_name)

    if family == "qwen3" and ds == "livecodebench":
        return (
            f"<|im_start|>system\n{qwen3_lcb.QWEN3_LCB_SYSTEM}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"Refine the code based on the feedback.\n\n"
            f"[Problem Description]\n{instruction}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Code:\n{current_code}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )

    if family == "qwen3":
        return (
            f"<|im_start|>system\n{baseline_prompts.CODING_REVISION_SYSTEM}\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"{baseline_prompts.CODING_REVISION_USER.format(instruction=instruction, code=current_code, feedback=feedback)}\n"
            f"<|im_end|>\n<|im_start|>assistant\n```python\n"
        )

    system = baseline_prompts.CODING_REVISION_SYSTEM
    if family == "llama31":
        system = system + _LLAMA_CODE_HINT

    return [
        {"role": "system", "content": system},
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
    del dataset
    family = get_prompt_family(model_name)

    if family == "qwen3":
        return (
            f"<|im_start|>system\n{sys_prompt}\n"
            f"<|im_end|>\n<|im_start|>user\nReview the following code.\n\n"
            f"Problem:\n{instruction}\n\n"
            f"Code:\n{current_code}\n\n"
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
