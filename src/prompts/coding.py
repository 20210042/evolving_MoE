"""Dataset-aware coding prompts for generation / refinement."""

from __future__ import annotations

from typing import List, Union

from prompts import baseline_prompts
from prompts import qwen3_lcb
from utils.domains import task_family

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


import string as _string


def _sni_punct(text: str) -> str:
    """공식 collator와 동일: 끝이 구두점이 아니면 마침표를 붙인다."""
    text = text.strip()
    if text and text[-1] not in _string.punctuation:
        text += "."
    return text


def sni_examples_block(examples: list | None, n: int = 2, kind: str = "Positive",
                       add_explanation: bool = False) -> str:
    """공식 Tk-Instruct 형식의 예시 블록.

    scripts/{train,eval}_tk_instruct.sh의 표준 설정은
    --num_pos_examples 2 --num_neg_examples 0 --add_explanation False 다.
    """
    out = []
    for idx, ex in enumerate((examples or [])[:n]):
        blk = f" {kind} Example {idx + 1} -\n"
        blk += f"Input: {_sni_punct(str(ex.get('input') or ''))}\n"
        blk += f" Output: {_sni_punct(str(ex.get('output') or ''))}\n"
        if add_explanation and ex.get("explanation"):
            blk += f" Explanation: {_sni_punct(str(ex['explanation']))}\n"
        out.append(blk + "\n")
    return "".join(out)


def sni_user_block(
    answer_line: str | None,
    instruction: str,
    *,
    positive_examples: list | None = None,
    negative_examples: list | None = None,
    num_pos: int = 2,
    num_neg: int = 0,
    add_explanation: bool = False,
) -> str:
    """SNI user 턴. **공식 Tk-Instruct 형식**을 따른다 —
    예시 블록 + "Now complete the following example -" + Input/Output 구분자.

    answer_line은 공식에 없는 우리 추가분이라, 있을 때만 예시 앞에 한 줄로 둔다.
    태스크 정의는 여기 넣지 않는다(system 턴, sni_system_block).
    """
    parts = []
    line = (answer_line or "").strip()
    if line:
        parts.append(line + "\n\n")
    parts.append(sni_examples_block(positive_examples, num_pos, "Positive", add_explanation))
    parts.append(sni_examples_block(negative_examples, num_neg, "Negative", add_explanation))
    parts.append("Now complete the following example -\n")
    parts.append(f"Input: {_sni_punct(instruction)}\n")
    parts.append("Output: ")
    return "".join(parts)


def sni_system_block(persona: str, definition: str | None) -> str:
    """SNI system 턴 = 정체성(페르소나) + 태스크 정의.

    정의를 **system에, 페르소나와 같은 층에** 둔다. 원 프로브(job 229352)는 정의 537자를
    user 턴에, 페르소나 90자를 system에 뒀는데 gemma가 user 지시를 더 강하게 따르므로
    정체성이 통째로 묻혔다. 정의 자체를 빼면 이번엔 "누가 태스크를 잘 알아맞히나"를 재게 되어
    (서술형은 무엇을 하라는 건지 알 방법이 없다) 관측 대상이 문체가 아니게 된다.
    그래서 정보는 다 주되 자리만 바꾼다 — 페르소나를 앞에 둬 정체성이 먼저 읽히게 한다.
    """
    d = (definition or "").strip()
    return f"{persona}\n\n{d}" if d else persona


def build_baseline_prompt(
    instruction: str,
    *,
    dataset: str,
    model_name: str,
    starter_code: str | None = None,
    domain: str | None = None,
    answer_line: str | None = None,
    definition: str | None = None,
    positive_examples: list | None = None,
    negative_examples: list | None = None,
    num_pos_examples: int = 2,
    num_neg_examples: int = 0,
    add_explanation: bool = False,
) -> Message:
    ds = (dataset or "mbpp").lower()
    family = task_family(dataset=ds, domain=domain)
    if family == "math":
        system = baseline_prompts.MATH_GEN_SYSTEM
        user = baseline_prompts.MATH_GEN_USER.format(instruction=instruction)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if family == "qasc":
        system = baseline_prompts.QASC_GEN_SYSTEM
        user = baseline_prompts.QASC_GEN_USER.format(instruction=instruction)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if family == "lbox":
        system = baseline_prompts.LBOX_GEN_SYSTEM
        user = baseline_prompts.LBOX_GEN_USER.format(instruction=instruction)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if family == "sni":
        system = sni_system_block(baseline_prompts.SNI_GEN_SYSTEM, definition)
        user = sni_user_block(answer_line, instruction,
                              positive_examples=positive_examples,
                              negative_examples=negative_examples,
                              num_pos=num_pos_examples, num_neg=num_neg_examples,
                              add_explanation=add_explanation)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

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
    approach: str | None = None,
    domain: str | None = None,
    answer_line: str | None = None,
    definition: str | None = None,
    positive_examples: list | None = None,
    negative_examples: list | None = None,
    num_pos_examples: int = 2,
    num_neg_examples: int = 0,
    add_explanation: bool = False,
) -> Message:
    """One-shot generation under a persona. system_prompt=정체성, approach(있으면)는
    user 턴에 프리앰블로 주입(Gemma가 user 지시를 더 잘 따름)."""
    ds = (dataset or "mbpp").lower()
    family = task_family(dataset=ds, domain=domain)
    if family == "math":
        persona_sys = system_prompt or baseline_prompts.MATH_GEN_SYSTEM
        user = baseline_prompts.MATH_GEN_USER.format(instruction=instruction)
        if approach:
            user = f"{approach}\n\n{user}"
        return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]
    if family == "qasc":
        persona_sys = system_prompt or baseline_prompts.QASC_GEN_SYSTEM
        user = baseline_prompts.QASC_GEN_USER.format(instruction=instruction)
        if approach:
            user = f"{approach}\n\n{user}"
        return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]
    if family == "lbox":
        persona_sys = system_prompt or baseline_prompts.LBOX_GEN_SYSTEM
        user = baseline_prompts.LBOX_GEN_USER.format(instruction=instruction)
        if approach:
            user = f"{approach}\n\n{user}"
        return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]
    if family == "sni":
        persona_sys = sni_system_block(
            system_prompt or baseline_prompts.SNI_GEN_SYSTEM, definition)
        user = sni_user_block(answer_line, instruction,
                              positive_examples=positive_examples,
                              negative_examples=negative_examples,
                              num_pos=num_pos_examples, num_neg=num_neg_examples,
                              add_explanation=add_explanation)
        if approach:
            user = f"{approach}\n\n{user}"
        return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]

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
    if approach:
        user = f"{approach}\n\n{user}"

    if family == "llama31":
        persona_sys = persona_sys + _LLAMA_CODE_HINT

    return [{"role": "system", "content": persona_sys}, {"role": "user", "content": user}]


def build_fewshot_block(examples: list[tuple[str, str]]) -> str:
    """(instruction, own_solved_code) 쌍을 build_expert_prompt의 approach= 프리앰블로.
    train_sft.py / generate_lora_binning.py가 동일 포맷을 공유해 train/inference 프롬프트를
    일치시킨다 (persona/few-shot을 배포에서만 추가하면 어댑터가 학습 안 해본 프롬프트 분포를
    보게 되어 품질이 흔들릴 수 있다)."""
    if not examples:
        return ""
    blocks = []
    for i, (instr, code) in enumerate(examples, 1):
        blocks.append(
            f"[Example {i} — a problem you solved before, in your own style]\n"
            f"Problem:\n{instr}\n\nYour solution:\n```python\n{code}\n```"
        )
    return (
        "Below are examples of problems you personally solved before. Solve the new "
        "problem in a similar style/approach.\n\n" + "\n\n".join(blocks)
    )


def build_refine_prompt(
    instruction: str,
    feedback: str,
    current_code: str,
    *,
    dataset: str,
    model_name: str,
    answer_line: str | None = None,
    definition: str | None = None,
    positive_examples: list | None = None,
    negative_examples: list | None = None,
    num_pos_examples: int = 2,
    num_neg_examples: int = 0,
    add_explanation: bool = False,
) -> Message:
    ds = (dataset or "mbpp").lower()
    family = get_prompt_family(model_name)
    task = task_family(dataset=ds)

    if task == "qasc":
        return [
            {"role": "system", "content": baseline_prompts.QASC_REVISION_SYSTEM},
            {
                "role": "user",
                "content": baseline_prompts.QASC_REVISION_USER.format(
                    instruction=instruction,
                    solution=current_code,
                    feedback=feedback,
                ),
            },
        ]

    if task == "lbox":
        return [
            {"role": "system", "content": baseline_prompts.LBOX_REVISION_SYSTEM},
            {
                "role": "user",
                "content": baseline_prompts.LBOX_REVISION_USER.format(
                    instruction=instruction,
                    solution=current_code,
                    feedback=feedback,
                ),
            },
        ]

    if task == "sni":
        return [
            {"role": "system",
             "content": sni_system_block(baseline_prompts.SNI_REVISION_SYSTEM, definition)},
            {
                "role": "user",
                "content": baseline_prompts.SNI_REVISION_USER.format(
                    instruction=sni_user_block(answer_line, instruction),
                    solution=current_code,
                    feedback=feedback,
                ),
            },
        ]

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
    answer_line: str | None = None,
    definition: str | None = None,
    positive_examples: list | None = None,
    negative_examples: list | None = None,
    num_pos_examples: int = 2,
    num_neg_examples: int = 0,
    add_explanation: bool = False,
) -> Message:
    ds = (dataset or "mbpp").lower()
    task = task_family(dataset=ds)
    family = get_prompt_family(model_name)

    if task == "qasc":
        return [
            {"role": "system", "content": sys_prompt or baseline_prompts.QASC_CRITIC_SYSTEM},
            {
                "role": "user",
                "content": baseline_prompts.QASC_CRITIC_USER.format(
                    instruction=instruction,
                    solution=current_code,
                ),
            },
        ]

    if task == "lbox":
        return [
            {"role": "system", "content": sys_prompt or baseline_prompts.LBOX_CRITIC_SYSTEM},
            {
                "role": "user",
                "content": baseline_prompts.LBOX_CRITIC_USER.format(
                    instruction=instruction,
                    solution=current_code,
                ),
            },
        ]

    if task == "sni":
        return [
            {"role": "system", "content": sni_system_block(
                sys_prompt or baseline_prompts.SNI_CRITIC_SYSTEM, definition)},
            {
                "role": "user",
                "content": baseline_prompts.SNI_CRITIC_USER.format(
                    instruction=sni_user_block(answer_line, instruction),
                    solution=current_code,
                ),
            },
        ]

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
