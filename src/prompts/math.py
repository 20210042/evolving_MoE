"""Math prompt builder — mirrors coding.py for math datasets."""

from __future__ import annotations

from typing import List, Union

from prompts import baseline_prompts

Message = Union[str, List[dict]]


def build_generation_prompt(instruction: str) -> Message:
    return [
        {"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM},
        {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)},
    ]
