"""Math prompt builder — mirrors coding.py for math datasets."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

from prompts import baseline_prompts

Message = Union[str, List[dict]]


def _normalize_domain_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        return key or None
    return None


def _iter_domain_candidates(metadata: Optional[Mapping[str, Any]]) -> Sequence[str]:
    if not metadata:
        return []

    candidates = []
    for field in ("category", "categories", "topic", "original_domain"):
        value = metadata.get(field)
        if isinstance(value, str):
            candidates.append(value)
            candidates.extend(part.strip() for part in value.split("->"))
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    candidates.append(v)
                    candidates.extend(part.strip() for part in v.split("->"))
    return candidates


def resolve_math_system_prompt(metadata: Optional[Mapping[str, Any]] = None) -> str:
    for candidate in _iter_domain_candidates(metadata):
        key = _normalize_domain_key(candidate)
        if key in baseline_prompts.MATH_GEN_SYSTEM_BY_DOMAIN:
            return baseline_prompts.MATH_GEN_SYSTEM_BY_DOMAIN[key]
    return baseline_prompts.MATH_GEN_SYSTEM


def build_generation_prompt(
    instruction: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Message:
    return [
        {"role": "system", "content": resolve_math_system_prompt(metadata)},
        {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)},
    ]
