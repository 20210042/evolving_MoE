"""Small dataset/domain classification helpers shared by prompts and pipelines."""

from __future__ import annotations


def task_family(dataset: str | None = None, domain: str | None = None) -> str:
    """Map dataset/domain metadata to the prompt/scoring family used by pipelines."""
    ds = (dataset or "").lower()
    dom = (domain or "").lower()
    if dom == "math" or ds in {"bigmath", "math", "numina_cot"}:
        return "math"
    if dom in {"qasc", "science_mc", "mc"} or ds == "qasc":
        return "qasc"
    if dom in {"lbox", "legal"} or ds == "lbox":
        return "lbox"
    return "coding"


def is_text_generation_task(dataset: str | None = None, domain: str | None = None) -> bool:
    return task_family(dataset=dataset, domain=domain) in {"math", "qasc", "lbox"}
