"""Super-NaturalInstructions prompt construction for SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_sni_prompt(item: dict[str, Any], *, num_positive_examples: int = 2) -> str:
    """Build definition + demonstrations + instruction in the requested order."""
    sections = [f"Definition:\n{str(item['definition']).strip()}"]
    for index, example in enumerate(item.get("positive_examples", [])[:num_positive_examples], start=1):
        sections.append(
            f"Positive Example {index}:\n"
            f"Input:\n{str(example['input']).strip()}\n\n"
            f"Output:\n{str(example['output']).strip()}"
        )
    sections.append(f"Instruction:\n{str(item['instruction']).strip()}")
    return "\n\n".join(sections)


def select_sni_target(item: dict[str, Any], *, seed: int) -> str:
    """Select one ground-truth answer pseudo-randomly and reproducibly per row."""
    answers = [str(answer) for answer in item.get("ground_truth", [])]
    if not answers:
        raise ValueError(f"SNI row {item.get('id', '<unknown>')} has no ground_truth answers.")
    key = f"{seed}:{item.get('id', '')}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(answers)
    return answers[index]


def load_sni_rows(local_dir: str, split: str) -> list[dict[str, Any]]:
    """Load the downloaded SNI JSONL split."""
    split_alias = {"validation": "valid", "val": "valid"}
    filename_split = split_alias.get(split.lower(), split.lower())
    path = Path(local_dir) / f"sni_{filename_split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing SNI split: {path}. Download Kcsp0042/SNI_split into --data_dir first."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows
