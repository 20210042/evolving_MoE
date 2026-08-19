#!/usr/bin/env python
"""Push best LBox legal-category LoRA adapters to Hugging Face Hub."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"
REPO_PREFIX = "Jongbin-kr/llama3_lbox_category"

UPLOAD_ALLOW_PATTERNS = [
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
]


@dataclass(frozen=True)
class SelectedCheckpoint:
    category: str
    source_dir: Path
    checkpoint_dir: Path
    reason: str
    repo_id: str


def category_from_dir(path: Path) -> str:
    prefix = "sft_lbox_legal_category_"
    suffix = "_step5000"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected checkpoint dir name: {name}")
    return name[len(prefix) : -len(suffix)]


def metric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def select_best_checkpoint(source_dir: Path) -> SelectedCheckpoint:
    category = category_from_dir(source_dir)
    state_path = source_dir / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved = {
        int(path.name.split("-")[-1]): path
        for path in source_dir.glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit()
    }
    if not saved:
        raise FileNotFoundError(f"No checkpoint-* directories under {source_dir}")

    best_model_checkpoint = state.get("best_model_checkpoint")
    best_metric = metric_value(state.get("best_metric"))
    if best_model_checkpoint and best_metric is not None:
        checkpoint_dir = ROOT / best_model_checkpoint
        return SelectedCheckpoint(
            category=category,
            source_dir=source_dir,
            checkpoint_dir=checkpoint_dir,
            reason=f"eval_loss={best_metric}",
            repo_id=f"{REPO_PREFIX}_{category}_step5000",
        )

    candidates: list[tuple[float, int]] = []
    for row in state.get("log_history", []):
        step = row.get("step")
        accuracy = metric_value(row.get("eval_mean_token_accuracy"))
        if isinstance(step, int) and step in saved and accuracy is not None:
            candidates.append((accuracy, step))
    if candidates:
        accuracy, step = max(candidates)
        reason = f"eval_mean_token_accuracy={accuracy}"
    else:
        step = max(saved)
        reason = "fallback_latest_saved"

    return SelectedCheckpoint(
        category=category,
        source_dir=source_dir,
        checkpoint_dir=saved[step],
        reason=reason,
        repo_id=f"{REPO_PREFIX}_{category}_step5000",
    )


def main() -> None:
    api = HfApi()
    selections = [
        select_best_checkpoint(path)
        for path in sorted(CHECKPOINT_ROOT.glob("sft_lbox_legal_category_*_step5000"))
    ]
    if not selections:
        raise FileNotFoundError("No sft_lbox_legal_category_*_step5000 directories found.")

    for selection in selections:
        print(
            f"push {selection.category}: {selection.checkpoint_dir} -> {selection.repo_id} "
            f"({selection.reason})",
            flush=True,
        )
        api.create_repo(selection.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=selection.repo_id,
            folder_path=str(selection.checkpoint_dir),
            allow_patterns=UPLOAD_ALLOW_PATTERNS,
            commit_message=f"Upload best LBox category LoRA checkpoint ({selection.reason})",
        )


if __name__ == "__main__":
    main()
