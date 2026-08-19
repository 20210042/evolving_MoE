#!/usr/bin/env python
"""Upload a completed mergoo MoE-on-LoRA checkpoint directory to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_README = """---
library_name: mergoo
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- mergoo
- moe
- lora
- llama
- router-training
---

# Mergoo MoE-on-LoRA

This checkpoint was built by composing category-specific LoRA experts with
`mergoo` and training only token-level router gates.

## Local Provenance

See `router_apply_summary.json` and `mergoo_compose_config.json` in this
repository for the exact local composition and router-application metadata.
"""


def ensure_model_card(folder: Path, overwrite: bool = False) -> None:
    readme = folder / "README.md"
    if readme.exists() and not overwrite:
        return

    summary_path = folder / "router_apply_summary.json"
    compose_path = folder / "mergoo_compose_config.json"
    extra = []
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        extra.append("\n## Router Apply Summary\n")
        extra.append("```json\n")
        extra.append(json.dumps(summary, indent=2))
        extra.append("\n```\n")
    if compose_path.exists():
        with compose_path.open() as f:
            compose = json.load(f)
        extra.append("\n## Mergoo Compose Config\n")
        extra.append("```json\n")
        extra.append(json.dumps(compose, indent=2))
        extra.append("\n```\n")

    readme.write_text(DEFAULT_README + "".join(extra), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/mergoo_lora_moe_5cat_top2_router_trained",
        help="Local completed full checkpoint directory to upload.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Target Hugging Face repo id, for example USER/mergoo-lora-moe-5cat.",
    )
    parser.add_argument("--private", action="store_true", help="Create/upload to a private repo.")
    parser.add_argument("--revision", default=None, help="Optional branch/revision to upload to.")
    parser.add_argument("--commit_message", default="Upload mergoo MoE-on-LoRA checkpoint")
    parser.add_argument("--overwrite_readme", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise EnvironmentError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before uploading.")

    ensure_model_card(checkpoint_dir, overwrite=args.overwrite_readme)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {checkpoint_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
