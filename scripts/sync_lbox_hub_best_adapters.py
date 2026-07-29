#!/usr/bin/env python3
"""Verify roster adapter Hub files and sync local best adapters when needed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_adapter_sha(api: HfApi, repo_id: str) -> str | None:
    info = api.model_info(repo_id, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename != "adapter_model.safetensors":
            continue
        lfs = sibling.lfs
        if lfs is None:
            return None
        return lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--sync", action="store_true", help="Upload mismatched local best adapters.")
    args = parser.parse_args()

    models: list[dict[str, Any]] = []
    for manifest_path in args.manifest:
        models.extend(json.loads(manifest_path.read_text(encoding="utf-8")))
    models = [model for model in models if model.get("family", "").startswith("roster_")]

    api = HfApi()
    mismatches = 0
    for model in models:
        repo_id = model["hub_model_id"]
        local_dir = Path(model["lora_path"])
        local_adapter = local_dir / "adapter_model.safetensors"
        local_sha = sha256(local_adapter)
        remote_sha = remote_adapter_sha(api, repo_id)
        matched = remote_sha == local_sha
        print(f"{repo_id}\t{'MATCH' if matched else 'MISMATCH'}\t{local_sha}\t{remote_sha or '-'}")
        if matched:
            continue
        mismatches += 1
        if args.sync:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=local_dir,
                ignore_patterns=["checkpoint-*", "checkpoint-*/**", ".git/**"],
                commit_message="Sync best validation-loss adapter after training",
            )
            synced_sha = remote_adapter_sha(api, repo_id)
            if synced_sha != local_sha:
                raise RuntimeError(f"Hub verification failed after syncing {repo_id}")
            print(f"{repo_id}\tSYNCED")
    print(f"Checked {len(models)} roster adapters; mismatches={mismatches}")
    if mismatches and not args.sync:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
