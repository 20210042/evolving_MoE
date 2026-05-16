"""Repository root resolution (for LiveCodeBench syspath, results/, legacy/)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """MetaAgentEvolution_Release root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def livecodebench_paths() -> list[Path]:
    root = repo_root()
    env = os.environ.get("LIVECODEBENCH_PATH")
    out: list[Path] = []
    if env:
        out.append(Path(env))
    out.extend(
        [
            root / "LiveCodeBench",
            root.parent / "MultiAgent" / "LiveCodeBench",
        ]
    )
    return [p for p in out if p.is_dir()]
