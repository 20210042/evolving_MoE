#!/usr/bin/env python3
"""Run evolution for multiple (dataset, seed) combinations via subprocess."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["livecodebench", "mbpp", "humaneval"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 1234])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for ds in args.datasets:
        cfg = ROOT / "configs" / f"{ds}.yaml"
        if not cfg.is_file():
            print(f"Skip {ds}: missing {cfg}", file=sys.stderr)
            continue
        for seed in args.seeds:
            run_id = f"{ds}_seed{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_evolution.py"),
                "--config",
                str(cfg),
                "--seed",
                str(seed),
                "--run_id",
                run_id,
            ]
            print("RUN:", " ".join(cmd))
            if not args.dry_run:
                subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
