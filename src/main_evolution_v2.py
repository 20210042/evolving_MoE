"""
Legacy entrypoint — prefer ``python scripts/run_evolution.py`` (YAML + action gate).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run_evolution.py"
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
