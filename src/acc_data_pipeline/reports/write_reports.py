"""Small JSON report writer wrapper.

Centralizes report output through the shared IO helper so CLI stages consistently create parent
directories and write formatted JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from acc_data_pipeline.utils.io import write_json


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
