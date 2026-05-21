from __future__ import annotations

from collections import Counter
from typing import Any


def split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("split", "unknown")) for record in records))
