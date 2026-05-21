from __future__ import annotations

from collections import Counter
from typing import Any


def count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(record.get(field, "unknown")) for record in records))


def load_report(
    records: list[dict[str, Any]],
    loader_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sources = ["APPS", "CodeContests", "TACO", "LiveCodeBench"]
    by_source = {source: 0 for source in sources}
    for record in records:
        by_source[record.get("source", "unknown")] = by_source.get(record.get("source"), 0) + 1
    schema_errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for report in loader_reports.values():
        schema_errors.extend(report.get("schema_errors", []))
        warnings.extend(report.get("warnings", []))
    return {
        "total_loaded": len(records),
        "by_source": by_source,
        "schema_errors": schema_errors,
        "warnings": warnings,
    }
