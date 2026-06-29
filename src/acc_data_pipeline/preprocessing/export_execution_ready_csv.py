"""CSV exporter for the final execution-ready JSONL dataset.

Converts selected fields into a labeling-friendly table, validates required paths, and writes a
small report describing skipped rows and source/category distributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from acc_data_pipeline.utils.io import ensure_parent, iter_jsonl, write_json


CSV_FIELDS = [
    "problem_id",
    "problem",
    "answer",
    "test_cases",
    "eval_spec",
    "source",
    "source_platform",
    "original_domain",
    "category",
]

REQUIRED_PATHS = [
    ("problem_id",),
    ("problem_statement", "raw"),
    ("reference_solutions",),
    ("test_cases",),
    ("eval_spec",),
    ("source",),
    ("source_platform",),
]


def _get_nested(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _path_name(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _string_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _original_domain(record: dict[str, Any]) -> Any:
    native_metadata = record.get("native_metadata")
    if not isinstance(native_metadata, dict):
        native_metadata = {}
    raw_fields = native_metadata.get("raw_fields")
    if not isinstance(raw_fields, dict):
        raw_fields = {}
    return native_metadata.get("tags") or raw_fields.get("cf_tags") or []


def _missing_required_fields(record: dict[str, Any]) -> list[str]:
    missing = []
    for path in REQUIRED_PATHS:
        value = _get_nested(record, path)
        if value is None:
            missing.append(_path_name(path))
    return missing


def make_csv_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "problem_id": _string_cell(record.get("problem_id")),
        "problem": _string_cell(_get_nested(record, ("problem_statement", "raw"))),
        "answer": _json_cell(record.get("reference_solutions")),
        "test_cases": _json_cell(record.get("test_cases")),
        "eval_spec": _json_cell(record.get("eval_spec")),
        "source": _string_cell(record.get("source")),
        "source_platform": _string_cell(record.get("source_platform")),
        "original_domain": _json_cell(_original_domain(record)),
        "category": "",
    }


def export_execution_ready_csv(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    ensure_parent(output_path)
    count = 0
    missing_records: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for line_no, record in enumerate(iter_jsonl(input_path), start=1):
            missing = _missing_required_fields(record)
            if missing:
                missing_records.append(
                    {
                        "line_no": line_no,
                        "problem_id": record.get("problem_id"),
                        "missing_fields": missing,
                    }
                )
                missing_counts.update(missing)
            writer.writerow(make_csv_row(record))
            count += 1

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "records_written": count,
        "missing_required_record_count": len(missing_records),
        "missing_required_field_counts": dict(sorted(missing_counts.items())),
        "missing_required_records": missing_records[:100],
        "missing_required_records_truncated": max(0, len(missing_records) - 100),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export execution-ready JSONL records to a CSV upload table."
    )
    parser.add_argument(
        "--input",
        default="data/processed/04_execution_ready.jsonl",
        help="Path to execution-ready JSONL records.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/04_execution_ready.csv",
        help="Path for the generated CSV file.",
    )
    parser.add_argument(
        "--report",
        default="data/reports/04_execution_ready_csv_report.json",
        help="Path for a JSON report with missing required field details.",
    )
    args = parser.parse_args(argv)

    report = export_execution_ready_csv(args.input, args.output, args.report)
    print(
        "Wrote "
        f"{report['records_written']} records to {report['output']} "
        f"from {report['input']}"
    )
    missing_count = report["missing_required_record_count"]
    if missing_count:
        print(
            "Missing required fields in "
            f"{missing_count} records. See {args.report}.",
            file=sys.stderr,
        )
        for field, field_count in report["missing_required_field_counts"].items():
            print(f"- {field}: {field_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
