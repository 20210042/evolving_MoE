"""CLI for loading and normalizing raw benchmark datasets.

Runs the selected dataset loaders, assigns stable problem IDs, writes 01_unified_raw.jsonl, and
emits a load report with source counts, warnings, and schema errors."""

from __future__ import annotations

import argparse
from typing import Any

from acc_data_pipeline.loaders import LOADER_BY_NAME
from acc_data_pipeline.reports.stats import load_report
from acc_data_pipeline.reports.write_reports import write_report
from acc_data_pipeline.utils.io import write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load and normalize raw coding benchmarks.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(LOADER_BY_NAME))
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    loader_reports: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        loader = LOADER_BY_NAME[dataset](args.raw_root)
        dataset_records = loader.load()
        records.extend(dataset_records)
        loader_reports[dataset] = {
            "loaded": len(dataset_records),
            "warnings": loader.warnings,
            "schema_errors": loader.schema_errors,
        }
    records.sort(key=lambda item: item["problem_id"])
    write_jsonl(args.output, records)
    write_report(args.report, load_report(records, loader_reports))


if __name__ == "__main__":
    main()
