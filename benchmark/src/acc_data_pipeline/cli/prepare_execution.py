from __future__ import annotations

import argparse

from acc_data_pipeline.execution.execution_interface import prepare_execution_records
from acc_data_pipeline.reports.write_reports import write_report
from acc_data_pipeline.utils.config import load_config
from acc_data_pipeline.utils.io import iter_jsonl, write_jsonl

DEFAULT_EXECUTION_CONFIG = {
    "default_language": "python",
    "supported_languages": ["python"],
    "default_timeout_seconds": 5,
    "default_memory_limit_mb": 512,
    "max_output_bytes": 200000,
    "comparison": {
        "default_type": "exact_or_token_match",
        "strip_trailing_whitespace": True,
        "normalize_final_newline": True,
        "case_sensitive": True,
        "numeric_tolerance": {"abs_tol": 1e-6, "rel_tol": 1e-6},
    },
    "unsupported": {"keep_records": True, "mark_requires_manual_review": True},
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare execution-ready records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, DEFAULT_EXECUTION_CONFIG)
    records, report = prepare_execution_records(list(iter_jsonl(args.input)), config)
    write_jsonl(args.output, records)
    write_report(args.report, report)


if __name__ == "__main__":
    main()
