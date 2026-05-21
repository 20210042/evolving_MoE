from __future__ import annotations

import argparse

from acc_data_pipeline.preprocessing.filter_algorithmic import DEFAULT_FILTER_CONFIG, filter_records
from acc_data_pipeline.reports.write_reports import write_report
from acc_data_pipeline.utils.config import load_config
from acc_data_pipeline.utils.io import iter_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filter algorithmic coding tasks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, DEFAULT_FILTER_CONFIG)
    kept, report = filter_records(list(iter_jsonl(args.input)), config)
    write_jsonl(args.output, kept)
    write_report(args.report, report)


if __name__ == "__main__":
    main()
