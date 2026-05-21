from __future__ import annotations

import argparse

from acc_data_pipeline.preprocessing.deduplicate import DEFAULT_DEDUP_CONFIG, deduplicate_records
from acc_data_pipeline.reports.write_reports import write_report
from acc_data_pipeline.utils.config import load_config
from acc_data_pipeline.utils.io import iter_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deduplicate normalized coding problems.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, DEFAULT_DEDUP_CONFIG)
    output, report = deduplicate_records(list(iter_jsonl(args.input)), config)
    write_jsonl(args.output, output)
    write_report(args.report, report)


if __name__ == "__main__":
    main()
