from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from acc_data_pipeline.reports.write_reports import write_report
from acc_data_pipeline.schemas.validation import validate_problem, validate_required_final_fields
from acc_data_pipeline.utils.io import iter_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate normalized problem schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    invalid: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []
    unsupported_eval_modes = 0
    empty_statements = 0
    empty_test_cases = 0
    eval_modes: Counter[str] = Counter()
    total = 0
    for total, record in enumerate(iter_jsonl(args.input), start=1):
        ok, errors = validate_problem(record)
        final_errors = validate_required_final_fields(record)
        if not ok:
            invalid.append({"problem_id": record.get("problem_id"), "errors": errors})
        if final_errors:
            missing_required.append({"problem_id": record.get("problem_id"), "errors": final_errors})
        mode = (record.get("eval_spec") or {}).get("eval_mode", "missing")
        eval_modes[mode] += 1
        if mode == "unsupported":
            unsupported_eval_modes += 1
        if not str(((record.get("problem_statement") or {}).get("raw") or "")).strip():
            empty_statements += 1
        if not record.get("test_cases"):
            empty_test_cases += 1
    report = {
        "input_count": total,
        "valid_count": total - len(invalid),
        "invalid_count": len(invalid),
        "invalid_records": invalid,
        "missing_required_fields": missing_required,
        "unsupported_eval_modes": unsupported_eval_modes,
        "empty_statements": empty_statements,
        "empty_test_cases": empty_test_cases,
        "by_eval_mode": dict(eval_modes),
    }
    write_report(args.report, report)


if __name__ == "__main__":
    main()
