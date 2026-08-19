#!/usr/bin/env python
"""Create derived LBox legal-category tag files with small categories merged."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MERGED_CATEGORY = "family_patent_special"
MERGED_NAME = "가사법 + 특허법"
SOURCE_CATEGORIES = {"family_case", "patent_ip"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_split(input_dir: Path, output_dir: Path, split: str) -> None:
    summary = json.loads((input_dir / f"lbox_{split}_legal_categories_summary.json").read_text(encoding="utf-8"))
    rows = load_jsonl(input_dir / f"lbox_{split}_legal_categories.jsonl")

    taxonomy = dict(summary["taxonomy"])
    for key in SOURCE_CATEGORIES:
        taxonomy.pop(key, None)
    taxonomy[MERGED_CATEGORY] = MERGED_NAME

    merged_rows = []
    for row in rows:
        row = dict(row)
        original_category = row.get("primary_category")
        if original_category in SOURCE_CATEGORIES:
            row["primary_category"] = MERGED_CATEGORY
            row["primary_category_name"] = MERGED_NAME
            row["merged_from_category"] = original_category
        merged_rows.append(row)

    counts = Counter(row.get("primary_category") for row in merged_rows)
    parse_counts = Counter(str(row.get("parse_status") or "ok") for row in merged_rows)
    output = output_dir / f"lbox_{split}_legal_categories.jsonl"
    write_jsonl(output, merged_rows)
    (output_dir / f"lbox_{split}_legal_categories_summary.json").write_text(
        json.dumps(
            {
                **summary,
                "output": str(output),
                "taxonomy": taxonomy,
                "counts": dict(counts),
                "parse_status_counts": dict(parse_counts),
                "merge_note": {
                    "merged_category": MERGED_CATEGORY,
                    "merged_name": MERGED_NAME,
                    "source_categories": sorted(SOURCE_CATEGORIES),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/lbox_legal_category_tags/gemma4_a4b")
    parser.add_argument("--output-dir", default="results/lbox_legal_category_tags/gemma4_a4b_family_patent_merged")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()
    for split in args.splits:
        merge_split(Path(args.input_dir), Path(args.output_dir), split)


if __name__ == "__main__":
    main()
