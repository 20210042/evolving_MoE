"""Tag-to-critic taxonomy analysis tool.

Maps raw benchmark tags into broader critic categories, computes raw and normalized tag counts,
and writes CSV/JSON distribution reports for data balancing decisions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from acc_data_pipeline.utils.io import ensure_parent, iter_jsonl, write_json


CRITIC_DEFINITIONS = {
    "Math / Fundamentals": {
        "description": "수식화, 조건 처리, overflow-like logic, edge case 진단",
        "raw_tags": {
            "Mathematics",
            "math",
            "Fundamentals",
        },
    },
    "Implementation / Constructive": {
        "description": "구현 실수, 조건 분기, constructive 구성 방식 오류 진단",
        "raw_tags": {
            "Implementation",
            "implementation",
            "Constructive algorithms",
            "constructive algorithms",
        },
    },
    "Data Structure / String": {
        "description": "자료구조 선택, 상태 유지, query/update invariant, string 처리 오류 진단",
        "raw_tags": {
            "Data structures",
            "Data Structures",
            "data structures",
            "String algorithms",
        },
    },
    "Greedy / Sorting": {
        "description": "정렬 기준, greedy choice, exchange argument 오류 진단",
        "raw_tags": {
            "Greedy algorithms",
            "greedy",
            "Sorting",
        },
    },
    "State-Space Search / DP / Graph": {
        "description": (
            "state 정의, recurrence, graph modeling, exhaustive search, pruning, "
            "path/cycle 오류 진단"
        ),
        "raw_tags": {
            "Dynamic programming",
            "dp",
            "Complete search",
            "brute force",
            "Graph algorithms",
        },
    },
}

BROAD_GENERAL_TAGS = {
    "Algorithms",
    "Fundamentals",
}

RAW_TAG_NORMALIZATION = {
    "Mathematics": "math",
    "math": "math",
    "Math": "math",
    "Mathematical": "math",
    "Basic Maths": "math",
    "Data structures": "data_structures",
    "Data Structures": "data_structures",
    "data structures": "data_structures",
    "Implementation": "implementation",
    "implementation": "implementation",
    "Greedy algorithms": "greedy",
    "greedy": "greedy",
    "Greedy": "greedy",
    "Dynamic programming": "dynamic_programming",
    "Dynamic Programming": "dynamic_programming",
    "dp": "dynamic_programming",
    "Constructive algorithms": "constructive",
    "constructive algorithms": "constructive",
    "Constructive": "constructive",
    "Complete search": "search_bruteforce",
    "brute force": "search_bruteforce",
    "Graph algorithms": "graph",
    "graphs": "graph",
    "Graph traversal": "graph",
    "dfs and similar": "graph",
    "String algorithms": "string",
    "Strings": "string",
    "strings": "string",
    "String": "string",
    "Sorting": "sorting",
    "sortings": "sorting",
    "Algorithms": "general",
    "Fundamentals": "general",
}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _normalize_tags(parsed)
    if isinstance(value, (list, tuple, set)):
        tags = []
        for item in value:
            if item is None:
                continue
            tag = str(item).strip()
            if tag:
                tags.append(tag)
        return tags
    return [str(value).strip()] if str(value).strip() else []


def extract_tags(record: dict[str, Any]) -> list[str]:
    native_metadata = _as_mapping(record.get("native_metadata"))
    raw_fields = _as_mapping(native_metadata.get("raw_fields"))
    tags = _normalize_tags(native_metadata.get("tags"))
    if not tags:
        tags = _normalize_tags(raw_fields.get("cf_tags"))
    return tags


def normalize_tag(raw_tag: str) -> str:
    if raw_tag in RAW_TAG_NORMALIZATION:
        return RAW_TAG_NORMALIZATION[raw_tag]
    normalized = raw_tag.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def critics_for_tags(raw_tags: list[str]) -> set[str]:
    raw_tag_set = set(raw_tags)
    critics = set()
    for critic_name, definition in CRITIC_DEFINITIONS.items():
        if raw_tag_set & definition["raw_tags"]:
            critics.add(critic_name)
    return critics


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def compute_distributions(input_path: str | Path) -> dict[str, Any]:
    raw_tag_counts: Counter[str] = Counter()
    normalized_tag_counts: Counter[str] = Counter()
    general_tag_counts: Counter[str] = Counter()
    critic_record_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    tagged_record_source_counts: Counter[str] = Counter()
    critic_source_counts: dict[str, Counter[str]] = {
        critic_name: Counter() for critic_name in CRITIC_DEFINITIONS
    }
    raw_to_normalized_counts: dict[str, Counter[str]] = {}
    total_records = 0
    records_with_tags = 0
    records_with_critic = 0
    multi_critic_records = 0

    for record in iter_jsonl(input_path):
        total_records += 1
        source = str(record.get("source_platform") or record.get("source") or "unknown")
        source_counts[source] += 1

        raw_tags = sorted(set(extract_tags(record)))
        if not raw_tags:
            continue
        records_with_tags += 1
        tagged_record_source_counts[source] += 1

        normalized_tags = {normalize_tag(tag) for tag in raw_tags}
        raw_tag_counts.update(raw_tags)
        normalized_tag_counts.update(normalized_tags)
        general_tag_counts.update(tag for tag in raw_tags if tag in BROAD_GENERAL_TAGS)
        for raw_tag in raw_tags:
            normalized = normalize_tag(raw_tag)
            raw_to_normalized_counts.setdefault(normalized, Counter()).update([raw_tag])

        critics = critics_for_tags(raw_tags)
        if critics:
            records_with_critic += 1
        if len(critics) > 1:
            multi_critic_records += 1
        for critic in critics:
            critic_record_counts[critic] += 1
            critic_source_counts[critic][source] += 1

    critic_record_counts = Counter(
        {critic: critic_record_counts.get(critic, 0) for critic in CRITIC_DEFINITIONS}
    )
    critic_tag_sum_counts = Counter(
        {
            critic: sum(raw_tag_counts.get(tag, 0) for tag in definition["raw_tags"])
            for critic, definition in CRITIC_DEFINITIONS.items()
        }
    )
    return {
        "input": str(input_path),
        "total_records": total_records,
        "records_with_tags": records_with_tags,
        "records_without_tags": total_records - records_with_tags,
        "records_with_critic": records_with_critic,
        "records_without_critic": total_records - records_with_critic,
        "multi_critic_record_count": multi_critic_records,
        "unique_raw_tag_count": len(raw_tag_counts),
        "unique_normalized_tag_count": len(normalized_tag_counts),
        "raw_tag_counts": _sorted_counts(raw_tag_counts),
        "normalized_tag_counts": _sorted_counts(normalized_tag_counts),
        "general_auxiliary_tag_counts": _sorted_counts(general_tag_counts),
        "critic_count_type": "included_raw_tag_count_sum",
        "critic_counts": _sorted_counts(critic_tag_sum_counts),
        "critic_tag_sum_counts": _sorted_counts(critic_tag_sum_counts),
        "critic_record_counts": _sorted_counts(critic_record_counts),
        "critic_definitions": {
            critic: {
                "description": definition["description"],
                "raw_tags": sorted(definition["raw_tags"]),
            }
            for critic, definition in CRITIC_DEFINITIONS.items()
        },
        "raw_to_normalized_counts": {
            tag: _sorted_counts(counts)
            for tag, counts in sorted(raw_to_normalized_counts.items())
        },
        "source_counts": _sorted_counts(source_counts),
        "tagged_record_source_counts": _sorted_counts(tagged_record_source_counts),
        "critic_source_counts": {
            critic: _sorted_counts(counts) for critic, counts in critic_source_counts.items()
        },
    }


def compute_tag_distribution(input_path: str | Path) -> dict[str, Any]:
    report = compute_distributions(input_path)
    return {
        "input": report["input"],
        "total_records": report["total_records"],
        "records_with_tags": report["records_with_tags"],
        "records_without_tags": report["records_without_tags"],
        "unique_raw_tag_count": report["unique_raw_tag_count"],
        "unique_normalized_tag_count": report["unique_normalized_tag_count"],
        "raw_tag_counts": report["raw_tag_counts"],
        "normalized_tag_counts": report["normalized_tag_counts"],
        "general_auxiliary_tag_counts": report["general_auxiliary_tag_counts"],
        "raw_to_normalized_counts": report["raw_to_normalized_counts"],
        "tagged_record_source_counts": report["tagged_record_source_counts"],
    }


def compute_critic_distribution(input_path: str | Path) -> dict[str, Any]:
    report = compute_distributions(input_path)
    return {
        "input": report["input"],
        "total_records": report["total_records"],
        "records_with_tags": report["records_with_tags"],
        "records_with_critic": report["records_with_critic"],
        "records_without_critic": report["records_without_critic"],
        "multi_critic_record_count": report["multi_critic_record_count"],
        "critic_count_type": report["critic_count_type"],
        "critic_counts": report["critic_counts"],
        "critic_tag_sum_counts": report["critic_tag_sum_counts"],
        "critic_record_counts": report["critic_record_counts"],
        "critic_definitions": report["critic_definitions"],
        "critic_source_counts": report["critic_source_counts"],
    }


def write_distribution_csv(
    counts: dict[str, int],
    output_path: str | Path,
    key_field: str,
    extra_rows: dict[str, dict[str, Any]] | None = None,
) -> None:
    ensure_parent(output_path)
    extra_rows = extra_rows or {}
    extra_fields = sorted({key for row in extra_rows.values() for key in row})
    fieldnames = [key_field, "count", *extra_fields]
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, count in counts.items():
            row = {key_field: key, "count": count}
            row.update(extra_rows.get(key, {}))
            writer.writerow(row)


def write_normalized_tag_counts_csv(report: dict[str, Any], output_path: str | Path) -> None:
    extra_rows = {
        tag: {
            "raw_tags": json.dumps(
                report["raw_to_normalized_counts"].get(tag, {}),
                ensure_ascii=False,
                sort_keys=True,
            )
        }
        for tag in report["normalized_tag_counts"]
    }
    write_distribution_csv(
        report["normalized_tag_counts"],
        output_path,
        "normalized_tag",
        extra_rows,
    )


def write_raw_tag_counts_csv(report: dict[str, Any], output_path: str | Path) -> None:
    extra_rows = {
        raw_tag: {"normalized_tag": normalize_tag(raw_tag)}
        for raw_tag in report["raw_tag_counts"]
    }
    write_distribution_csv(
        report["raw_tag_counts"],
        output_path,
        "raw_tag",
        extra_rows,
    )


def write_critic_counts_csv(report: dict[str, Any], output_path: str | Path) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "critic",
                "tag_sum_count",
                "record_count",
                "description",
                "raw_tags",
            ],
        )
        writer.writeheader()
        for critic, tag_sum_count in report["critic_tag_sum_counts"].items():
            writer.writerow(
                {
                    "critic": critic,
                    "tag_sum_count": tag_sum_count,
                    "record_count": report["critic_record_counts"][critic],
                    "description": report["critic_definitions"][critic]["description"],
                    "raw_tags": json.dumps(
                        report["critic_definitions"][critic]["raw_tags"],
                        ensure_ascii=False,
                    ),
                }
            )


def write_tag_counts_csv(report: dict[str, Any], output_path: str | Path) -> None:
    write_normalized_tag_counts_csv(report, output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count normalized tag and critic distributions."
    )
    parser.add_argument(
        "--input",
        default="data/processed/04_execution_ready.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--tag-report",
        default="src/acc_data_pipeline/critic/normalized_tag_distribution.json",
        help="Output JSON report path for normalized tag distribution.",
    )
    parser.add_argument(
        "--tag-csv",
        default="src/acc_data_pipeline/critic/normalized_tag_distribution.csv",
        help="Output CSV path with one row per normalized tag.",
    )
    parser.add_argument(
        "--raw-tag-csv",
        default="src/acc_data_pipeline/critic/raw_tag_distribution.csv",
        help="Output CSV path with one row per raw tag.",
    )
    parser.add_argument(
        "--critic-report",
        default="src/acc_data_pipeline/critic/critic_distribution.json",
        help="Output JSON report path for critic distribution.",
    )
    parser.add_argument(
        "--critic-csv",
        default="src/acc_data_pipeline/critic/critic_distribution.csv",
        help="Output CSV path with one row per critic.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top normalized tags to print to stdout.",
    )
    args = parser.parse_args(argv)

    report = compute_distributions(args.input)
    tag_report = {
        "input": report["input"],
        "total_records": report["total_records"],
        "records_with_tags": report["records_with_tags"],
        "records_without_tags": report["records_without_tags"],
        "unique_raw_tag_count": report["unique_raw_tag_count"],
        "unique_normalized_tag_count": report["unique_normalized_tag_count"],
        "raw_tag_counts": report["raw_tag_counts"],
        "normalized_tag_counts": report["normalized_tag_counts"],
        "general_auxiliary_tag_counts": report["general_auxiliary_tag_counts"],
        "raw_to_normalized_counts": report["raw_to_normalized_counts"],
        "tagged_record_source_counts": report["tagged_record_source_counts"],
    }
    critic_report = {
        "input": report["input"],
        "total_records": report["total_records"],
        "records_with_tags": report["records_with_tags"],
        "records_with_critic": report["records_with_critic"],
        "records_without_critic": report["records_without_critic"],
        "multi_critic_record_count": report["multi_critic_record_count"],
        "critic_count_type": report["critic_count_type"],
        "critic_counts": report["critic_counts"],
        "critic_tag_sum_counts": report["critic_tag_sum_counts"],
        "critic_record_counts": report["critic_record_counts"],
        "critic_definitions": report["critic_definitions"],
        "critic_source_counts": report["critic_source_counts"],
    }

    write_json(args.tag_report, tag_report)
    write_json(args.critic_report, critic_report)
    write_normalized_tag_counts_csv(tag_report, args.tag_csv)
    write_raw_tag_counts_csv(tag_report, args.raw_tag_csv)
    write_critic_counts_csv(critic_report, args.critic_csv)

    print(f"Input: {report['input']}")
    print(f"Total records: {report['total_records']}")
    print(f"Records with tags: {report['records_with_tags']}")
    print(f"Records without tags: {report['records_without_tags']}")
    print(f"Records with critic: {report['records_with_critic']}")
    print(f"Records without critic: {report['records_without_critic']}")
    print(f"Unique raw tags: {report['unique_raw_tag_count']}")
    print(f"Unique normalized tags: {report['unique_normalized_tag_count']}")
    print(f"Wrote tag report: {args.tag_report}")
    print(f"Wrote tag CSV: {args.tag_csv}")
    print(f"Wrote raw tag CSV: {args.raw_tag_csv}")
    print(f"Wrote critic report: {args.critic_report}")
    print(f"Wrote critic CSV: {args.critic_csv}")
    print("Critic distribution by included raw-tag count sum:")
    for critic, count in report["critic_tag_sum_counts"].items():
        record_count = report["critic_record_counts"][critic]
        print(f"{critic}\t{count}\trecord_count={record_count}")
    if args.top > 0:
        print(f"Top {args.top} normalized tags:")
        for tag, count in list(report["normalized_tag_counts"].items())[: args.top]:
            print(f"{tag}\t{count}")


if __name__ == "__main__":
    main()
