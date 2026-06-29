"""Duplicate and near-duplicate removal for normalized benchmark records.

Records are grouped by exact hashes, source URLs, and configurable n-gram Jaccard similarity,
then a canonical problem is selected using source priority and data-completeness signals."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from acc_data_pipeline.utils.text import jaccard, normalize_statement, sha256_text, token_ngrams

DEFAULT_DEDUP_CONFIG: dict[str, Any] = {
    "source_priority": ["LiveCodeBench", "TACO", "CodeContests", "APPS"],
    "exact": {"use_statement_hash": True, "use_source_url": True, "use_examples": False},
    "near_duplicate": {
        "enabled": True,
        "method": "token_ngram_jaccard",
        "ngram_size": 5,
        "threshold": 0.85,
    },
    "canonical_selection": {
        "prefer_more_test_cases": True,
        "prefer_reference_solution": True,
        "prefer_native_tags": True,
        "prefer_source_url": True,
    },
}


@dataclass
class UnionFind:
    parent: dict[str, str]

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def deduplicate_records(
    records: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = merge_config(DEFAULT_DEDUP_CONFIG, config or {})
    for record in records:
        normalized = normalize_statement((record.get("problem_statement") or {}).get("raw"))
        record.setdefault("dedup", {})
        record["dedup"]["normalized_statement_hash"] = sha256_text(normalized)
        record["_dedup_normalized_statement"] = normalized
        record["_dedup_ngrams"] = token_ngrams(
            normalized, int(config["near_duplicate"].get("ngram_size", 5))
        )

    uf = UnionFind(parent={record["problem_id"]: record["problem_id"] for record in records})
    methods: dict[tuple[str, str], str] = {}
    exact_groups = exact_duplicate_groups(records, config)
    for group_records, method in exact_groups:
        union_group(uf, group_records, methods, method)
    if config["near_duplicate"].get("enabled", True):
        detect_near_duplicates(records, config, uf, methods)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[uf.find(record["problem_id"])].append(record)

    outputs: list[dict[str, Any]] = []
    duplicate_items: list[dict[str, Any]] = []
    source_pair_counter: Counter[str] = Counter()
    duplicate_group_index = 0
    for group in grouped.values():
        if len(group) == 1:
            record = cleanup_temp(group[0])
            record["dedup"].update(
                {
                    "canonical_problem_id": record["problem_id"],
                    "dedup_group_id": None,
                    "is_duplicate": False,
                    "duplicate_of": None,
                    "dedup_method": record["dedup"].get("dedup_method"),
                }
            )
            outputs.append(record)
            continue
        duplicate_group_index += 1
        group_id = f"dedup_group_{duplicate_group_index:06d}"
        canonical = select_canonical(group, config)
        for record in group:
            pair = tuple(sorted((canonical.get("source"), record.get("source"))))
            if record["problem_id"] != canonical["problem_id"]:
                source_pair_counter[f"{pair[0]}::{pair[1]}"] += 1
            method = lookup_method(methods, canonical["problem_id"], record["problem_id"])
            record["dedup"].update(
                {
                    "canonical_problem_id": canonical["problem_id"],
                    "dedup_group_id": group_id,
                    "is_duplicate": record["problem_id"] != canonical["problem_id"],
                    "duplicate_of": None
                    if record["problem_id"] == canonical["problem_id"]
                    else canonical["problem_id"],
                    "dedup_method": method,
                }
            )
            duplicate_items.append(
                {
                    "problem_id": record["problem_id"],
                    "source": record.get("source"),
                    "canonical_problem_id": canonical["problem_id"],
                    "dedup_group_id": group_id,
                    "is_duplicate": record["problem_id"] != canonical["problem_id"],
                    "dedup_method": method,
                }
            )
        outputs.append(cleanup_temp(canonical))

    outputs.sort(key=lambda item: item["problem_id"])
    report = {
        "input_count": len(records),
        "output_count": len(outputs),
        "num_duplicate_groups": duplicate_group_index,
        "num_removed_duplicates": len(records) - len(outputs),
        "duplicates_by_source_pair": dict(source_pair_counter),
        "near_duplicate_threshold": config["near_duplicate"].get("threshold", 0.85),
        "duplicates": duplicate_items,
    }
    return outputs, report


def merge_config(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = {**base}
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def exact_duplicate_groups(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[tuple[list[dict[str, Any]], str]]:
    groups: list[tuple[list[dict[str, Any]], str]] = []
    exact = config.get("exact", {})
    if exact.get("use_statement_hash", True):
        groups.extend(group_by_key(records, lambda r: r["dedup"].get("normalized_statement_hash"), "exact_statement_hash"))
    if exact.get("use_source_url", True):
        groups.extend(group_by_key(records, lambda r: r.get("source_url"), "exact_source_url"))
    if exact.get("use_examples", True):
        groups.extend(group_by_key(records, example_key, "exact_examples"))
    return groups


def group_by_key(
    records: list[dict[str, Any]], key_fn: Any, method: str
) -> list[tuple[list[dict[str, Any]], str]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key:
            buckets[str(key)].append(record)
    return [(items, method) for items in buckets.values() if len(items) > 1]


def example_key(record: dict[str, Any]) -> str | None:
    examples = record.get("examples") or []
    if not examples:
        return None
    joined = "|".join(
        f"{example.get('input')}=>{example.get('output')}"
        for example in examples
        if isinstance(example, dict)
    )
    return sha256_text(joined) if joined else None


def union_group(
    uf: UnionFind,
    group_records: list[dict[str, Any]],
    methods: dict[tuple[str, str], str],
    method: str,
) -> None:
    first = group_records[0]["problem_id"]
    for record in group_records[1:]:
        other = record["problem_id"]
        uf.union(first, other)
        methods[tuple(sorted((first, other)))] = method


def detect_near_duplicates(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    uf: UnionFind,
    methods: dict[tuple[str, str], str],
) -> None:
    ngram_size = int(config["near_duplicate"].get("ngram_size", 5))
    threshold = float(config["near_duplicate"].get("threshold", 0.85))
    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        for gram in record.get("_dedup_ngrams", set()):
            if len(inverted[gram]) < 200:
                inverted[gram].append(idx)
    compared: set[tuple[int, int]] = set()
    for idx, record in enumerate(records):
        candidate_counts: Counter[int] = Counter()
        grams = record.get("_dedup_ngrams", set())
        for gram in grams:
            for other_idx in inverted.get(gram, []):
                if other_idx <= idx:
                    continue
                candidate_counts[other_idx] += 1
        for other_idx, overlap_hint in candidate_counts.items():
            pair = (idx, other_idx)
            if pair in compared:
                continue
            compared.add(pair)
            other = records[other_idx]
            other_grams = other.get("_dedup_ngrams", set())
            if not grams or not other_grams:
                continue
            min_needed = int(threshold * max(len(grams), len(other_grams)) / 2)
            if overlap_hint < max(1, min_needed):
                continue
            score = jaccard(grams, other_grams)
            if score >= threshold:
                left = record["problem_id"]
                right = other["problem_id"]
                uf.union(left, right)
                methods[tuple(sorted((left, right)))] = (
                    f"near_duplicate_token_{ngram_size}gram_jaccard"
                )


def select_canonical(group: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    priority = {source: idx for idx, source in enumerate(config.get("source_priority", []))}

    def score(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            priority.get(record.get("source"), 999),
            -len(record.get("test_cases") or []),
            -len(record.get("reference_solutions") or []),
            -statement_completeness(record),
            -len(((record.get("native_metadata") or {}).get("tags") or [])),
            0 if record.get("source_url") else 1,
            record.get("problem_id", ""),
        )

    return sorted(group, key=score)[0]


def statement_completeness(record: dict[str, Any]) -> int:
    statement = record.get("problem_statement") or {}
    return sum(1 for key in ("raw", "description", "input_format", "output_format", "constraints", "notes") if statement.get(key))


def lookup_method(methods: dict[tuple[str, str], str], left: str, right: str) -> str | None:
    if left == right:
        return "canonical"
    return methods.get(tuple(sorted((left, right)))) or "duplicate_group"


def cleanup_temp(record: dict[str, Any]) -> dict[str, Any]:
    record.pop("_dedup_normalized_statement", None)
    record.pop("_dedup_ngrams", None)
    return record
