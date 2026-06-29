"""TACO dataset loader.

Reads BAAI/TACO-style JSON or JSONL records, extracts statements, solutions, tags, and tests, and
converts function-call or stdin/stdout problems into the shared normalized schema."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

from acc_data_pipeline.loaders.base import (
    BaseLoader,
    build_problem_record,
    build_reference_solutions,
    make_problem_id,
    safe_id,
    stable_hash,
    tests_from_field,
)
from acc_data_pipeline.utils.io import parse_maybe_json


class TACOLoader(BaseLoader):
    source = "TACO"
    dataset_candidates = ("baai_taco", "taco", "TACO")

    def load(self) -> list[dict[str, Any]]:
        root = self.dataset_dir()
        if root is None:
            self.warnings.append(f"dataset_dir_not_found:{self.raw_root}/baai_taco")
            return []
        baai_records = self._load_baai_taco_records(root)
        if baai_records:
            return baai_records
        records: list[dict[str, Any]] = []
        for path, raw in self.iter_json_records(root):
            try:
                record = self._convert_record(raw, path)
                if record:
                    records.append(record)
            except Exception as exc:
                self.record_error(str(raw.get("id") or raw.get("problem_id") or path), exc)
        if not records:
            self.warnings.append(
                "no_taco_algorithmic_records_found:"
                "expected baai_taco parquet/arrow files or algorithmic json/jsonl records"
            )
        return records

    def _load_baai_taco_records(self, root: Path) -> list[dict[str, Any]]:
        parquet_files = sorted((root / "ALL").glob("*.parquet"))
        if not parquet_files:
            parquet_files = sorted(root.glob("**/*.parquet"))
        if not parquet_files:
            return []
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ModuleNotFoundError:
            self.warnings.append("baai_taco_requires_pyarrow:install pyarrow in the active env")
            return []
        records: list[dict[str, Any]] = []
        columns = [
            "question",
            "solutions",
            "starter_code",
            "input_output",
            "difficulty",
            "raw_tags",
            "name",
            "source",
            "tags",
            "skill_types",
            "url",
            "time_limit",
            "memory_limit",
            "Expected Auxiliary Space",
            "Expected Time Complexity",
        ]
        for path in parquet_files:
            try:
                parquet_file = pq.ParquetFile(path)
                available = [name for name in columns if name in parquet_file.schema_arrow.names]
                row_offset = 0
                for batch in parquet_file.iter_batches(batch_size=1000, columns=available):
                    for row_index, raw in enumerate(batch.to_pylist(), start=row_offset):
                        try:
                            record = self._convert_baai_taco_record(raw, path, row_index)
                            if record:
                                records.append(record)
                        except Exception as exc:
                            self.record_error(f"{path.name}:{row_index}", exc)
                    row_offset += batch.num_rows
            except Exception as exc:
                self.warnings.append(f"baai_taco_parquet_read_failed:{path}:{exc}")
        if records:
            self.warnings.append(f"baai_taco_parquet_loaded:{len(records)}")
        return records

    def _convert_baai_taco_record(
        self, raw: dict[str, Any], path: Path, index: int
    ) -> dict[str, Any] | None:
        statement = str(raw.get("question") or "").strip()
        if not statement:
            return None
        raw_io = parse_taco_value(raw.get("input_output"), default={})
        if not raw_io:
            return None
        max_tests = int(os.environ.get("TACO_MAX_TESTS", "8"))
        raw_io, total_tests = limit_test_payload(raw_io, max_tests)
        split = infer_split_from_path(path)
        source_problem_id = (
            raw.get("url")
            or raw.get("name")
            or stable_hash({"path": str(path), "index": index, "question": statement})
        )
        problem_id = make_problem_id(self.source, split, safe_id(source_problem_id))
        fn_name = raw_io.get("fn_name") if isinstance(raw_io, dict) else None
        eval_mode = "function_call" if fn_name else "stdin_stdout"
        tests = tests_from_field(problem_id, raw_io, "unknown", eval_mode)
        if not tests:
            return None
        solution_payload = parse_taco_value(raw.get("solutions"), default=[])
        total_solutions = count_solution_payload(solution_payload)
        solution_payload = limit_solution_payload(
            solution_payload,
            int(os.environ.get("TACO_MAX_SOLUTIONS", "3")),
        )
        references = build_reference_solutions(
            problem_id,
            solution_payload,
            default_language="python",
        )
        tags = collect_tags(raw)
        raw_fields = {
            "path": str(path),
            "row_index": index,
            "source": raw.get("source"),
            "raw_tags": parse_taco_value(raw.get("raw_tags"), default=raw.get("raw_tags")),
            "skill_types": parse_taco_value(raw.get("skill_types"), default=raw.get("skill_types")),
            "time_limit": raw.get("time_limit"),
            "memory_limit": raw.get("memory_limit"),
            "expected_time_complexity": raw.get("Expected Time Complexity"),
            "expected_auxiliary_space": raw.get("Expected Auxiliary Space"),
            "input_output_total_tests": total_tests,
            "solutions_total": total_solutions,
        }
        return build_problem_record(
            source=self.source,
            split=split,
            source_problem_id=safe_id(source_problem_id),
            raw_statement=statement,
            title=raw.get("name"),
            test_cases=tests,
            reference_solutions=references,
            starter_code=raw.get("starter_code") or None,
            eval_mode=eval_mode,
            entry_point=fn_name,
            source_url=raw.get("url"),
            source_platform=raw.get("source") or "BAAI/TACO",
            difficulty=raw.get("difficulty"),
            tags=tags,
            original_task_type="algorithmic_code_generation",
            raw_fields=raw_fields,
            timeout_seconds=parse_time_limit_seconds(raw.get("time_limit")) or 5,
            memory_limit_mb=parse_memory_limit_mb(raw.get("memory_limit")) or 512,
        )

    def _convert_record(self, raw: dict[str, Any], path: Path) -> dict[str, Any] | None:
        statement = (
            raw.get("question")
            or raw.get("statement")
            or raw.get("problem_statement")
            or raw.get("description")
            or ""
        )
        if not str(statement).strip():
            return None
        has_eval_evidence = any(
            key in raw
            for key in (
                "input_output",
                "tests",
                "test_cases",
                "public_tests",
                "private_tests",
                "solutions",
            )
        )
        if not has_eval_evidence:
            return None
        split = raw.get("split") or infer_split_from_path(path)
        source_problem_id = (
            raw.get("id")
            or raw.get("problem_id")
            or raw.get("source_problem_id")
            or stable_hash({"path": str(path), "statement": statement})
        )
        raw_io = raw.get("input_output") or raw.get("tests") or raw.get("test_cases")
        raw_io_dict = raw_io if isinstance(raw_io, dict) else {}
        fn_name = raw.get("fn_name") or raw.get("entry_point") or raw_io_dict.get("fn_name")
        special_judge = bool(raw.get("special_judge") or raw.get("requires_special_judge"))
        eval_mode = "function_call" if fn_name else "special_judge" if special_judge else "stdin_stdout"
        problem_id = make_problem_id(self.source, split, safe_id(source_problem_id))
        tests = tests_from_field(problem_id, raw_io, "unknown", eval_mode)
        tests.extend(tests_from_field(problem_id, raw.get("public_tests"), "public", eval_mode))
        tests.extend(tests_from_field(problem_id, raw.get("private_tests"), "hidden", eval_mode))
        tags = collect_tags(raw)
        references = build_reference_solutions(problem_id, raw.get("solutions") or raw.get("reference_solutions"))
        return build_problem_record(
            source=self.source,
            split=split,
            source_problem_id=safe_id(source_problem_id),
            raw_statement=str(statement),
            title=raw.get("title"),
            test_cases=tests,
            reference_solutions=references,
            eval_mode=eval_mode,
            entry_point=fn_name,
            source_url=raw.get("url") or raw.get("source_url"),
            source_platform=raw.get("source") or raw.get("source_platform") or "TACO",
            difficulty=raw.get("difficulty"),
            tags=tags,
            raw_fields={"path": str(path), **raw},
        )


def collect_tags(raw: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "tags",
        "raw_tags",
        "skill_types",
        "topics",
        "topic",
        "skills",
        "algorithms",
        "algorithm_tags",
    ):
        value = parse_taco_value(raw.get(key), default=raw.get(key))
        if isinstance(value, list):
            tags.extend(str(item) for item in value)
        elif value:
            tags.append(str(value))
    return sorted(set(tags))


def infer_split_from_path(path: Path) -> str:
    text = str(path).lower()
    for split in ("train", "valid", "validation", "test"):
        if split in text:
            return "valid" if split == "validation" else split
    return "unknown"


def parse_taco_value(value: Any, default: Any = None) -> Any:
    parsed = parse_maybe_json(value, default=None)
    if parsed is None:
        return default
    if parsed is not value:
        return parsed
    if isinstance(parsed, str):
        stripped = parsed.strip()
        if not stripped:
            return default
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return parsed
    return parsed


def limit_test_payload(value: Any, limit: int) -> tuple[Any, int | None]:
    if limit < 0:
        return value, count_test_payload(value)
    if isinstance(value, dict):
        inputs = value.get("inputs", value.get("input", []))
        outputs = value.get("outputs", value.get("output", value.get("expected_output", [])))
        if isinstance(inputs, list) and isinstance(outputs, list):
            total = min(len(inputs), len(outputs))
            limited = dict(value)
            if "inputs" in limited:
                limited["inputs"] = inputs[:limit]
            elif "input" in limited:
                limited["input"] = inputs[:limit]
            if "outputs" in limited:
                limited["outputs"] = outputs[:limit]
            elif "output" in limited:
                limited["output"] = outputs[:limit]
            elif "expected_output" in limited:
                limited["expected_output"] = outputs[:limit]
            return limited, total
        return value, count_test_payload(value)
    if isinstance(value, list):
        return value[:limit], len(value)
    return value, count_test_payload(value)


def count_test_payload(value: Any) -> int | None:
    if isinstance(value, dict):
        inputs = value.get("inputs", value.get("input", []))
        outputs = value.get("outputs", value.get("output", value.get("expected_output", [])))
        if isinstance(inputs, list) and isinstance(outputs, list):
            return min(len(inputs), len(outputs))
        if "input" in value and ("output" in value or "expected_output" in value):
            return 1
    if isinstance(value, list):
        return len(value)
    return None


def limit_solution_payload(value: Any, limit: int) -> Any:
    if limit < 0:
        return value
    if isinstance(value, dict):
        limited = dict(value)
        for key in ("solution", "solutions", "code"):
            item = limited.get(key)
            if isinstance(item, list):
                limited[key] = item[:limit]
            elif isinstance(item, str) and limit == 0:
                limited[key] = []
        for key in ("language", "languages"):
            item = limited.get(key)
            if isinstance(item, list):
                limited[key] = item[:limit]
        return limited
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, str):
        return value if limit else []
    return value


def count_solution_payload(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("solution", "solutions", "code"):
            item = value.get(key)
            if isinstance(item, list):
                return len(item)
            if isinstance(item, str) and item.strip():
                return 1
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        return 1
    return None


def parse_time_limit_seconds(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def parse_memory_limit_mb(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    amount = float(match.group(0))
    if "gb" in text or "gib" in text:
        amount *= 1024
    elif "kb" in text or "kib" in text:
        amount /= 1024
    return max(1, int(amount))
