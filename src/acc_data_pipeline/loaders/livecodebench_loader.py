"""LiveCodeBench dataset loader.

Decodes plain, pickled, base64, and compressed testcase payloads where needed, limits oversized test
data, and normalizes LiveCodeBench task variants into the common problem schema."""

from __future__ import annotations

import base64
import os
import pickle
import re
import zlib
from pathlib import Path
from typing import Any, Iterable

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


class LiveCodeBenchLoader(BaseLoader):
    source = "LiveCodeBench"
    dataset_candidates = ("livecodebench", "LiveCodeBench")

    def load(self) -> list[dict[str, Any]]:
        root = self.dataset_dir()
        if root is None:
            self.warnings.append(f"dataset_dir_not_found:{self.raw_root}/livecodebench")
            return []
        records: list[dict[str, Any]] = []
        for path, raw in self.iter_json_records(root):
            try:
                record = self._convert_record(raw, path)
                if record:
                    records.append(record)
            except Exception as exc:
                self.record_error(str(raw.get("question_id") or raw.get("id") or path), exc)
        if records:
            return records
        records.extend(self._load_huggingface_dataset(root))
        return records

    def _load_huggingface_dataset(self, root: Path) -> list[dict[str, Any]]:
        try:
            from datasets import load_from_disk  # type: ignore
        except ModuleNotFoundError:
            if any(root.rglob("*.arrow")):
                self.warnings.append(
                    "livecodebench_arrow_requires_optional_dependency:"
                    "install datasets/pyarrow or provide json/jsonl export"
                )
            return []
        records: list[dict[str, Any]] = []
        for dataset_dir in [root, *[path for path in root.iterdir() if path.is_dir()]]:
            try:
                dataset = load_from_disk(str(dataset_dir))
            except Exception:
                continue
            for split, table in dataset.items() if hasattr(dataset, "items") else [("unknown", dataset)]:
                for raw in table:
                    raw = dict(raw)
                    raw.setdefault("split", split)
                    record = self._convert_record(raw, dataset_dir)
                    if record:
                        records.append(record)
            if records:
                break
        return records

    def _convert_record(self, raw: dict[str, Any], path: Path) -> dict[str, Any] | None:
        if path.name in {"dataset_info.json", "dataset_dict.json", "state.json"}:
            return None
        if "features" in raw and "splits" in raw:
            return None
        statement = (
            raw.get("question_content")
            or raw.get("problem_statement")
            or raw.get("statement")
            or raw.get("question")
            or raw.get("description")
            or ""
        )
        if not str(statement).strip():
            return None
        task_variant = infer_task_variant(raw)
        task_family = map_task_family(task_variant)
        split = raw.get("split") or infer_split_from_path(path)
        source_problem_id = (
            raw.get("question_id")
            or raw.get("id")
            or raw.get("problem_id")
            or raw.get("contest_id")
            or stable_hash({"path": str(path), "statement": statement})
        )
        metadata = parse_maybe_json(raw.get("metadata"), default={})
        if not isinstance(metadata, dict):
            metadata = {}
        entry_point = (
            raw.get("fn_name")
            or raw.get("entry_point")
            or metadata.get("func_name")
            or metadata.get("entry_point")
        )
        eval_mode = infer_eval_mode(task_variant, entry_point, raw)
        problem_id = make_problem_id(self.source, split, safe_id(source_problem_id))
        tests = []
        for field_name, visibility in (
            ("public_test_cases", "public"),
            ("private_test_cases", "hidden"),
            ("public_tests", "public"),
            ("private_tests", "hidden"),
            ("tests", "unknown"),
            ("test_cases", "unknown"),
        ):
            raw_cases = decode_livecodebench_test_cases(raw.get(field_name))
            if visibility == "public":
                raw_cases = limit_test_payload(
                    raw_cases, int(os.environ.get("LIVECODEBENCH_MAX_PUBLIC_TESTS", "8"))
                )
            elif visibility == "hidden":
                raw_cases = limit_test_payload(
                    raw_cases, int(os.environ.get("LIVECODEBENCH_MAX_PRIVATE_TESTS", "8"))
                )
            tests.extend(
                tests_from_field(
                    problem_id,
                    raw_cases,
                    visibility,
                    eval_mode,
                )
            )
        references = build_reference_solutions(
            problem_id, raw.get("solutions") or raw.get("reference_solutions")
        )
        return build_problem_record(
            source=self.source,
            split=split,
            source_problem_id=safe_id(source_problem_id),
            raw_statement=str(statement),
            task_family=task_family,
            title=raw.get("question_title") or raw.get("title"),
            test_cases=tests,
            reference_solutions=references,
            starter_code=raw.get("starter_code"),
            eval_mode=eval_mode,
            entry_point=entry_point,
            source_url=raw.get("url") or raw.get("source_url") or infer_source_url(raw),
            source_platform=raw.get("platform") or raw.get("source_platform"),
            difficulty=raw.get("difficulty"),
            tags=collect_tags(raw, metadata),
            original_task_type=task_variant,
            raw_fields={"path": str(path), **raw},
        )


def infer_task_variant(raw: dict[str, Any]) -> str:
    for key in ("task_variant", "task", "task_type", "original_task_type"):
        value = raw.get(key)
        if value:
            return str(value)
    return "code_generation"


def map_task_family(task_variant: str) -> str:
    normalized = task_variant.lower()
    if "self" in normalized and "repair" in normalized:
        return "self_repair"
    if "execution" in normalized or "execute" in normalized:
        return "pure_code_execution"
    if "output" in normalized and "prediction" in normalized:
        return "test_output_prediction"
    if normalized in {"code_generation", "generation", "coding"}:
        return "algorithmic_code_generation"
    return "unknown"


def infer_eval_mode(task_variant: str, entry_point: Any, raw: dict[str, Any]) -> str:
    family = map_task_family(task_variant)
    if family == "self_repair":
        return "self_repair"
    if family in {"pure_code_execution", "test_output_prediction"}:
        return "unsupported"
    if raw.get("special_judge") or raw.get("requires_special_judge"):
        return "special_judge"
    return "function_call" if entry_point else "stdin_stdout"


def collect_tags(raw: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for source in (raw, metadata):
        for key in ("tags", "topics", "topic", "skills", "algorithms"):
            value = source.get(key)
            if isinstance(value, list):
                tags.extend(str(item) for item in value)
            elif value:
                tags.append(str(value))
    return sorted(set(tags))


def decode_livecodebench_test_cases(value: Any) -> Any:
    parsed = parse_maybe_json(value, default=None)
    if isinstance(parsed, (dict, list)) or not isinstance(value, str):
        return parsed
    text = value.strip()
    if not text:
        return None
    try:
        padded = text + "=" * ((4 - len(text) % 4) % 4)
        decompressed = zlib.decompress(base64.b64decode(padded))
        unpickled = pickle.loads(decompressed)
    except Exception:
        return value
    return parse_maybe_json(unpickled, default=unpickled)


def limit_test_payload(value: Any, limit: int) -> Any:
    if limit < 0:
        return value
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, dict):
        limited = dict(value)
        inputs = limited.get("inputs", limited.get("input"))
        outputs = limited.get("outputs", limited.get("output", limited.get("expected_output")))
        if isinstance(inputs, list) and isinstance(outputs, list):
            for key in ("inputs", "input"):
                if key in limited:
                    limited[key] = inputs[:limit]
            for key in ("outputs", "output", "expected_output"):
                if key in limited:
                    limited[key] = outputs[:limit]
        return limited
    return value


def infer_source_url(raw: dict[str, Any]) -> str | None:
    platform = str(raw.get("platform") or raw.get("source_platform") or "").lower()
    question_id = str(raw.get("question_id") or raw.get("id") or "")
    match = re.fullmatch(r"(\d+)_([A-Za-z0-9]+)", question_id)
    if "codeforces" in platform and match:
        contest_id, index = match.groups()
        return f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    return None


def infer_split_from_path(path: Path) -> str:
    text = str(path).lower()
    for split in ("train", "valid", "validation", "test"):
        if split in text:
            return "valid" if split == "validation" else split
    return "unknown"
