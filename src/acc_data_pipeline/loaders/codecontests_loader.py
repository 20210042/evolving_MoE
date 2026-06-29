"""CodeContests dataset loader.

Prefers exported JSONL when available, otherwise falls back to recordio/protobuf parsing, infers source
URLs/tags where possible, and converts public/private/generated tests into normalized records."""

from __future__ import annotations

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
from acc_data_pipeline.utils.io import iter_jsonl
from acc_data_pipeline.utils.protobuf_wire import first_int, first_text, parse_message


class CodeContestsLoader(BaseLoader):
    source = "CodeContests"
    dataset_candidates = ("dm-code_contests", "codecontests", "CodeContests")

    def load(self) -> list[dict[str, Any]]:
        exported_records = self._load_exported_jsonl()
        if exported_records:
            return exported_records
        root = self.dataset_dir()
        if root is None:
            self.warnings.append(f"dataset_dir_not_found:{self.raw_root}/dm-code_contests")
            return []
        records: list[dict[str, Any]] = []
        found_json = False
        for path, raw in self.iter_json_records(root):
            found_json = True
            try:
                record = self._convert_record(raw, path)
                if record:
                    records.append(record)
            except Exception as exc:
                self.record_error(str(raw.get("id") or raw.get("name") or path), exc)
        if not found_json and any(root.glob("*.riegeli*")):
            records.extend(self._load_riegeli_records(root))
        return records

    def _load_exported_jsonl(self) -> list[dict[str, Any]]:
        export_path = os.environ.get("CODECONTESTS_EXPORT_JSONL")
        if not export_path:
            return []
        path = Path(export_path)
        if not path.exists():
            self.warnings.append(f"codecontests_export_jsonl_not_found:{path}")
            return []
        records: list[dict[str, Any]] = []
        for raw in iter_jsonl(path):
            try:
                record = self._convert_record(raw, path)
                if record:
                    records.append(record)
            except Exception as exc:
                self.record_error(str(raw.get("id") or raw.get("name") or path), exc)
        if records:
            self.warnings.append(f"codecontests_export_jsonl_loaded:{path}:{len(records)}")
        return records

    def _load_riegeli_records(self, root: Path) -> list[dict[str, Any]]:
        try:
            from riegeli import records as riegeli_records  # type: ignore
        except ModuleNotFoundError:
            self.warnings.append(
                "codecontests_riegeli_requires_optional_dependency:"
                "install riegeli or provide json/jsonl export"
            )
            return []
        output: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.riegeli*")):
            split = "test" if "test" in path.name else "train"
            try:
                with riegeli_records.RecordReader(open(path, "rb")) as reader:
                    for index, payload in enumerate(reader.read_records()):
                        raw = self._decode_riegeli_problem(payload, path, index, split)
                        if raw:
                            record = self._convert_record(raw, path)
                            if record:
                                output.append(record)
            except Exception as exc:
                self.warnings.append(f"codecontests_riegeli_read_failed:{path.name}:{exc}")
        if output:
            self.warnings.append("codecontests_riegeli_loaded_with_builtin_proto_decoder")
        return output

    def _decode_riegeli_problem(
        self, payload: bytes | str, path: Path, index: int, split: str
    ) -> dict[str, Any] | None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        fields = parse_message(payload)
        description = first_text(fields, 2)
        name = first_text(fields, 1) or f"{path.stem}_{index:06d}"
        if not description:
            return None
        return {
            "name": name,
            "description": description,
            "public_tests": decode_codecontests_tests(fields.get(4, [])),
            "private_tests": decode_codecontests_tests(fields.get(5, [])),
            "generated_tests": decode_codecontests_tests(fields.get(18, [])),
            "source": codecontests_source_name(first_int(fields, 6)),
            "difficulty": codecontests_difficulty_name(first_int(fields, 7)),
            "solutions": decode_codecontests_solutions(fields.get(8, [])),
            "incorrect_solutions": decode_codecontests_solutions(fields.get(19, [])),
            "split": split,
            "id": f"{path.stem}_{index:06d}",
        }

    def _convert_record(self, raw: dict[str, Any], path: Path) -> dict[str, Any] | None:
        description = (
            raw.get("description")
            or raw.get("statement")
            or raw.get("problem_statement")
            or raw.get("question")
            or ""
        )
        if isinstance(description, bytes):
            description = description.decode("utf-8", errors="replace")
        if not str(description).strip():
            return None
        split = raw.get("split") or infer_split_from_path(path)
        source_problem_id = (
            raw.get("id")
            or raw.get("name")
            or raw.get("source_problem_id")
            or stable_hash({"path": str(path), "description": description})
        )
        problem_id = make_problem_id(self.source, split, safe_id(source_problem_id))
        special_judge = bool(
            raw.get("special_judge")
            or raw.get("requires_special_judge")
            or raw.get("custom_checker")
        )
        eval_mode = "special_judge" if special_judge else "stdin_stdout"
        tests = []
        tests.extend(tests_from_field(problem_id, raw.get("public_tests"), "public", eval_mode))
        tests.extend(tests_from_field(problem_id, raw.get("private_tests"), "hidden", eval_mode))
        tests.extend(tests_from_field(problem_id, raw.get("generated_tests"), "generated", eval_mode))
        if not tests:
            tests.extend(tests_from_field(problem_id, raw.get("tests"), "unknown", eval_mode))
        references = build_reference_solutions(problem_id, raw.get("solutions") or raw.get("correct_solutions"))
        raw_fields = {
            "path": str(path),
            "incorrect_solutions": raw.get("incorrect_solutions"),
            "solutions_total": raw.get("solutions_total"),
            "incorrect_solutions_total": raw.get("incorrect_solutions_total"),
            "public_tests_total": raw.get("public_tests_total"),
            "private_tests_total": raw.get("private_tests_total"),
            "generated_tests_total": raw.get("generated_tests_total"),
            "source": raw.get("source"),
            "difficulty": raw.get("difficulty"),
            "num_public_tests": len(raw.get("public_tests", {}).get("inputs", []))
            if isinstance(raw.get("public_tests"), dict)
            else None,
            "time_limit_seconds": raw.get("time_limit_seconds"),
            "memory_limit_bytes": raw.get("memory_limit_bytes"),
            "input_file": raw.get("input_file"),
            "output_file": raw.get("output_file"),
            "cf_contest_id": raw.get("cf_contest_id"),
            "cf_index": raw.get("cf_index"),
            "cf_points": raw.get("cf_points"),
            "cf_rating": raw.get("cf_rating"),
            "cf_tags": raw.get("cf_tags"),
        }
        tags = []
        for key in ("tags", "cf_tags"):
            value = raw.get(key)
            if isinstance(value, list):
                tags.extend(str(tag) for tag in value)
            elif value:
                tags.append(str(value))
        return build_problem_record(
            source=self.source,
            split=split,
            source_problem_id=safe_id(source_problem_id),
            raw_statement=str(description),
            title=raw.get("name") or raw.get("title"),
            test_cases=tests,
            reference_solutions=references,
            eval_mode=eval_mode,
            source_url=raw.get("url") or raw.get("source_url") or infer_source_url(raw),
            source_platform=raw.get("source") or raw.get("platform") or raw.get("source_platform"),
            difficulty=raw.get("difficulty"),
            tags=tags,
            raw_fields=raw_fields,
        )


def infer_split_from_path(path: Path) -> str:
    text = str(path).lower()
    for split in ("train", "valid", "validation", "test"):
        if split in text:
            return "valid" if split == "validation" else split
    return "unknown"


def infer_source_url(raw: dict[str, Any]) -> str | None:
    source = str(raw.get("source") or raw.get("platform") or "").lower()
    contest_id = raw.get("cf_contest_id")
    index = raw.get("cf_index")
    if "codeforces" in source and contest_id and index:
        return f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    if "codeforces" in source:
        for key in ("name", "title", "source_problem_id", "id"):
            value = str(raw.get(key) or "")
            match = re.search(r"\b(\d{3,6})[\s_.-]+([A-Za-z][A-Za-z0-9]?)\b", value)
            if match:
                return (
                    "https://codeforces.com/problemset/problem/"
                    f"{match.group(1)}/{match.group(2).upper()}"
                )
    return None


def decode_codecontests_tests(messages: list[Any]) -> dict[str, list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    for message in messages:
        if not isinstance(message, bytes):
            continue
        try:
            fields = parse_message(message)
        except Exception:
            continue
        case_input = first_text(fields, 1)
        case_output = first_text(fields, 2)
        if case_input is not None and case_output is not None:
            inputs.append(case_input)
            outputs.append(case_output)
    return {"inputs": inputs, "outputs": outputs}


def decode_codecontests_solutions(messages: list[Any]) -> dict[str, list[Any]]:
    languages: list[Any] = []
    solutions: list[str] = []
    for message in messages:
        if not isinstance(message, bytes):
            continue
        try:
            fields = parse_message(message)
        except Exception:
            continue
        language = first_int(fields, 1)
        solution = first_text(fields, 2)
        if solution:
            languages.append(language)
            solutions.append(solution)
    return {"language": languages, "solution": solutions}


def codecontests_source_name(value: int | None) -> str | None:
    mapping = {
        0: "UNKNOWN_SOURCE",
        1: "CODECHEF",
        2: "CODEFORCES",
        3: "HACKEREARTH",
        4: "CODEJAM",
        6: "ATCODER",
        7: "AIZU",
    }
    return mapping.get(value, None if value is None else f"source_{value}")


def codecontests_difficulty_name(value: int | None) -> str | None:
    mapping = {
        0: "unknown",
        1: "easy",
        2: "medium",
        3: "hard",
        4: "harder",
        5: "hardest",
        6: "external",
        7: "A",
        8: "B",
        9: "C",
        10: "D",
        11: "E",
        12: "F",
        13: "G",
        14: "H",
        15: "I",
        16: "J",
        17: "K",
        19: "L",
        20: "M",
        21: "N",
        22: "O",
        23: "P",
        24: "Q",
        25: "R",
        26: "S",
        27: "T",
        28: "U",
        29: "V",
    }
    return mapping.get(value, None if value is None else f"difficulty_{value}")
