"""Shared loader infrastructure for all benchmark sources.

Provides stable ID generation, safe source identifiers, test-case construction, reference-solution
packing, native metadata defaults, and generic JSON/JSONL traversal utilities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from acc_data_pipeline.schemas.problem import (
    apply_problem_defaults,
    default_dedup,
    default_eval_spec,
    default_native_metadata,
    default_quality_flags,
    dump_problem,
)
from acc_data_pipeline.utils.io import compact_value, load_json_records, parse_maybe_json, read_json


SOURCE_KEYS = {
    "APPS": "apps",
    "CodeContests": "codecontests",
    "TACO": "taco",
    "LiveCodeBench": "livecodebench",
}


def safe_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        text = "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_problem_id(source: str, split: str, source_problem_id: str) -> str:
    return f"{SOURCE_KEYS[source]}__{safe_id(split)}__{safe_id(source_problem_id)}"


def resolve_dataset_dir(raw_root: str | Path, candidates: Iterable[str]) -> Path | None:
    root = Path(raw_root)
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def as_list(value: Any) -> list[Any]:
    value = parse_maybe_json(value, default=[])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_language(value: Any) -> str:
    if value is None:
        return "python"
    if isinstance(value, int):
        # DeepMind CodeContests stores language as numeric enums. Python is
        # commonly 3, but keep unknown values traceable instead of guessing.
        return "python" if value == 3 else f"language_{value}"
    text = str(value).strip().lower()
    if text in {"py", "python3", "python 3"}:
        return "python"
    return text or "python"


def io_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def function_args(value: Any) -> dict[str, Any]:
    value = parse_maybe_json(value, default=value)
    if isinstance(value, dict) and ("args" in value or "kwargs" in value):
        return {"args": value.get("args", []), "kwargs": value.get("kwargs", {})}
    if isinstance(value, list):
        return {"args": value, "kwargs": {}}
    return {"args": [value], "kwargs": {}}


def extract_examples_from_statement(statement: str) -> list[dict[str, Any]]:
    if not statement:
        return []
    statement = normalize_sample_labels(statement)
    examples: list[dict[str, Any]] = []
    sample_pattern = re.compile(
        r"(?:^|\n)\s*sample\s+input\s*\d*\s*:?\s*\n"
        r"(?P<input>.*?)"
        r"\n\s*sample\s+output\s*\d*\s*:?\s*\n"
        r"(?P<output>.*?)(?=\n\s*(?:[-#*_\s]*)(?:sample\s+input|note|notes|explanation)\b|\Z)",
        flags=re.I | re.S,
    )
    for match in sample_pattern.finditer(statement):
        add_example(examples, match.group("input"), match.group("output"))
    if examples:
        return examples[:10]

    example_section_pattern = re.compile(
        r"(?:^|\n)\s*[-#*_\s]*examples?(?:\s+\d+)?\s*:?\s*[-#*_\s]*\n"
        r"(?P<body>.*?)(?=\n\s*[-#*_\s]*(?:note|notes|constraints)\b|\Z)",
        flags=re.I | re.S,
    )
    io_pattern = re.compile(
        r"(?:^|\n)\s*input\s*:?\s*\n"
        r"(?P<input>.*?)"
        r"\n\s*output\s*:?\s*\n"
        r"(?P<output>.*?)(?=\n\s*input\s*:?\s*\n|\Z)",
        flags=re.I | re.S,
    )
    inline_io_pattern = re.compile(
        r"(?:^|\n)\s*input\s*:?\s*(?P<input>.+?)"
        r"\s+output\s*:?\s*(?P<output>.*?)(?=\n\s*(?:input|example|explanation|note)\b|\Z)",
        flags=re.I | re.S,
    )
    for section in example_section_pattern.finditer(statement):
        body = section.group("body")
        found = False
        for match in io_pattern.finditer(body):
            add_example(examples, match.group("input"), match.group("output"))
            found = True
        if not found:
            for match in inline_io_pattern.finditer(body):
                add_example(examples, match.group("input"), match.group("output"))
    return examples[:10]


def add_example(examples: list[dict[str, Any]], input_text: str, output_text: str) -> None:
    input_text = strip_example_text(input_text)
    output_text = strip_example_text(output_text)
    if input_text or output_text:
        examples.append({"input": input_text, "output": output_text, "explanation": None})


def strip_example_text(text: str) -> str:
    return re.sub(
        r"\n\s*[-#*_\s]*(?:note|notes|explanation)\b.*\Z",
        "",
        text.strip(),
        flags=re.I | re.S,
    )


def normalize_sample_labels(statement: str) -> str:
    value = statement.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"(?i)(?<!\n)(sample\s+input\s*\d*\s*:?)", r"\n\1", value)
    value = re.sub(r"(?i)(?<!\n)(sample\s+output\s*\d*\s*:?)", r"\n\1", value)
    return value


def split_statement_sections(statement: str) -> dict[str, Any]:
    value = normalize_sample_labels(statement or "")
    buckets: dict[str, list[str]] = {
        "description": [],
        "input_format": [],
        "output_format": [],
        "constraints": [],
        "notes": [],
    }
    current: str | None = "description"
    in_examples = False
    for line in value.splitlines():
        heading = statement_heading(line)
        if heading in {"example", "examples"} or heading.startswith("sample input"):
            in_examples = True
            current = None
            continue
        if in_examples:
            if heading in {"note", "notes", "explanation"}:
                in_examples = False
                current = "notes"
                continue
            if heading == "constraints":
                in_examples = False
                current = "constraints"
                continue
            else:
                continue
        elif heading == "input":
            current = "input_format"
            continue
        elif heading == "output":
            current = "output_format"
            continue
        elif heading == "constraints":
            current = "constraints"
            continue
        elif heading in {"note", "notes", "explanation"}:
            current = "notes"
            continue
        if current:
            buckets[current].append(line)
    return {
        "description": clean_statement_section("\n".join(buckets["description"])),
        "input_format": clean_statement_section("\n".join(buckets["input_format"])),
        "output_format": clean_statement_section("\n".join(buckets["output_format"])),
        "constraints": clean_statement_section("\n".join(buckets["constraints"]))
        or infer_constraints_from_text(
            "\n".join(buckets["input_format"]),
            "\n".join(buckets["output_format"]),
        ),
        "notes": clean_statement_section("\n".join(buckets["notes"])),
        "examples": extract_examples_from_statement(value),
    }


def statement_heading(line: str) -> str:
    text = re.sub(r"<[^>]+>", " ", line.strip())
    text = text.strip().strip("-#*_").strip()
    text = text.strip("：")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(":").lower()
    if text in {"input format", "input description", "input specification"}:
        return "input"
    if text in {"output format", "output description", "output specification"}:
        return "output"
    if re.fullmatch(r"sample input\s*\d*", text):
        return "sample input"
    if re.fullmatch(r"sample output\s*\d*", text):
        return "sample output"
    if re.fullmatch(r"examples?\s*\d*", text):
        return "example" if text.startswith("example") and not text.startswith("examples") else "examples"
    return text


def clean_statement_section(text: str) -> str | None:
    text = text.strip()
    return text or None


def infer_constraints_from_text(*sections: str) -> str | None:
    pieces: list[str] = []
    for section in sections:
        if not section:
            continue
        for match in re.finditer(r"\([^()\n]{0,240}\)", section):
            value = match.group(0).strip()
            if looks_like_constraint(value):
                pieces.append(value)
        for line in section.splitlines():
            value = line.strip()
            if not value:
                continue
            lowered = value.lower()
            if "guaranteed" in lowered or (looks_like_constraint(value) and "(" not in value):
                pieces.append(value)
    unique: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        compact = re.sub(r"\s+", " ", piece).strip()
        if compact and compact not in seen:
            unique.append(compact)
            seen.add(compact)
    return "\n".join(unique) if unique else None


def looks_like_constraint(text: str) -> bool:
    return bool(
        re.search(
            r"(?:≤|≥|<=|>=|\\leq?|\\geq?|\b\d+\s*[<≤]\s*[A-Za-z_]|[A-Za-z_]\w*\s*[<≤]\s*\d)",
            text,
        )
    )


def make_test_cases(
    problem_id: str,
    inputs: Iterable[Any],
    outputs: Iterable[Any],
    visibility: str = "unknown",
    eval_mode: str = "stdin_stdout",
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (case_input, case_output) in enumerate(zip(list(inputs), list(outputs))):
        if eval_mode == "function_call":
            input_value = {"kind": "function_args", "value": function_args(case_input)}
            output_value = {"kind": "return_value", "value": parse_maybe_json(case_output, case_output)}
        else:
            input_value = {"kind": "stdin", "value": io_to_string(case_input)}
            output_value = {"kind": "stdout", "value": io_to_string(case_output)}
        cases.append(
            {
                "case_id": f"{problem_id}__{visibility}__{index:04d}",
                "visibility": visibility,
                "input": input_value,
                "expected_output": output_value,
                "metadata": metadata or {},
            }
        )
    return cases


def tests_from_field(
    problem_id: str,
    field: Any,
    visibility: str,
    eval_mode: str,
) -> list[dict[str, Any]]:
    data = parse_maybe_json(field, default=None)
    if not data:
        return []
    if isinstance(data, dict):
        inputs = data.get("inputs", data.get("input", []))
        outputs = data.get("outputs", data.get("output", data.get("expected_output", [])))
        if isinstance(inputs, list) and isinstance(outputs, list):
            return make_test_cases(problem_id, inputs, outputs, visibility, eval_mode)
        if "input" in data and ("output" in data or "expected_output" in data):
            return make_test_cases(
                problem_id,
                [data.get("input")],
                [data.get("output", data.get("expected_output"))],
                visibility,
                eval_mode,
            )
    if isinstance(data, list):
        inputs: list[Any] = []
        outputs: list[Any] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "input" in item and ("output" in item or "expected_output" in item):
                inputs.append(item.get("input"))
                outputs.append(item.get("output", item.get("expected_output")))
        return make_test_cases(problem_id, inputs, outputs, visibility, eval_mode)
    return []


def build_reference_solutions(
    problem_id: str,
    solutions: Any,
    default_language: str = "python",
    source: str = "benchmark",
) -> list[dict[str, Any]]:
    parsed = parse_maybe_json(solutions, default=[])
    refs: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        code_values = parsed.get("solution") or parsed.get("solutions") or parsed.get("code") or []
        language_values = parsed.get("language") or parsed.get("languages") or []
        if isinstance(code_values, str):
            code_values = [code_values]
        if not isinstance(language_values, list):
            language_values = [language_values] * len(code_values)
        for index, code in enumerate(code_values):
            if not isinstance(code, str) or not code.strip():
                continue
            language = language_values[index] if index < len(language_values) else default_language
            refs.append(
                {
                    "solution_id": f"{problem_id}__ref__{len(refs):04d}",
                    "language": normalize_language(language),
                    "code": code,
                    "is_known_correct": True,
                    "source": source,
                    "metadata": {},
                }
            )
        return refs
    for item in as_list(parsed):
        if isinstance(item, str):
            code = item
            language = default_language
            metadata: dict[str, Any] = {}
        elif isinstance(item, dict):
            code = item.get("code") or item.get("solution") or item.get("source_code") or ""
            language = normalize_language(item.get("language", default_language))
            metadata = compact_value({k: v for k, v in item.items() if k not in {"code", "solution"}})
        else:
            continue
        if not isinstance(code, str) or not code.strip():
            continue
        refs.append(
            {
                "solution_id": f"{problem_id}__ref__{len(refs):04d}",
                "language": language,
                "code": code,
                "is_known_correct": True,
                "source": source,
                "metadata": metadata,
            }
        )
    return refs


def build_problem_record(
    *,
    source: str,
    split: str,
    source_problem_id: str,
    raw_statement: str,
    task_family: str = "algorithmic_code_generation",
    title: str | None = None,
    description: str | None = None,
    input_format: str | None = None,
    output_format: str | None = None,
    constraints: str | None = None,
    notes: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    test_cases: list[dict[str, Any]] | None = None,
    reference_solutions: list[dict[str, Any]] | None = None,
    starter_code: str | None = None,
    eval_mode: str = "stdin_stdout",
    entry_point: str | None = None,
    source_url: str | None = None,
    source_platform: str | None = None,
    difficulty: str | None = None,
    tags: list[str] | None = None,
    original_task_type: str | None = None,
    raw_fields: dict[str, Any] | None = None,
    timeout_seconds: float = 5,
    memory_limit_mb: int | None = 512,
) -> dict[str, Any]:
    problem_id = make_problem_id(source, split, source_problem_id)
    statement_parts = split_statement_sections(raw_statement or "")
    description = description if description is not None else statement_parts["description"]
    input_format = input_format if input_format is not None else statement_parts["input_format"]
    output_format = output_format if output_format is not None else statement_parts["output_format"]
    constraints = constraints if constraints is not None else statement_parts["constraints"]
    notes = notes if notes is not None else statement_parts["notes"]
    examples = examples if examples is not None else statement_parts["examples"]
    references = reference_solutions or []
    tests = test_cases or []
    eval_spec = default_eval_spec(eval_mode)
    eval_spec["entry_point"] = entry_point
    eval_spec["timeout_seconds"] = timeout_seconds
    eval_spec["memory_limit_mb"] = memory_limit_mb
    eval_spec["requires_special_judge"] = eval_mode == "special_judge"
    native = default_native_metadata()
    native.update(
        {
            "difficulty": difficulty,
            "tags": sorted(set(tags or [])),
            "original_task_type": original_task_type,
            "raw_fields": compact_value(raw_fields or {}),
        }
    )
    flags = default_quality_flags()
    flags.update(
        {
            "has_statement": bool(str(raw_statement or "").strip()),
            "has_tests": bool(tests),
            "has_reference_solution": bool(references),
            "has_native_tags": bool(tags),
            "is_algorithmic": task_family in {"algorithmic_code_generation", "self_repair"},
            "is_supported_for_execution": eval_mode not in {"unsupported", "special_judge"},
            "requires_manual_review": eval_mode in {"unsupported", "special_judge"},
            "warnings": ["special_judge_not_implemented"] if eval_mode == "special_judge" else [],
        }
    )
    record = {
        "schema_version": "0.1",
        "problem_id": problem_id,
        "source": source,
        "source_problem_id": safe_id(source_problem_id),
        "source_url": source_url,
        "source_platform": source_platform,
        "split": safe_id(split),
        "task_family": task_family,
        "title": title,
        "problem_statement": {
            "raw": raw_statement or "",
            "description": description if description is not None else raw_statement or "",
            "input_format": input_format,
            "output_format": output_format,
            "constraints": constraints,
            "notes": notes,
        },
        "examples": examples,
        "test_cases": tests,
        "reference_solutions": references,
        "starter_code": starter_code,
        "eval_spec": eval_spec,
        "native_metadata": native,
        "dedup": default_dedup(),
        "quality_flags": flags,
    }
    return dump_problem(apply_problem_defaults(record))


class BaseLoader:
    source: str = "unknown"
    dataset_candidates: tuple[str, ...] = ()

    def __init__(self, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)
        self.warnings: list[str] = []
        self.schema_errors: list[dict[str, Any]] = []

    def dataset_dir(self) -> Path | None:
        return resolve_dataset_dir(self.raw_root, self.dataset_candidates)

    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def iter_json_records(self, root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
        for path in sorted(root.rglob("*")):
            if ".git" in path.parts or path.suffix not in {".json", ".jsonl"}:
                continue
            try:
                for record in load_json_records(path):
                    yield path, record
            except Exception as exc:
                self.warnings.append(f"failed_to_parse_json:{path}:{exc}")

    def record_error(self, source_problem_id: str, error: Exception) -> None:
        self.schema_errors.append(
            {
                "source": self.source,
                "source_problem_id": source_problem_id,
                "error": str(error),
            }
        )

    @staticmethod
    def read_json(path: Path, default: Any = None) -> Any:
        return read_json(path, default)
