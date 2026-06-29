"""APPS dataset loader.

Walks the APPS split/problem directory layout, reads question.txt, input_output.json, metadata, and
solutions, then emits normalized stdin/stdout problem records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from acc_data_pipeline.loaders.base import (
    BaseLoader,
    build_problem_record,
    build_reference_solutions,
    make_problem_id,
    make_test_cases,
)
from acc_data_pipeline.utils.io import read_json, read_text


class APPSLoader(BaseLoader):
    source = "APPS"
    dataset_candidates = ("APPS", "apps")

    def load(self) -> list[dict[str, Any]]:
        root = self.dataset_dir()
        if root is None:
            self.warnings.append(f"dataset_dir_not_found:{self.raw_root}/APPS")
            return []
        records: list[dict[str, Any]] = []
        for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            split = split_dir.name
            for problem_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                source_problem_id = problem_dir.name
                question_path = problem_dir / "question.txt"
                if not question_path.exists():
                    continue
                try:
                    record = self._load_problem(problem_dir, split, source_problem_id)
                    records.append(record)
                except Exception as exc:
                    self.record_error(source_problem_id, exc)
        return records

    def _load_problem(self, problem_dir: Path, split: str, source_problem_id: str) -> dict[str, Any]:
        statement = read_text(problem_dir / "question.txt")
        input_output = read_json(problem_dir / "input_output.json", default={}) or {}
        metadata = read_json(problem_dir / "metadata.json", default={}) or {}
        solutions = read_json(problem_dir / "solutions.json", default=[]) or []
        starter_path = problem_dir / "starter_code.py"
        starter_code = read_text(starter_path) if starter_path.exists() else None

        fn_name = input_output.get("fn_name")
        eval_mode = "function_call" if fn_name else "stdin_stdout"
        problem_id = make_problem_id(self.source, split, source_problem_id)
        tests = make_test_cases(
            problem_id,
            input_output.get("inputs", []),
            input_output.get("outputs", []),
            visibility="unknown",
            eval_mode=eval_mode,
            metadata={"source_file": "input_output.json"},
        )
        references = build_reference_solutions(problem_id, solutions)
        raw_fields = {
            "metadata": metadata,
            "input_output": {
                "fn_name": fn_name,
                "num_inputs": len(input_output.get("inputs", []) or []),
                "num_outputs": len(input_output.get("outputs", []) or []),
            },
            "problem_dir": str(problem_dir),
        }
        return build_problem_record(
            source=self.source,
            split=split,
            source_problem_id=source_problem_id,
            raw_statement=statement,
            test_cases=tests,
            reference_solutions=references,
            starter_code=starter_code,
            eval_mode=eval_mode,
            entry_point=fn_name,
            source_url=metadata.get("url"),
            source_platform="APPS",
            difficulty=metadata.get("difficulty"),
            raw_fields=raw_fields,
        )
