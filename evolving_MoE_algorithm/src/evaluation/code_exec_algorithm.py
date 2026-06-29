"""Code execution helpers for ACC algorithm evaluation.

Extends the original code-execution utilities with stdin/stdout judging for normalized ACC test
cases. It extracts generated code from model text, runs submissions in temporary files with timeouts,
normalizes stdout according to `eval_spec`, and returns pass/fail scores on the existing 0/100 scale."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from io import StringIO
from typing import Any

from utils.helpers import extract_code_block


def clean_extracted_code(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    code = extract_code_block(text)
    if not code:
        code = (
            text.replace("```python\n", "")
            .replace("```python", "")
            .replace("```\n", "")
            .replace("```", "")
        )
    for marker in ["END SOLUTION", "### SUCCESS", "### END"]:
        if marker in code:
            code = code.split(marker)[0]
    return code.strip()


def extract_helper_code(ground_truth: str | None) -> str:
    if not ground_truth:
        return ""
    lines = ground_truth.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    helper_lines = []
    in_class = False
    class_indent = 0
    for line in lines:
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if stripped.startswith("import ") or stripped.startswith("from "):
            helper_lines.append(line)
            continue
        if stripped.startswith("class "):
            in_class = True
            class_indent = current_indent
            helper_lines.append(line)
            continue
        if in_class:
            if stripped and current_indent <= class_indent:
                in_class = False
            else:
                helper_lines.append(line)
    return "\n".join(helper_lines)


def unsafe_execute(code, test_code, entry_point, result_queue, expected_name=None):
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        try:
            local_env = {}
            try:
                exec(code, globals(), local_env)
            except Exception:
                result_queue.put(0.0)
                return
            with tempfile.TemporaryDirectory() as tmpdir:
                original_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    if "def check(" in test_code and entry_point:
                        if entry_point not in local_env:
                            found = False
                            for key in local_env:
                                if callable(local_env[key]) and key != "check":
                                    local_env[entry_point] = local_env[key]
                                    found = True
                                    break
                            if not found:
                                result_queue.put(0.0)
                                return
                        exec(test_code, globals(), local_env)
                        exec(f"check({entry_point})", globals(), local_env)
                    else:
                        if expected_name and expected_name not in local_env:
                            for key in list(local_env.keys()):
                                if callable(local_env[key]):
                                    local_env[expected_name] = local_env[key]
                                    break
                        exec(test_code, globals(), local_env)
                    result_queue.put(100.0)
                finally:
                    os.chdir(original_cwd)
        except Exception:
            result_queue.put(0.0)


def evaluate_code_score(
    prediction: str,
    test_code: str,
    entry_point: str | None = None,
    expected_name: str | None = None,
    helper_code: str | None = None,
    *,
    timeout: float = 3.0,
) -> float:
    code = clean_extracted_code(prediction)
    if helper_code:
        code = helper_code + "\n\n" + code
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=unsafe_execute,
        args=(code, test_code, entry_point, queue, expected_name),
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return 0.0
    if not queue.empty():
        return queue.get()
    return 0.0


def case_value(field: Any) -> str:
    if isinstance(field, dict):
        return str(field.get("value", ""))
    return str(field if field is not None else "")


def normalize_output(value: str, comparison: dict[str, Any]) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if comparison.get("strip_trailing_whitespace", True):
        value = "\n".join(line.rstrip() for line in value.split("\n")).rstrip()
    if not comparison.get("case_sensitive", True):
        value = value.lower()
    return value


def outputs_match(actual: str, expected: str, eval_spec: dict[str, Any]) -> bool:
    comparison = eval_spec.get("comparison") or {}
    comparison_type = comparison.get("type", "exact_or_token_match")
    actual_norm = normalize_output(actual, comparison)
    expected_norm = normalize_output(expected, comparison)
    if actual_norm == expected_norm:
        return True
    if comparison_type == "exact_or_token_match":
        return actual_norm.split() == expected_norm.split()
    return False


def evaluate_stdin_stdout_score(
    prediction: str,
    test_cases: list[dict[str, Any]],
    eval_spec: dict[str, Any] | None = None,
    *,
    timeout: float = 3.0,
) -> float:
    code = clean_extracted_code(prediction)
    if not code or not test_cases:
        return 0.0
    eval_spec = eval_spec or {}
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "submission.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        for case in test_cases:
            stdin = case_value(case.get("input"))
            expected = case_value(case.get("expected_output"))
            try:
                completed = subprocess.run(
                    [sys.executable, code_path],
                    input=stdin,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return 0.0
            if completed.returncode != 0:
                return 0.0
            if not outputs_match(completed.stdout, expected, eval_spec):
                return 0.0
    return 100.0
