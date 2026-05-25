"""Sandbox code execution scoring (HumanEval / MBPP asserts)."""

from __future__ import annotations

import contextlib
import multiprocessing
from io import StringIO

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

            import os
            import tempfile

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
    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(code, test_code, entry_point, queue, expected_name),
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return 0.0

    if not queue.empty():
        return queue.get()
    return 0.0
