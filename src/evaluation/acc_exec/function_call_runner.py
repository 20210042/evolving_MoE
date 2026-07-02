from __future__ import annotations

import json
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .base import ExecutionRunner, unsupported_result
from .comparator import compare_outputs
from .sandbox import run_python_file


def _decode(v: Any) -> Any:
    """Dataset stores every leaf scalar JSON-string-encoded (e.g. "9"->9, '"aa"'->"aa");
    genuine strings that are not valid JSON (e.g. "EBG13 rknzcyr.") are left as-is.
    Recurse through lists/dicts."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    if isinstance(v, list):
        return [_decode(x) for x in v]
    if isinstance(v, dict):
        return {k: _decode(x) for k, x in v.items()}
    return v


def _unwrap(v: Any) -> Any:
    """expected_output.value wraps the single return value in a 1-element list."""
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


# LeetCode-style `class Solution` snippets (~27% of function_call refs) use List/Optional/
# collections/etc. WITHOUT importing them → NameError at import. Inject a common header.
_IMPORT_HEADER = (
    "from typing import List, Optional, Dict, Tuple, Set, Any, Union\n"
    "import collections, math, bisect, heapq, itertools, functools, re, string\n"
    "from collections import defaultdict, Counter, deque, OrderedDict\n"
    "from functools import lru_cache, reduce\n"
)


class FunctionCallRunner(ExecutionRunner):
    def can_run(self, problem: dict[str, Any]) -> bool:
        spec = problem.get("eval_spec") or {}
        return spec.get("eval_mode") == "function_call" and bool(spec.get("entry_point"))

    def run(
        self,
        problem: dict[str, Any],
        candidate_code: str,
        language: str = "python",
        solution_id: str = "candidate",
    ) -> dict[str, Any]:
        if language != "python":
            return unsupported_result(problem, solution_id, "unsupported_language")
        if not self.can_run(problem):
            return unsupported_result(problem, solution_id)
        eval_spec = problem.get("eval_spec") or {}
        entry_point = eval_spec.get("entry_point")
        comparison = eval_spec.get("comparison") or {}
        passed = 0
        failed_tests: list[dict[str, Any]] = []
        total_runtime = 0.0
        stderr_parts: list[str] = []
        status = "accepted"
        with TemporaryDirectory(prefix="acc_eval_fc_") as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "candidate.py").write_text(_IMPORT_HEADER + candidate_code, encoding="utf-8")
            for case in problem.get("test_cases") or []:
                payload = (case.get("input") or {}).get("value") or {}
                args = _decode(payload.get("args", [])) if isinstance(payload, dict) else []
                kwargs = _decode(payload.get("kwargs", {})) if isinstance(payload, dict) else {}
                wrapper = make_wrapper(entry_point, args, kwargs)
                wrapper_path = tmp_path / "runner.py"
                wrapper_path.write_text(wrapper, encoding="utf-8")
                result = run_python_file(
                    wrapper_path,
                    timeout_seconds=float(eval_spec.get("timeout_seconds", 5)),
                    memory_limit_mb=eval_spec.get("memory_limit_mb", 512),
                    cwd=tmp_path,
                )
                total_runtime += result.runtime_seconds
                if result.stderr:
                    stderr_parts.append(result.stderr)
                if result.timed_out:
                    status = "timeout"
                    failed_tests.append(failed_case(case, {"kind": "return_value", "value": None}, "timeout"))
                    continue
                if result.returncode != 0:
                    status = "runtime_error"
                    failed_tests.append(
                        failed_case(case, {"kind": "return_value", "value": None}, result.stderr)
                    )
                    continue
                try:
                    actual = json.loads(result.stdout)
                except json.JSONDecodeError:
                    actual = result.stdout
                expected = _unwrap(_decode((case.get("expected_output") or {}).get("value")))
                if actual == expected or compare_outputs(actual, expected, comparison):
                    passed += 1
                else:
                    if status == "accepted":
                        status = "wrong_answer"
                    failed_tests.append(failed_case(case, {"kind": "return_value", "value": actual}, None))
        total = len(problem.get("test_cases") or [])
        return {
            "problem_id": problem.get("problem_id"),
            "solution_id": solution_id,
            "passed": passed == total and total > 0,
            "status": "accepted" if passed == total and total > 0 else status,
            "num_tests_passed": passed,
            "num_tests_total": total,
            "failed_tests": failed_tests,
            "runtime_seconds": total_runtime,
            "memory_mb": None,
            "stderr": "\n".join(stderr_parts) if stderr_parts else None,
        }


def make_wrapper(entry_point: str, args: Any, kwargs: Any) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False)
    return textwrap.dedent(
        f"""
        import json
        import candidate

        payload = json.loads({payload!r})
        if hasattr(candidate, {entry_point!r}):
            func = getattr(candidate, {entry_point!r})       # module-level function
        else:
            func = getattr(candidate.Solution(), {entry_point!r})  # class Solution method (~27%)
        result = func(*payload.get("args", []), **payload.get("kwargs", {{}}))
        print(json.dumps(result, ensure_ascii=False))
        """
    )


def failed_case(case: dict[str, Any], actual: dict[str, Any], error: str | None) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "input": case.get("input"),
        "expected_output": case.get("expected_output"),
        "actual_output": actual,
        "error_message": error,
    }
