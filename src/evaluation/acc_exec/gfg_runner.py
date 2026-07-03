"""Runner for reverse-engineered geeksforgeeks 'complete the function' problems.

gfg problems have example-text inputs (`param = value`) and no driver. During dataset
prep (scripts/gfg_recover.py) we recover the ones whose known-correct ref passes, tagging
them eval_mode="gfg_function" with entry_point + gfg_params (the method signature). This
runner replays that driver on a *candidate* (model) solution: parse the pseudo-stdin ->
args in signature order -> call Solution().method -> format output -> token-compare.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .base import ExecutionRunner, unsupported_result
from .sandbox import run_python_file

_IMPORT_HEADER = (
    "from typing import List, Optional, Dict, Tuple, Set, Any, Union\n"
    "import collections, math, bisect, heapq, itertools, functools, re, string\n"
    "from collections import defaultdict, Counter, deque, OrderedDict\n"
    "from functools import lru_cache, reduce\n"
)
_COUNT_NAMES = {"n", "m", "size", "len", "length"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_value(raw: str) -> Any:
    s = raw.strip().rstrip(".").strip().replace("{", "[").replace("}", "]")
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return raw.strip()


def parse_pseudo_stdin(text: str) -> dict:
    text = text.replace("\r", "")
    chunks = []
    for line in text.split("\n"):
        depth = last = 0
        for i, ch in enumerate(line):
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            elif ch == "," and depth == 0:
                chunks.append(line[last:i]); last = i + 1
        chunks.append(line[last:])
    out = {}
    for c in chunks:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*((?:\[\s*\])*)\s*=\s*(.*)", c, re.S)
        if m:
            out[_norm(m.group(1))] = parse_value(m.group(3))
    return out


def build_args(params, kv):
    matched = {p: kv[_norm(p)] for p in params if _norm(p) in kv}
    iterables = [v for v in matched.values() if isinstance(v, (list, str, tuple))]
    args = []
    for p in params:
        if p in matched:
            args.append(matched[p])
        elif _norm(p) in _COUNT_NAMES and len(iterables) == 1:
            args.append(len(iterables[0]))
        else:
            return None
    return args


_WRAP = """
{header}
import json, sys
{code}

payload = json.loads({payload!r})
if "Solution" in dir():
    fn = getattr(Solution(), {method!r})
else:
    fn = globals()[{method!r}]
ret = fn(*payload["args"])
if isinstance(ret, bool):
    print(1 if ret else 0)
elif isinstance(ret, (list, tuple)):
    print(" ".join(map(str, ret)))
else:
    print(ret)
"""


class GfgRunner(ExecutionRunner):
    def can_run(self, problem: dict[str, Any]) -> bool:
        spec = problem.get("eval_spec") or {}
        return spec.get("eval_mode") == "gfg_function" and bool(spec.get("entry_point"))

    def run(self, problem, candidate_code, language="python", solution_id="candidate"):
        if language != "python":
            return unsupported_result(problem, solution_id, "unsupported_language")
        if not self.can_run(problem):
            return unsupported_result(problem, solution_id)
        spec = problem["eval_spec"]
        method = spec["entry_point"]
        params = spec.get("gfg_params") or []
        passed = 0
        status = "accepted"
        stderr_parts = []
        cases = problem.get("test_cases") or []
        with TemporaryDirectory(prefix="gfg_") as tmp:
            for case in cases:
                kv = parse_pseudo_stdin(str((case.get("input") or {}).get("value") or ""))
                args = build_args(params, kv)
                if args is None:
                    status = "runtime_error"
                    break
                wp = Path(tmp) / "run.py"
                wp.write_text(_WRAP.format(header=_IMPORT_HEADER, code=candidate_code,
                                           payload=json.dumps({"args": args}), method=method),
                              encoding="utf-8")
                r = run_python_file(wp, timeout_seconds=float(spec.get("timeout_seconds", 5)),
                                    memory_limit_mb=spec.get("memory_limit_mb", 512), cwd=Path(tmp))
                if r.stderr:
                    stderr_parts.append(r.stderr)
                if r.timed_out:
                    status = "timeout"; break
                if r.returncode != 0:
                    status = "runtime_error"; break
                exp = (case.get("expected_output") or {}).get("value")
                if r.stdout.split() == str(exp).split():
                    passed += 1
                else:
                    status = "wrong_answer"; break
        total = len(cases)
        return {
            "problem_id": problem.get("problem_id"),
            "solution_id": solution_id,
            "passed": passed == total and total > 0,
            "status": "accepted" if passed == total and total > 0 else status,
            "num_tests_passed": passed,
            "num_tests_total": total,
            "stderr": "\n".join(stderr_parts) if stderr_parts else None,
        }
