"""Reverse-engineer geeksforgeeks pseudo-stdin -> function call, gated by self-consistency.

gfg problems are "complete the function" (class Solution method) with example-text
inputs (`param = value`) and NO driver (broken in TACO itself). This synthesizes a
driver: parse the `param = value` pairs, match to the ref method's signature, call it,
format the return the way gfg would print it, and token-compare to the expected string.

SAFE: a problem is only "recovered" if the KNOWN-CORRECT reference now passes its own
tests — so a mis-parse can never create a false eval signal (it just stays dropped).

Usage: python scripts/gfg_recover.py <train.jsonl> [n_sample]   (n_sample=0 → all gfg)
"""
import sys, os, re, ast, json, textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.acc_exec.sandbox import run_python_file  # noqa: E402

_IMPORT_HEADER = (
    "from typing import List, Optional, Dict, Tuple, Set, Any, Union\n"
    "import collections, math, bisect, heapq, itertools, functools, re, string\n"
    "from collections import defaultdict, Counter, deque, OrderedDict\n"
    "from functools import lru_cache, reduce\n"
)


def J(x):
    return json.loads(x) if isinstance(x, str) else x


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_value(raw):
    """`{..}`->`[..]`, strip junk, literal_eval; fallback to raw string."""
    s = raw.strip().rstrip(".").strip()
    s = s.replace("{", "[").replace("}", "]")
    for parser in (ast.literal_eval,):
        try:
            return parser(s)
        except Exception:
            pass
    # bare int/float
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return raw.strip()  # bare string (unquoted)


def parse_pseudo_stdin(text):
    """Return {normalized_varname: value}. Splits on newlines and top-level commas."""
    text = text.replace("\r", "")
    # split on newline first
    chunks = []
    for line in text.split("\n"):
        # split top-level commas (not inside [] or {})
        depth = 0; last = 0
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
        m = re.match(r"\s*([A-Za-z_]\w*)\s*((?:\[\s*\])*)\s*=\s*(.*)", c, re.S)  # var / var[] / var[][]
        if m:
            out[_norm(m.group(1))] = parse_value(m.group(3))
    return out


_COUNT_NAMES = {"n", "m", "size", "len", "length"}


def build_args(params, kv):
    """Map pseudo-stdin dict -> positional args in signature order.
    Derives a missing length param (n/m/..) from the single iterable arg (gfg convention)."""
    matched = {p: kv[_norm(p)] for p in params if _norm(p) in kv}
    iterables = [v for v in matched.values() if isinstance(v, (list, str, tuple))]
    args = []
    for p in params:
        if p in matched:
            args.append(matched[p])
        elif _norm(p) in _COUNT_NAMES and len(iterables) == 1:
            args.append(len(iterables[0]))          # n = len(arr)
        else:
            return None
    return args


def extract_methods(code):
    """[(name, [param_names_excluding_self])] for each def in the ref."""
    methods = []
    for m in re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", code):
        name = m.group(1)
        params = [p.split(":")[0].split("=")[0].strip() for p in m.group(2).split(",")]
        params = [p for p in params if p and p != "self"]
        methods.append((name, params))
    return methods


def pick_method(methods, keys):
    """Method whose params best match the pseudo-stdin var names."""
    best, best_score = None, (-1, -1)
    for name, params in methods:
        if not params:
            continue
        score = sum(1 for p in params if _norm(p) in keys)
        # prefer full coverage, then more params
        score = (score, len(params)) if score == len(params) else (score, 0)
        if score > best_score:
            best, best_score = (name, params), score
    return best


_WRAP = """
{header}
import json, sys
{code}

payload = json.loads({payload!r})
args = payload["args"]
mod = sys.modules["__main__"]
if "Solution" in dir():
    inst = Solution()
    fn = getattr(inst, {method!r})
else:
    fn = globals()[{method!r}]
ret = fn(*args)
if isinstance(ret, bool):
    print(1 if ret else 0)
elif isinstance(ret, (list, tuple)):
    print(" ".join(map(str, ret)))
else:
    print(ret)
"""


def try_recover_case(code, method, args, expected, timeout=5):
    with TemporaryDirectory(prefix="gfg_") as tmp:
        wp = Path(tmp) / "run.py"
        wp.write_text(_WRAP.format(header=_IMPORT_HEADER, code=code,
                                   payload=json.dumps({"args": args}), method=method),
                      encoding="utf-8")
        r = run_python_file(wp, timeout_seconds=timeout, memory_limit_mb=512, cwd=Path(tmp))
        if r.timed_out:
            return "timeout"
        if r.returncode != 0:
            return "runtime_error"
        return "PASS" if r.stdout.split() == str(expected).split() else "wrong_answer"


def recover(rec):
    es = J(rec.get("eval_spec")) or {}
    refs = [s for s in (J(rec.get("reference_solutions")) or [])
            if isinstance(s, dict) and s.get("is_known_correct") and s.get("language") == "python"]
    cases = J(rec.get("test_cases")) or []
    if not refs or not cases:
        return "no_ref_or_case"
    code = refs[0]["code"]
    methods = extract_methods(code)
    if not methods:
        return "no_method"
    # use first case to pick the method
    first_keys = set(parse_pseudo_stdin(str((cases[0].get("input") or {}).get("value") or "")))
    picked = pick_method(methods, first_keys)
    if not picked:
        return "no_method_match"
    method, params = picked
    # validate ref on all cases
    for c in cases:
        kv = parse_pseudo_stdin(str((c.get("input") or {}).get("value") or ""))
        args = build_args(params, kv)
        if args is None:
            return "param_unmatched"
        exp = (c.get("expected_output") or {}).get("value")
        res = try_recover_case(code, method, args, exp)
        if res != "PASS":
            return res
    return "PASS"


def _one(r):
    return (r.get("problem_id") or r.get("id"), recover(r))


def main():
    from concurrent.futures import ProcessPoolExecutor
    inp = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_ids = sys.argv[3] if len(sys.argv) > 3 else None
    gfg = []
    with open(inp) as fh:
        for line in fh:
            r = json.loads(line)
            if "geeksforgeeks" in (r.get("source_platform") or "").lower() and \
               (J(r.get("eval_spec")) or {}).get("eval_mode") == "stdin_stdout":
                gfg.append(r)
    if n:
        import random; random.seed(0); random.shuffle(gfg); gfg = gfg[:n]
    res = Counter(); recovered = []
    workers = min(16, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pid, cat in pool.map(_one, gfg, chunksize=8):
            res[cat] += 1
            if cat == "PASS":
                recovered.append(pid)
    tot = sum(res.values()); pas = res.get("PASS", 0)
    print(f"gfg stdin recovered: {pas}/{tot} = {100*pas/tot:.0f}%")
    print("breakdown:", dict(res.most_common()))
    if out_ids:
        Path(out_ids).write_text("\n".join(recovered), encoding="utf-8")
        print("recovered ids ->", out_ids)


if __name__ == "__main__":
    main()
