"""Build the corrected, self-consistent coding dataset from Minji/QuantCat (=TACO).

Keep only problems whose KNOWN-CORRECT reference passes its own tests under the FIXED
acc_exec runner (full cases). geeksforgeeks problems are recovered via the reverse-
engineered driver (scripts/gfg_recover) and re-tagged eval_mode="gfg_function" with
entry_point + gfg_params. Emits normalized records ready for evolution scoring
(score_one -> kind "acc").

Usage: python scripts/build_acc_selfconsistent.py <in.jsonl> <out.jsonl> [workers]
"""
import sys, os, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluation.acc_exec import ExecutionInterface  # noqa: E402
from gfg_recover import (J, extract_methods, pick_method, parse_pseudo_stdin,  # noqa: E402
                         build_args, try_recover_case)

KEEP = ["id", "problem_id", "instruction", "starter_code", "main_critic_category",
        "critic_categories", "source_platform", "source"]


def _ref_code(rec):
    refs = [s for s in (J(rec.get("reference_solutions")) or [])
            if isinstance(s, dict) and s.get("is_known_correct") and s.get("language") == "python"]
    return refs[0]["code"] if refs else None


def process(rec):
    es = J(rec.get("eval_spec")) or {}
    cases = J(rec.get("test_cases")) or []
    code = _ref_code(rec)
    if not code or not cases:
        return ("no_ref", None)
    base = {k: rec.get(k) for k in KEEP}
    base.update({"domain": "coding", "dataset": "acc", "scoring_kind": "acc"})
    is_gfg = "geeksforgeeks" in (rec.get("source_platform") or "").lower()

    if is_gfg and es.get("eval_mode") == "stdin_stdout":
        methods = extract_methods(code)
        keys = set(parse_pseudo_stdin(str((cases[0].get("input") or {}).get("value") or "")))
        picked = pick_method(methods, keys) if methods else None
        if not picked:
            return ("gfg_unrecovered", None)
        method, params = picked
        for c in cases:  # validate ref on ALL cases
            args = build_args(params, parse_pseudo_stdin(str((c.get("input") or {}).get("value") or "")))
            if args is None or try_recover_case(code, method, args,
                                                (c.get("expected_output") or {}).get("value")) != "PASS":
                return ("gfg_unrecovered", None)
        base["eval_spec"] = {**es, "eval_mode": "gfg_function",
                             "entry_point": method, "gfg_params": params}
        base["test_cases"] = cases
        return ("gfg_recovered", base)

    # non-gfg: run the ref through the fixed runner (all cases)
    prob = {"eval_spec": es, "test_cases": cases, "problem_id": rec.get("problem_id")}
    out = ExecutionInterface().run(prob, code, solution_id="ref")
    if not out.get("passed"):
        return (f"drop_{out.get('status', 'fail')}", None)
    base["eval_spec"] = es
    base["test_cases"] = cases
    return ("kept", base)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else min(16, os.cpu_count() or 4)
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    print(f"input={len(recs)} workers={workers}", flush=True)
    stats = Counter()
    n_out = 0
    with open(outp, "w") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (cat, rec) in enumerate(pool.map(process, recs, chunksize=8)):
            stats[cat] += 1
            if rec is not None:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_out += 1
            if (i + 1) % 3000 == 0:
                print(f"  ...{i+1}/{len(recs)} kept={n_out}", flush=True)
    print(f"\nDONE: emitted {n_out}/{len(recs)} self-consistent -> {outp}")
    print("breakdown:", dict(stats.most_common()))


if __name__ == "__main__":
    main()
