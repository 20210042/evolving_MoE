"""Full self-consistency census of the QuantCat/Algorithm-Dataset coding set.

For every problem, run its first known-correct python reference through the fixed
acc_exec runner (1 case, for speed — the dominant breakage modes fail on case 1)
and classify: PASS / py2_syntax / java_ref / tab_error / syntax_other /
runtime_error / wrong_answer / no_ref. Writes per-problem rows + prints an
aggregate broken-rate by (source_platform x eval_mode).

Usage: python scripts/audit_acc_self_consistency.py <train.jsonl> <out.jsonl> [max_workers]
"""
import sys, json, os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.acc_exec import ExecutionInterface  # noqa: E402


def J(x):
    return json.loads(x) if isinstance(x, str) else x


def classify(rec):
    es = J(rec.get("eval_spec")) or {}
    mode = es.get("eval_mode")
    refs = [s for s in (J(rec.get("reference_solutions")) or [])
            if isinstance(s, dict) and s.get("is_known_correct") and s.get("language") == "python"]
    sp = rec.get("source_platform") or rec.get("source") or "?"
    pid = rec.get("problem_id") or rec.get("id")
    base = {"id": pid, "source_platform": sp, "eval_mode": mode}
    if not refs:
        return {**base, "category": "no_ref"}
    cases = (J(rec.get("test_cases")) or [])[:1]
    if not cases:
        return {**base, "category": "no_case"}
    prob = {**rec, "eval_spec": es, "test_cases": cases}
    code = refs[0]["code"]
    try:
        out = ExecutionInterface().run(prob, code, solution_id="ref")
    except Exception as e:
        return {**base, "category": f"exc:{type(e).__name__}"}
    if out.get("passed"):
        return {**base, "category": "PASS"}
    st = out.get("status")
    err = out.get("stderr") or ""
    if st == "runtime_error":
        if "SyntaxError" in err and ("print " in code or "raw_input" in code):
            cat = "py2_syntax"
        elif "TabError" in err:
            cat = "tab_error"
        elif "import java" in code or "public class" in code:
            cat = "java_ref"
        elif "SyntaxError" in err:
            cat = "syntax_other"
        else:
            cat = "runtime_error"
    else:
        cat = st or "fail"
    return {**base, "category": cat}


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else min(16, os.cpu_count() or 4)
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    print(f"records={len(recs)} workers={workers}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(classify, r) for r in recs]
        for i, f in enumerate(as_completed(futs)):
            results.append(f.result())
            if (i + 1) % 2000 == 0:
                print(f"  ...{i+1}/{len(recs)}", flush=True)
    with open(outp, "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # aggregate
    by_sp_mode = defaultdict(Counter)
    cat_tot = Counter()
    for r in results:
        by_sp_mode[(r["source_platform"], r["eval_mode"])][r["category"]] += 1
        cat_tot[r["category"]] += 1
    total = len(results)
    passed = cat_tot.get("PASS", 0)
    print(f"\n=== CENSUS: {total} problems, self-consistent(PASS)={passed} ({100*passed/total:.1f}%) ===")
    print("category totals:", dict(cat_tot.most_common()))
    print("\n=== broken-rate by (source_platform, eval_mode), sorted by n ===")
    rows = sorted(by_sp_mode.items(), key=lambda kv: -sum(kv[1].values()))
    for (sp, mode), cc in rows:
        n = sum(cc.values()); brk = n - cc.get("PASS", 0)
        print(f"[{sp} / {mode}] n={n} broken={brk} ({100*brk/n:.0f}%)  {dict(cc)}")


if __name__ == "__main__":
    main()
