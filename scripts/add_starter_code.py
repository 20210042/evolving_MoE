"""Add a starter_code stub to function_call / gfg records so the model writes the
right entry_point signature (our dump dropped TACO's starter_code → getattr fails).

gfg: from stored entry_point + gfg_params. function_call: extract the `def <ep>(...)`
signature from the original reference solution (joined by id).

Usage: python scripts/add_starter_code.py <acc_train.jsonl> <original.jsonl>
"""
import sys, json, re


def J(x):
    return json.loads(x) if isinstance(x, str) else x


def main():
    acc_path, orig_path = sys.argv[1], sys.argv[2]

    # id -> (ref_code) for function_call signature extraction
    ref_by_id = {}
    with open(orig_path) as fh:
        for l in fh:
            r = json.loads(l)
            es = J(r.get("eval_spec")) or {}
            if es.get("eval_mode") != "function_call":
                continue
            refs = [s for s in (J(r.get("reference_solutions")) or [])
                    if isinstance(s, dict) and s.get("is_known_correct") and s.get("language") == "python"]
            if refs:
                ref_by_id[r.get("problem_id") or r.get("id")] = refs[0]["code"]

    rows = [json.loads(l) for l in open(acc_path)]
    n_fc = n_gfg = n_miss = 0
    for r in rows:
        es = r["eval_spec"]
        mode = es.get("eval_mode")
        ep = es.get("entry_point")
        if mode == "gfg_function":
            params = ", ".join(["self"] + (es.get("gfg_params") or []))
            r["starter_code"] = f"class Solution:\n\tdef {ep}({params}):\n\t\t# complete this function\n\t\t"
            n_gfg += 1
        elif mode == "function_call":
            code = ref_by_id.get(r.get("problem_id") or r.get("id"), "")
            m = re.search(rf"def\s+{re.escape(ep or '')}\s*\(([^)]*)\)", code)
            if not m:
                n_miss += 1
                continue
            sig = m.group(1)
            if "class Solution" in code:
                r["starter_code"] = f"class Solution:\n\tdef {ep}({sig}):\n\t\t# complete this function\n\t\t"
            else:
                r["starter_code"] = f"def {ep}({sig}):\n\t# complete this function\n\t"
            n_fc += 1

    with open(acc_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"starter_code 추가: function_call {n_fc}, gfg {n_gfg}, function_call 시그니처 못찾음 {n_miss}")


if __name__ == "__main__":
    main()
