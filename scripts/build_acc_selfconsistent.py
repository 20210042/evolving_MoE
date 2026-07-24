"""Build the corrected, self-consistent coding dataset from Minji/QuantCat (=TACO).

Keep only problems whose KNOWN-CORRECT reference passes its own tests under the FIXED
acc_exec runner (full cases). geeksforgeeks problems are recovered via the reverse-
engineered driver (scripts/gfg_recover) and re-tagged eval_mode="gfg_function" with
entry_point + gfg_params. Emits normalized records ready for evolution scoring
(score_one -> kind "acc").

레거시 사용: python scripts/build_acc_selfconsistent.py <in.jsonl> <out.jsonl> [workers]
  (플래그를 하나도 주지 않으면 기존 동작과 byte-identical: refs[0]만 시도, 솔루션 미보존)

SFT 자산용 사용:
  python scripts/build_acc_selfconsistent.py IN... -o out.jsonl \
      --dedupe-problem-id --ref-fallback --keep-solution

  --keep-solution      검증에 통과한 그 ref 코드를 solution/ground_truth로 보존한다.
                       이게 없으면 하위 SFT 빌더가 "참조 솔루션이 없다"고 오판하고
                       에이전트 생성코드를 타깃으로 승격하게 된다(실제로 그랬다).
  --ref-fallback       refs[0]이 실행검증에 실패하면 다음 known-correct ref로 넘어간다.
                       canonical은 어디까지나 refs[0] — 폴백은 커버리지 손실 방지용.
  --dedupe-problem-id  입력 여러 개(원본 train/validation/test)를 합쳐 problem_id 단위로
                       1행만 남긴다. 원본 split은 critic별 확장이라 problem_id가 split을
                       가로지른다(3,636건) — 행 단위로 자르면 홀드아웃이 새어나간다.
"""
import argparse
import sys, os, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluation.acc_exec import ExecutionInterface  # noqa: E402
from gfg_recover import (J, extract_methods, pick_method, parse_pseudo_stdin,  # noqa: E402
                         build_args, try_recover_case)

KEEP = ["id", "problem_id", "instruction", "starter_code", "main_critic_category",
        "critic_categories", "source_platform", "source"]


def _ref_codes(rec):
    """known-correct python ref들을 파일 순서 그대로. refs[0]이 canonical."""
    return [s["code"] for s in (J(rec.get("reference_solutions")) or [])
            if isinstance(s, dict) and s.get("is_known_correct") and s.get("language") == "python"
            and s.get("code")]


def _ref_code(rec):
    codes = _ref_codes(rec)
    return codes[0] if codes else None


def _try_gfg(code, es, cases):
    """gfg 역설계: 성공하면 (eval_spec, ) 아니면 None."""
    methods = extract_methods(code)
    keys = set(parse_pseudo_stdin(str((cases[0].get("input") or {}).get("value") or "")))
    picked = pick_method(methods, keys) if methods else None
    if not picked:
        return None
    method, params = picked
    for c in cases:  # validate ref on ALL cases
        args = build_args(params, parse_pseudo_stdin(str((c.get("input") or {}).get("value") or "")))
        if args is None or try_recover_case(code, method, args,
                                            (c.get("expected_output") or {}).get("value")) != "PASS":
            return None
    return {**es, "eval_mode": "gfg_function", "entry_point": method, "gfg_params": params}


def process(rec, keep_solution=False, ref_fallback=False):
    es = J(rec.get("eval_spec")) or {}
    cases = J(rec.get("test_cases")) or []
    codes = _ref_codes(rec)
    if not ref_fallback:
        codes = codes[:1]
    if not codes or not cases:
        return ("no_ref", None)
    base = {k: rec.get(k) for k in KEEP}
    base.update({"domain": "coding", "dataset": "acc", "scoring_kind": "acc"})
    if rec.get("main_critic_categories_seen"):       # dedupe가 합쳐둔 축들
        base["main_critic_categories_seen"] = rec["main_critic_categories_seen"]
    is_gfg = "geeksforgeeks" in (rec.get("source_platform") or "").lower()

    def emit(cat, code, spec, idx):
        base["eval_spec"] = spec
        base["test_cases"] = cases
        if keep_solution:
            base["solution"] = code          # SFT completion = 실행검증된 참조 솔루션
            base["ground_truth"] = code
            base["ref_index"] = idx          # 0 = canonical, >0 = 폴백으로 승격됨
        return (cat, base)

    last = "fail"
    for idx, code in enumerate(codes):
        if is_gfg and es.get("eval_mode") == "stdin_stdout":
            spec = _try_gfg(code, es, cases)
            if spec is not None:
                return emit("gfg_recovered", code, spec, idx)
            last = "gfg_unrecovered"
            continue
        # non-gfg: run the ref through the fixed runner (all cases)
        prob = {"eval_spec": es, "test_cases": cases, "problem_id": rec.get("problem_id")}
        out = ExecutionInterface().run(prob, code, solution_id="ref")
        if out.get("passed"):
            return emit("kept", code, es, idx)
        last = f"drop_{out.get('status', 'fail')}"
    return (last, None)


def dedupe(recs):
    """problem_id 단위 1행. 중복 행이 갖고 있던 main_critic_category는 전부 보존한다
    (human-prior 분할이 어느 축을 쓸지는 하위에서 결정)."""
    by_pid = {}
    for r in recs:
        pid = str(r.get("problem_id"))
        if pid in by_pid:
            seen = by_pid[pid].setdefault("main_critic_categories_seen", [])
            mc = r.get("main_critic_category")
            if mc and mc not in seen:
                seen.append(mc)
            continue
        r = dict(r)
        mc = r.get("main_critic_category")
        r["main_critic_categories_seen"] = [mc] if mc else []
        by_pid[pid] = r
    return list(by_pid.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="원본 jsonl (여러 개면 합침)")
    ap.add_argument("-o", "--out", default=None, help="출력 jsonl")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--keep-solution", action="store_true")
    ap.add_argument("--ref-fallback", action="store_true")
    ap.add_argument("--dedupe-problem-id", action="store_true")
    a = ap.parse_args()

    # 레거시 위치인자 형태: <in> <out> [workers]
    inputs, outp, workers = a.inputs, a.out, a.workers
    if outp is None:
        if len(inputs) < 2:
            ap.error("출력 경로가 필요하다 (-o 또는 레거시 위치인자)")
        *inputs, outp = inputs
        if outp.isdigit():                           # <in> <out> <workers>
            workers, outp = int(outp), inputs.pop()
    workers = workers or min(16, os.cpu_count() or 4)

    recs = []
    for p in inputs:
        n0 = len(recs)
        recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        print(f"  + {p}: {len(recs) - n0}", flush=True)
    if a.dedupe_problem_id:
        n0 = len(recs)
        recs = dedupe(recs)
        print(f"dedupe problem_id: {n0} -> {len(recs)}", flush=True)
    print(f"input={len(recs)} workers={workers} keep_solution={a.keep_solution} "
          f"ref_fallback={a.ref_fallback}", flush=True)

    stats = Counter()
    ref_idx = Counter()
    n_out = 0
    fn = partial(process, keep_solution=a.keep_solution, ref_fallback=a.ref_fallback)
    with open(outp, "w") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (cat, rec) in enumerate(pool.map(fn, recs, chunksize=8)):
            stats[cat] += 1
            if rec is not None:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_out += 1
                ref_idx[rec.get("ref_index", 0)] += 1
            if (i + 1) % 3000 == 0:
                print(f"  ...{i+1}/{len(recs)} kept={n_out}", flush=True)
    print(f"\nDONE: emitted {n_out}/{len(recs)} self-consistent -> {outp}")
    print("breakdown:", dict(stats.most_common()))
    if a.ref_fallback:
        print("채택된 ref 인덱스:", dict(sorted(ref_idx.items())))


if __name__ == "__main__":
    main()
