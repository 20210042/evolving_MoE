#!/usr/bin/env python3
"""Score a binning run (one record per problem with per-expert outputs) and emit
the per-expert solve labels — the training signal for the weight-level MoE.

Input  : inference jsonl from `run_inference.py --pipeline binning`, i.e. lines of
         {"id", "dataset", "expert_outputs": {expert_id: output, ...}}.
Output : <input>.binned.jsonl  — {"id", "dataset", "solved_by": [...], "n_solved",
                                   "per_expert": {expert_id: 0|1}}
         <input>.binned.summary.json — per-expert pass@1, union UB, coverage histogram.

Reuses orchestrator._score_pair + its wall-clock-capped pool so a hung math_verify
comparison can never freeze the whole scoring pass (unfinished -> unsolved).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import as_completed, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from orchestrator import (  # noqa: E402
    _score_pair,
    _SCORE_MIN_TIMEOUT,
    _SCORE_PER_PAIR_TIMEOUT,
    _SCORE_WORKERS,
)


def _data_dir_from_config() -> str | None:
    base = ROOT / "configs" / "base.yaml"
    if not base.is_file():
        return None
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("data_dir")


def _score_chunk(score_args: list, workers: int) -> dict:
    """Score a list of (item, code, lcb_to, code_to, lcb_ver, key) args.
    Returns {key: score}; unfinished (hung) tasks default to 0.0."""
    results: dict = {}
    if workers <= 1 or len(score_args) <= 1:
        for a in score_args:
            _pid, key, sc = _score_pair(a)
            results[key] = sc
        return results

    timeout_s = max(_SCORE_MIN_TIMEOUT, _SCORE_PER_PAIR_TIMEOUT * len(score_args) / workers)
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = {pool.submit(_score_pair, a): a[5] for a in score_args}
    try:
        for fut in as_completed(futures, timeout=timeout_s):
            _pid, key, sc = fut.result()
            results[key] = sc
    except FuturesTimeoutError:
        unfinished = sum(1 for f in futures if not f.done())
        logging.error("Score chunk wall-clock timeout (%.0fs): %d unfinished -> 0.0",
                      timeout_s, unfinished)
    finally:
        for proc in list(getattr(pool, "_processes", {}).values()):
            if proc.is_alive():
                proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    for key in futures.values():
        results.setdefault(key, 0.0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a binning inference jsonl")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=3.0, help="per-problem code exec timeout")
    parser.add_argument("--lcb_timeout", type=int, default=10)
    parser.add_argument("--lcb_release_version", type=str, default="release_v5")
    parser.add_argument("--workers", type=int, default=_SCORE_WORKERS)
    parser.add_argument("--chunk", type=int, default=2000, help="problems per scoring chunk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

    input_path = Path(args.input)
    if not input_path.is_file():
        logging.error("Input not found: %s", input_path)
        sys.exit(1)

    records = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    logging.info("Loaded %d binning records.", len(records))

    dataset_name = args.dataset or (records[0].get("dataset") if records else None) or "mbpp"

    # Gold reference (needed for math gold / mbpp test_list).
    ref_items: dict = {}
    data_dir = args.data_dir or _data_dir_from_config()
    if data_dir:
        try:
            from data.loader import get_dataset
            items = get_dataset(dataset_name, split=args.split, local_dir=data_dir)
            ref_items = {str(it["id"]): it for it in items}
            logging.info("Loaded %d reference items (%s/%s).", len(ref_items), dataset_name, args.split)
        except Exception as exc:
            logging.warning("Could not load reference dataset: %s", exc)

    expert_ids: list = []
    seen = set()
    for r in records:
        for eid in (r.get("expert_outputs") or {}):
            if eid not in seen:
                seen.add(eid)
                expert_ids.append(eid)
    logging.info("Experts (%d): %s", len(expert_ids), ", ".join(expert_ids))

    workers = max(1, min(args.workers, _SCORE_WORKERS))
    out_path = input_path.with_suffix(".binned.jsonl")
    per_expert_pass = Counter()
    coverage = Counter()  # n_solved -> #problems
    total = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for start in range(0, len(records), args.chunk):
            chunk = records[start:start + args.chunk]
            score_args = []
            for r in chunk:
                pid = str(r.get("id"))
                item = ref_items.get(pid, dict(r))
                if "dataset" not in item:
                    item = dict(item)
                    item["dataset"] = r.get("dataset") or dataset_name
                for eid, code in (r.get("expert_outputs") or {}).items():
                    # key = (pid, eid) so results map back uniquely.
                    score_args.append(
                        (item, code, args.lcb_timeout, args.timeout,
                         args.lcb_release_version, (pid, eid))
                    )
            scores = _score_chunk(score_args, workers)

            for r in chunk:
                pid = str(r.get("id"))
                per_expert = {}
                solved_by = []
                for eid in (r.get("expert_outputs") or {}):
                    ok = scores.get((pid, eid), 0.0) >= 100.0 - 1e-6
                    per_expert[eid] = int(ok)
                    if ok:
                        solved_by.append(eid)
                        per_expert_pass[eid] += 1
                coverage[len(solved_by)] += 1
                total += 1
                out_f.write(json.dumps({
                    "id": r.get("id"),
                    "dataset": r.get("dataset") or dataset_name,
                    "solved_by": solved_by,
                    "n_solved": len(solved_by),
                    "per_expert": per_expert,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            logging.info("Scored %d/%d problems.", min(start + args.chunk, len(records)), len(records))

    union_ub = sum(c for k, c in coverage.items() if k > 0)
    summary = {
        "input": str(input_path),
        "dataset": dataset_name,
        "split": args.split,
        "total": total,
        "experts": expert_ids,
        "per_expert_pass_at_1": {e: per_expert_pass[e] / total * 100.0 for e in expert_ids},
        "per_expert_solved": dict(per_expert_pass),
        "union_ub": union_ub,
        "union_ub_pct": union_ub / total * 100.0 if total else 0.0,
        "coverage_histogram": {str(k): coverage[k] for k in sorted(coverage)},
    }
    summary_path = input_path.with_suffix(".binned.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info("=== Binning summary ===")
    for e in expert_ids:
        logging.info("  %-12s pass@1 = %.2f%% (%d)", e,
                     summary["per_expert_pass_at_1"][e], per_expert_pass[e])
    logging.info("  Union UB: %.2f%% (%d/%d)", summary["union_ub_pct"], union_ub, total)
    logging.info("  Coverage (n_solved -> #problems): %s", summary["coverage_histogram"])
    logging.info("Labels -> %s", out_path)
    logging.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
