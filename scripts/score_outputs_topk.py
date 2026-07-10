#!/usr/bin/env python3
"""Score top-k routing outputs: pass@1 = union over the k routed experts' candidates
(a problem is solved if ANY of its k picks passes). Legit for coding — execution picks
the winner. Also reports the top-1 (first pick) rate and how much the extra picks add."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.scorer import score_one  # noqa: E402
import yaml  # noqa: E402


def _data_dir_from_config() -> Optional[str]:
    base = ROOT / "configs" / "base.yaml"
    if not base.is_file():
        return None
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("data_dir")


def _passes(item, code, timeout) -> bool:
    return score_one(item, code, code_timeout=timeout) >= 100.0 - 1e-6


def main() -> None:
    ap = argparse.ArgumentParser(description="Score top-k routing outputs (union pass@1)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    data_dir = args.data_dir or _data_dir_from_config()

    ref_items: dict = {}
    if data_dir:
        try:
            from data.loader import get_dataset
            items = get_dataset(args.dataset or "acc", split=args.split, local_dir=data_dir)
            ref_items = {str(it["id"]): it for it in items}
            logging.info("Loaded %d reference items.", len(ref_items))
        except Exception as exc:
            logging.warning("Could not load reference dataset: %s", exc)

    total = passed_union = passed_top1 = 0
    recovered = 0  # top-1 failed but a later pick passed

    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        item_id = str(rec.get("id", ""))
        dataset_name = rec.get("dataset") or args.dataset or "acc"
        item = ref_items.get(item_id, rec)
        if "dataset" not in item:
            item = dict(item); item["dataset"] = dataset_name

        cands = rec.get("candidates")
        if not cands:  # fall back to single-code record
            code = rec.get("final_output") or rec.get("code") or ""
            cands = [{"expert": None, "code": code}]

        total += 1
        results = [_passes(item, c.get("code", ""), args.timeout) for c in cands]
        if results and results[0]:
            passed_top1 += 1
        if any(results):
            passed_union += 1
            if results and not results[0]:
                recovered += 1

    if total == 0:
        logging.error("No records."); sys.exit(1)

    logging.info("=== top-k routing scoring: %s ===", Path(args.input).name)
    logging.info("  Total          : %d", total)
    logging.info("  top-1 pass@1   : %.2f%% (%d)", 100 * passed_top1 / total, passed_top1)
    logging.info("  UNION pass@1   : %.2f%% (%d)", 100 * passed_union / total, passed_union)
    logging.info("  recovered by 2nd+ pick: %d (+%.2f pp)", recovered, 100 * recovered / total)

    summary = {
        "input": str(args.input), "dataset": args.dataset, "split": args.split,
        "total": total, "passed_top1": passed_top1, "passed_union": passed_union,
        "pass_at_1_top1": 100 * passed_top1 / total,
        "pass_at_1_union": 100 * passed_union / total,
        "recovered_by_extra_picks": recovered,
    }
    Path(args.input).with_suffix(".topk_score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"UNION pass@1: {100*passed_union/total:.2f}%  (top-1 {100*passed_top1/total:.2f}%, +{recovered} recovered)")


if __name__ == "__main__":
    main()
