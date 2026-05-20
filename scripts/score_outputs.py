#!/usr/bin/env python3
"""Score inference outputs (JSONL) and report pass@1."""

from typing import Optional
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:
    yaml = None

from meta_agent_evo.evaluation.scorer import score_one


def _data_dir_from_config() -> Optional[str]:
    base = ROOT / "configs" / "base.yaml"
    if yaml is None or not base.is_file():
        return None
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("data_dir")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score inference JSONL output")
    parser.add_argument("--input", type=str, required=True, help="Path to inference_output.jsonl")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (mbpp|humaneval|livecodebench)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override configs/base.yaml data_dir (for MBPP test_list reload)",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    data_dir = args.data_dir or _data_dir_from_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    input_path = Path(args.input)
    if not input_path.is_file():
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Optional: load reference items for test_list (needed for MBPP asserts)
    ref_items: dict = {}
    if data_dir:
        try:
            from meta_agent_evo.data.loader import get_dataset

            dataset_name = args.dataset or "mbpp"
            items = get_dataset(dataset_name, split=args.split, local_dir=data_dir)
            ref_items = {str(it["id"]): it for it in items}
            logging.info("Loaded %d reference items from %s.", len(ref_items), dataset_name)
        except Exception as exc:
            logging.warning("Could not load reference dataset: %s", exc)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    total = 0
    passed = 0
    failures: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping malformed JSON line.")
            continue

        item_id = str(record.get("id", ""))
        code = record.get("final_code") or record.get("final_output") or record.get("code") or record.get("prediction", "")
        dataset_name = record.get("dataset") or args.dataset or "mbpp"

        item = ref_items.get(item_id, record)
        if "dataset" not in item:
            item = dict(item)
            item["dataset"] = dataset_name

        score = score_one(item, code, code_timeout=args.timeout)
        total += 1
        if score >= 100.0 - 1e-6:
            passed += 1
        else:
            failures.append(item_id)

    if total == 0:
        logging.error("No valid records found in %s", input_path)
        sys.exit(1)

    pass_at_1 = passed / total * 100.0
    logging.info("=== Results: %s ===", input_path.name)
    logging.info("  Total : %d", total)
    logging.info("  Passed: %d", passed)
    logging.info("  Pass@1: %.2f%%", pass_at_1)
    if failures:
        logging.info("  Failed IDs (first 20): %s", failures[:20])

    summary = {
        "input": str(input_path),
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "passed": passed,
        "pass_at_1": pass_at_1,
    }
    summary_path = input_path.with_suffix(".score.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary written to %s", summary_path)

    print(f"Pass@1: {pass_at_1:.2f}%  ({passed}/{total})")


if __name__ == "__main__":
    main()
