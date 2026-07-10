#!/usr/bin/env python3
"""Build local JSONL files for QASC evolution/eval.

Output files follow data.loader.get_dataset(local_dir=...) naming:
  export/qasc/qasc_train.jsonl
  export/qasc/qasc_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset


def convert_item(item: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    qid = item.get("id") or item.get("question_id") or f"qasc_{split}_{idx}"
    instruction = item.get("formatted_question") or item.get("question") or ""
    gold = str(item.get("answerKey") or item.get("answer") or "").strip().upper()
    if not instruction:
        raise ValueError(f"Missing formatted question for {qid}")
    if gold not in set("ABCDEFGH"):
        raise ValueError(f"Missing/invalid answerKey for {qid}: {gold!r}")
    return {
        "id": str(qid),
        "instruction": instruction,
        "ground_truth": gold,
        "domain": "qasc",
        "dataset": "qasc",
        "scoring_kind": "qasc",
        "num_choices": 8,
    }


def write_split(split: str, out_dir: Path) -> int:
    ds = load_dataset("allenai/qasc", split=split)
    out_path = out_dir / f"qasc_{split}.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            rec = convert_item(dict(item), split, idx)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QASC local JSONL files")
    ap.add_argument("--out_dir", default="export/qasc")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        write_split(split, out_dir)

    sample = next((out_dir / "qasc_train.jsonl").open(encoding="utf-8"))
    print("sample:", sample.strip()[:500])


if __name__ == "__main__":
    main()
