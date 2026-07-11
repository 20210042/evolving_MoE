#!/usr/bin/env python3
"""Build local JSONL files for LBox Open legal EM evolution/eval.

Phase 1 includes classification tasks only:
  - casename_classification
  - casename_classification_plus
  - statute_classification
  - statute_classification_plus

Output files follow data.loader.get_dataset(local_dir=...) naming:
  export/lbox/lbox_train.jsonl
  export/lbox/lbox_valid.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


TASK_CONFIGS = [
    ("casename", "casename_classification"),
    ("casename", "casename_classification_plus"),
    ("statute", "statute_classification"),
    ("statute", "statute_classification_plus"),
]

INSTRUCTIONS = {
    "casename": "다음 사실관계에 해당하는 사건명 또는 죄명을 정확히 한 줄로 답하라.",
    "statute": "다음 사실관계에 적용되는 법조문을 모두 나열하라(예: 형법 제298조).",
}


def _nonempty(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(str(x).strip() for x in value)
    return bool(str(value or "").strip())


def convert_item(task_type: str, config: str, split: str, item: Dict[str, Any]) -> Dict[str, Any] | None:
    facts = str(item.get("facts") or "").strip()
    if not facts:
        return None

    if task_type == "casename":
        gold: Any = str(item.get("casename") or "").strip()
    elif task_type == "statute":
        gold = item.get("statutes") or []
        if isinstance(gold, str):
            gold = [gold]
        gold = [str(x).strip() for x in gold if str(x).strip()]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if not _nonempty(gold):
        return None

    raw_id = item.get("id")
    rec_id = f"{task_type}_{config}_{split}_{raw_id}"
    instruction = f"{INSTRUCTIONS[task_type]}\n\n사실관계:\n{facts}"
    return {
        "id": rec_id,
        "task_type": task_type,
        "task_config": config,
        "casetype": item.get("casetype"),
        "facts": facts,
        "ground_truth": gold,
        "instruction": instruction,
        "domain": "lbox",
        "dataset": "lbox",
        "scoring_kind": "lbox",
    }


def iter_records(split: str) -> Iterable[Dict[str, Any]]:
    for task_type, config in TASK_CONFIGS:
        ds = load_dataset("lbox/lbox_open", config, split=split)
        emitted = 0
        skipped = 0
        for item in ds:
            rec = convert_item(task_type, config, split, dict(item))
            if rec is None:
                skipped += 1
                continue
            emitted += 1
            yield rec
        print(f"{split}/{config}: emitted={emitted} skipped={skipped}")


def write_split(split: str, out_dir: Path, seed: int) -> int:
    rows: List[Dict[str, Any]] = list(iter_records(split))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out_path = out_dir / f"lbox_{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LBox Open local JSONL files")
    ap.add_argument("--out_dir", default="export/lbox")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        write_split(split, out_dir, args.seed)

    sample_path = out_dir / f"lbox_{args.splits[0]}.jsonl"
    sample = next(sample_path.open(encoding="utf-8"))
    print("sample:", sample.strip()[:500])


if __name__ == "__main__":
    main()
