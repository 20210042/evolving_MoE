#!/usr/bin/env python3
"""Push local QASC/LBox exports to Hugging Face datasets.

The Hub dataset viewer needs a stable column schema. QASC is loaded from the
original HF source so the blind test split can be included. LBox is loaded from
the local JSONL exports and normalizes mixed string/list labels into explicit
text/list/raw columns.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import HfApi


QASC_FEATURES = Features(
    {
        "id": Value("string"),
        "question": Value("string"),
        "choices_text": Sequence(Value("string")),
        "choices_label": Sequence(Value("string")),
        "instruction": Value("string"),
        "ground_truth": Value("string"),
        "has_label": Value("bool"),
        "fact1": Value("string"),
        "fact2": Value("string"),
        "combinedfact": Value("string"),
        "domain": Value("string"),
        "dataset": Value("string"),
        "scoring_kind": Value("string"),
        "num_choices": Value("int32"),
    }
)


LBOX_FEATURES = Features(
    {
        "id": Value("string"),
        "task_type": Value("string"),
        "task_config": Value("string"),
        "casetype": Value("string"),
        "facts": Value("string"),
        "instruction": Value("string"),
        "ground_truth_text": Value("string"),
        "ground_truth_items": Sequence(Value("string")),
        "ground_truth_raw_json": Value("string"),
        "domain": Value("string"),
        "dataset": Value("string"),
        "scoring_kind": Value("string"),
    }
)


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value)


def build_qasc() -> DatasetDict:
    splits = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset("allenai/qasc", split=split)
        rows = []
        for item in ds:
            choices = item.get("choices") or {}
            answer = normalize_text(item.get("answerKey") or item.get("answer")).strip().upper()
            rows.append(
                {
                    "id": normalize_text(item.get("id")),
                    "question": normalize_text(item.get("question")),
                    "choices_text": [normalize_text(x) for x in choices.get("text", [])],
                    "choices_label": [normalize_text(x) for x in choices.get("label", [])],
                    "instruction": normalize_text(item.get("formatted_question") or item.get("question")),
                    "ground_truth": answer,
                    "has_label": bool(answer),
                    "fact1": normalize_text(item.get("fact1")),
                    "fact2": normalize_text(item.get("fact2")),
                    "combinedfact": normalize_text(item.get("combinedfact")),
                    "domain": "qasc",
                    "dataset": "qasc",
                    "scoring_kind": "qasc",
                    "num_choices": len(choices.get("label", [])),
                }
            )
        splits[split] = Dataset.from_list(rows, features=QASC_FEATURES)
    return DatasetDict(splits)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_lbox_ground_truth(value: Any) -> tuple[str, list[str], str]:
    if isinstance(value, list):
        items = [normalize_text(x).strip() for x in value if normalize_text(x).strip()]
        text = ", ".join(items)
    else:
        text = normalize_text(value).strip()
        items = [text] if text else []
    return text, items, json.dumps(value, ensure_ascii=False)


def build_lbox(export_dir: Path) -> DatasetDict:
    split_files = {
        "train": export_dir / "lbox_train.jsonl",
        "valid": export_dir / "lbox_valid.jsonl",
        "test": export_dir / "lbox_test.jsonl",
    }
    splits = {}
    for split, path in split_files.items():
        rows = []
        for item in read_jsonl(path):
            gt_text, gt_items, gt_raw = normalize_lbox_ground_truth(item.get("ground_truth"))
            rows.append(
                {
                    "id": normalize_text(item.get("id")),
                    "task_type": normalize_text(item.get("task_type")),
                    "task_config": normalize_text(item.get("task_config")),
                    "casetype": normalize_text(item.get("casetype")),
                    "facts": normalize_text(item.get("facts")),
                    "instruction": normalize_text(item.get("instruction")),
                    "ground_truth_text": gt_text,
                    "ground_truth_items": gt_items,
                    "ground_truth_raw_json": gt_raw,
                    "domain": normalize_text(item.get("domain") or "lbox"),
                    "dataset": normalize_text(item.get("dataset") or "lbox"),
                    "scoring_kind": normalize_text(item.get("scoring_kind") or "lbox"),
                }
            )
        splits[split] = Dataset.from_list(rows, features=LBOX_FEATURES)
    return DatasetDict(splits)


def upload_readme(repo_id: str, text: str) -> None:
    api = HfApi()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "README.md"
        path.write_text(text, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )


def qasc_card() -> str:
    return """---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices_text
    sequence: string
  - name: choices_label
    sequence: string
  - name: instruction
    dtype: string
  - name: ground_truth
    dtype: string
  - name: has_label
    dtype: bool
  - name: fact1
    dtype: string
  - name: fact2
    dtype: string
  - name: combinedfact
    dtype: string
  - name: domain
    dtype: string
  - name: dataset
    dtype: string
  - name: scoring_kind
    dtype: string
  - name: num_choices
    dtype: int32
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# QASC formatted for MAE / SFT evaluation

This dataset mirrors `allenai/qasc` with columns kept convenient for local
inference and scoring.

- `instruction` is the formatted multiple-choice prompt used by this repo.
- `ground_truth` is the answer letter when available.
- `has_label=false` on the official blind test split, whose `answerKey` is not public.
- `id` is preserved for joining with inference outputs.
"""


def lbox_card() -> str:
    return """---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: task_type
    dtype: string
  - name: task_config
    dtype: string
  - name: casetype
    dtype: string
  - name: facts
    dtype: string
  - name: instruction
    dtype: string
  - name: ground_truth_text
    dtype: string
  - name: ground_truth_items
    sequence: string
  - name: ground_truth_raw_json
    dtype: string
  - name: domain
    dtype: string
  - name: dataset
    dtype: string
  - name: scoring_kind
    dtype: string
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: valid
    path: data/valid-*
  - split: test
    path: data/test-*
---

# LBox Open classification formatted for MAE / SFT evaluation

This dataset is built from local `export/lbox` JSONL files.

- `id` is preserved for joining with inference outputs.
- `instruction` is the exact prompt body used by this repo.
- `ground_truth_items` stores one or more labels with a stable list schema.
- `ground_truth_text` is a display-friendly joined label string.
- `ground_truth_raw_json` preserves the original local label object.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasc_repo", default="jongbin-kr/qasc")
    parser.add_argument("--lbox_repo", default="jongbin-kr/lbox")
    parser.add_argument("--lbox_export_dir", default="export/lbox")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_qasc", action="store_true")
    parser.add_argument("--skip_lbox", action="store_true")
    args = parser.parse_args()

    if not args.skip_qasc:
        qasc = build_qasc()
        print(qasc)
        if not args.dry_run:
            qasc.push_to_hub(args.qasc_repo, private=args.private)
            upload_readme(args.qasc_repo, qasc_card())

    if not args.skip_lbox:
        lbox = build_lbox(Path(args.lbox_export_dir))
        print(lbox)
        if not args.dry_run:
            lbox.push_to_hub(args.lbox_repo, private=args.private)
            upload_readme(args.lbox_repo, lbox_card())


if __name__ == "__main__":
    main()
