#!/usr/bin/env python3
"""Measure LBox methods on problem subsets solved by each pre-SFT persona."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def strip_thinking_channels(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"^thought\n.*?\n\n(?=[^ \t])", "", text, count=1, flags=re.DOTALL)
    return text.strip()


def norm_lbox(value: object) -> str:
    text = re.sub(r"[\"'`“”‘’]", "", str(value or ""))
    return re.sub(r"[\s·ㆍ・\-\(\)\[\]{}.,:;，。]", "", text).lower()


STATUTE_RE = re.compile(
    r"([가-힣A-Za-z0-9·ㆍ()]+법)\s*제\s*\d+(?:조|항|호)(?:의\s*\d+)?(?:\s*제\s*\d+(?:항|호))*"
)


def score_lbox(item: dict, prediction: str) -> bool:
    text = strip_thinking_channels(prediction).strip()
    if item["task_type"] == "casename":
        matches = re.findall(r"(?:죄명|범죄명|정답|답)\s*[:：]\s*(.+)", text)
        candidate = matches[-1] if matches else text.splitlines()[-1] if text.splitlines() else text
        candidate = re.sub(r"^(?:정답은|답은)\s*", "", candidate.strip())
        candidate = re.sub(r"(?:입니다|이다)[.。]?$", "", candidate.strip())
        candidate = candidate.strip(" \t\r\n\"'`.,:;")
        return norm_lbox(candidate) == norm_lbox(item["ground_truth"])

    gold = item["ground_truth"]
    gold_items = gold if isinstance(gold, list) else [gold]
    gold_set = {norm_lbox(value) for value in gold_items if norm_lbox(value)}
    found = {norm_lbox(match.group(0)) for match in STATUTE_RE.finditer(text)}
    pred_norm = norm_lbox(text)
    for value in gold_items:
        normalized = norm_lbox(value)
        if normalized and normalized in pred_norm:
            found.add(normalized)
    return bool(gold_set) and {value for value in found if value} == gold_set


DEFAULT_METHODS = {
    "LLM Router + Persona prompt LLM (No SFT, Gemma 4 26B-A4B)": (
        "results/lbox_pre_sft_routed_test/seed20210311/inference_test_routed_top1.jsonl",
        "final_output",
    ),
    "MLP Router + persona fine-tuned LLM": (
        "results/lbox_routed_top1_persona_eval_ep5/20260806_seed42_hs_mean/predictions.jsonl",
        "pass_score",
    ),
    "MLP Router + Legal-category fine-tuned LLM": (
        "results/lbox_routed_top1_legal_category/20260807_seed42_hs_mean/predictions.jsonl",
        "pass_score",
    ),
    "MLP Router + Task-prior fine-tuned LLM": (
        "results/lbox_task_prior_routed_top1/20260730_125326/predictions.jsonl",
        "pass_score",
    ),
    "Dense SFT": (
        "results/qasc_lbox_sft_eval/"
        "lbox_sft_llama3_finetuned_lbox_baseline_eval500_full_eval_snapshot_"
        "checkpoint-12000_baseline_208278.jsonl",
        "pass_score",
    ),
    "vanilla Llama3-8B": (
        "results/qasc_lbox_sft_eval/"
        "lbox_Llama-3.1-8B-Instruct_vanilla_baseline_208397.jsonl",
        "pass_score",
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binned",
        type=Path,
        default=REPO
        / "results/lbox_test_roster_binning/20260730_135227/"
        "binning_test_pre_sft.binned.jsonl",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPO / "results/lbox_binning_seed20210311/agent_mapping.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO / "export/lbox/lbox_test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/lbox_pre_sft_persona_subset_analysis",
    )
    parser.add_argument(
        "--sparse-predictions",
        type=Path,
        help="Optional problem-level Sparse-upcycled MoE JSONL containing id/prediction.",
    )
    args = parser.parse_args()

    binned = load_jsonl(args.binned)
    references = {row["id"]: row for row in load_jsonl(args.reference)}
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    expert_ids = list(binned[0]["per_expert"])

    subsets: dict[str, set[str]] = {
        expert_id: {
            row["id"] for row in binned if int(row["per_expert"][expert_id]) > 0
        }
        for expert_id in expert_ids
    }
    for expert_id in expert_ids:
        subsets[f"{expert_id}_low5"] = {
            row["id"]
            for row in binned
            if int(row["n_solved"]) <= 5 and int(row["per_expert"][expert_id]) > 0
        }
    subsets["generalist_high6"] = {
        row["id"] for row in binned if int(row["n_solved"]) >= 6
    }
    subsets["all_failed"] = {row["id"] for row in binned if int(row["n_solved"]) == 0}

    methods = dict(DEFAULT_METHODS)
    if args.sparse_predictions:
        methods = {
            **{
                key: value
                for key, value in methods.items()
                if key not in {"Dense SFT", "vanilla Llama3-8B"}
            },
            "Sparse-upcycled MoE": (str(args.sparse_predictions), "prediction"),
            "Dense SFT": methods["Dense SFT"],
            "vanilla Llama3-8B": methods["vanilla Llama3-8B"],
        }

    correctness: dict[str, dict[str, bool]] = {}
    sources: dict[str, str] = {}
    expected_ids = {row["id"] for row in binned}
    for method, (raw_path, score_field) in methods.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO / path
        rows = load_jsonl(path)
        if score_field in {"final_output", "prediction"}:
            by_id = {
                row["id"]: score_lbox(references[row["id"]], row[score_field])
                for row in rows
            }
        else:
            by_id = {row["id"]: float(row[score_field]) > 0 for row in rows}
        if set(by_id) != expected_ids:
            missing = len(expected_ids - set(by_id))
            extra = len(set(by_id) - expected_ids)
            raise ValueError(f"ID mismatch for {method}: missing={missing}, extra={extra}")
        correctness[method] = by_id
        sources[method] = str(path.relative_to(REPO))

    columns = [f"{expert_id}_low5" for expert_id in expert_ids] + [
        "generalist_high6",
        "all_failed",
    ]

    def column_name(key: str) -> str:
        if key.endswith("_low5"):
            return f'{mapping[key.removesuffix("_low5")]["name"]} (low5)'
        if key == "generalist_high6":
            return "High6 Generalist (solved by >=6 personas)"
        if key == "all_failed":
            return "All personas failed"
        return mapping[key]["name"]

    results = []
    for method, by_id in correctness.items():
        total_correct = sum(by_id.values())
        row = {
            "method": method,
            "overall_n": len(expected_ids),
            "overall_correct": total_correct,
            "overall_accuracy": 100.0 * total_correct / len(expected_ids),
            "subsets": {},
        }
        for column in columns:
            ids = subsets[column]
            correct = sum(by_id[item_id] for item_id in ids)
            row["subsets"][column] = {
                "n": len(ids),
                "correct": correct,
                "accuracy": 100.0 * correct / len(ids),
            }
        results.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": (
            "Each persona column is the overlapping set of test examples solved by that "
            "pre-SFT prompt persona. all_failed is the set solved by none of the personas."
        ),
        "total": len(expected_ids),
        "columns": [
            {
                "key": key,
                "name": column_name(key),
                "n": len(subsets[key]),
            }
            for key in columns
        ],
        "sources": sources,
        "results": results,
        "unavailable_method": (
            None
            if args.sparse_predictions
            else "Sparse-upcycled MoE: no problem-level prediction artifact was supplied/found"
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "overall_accuracy"] + [
            f"{key}_accuracy" for key in columns
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result["method"],
                    "overall_accuracy": f'{result["overall_accuracy"]:.2f}',
                    **{
                        f"{key}_accuracy": f'{result["subsets"][key]["accuracy"]:.2f}'
                        for key in columns
                    },
                }
            )

    headers = ["Method", f"Overall [{len(expected_ids):,}]"] + [
        f"{column_name(key)} [{len(subsets[key]):,}]"
        for key in columns
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for result in results:
        values = [result["method"], f'{result["overall_accuracy"]:.2f}'] + [
            f'{result["subsets"][key]["accuracy"]:.2f}' for key in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'results.csv'}")
    print(f"Wrote {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
