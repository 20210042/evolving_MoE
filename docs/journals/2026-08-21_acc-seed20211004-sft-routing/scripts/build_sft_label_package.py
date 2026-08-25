#!/usr/bin/env python3
"""Build an expert-SFT label package from a final roster and binned results.

The binned file decides which examples each expert sees.  The roster supplies the
expert-specific system prompt.  The source JSONL supplies the original instruction
and a supervised target (``solution`` or ``ground_truth``); binning outputs alone do
not contain training targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_roster(path: Path) -> list[dict]:
    raw = json.load(path.open(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("roster", raw.get("agents"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a roster list (or a roster/agents key).")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--binned", required=True, type=Path)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=None, help="Override the dataset field from binned rows.")
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help="Keep only the binned/source ID intersection (for corpora filtered to verified references).",
    )
    args = parser.parse_args()

    # Some coding test specifications contain very large integer literals.
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

    roster = load_roster(args.roster)
    binned = load_jsonl(args.binned)
    if not roster or not binned or not args.source_jsonl.stat().st_size:
        raise ValueError("roster, binned, and source JSONL must all be non-empty.")

    mapping: dict[str, dict] = {}
    for agent in roster:
        agent_id = str(agent.get("id") or "").strip()
        if not agent_id:
            raise ValueError("Every roster entry must have a non-empty id.")
        if agent_id in mapping:
            raise ValueError(f"Duplicate roster id: {agent_id}")
        prompt = str(agent.get("system_prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Roster expert {agent_id!r} has no system_prompt.")
        mapping[agent_id] = {
            key: agent[key]
            for key in ("name", "persona_name", "system_prompt", "strengths", "approach")
            if agent.get(key) is not None
        }

    expert_ids = set(mapping)
    labels: list[dict] = []
    label_ids: set[str] = set()
    solve_counts = {expert_id: 0 for expert_id in mapping}
    for row in binned:
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in label_ids:
            raise ValueError(f"Missing or duplicate binned id: {row_id!r}")
        per_expert = row.get("per_expert") or {}
        if set(per_expert) != expert_ids:
            missing = sorted(expert_ids - set(per_expert))
            extra = sorted(set(per_expert) - expert_ids)
            raise ValueError(f"Expert mismatch for {row_id}: missing={missing}, extra={extra}")
        binary = {expert_id: int(bool(per_expert[expert_id])) for expert_id in mapping}
        for expert_id, solved in binary.items():
            solve_counts[expert_id] += solved
        labels.append({
            "id": row_id,
            "dataset": args.dataset or row.get("dataset"),
            "n_solved": sum(binary.values()),
            "per_expert": binary,
        })
        label_ids.add(row_id)

    # ACC rows can carry large test suites; retain only join/target metadata in memory.
    source_ids: set[str] = set()
    source_ids_with_target: set[str] = set()
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "").strip()
            if not row_id or row_id in source_ids:
                raise ValueError(f"Missing or duplicate source id: {row_id!r}")
            source_ids.add(row_id)
            if row.get("solution") is not None or row.get("ground_truth") is not None:
                source_ids_with_target.add(row_id)

    original_label_count = len(labels)
    missing_source = sorted(label_ids - source_ids)
    if missing_source and not args.allow_missing_source:
        preview = ", ".join(missing_source[:5])
        raise ValueError(
            f"Source JSONL is missing {len(missing_source)}/{len(label_ids)} binned ids; examples: {preview}"
        )
    if missing_source:
        labels = [row for row in labels if row["id"] in source_ids]
        label_ids = {row["id"] for row in labels}
        solve_counts = {
            expert_id: sum(row["per_expert"][expert_id] for row in labels)
            for expert_id in mapping
        }
        print(
            f"WARNING: kept {len(labels)}/{original_label_count} binned rows with source targets; "
            f"skipped {len(missing_source)} missing source ids.",
            file=sys.stderr,
        )
    missing_targets = [
        row_id for row_id in label_ids if row_id not in source_ids_with_target
    ]
    if missing_targets:
        preview = ", ".join(sorted(missing_targets)[:5])
        raise ValueError(f"{len(missing_targets)} matched source rows have no solution/ground_truth: {preview}")

    for expert_id, count in solve_counts.items():
        mapping[expert_id]["train_pass_at_1"] = count / len(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "binning_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "agent_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
    summary = {
        "source_train_jsonl": str(args.source_jsonl),
        "roster_path": str(args.roster),
        "binned_path": str(args.binned),
        "n_examples": len(labels),
        "n_binned_input": original_label_count,
        "n_missing_source_skipped": len(missing_source),
        "n_experts": len(mapping),
        "expert_solve_counts": solve_counts,
        "prompt_mode": "persona system_prompt from final roster",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(labels)} labels x {len(mapping)} experts to {args.output_dir}")


if __name__ == "__main__":
    main()
