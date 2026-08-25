#!/usr/bin/env python3
"""Aggregate complete per-expert SFT traversals into an oracle-union upper bound."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def aggregate(test_path: Path, parts_dir: Path, experts: list[str]) -> tuple[list[dict], dict]:
    test_rows = load_jsonl(test_path)
    ids = [str(r["id"]) for r in test_rows]
    predictions: dict[str, dict[str, dict]] = {}
    for expert in experts:
        path = parts_dir / f"{expert}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing expert output: {path}")
        rows = load_jsonl(path)
        by_id = {str(r["id"]): r for r in rows}
        if set(by_id) != set(ids) or len(rows) != len(ids):
            raise RuntimeError(f"incomplete or duplicate output for {expert}: {len(rows)}/{len(ids)}")
        predictions[expert] = by_id

    per_expert_solved = Counter()
    coverage = Counter()
    oracle_rows = []
    for item in test_rows:
        pid = str(item["id"])
        solved_by = []
        for expert in experts:
            if float(predictions[expert][pid]["pass_score"]) > 0:
                solved_by.append(expert)
                per_expert_solved[expert] += 1
        coverage[len(solved_by)] += 1
        chosen = solved_by[0] if solved_by else experts[0]
        oracle_rows.append({
            "id": item["id"],
            "solved_by": solved_by,
            "n_solved": len(solved_by),
            "oracle_expert": chosen if solved_by else None,
            "oracle_prediction": predictions[chosen][pid]["prediction"] if solved_by else None,
            "oracle_pass": int(bool(solved_by)),
            "per_expert": {e: int(e in solved_by) for e in experts},
        })

    total = len(test_rows)
    union = sum(r["oracle_pass"] for r in oracle_rows)
    best_expert = max(experts, key=lambda e: per_expert_solved[e])
    summary = {
        "definition": "oracle union over actual generations from all trained SFT LoRA experts",
        "total": total,
        "experts": experts,
        "per_expert_solved": dict(per_expert_solved),
        "per_expert_pass_at_1": {e: per_expert_solved[e] / total for e in experts},
        "best_single_expert": best_expert,
        "best_single_pass_at_1": per_expert_solved[best_expert] / total,
        "oracle_union_solved": union,
        "oracle_union_pass_at_1": union / total,
        "router_headroom_absolute": union / total,
        "coverage_histogram": {str(k): coverage[k] for k in sorted(coverage)},
        "unique_solve_by_expert": {
            e: sum(r["solved_by"] == [e] for r in oracle_rows) for e in experts
        },
    }
    return oracle_rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--parts-dir", default="results/acc/seed20211004/sft_oracle/parts")
    ap.add_argument("--router-config", default="checkpoints/router/acc_seed20211004_top1_set/router_config.json")
    ap.add_argument("--output-dir", default="results/acc/seed20211004/sft_oracle")
    a = ap.parse_args()
    cfg = json.loads(Path(a.router_config).read_text())
    oracle_rows, summary = aggregate(Path(a.test_jsonl), Path(a.parts_dir), cfg["experts"])
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "oracle_union_test.jsonl").open("w", encoding="utf-8") as f:
        for row in oracle_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "oracle_union_test.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
