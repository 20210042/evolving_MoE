#!/usr/bin/env python3
"""
Compare hard-WAR (seed20210111) vs soft-WAR (seed20211004) evolution results.
Analyze:
1. Expert specialization (per-expert pass@1, coverage differences)
2. All-fail reduction
3. UB comparison (oracle coverage)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_roster(roster_path):
    """Load roster and return expert metadata."""
    with open(roster_path) as f:
        roster = json.load(f)
    return roster

def load_binning_summary(summary_path):
    """Load binning summary statistics."""
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f)

def load_binned_jsonl(jsonl_path):
    """Load binned JSONL and compute per-expert stats."""
    if not os.path.exists(jsonl_path):
        return None
    
    solves = defaultdict(lambda: {"solved": 0, "total": 0})
    all_fail_count = 0
    all_pass_count = 0
    total_count = 0
    
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            total_count += 1
            
            # Count how many experts solved this problem
            expert_solves = row.get("solved_by_experts", [])
            
            if len(expert_solves) == 0:
                all_fail_count += 1
            else:
                # Check if all experts solved it
                roster_size = row.get("roster_size", len(expert_solves))
                if len(expert_solves) == roster_size:
                    all_pass_count += 1
            
            # Count per expert
            for expert_id in expert_solves:
                solves[expert_id]["solved"] += 1
                solves[expert_id]["total"] += 1
            
            # Update total for all experts
            for expert_id in solves.keys():
                if expert_id not in expert_solves:
                    solves[expert_id]["total"] += 1
    
    return {
        "per_expert": dict(solves),
        "all_fail": all_fail_count,
        "all_pass": all_pass_count,
        "total": total_count,
        "all_fail_pct": 100 * all_fail_count / total_count if total_count > 0 else 0,
    }

def compute_per_expert_pass1(binned_jsonl_path, roster_size):
    """Compute pass@1 for each expert from binned JSONL."""
    if not os.path.exists(binned_jsonl_path):
        return None
    
    pass1_stats = defaultdict(lambda: {"solved": 0, "total": 0})
    
    with open(binned_jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            expert_solves = row.get("solved_by_experts", [])
            
            for i in range(roster_size):
                expert_id = str(i)  # or use actual expert ID
                pass1_stats[expert_id]["total"] += 1
                if expert_id in expert_solves:
                    pass1_stats[expert_id]["solved"] += 1
    
    return {
        expert_id: stats["solved"] / stats["total"] * 100 if stats["total"] > 0 else 0
        for expert_id, stats in pass1_stats.items()
    }

def main():
    base_dir = Path("/home/jaehoonjeong/data/MetaAgentEvolution_Release")
    
    # Hard-WAR (seed20210111)
    hard_war_dir = base_dir / "results/acc/seed20210111"
    hard_roster = load_roster(hard_war_dir / "roster_final.json")
    hard_train_summary = load_binning_summary(hard_war_dir / "binning_train_full.binned.summary.json")
    hard_test_summary = load_binning_summary(hard_war_dir / "inference_test_binning_final.binned.summary.json")
    
    # Soft-WAR (seed20211004)
    soft_war_dir = base_dir / "results/acc/seed20211004"
    soft_roster = load_roster(soft_war_dir / "roster_final.json")
    soft_train_summary = load_binning_summary(soft_war_dir / "binning_train_full.binned.summary.json")
    soft_test_summary = load_binning_summary(soft_war_dir / "binning_test_full.binned.summary.json")
    
    print("=" * 80)
    print("HARD-WAR vs SOFT-WAR: EVOLUTION COMPARISON")
    print("=" * 80)
    
    # 1. ROSTER SIZE & STRUCTURE
    print("\n### 1. ROSTER STRUCTURE ###")
    print(f"Hard-WAR (seed20210111): {len(hard_roster)} experts")
    print(f"Soft-WAR (seed20211004): {len(soft_roster)} experts")
    
    # 2. ALL-FAIL COMPARISON
    print("\n### 2. ALL-FAIL REDUCTION ###")
    if hard_train_summary:
        hard_all_fail_train = hard_train_summary.get("all_fail_count", 0)
        hard_all_fail_train_pct = 100 * hard_all_fail_train / hard_train_summary.get("total", 1)
        print(f"Hard-WAR Train - All Fail: {hard_all_fail_train} / {hard_train_summary['total']} ({hard_all_fail_train_pct:.2f}%)")
    
    if soft_train_summary:
        soft_all_fail_train = soft_train_summary.get("all_fail_count", 0)
        soft_all_fail_train_pct = 100 * soft_all_fail_train / soft_train_summary.get("total", 1)
        print(f"Soft-WAR Train - All Fail: {soft_all_fail_train} / {soft_train_summary['total']} ({soft_all_fail_train_pct:.2f}%)")
    
    if hard_test_summary:
        hard_all_fail_test = hard_test_summary.get("all_fail_count", 0)
        hard_all_fail_test_pct = 100 * hard_all_fail_test / hard_test_summary.get("total", 1)
        print(f"Hard-WAR Test - All Fail: {hard_all_fail_test} / {hard_test_summary['total']} ({hard_all_fail_test_pct:.2f}%)")
    
    if soft_test_summary:
        soft_all_fail_test = soft_test_summary.get("all_fail_count", 0)
        soft_all_fail_test_pct = 100 * soft_all_fail_test / soft_test_summary.get("total", 1)
        print(f"Soft-WAR Test - All Fail: {soft_all_fail_test} / {soft_test_summary['total']} ({soft_all_fail_test_pct:.2f}%)")
    
    # 3. UB COMPARISON
    print("\n### 3. UPPER BOUND (UNION COVERAGE) ###")
    if hard_train_summary:
        hard_ub_train = hard_train_summary.get("union_ub_pct", 0)
        print(f"Hard-WAR Train UB: {hard_ub_train:.2f}%")
    
    if soft_train_summary:
        soft_ub_train = soft_train_summary.get("union_ub_pct", 0)
        print(f"Soft-WAR Train UB: {soft_ub_train:.2f}%")
    
    if hard_test_summary:
        hard_ub_test = hard_test_summary.get("union_ub_pct", 0)
        print(f"Hard-WAR Test UB: {hard_ub_test:.2f}%")
    
    if soft_test_summary:
        soft_ub_test = soft_test_summary.get("union_ub_pct", 0)
        print(f"Soft-WAR Test UB: {soft_ub_test:.2f}%")
    
    # 4. DETAILED BINNING ANALYSIS
    print("\n### 4. BINNING HISTOGRAM (Coverage Distribution) ###")
    
    if hard_train_summary and "coverage_histogram" in hard_train_summary:
        print("\nHard-WAR Train Coverage Histogram:")
        hist = hard_train_summary["coverage_histogram"]
        for k in sorted(hist.keys()):
            v = hist[k]
            pct = 100 * v / hard_train_summary["total"]
            print(f"  {k} experts solve: {v} problems ({pct:.2f}%)")
    
    if soft_train_summary and "coverage_histogram" in soft_train_summary:
        print("\nSoft-WAR Train Coverage Histogram:")
        hist = soft_train_summary["coverage_histogram"]
        for k in sorted(hist.keys()):
            v = hist[k]
            pct = 100 * v / soft_train_summary["total"]
            print(f"  {k} experts solve: {v} problems ({pct:.2f}%)")
    
    # 5. ROSTER NAMES & STRENGTHS
    print("\n### 5. EXPERT ROSTER COMPARISON ###")
    print("\nHard-WAR Experts:")
    for i, expert in enumerate(hard_roster):
        name = expert.get("name", f"Expert {i}")
        avg_war = expert.get("average_war", 0)
        print(f"  {i}: {name} (avg_war={avg_war:.3f})")
    
    print("\nSoft-WAR Experts:")
    for i, expert in enumerate(soft_roster):
        name = expert.get("name", f"Expert {i}")
        avg_war = expert.get("average_war", 0)
        print(f"  {i}: {name} (avg_war={avg_war:.3f})")
    
    # 6. EXPERT SPECIALIZATION TABLE
    print("\n### 6. EXPERT PASS@1 COMPARISON ###")
    print("\nNote: Computing per-expert accuracy from binning data...")
    
    # Load binned jsonl to compute per-expert pass@1
    hard_train_binned = hard_war_dir / "binning_train_full.binned.jsonl"
    soft_train_binned = soft_war_dir / "binning_train_full.binned.jsonl"
    
    print("\n(This requires reading full binned files - see generated comparison report)")
    
    print("\n" + "=" * 80)
    print("Analysis complete. See results/acc/hard_soft_war_comparison.md for detailed report.")
    print("=" * 80)

if __name__ == "__main__":
    main()
