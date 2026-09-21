#!/usr/bin/env python
"""Measure spontaneous alignment between 4x1 routing and dataset priors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_trace(directory: Path, variant: str) -> dict[str, dict]:
    rows = {}
    for path in sorted(directory.glob(f"{variant}_rank*.jsonl")):
        for row in read_jsonl(path):
            rows[str(row["id"])] = row
    return rows


def lbox_task(row: dict) -> str:
    return "statute" if row["task_type"] == "statute" else f"casename_{row['casetype']}"


def label_maps(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    if args.dataset == "sni":
        source = {str(row["id"]): row for row in read_jsonl(args.source)}
        return {
            "category": {key: str(row["category"]) for key, row in source.items()},
            "domain": {key: str(row["sni_domain"]) for key, row in source.items()},
            "task": {key: str(row["task_name"]) for key, row in source.items()},
        }

    source = {str(row["id"]): row for row in read_jsonl(args.source)}
    result = {"task_prior": {key: lbox_task(row) for key, row in source.items()}}
    if args.category_labels and args.category_labels.is_file():
        result["category_prior"] = {
            str(row["id"]): str(row["primary_category"])
            for row in read_jsonl(args.category_labels)
            if row.get("primary_category")
        }
    if args.human_prior and args.human_prior.is_file():
        selected = {}
        for row in read_jsonl(args.human_prior):
            route = next((item for item in row.get("history", []) if item.get("stage") == "routing"), None)
            if route and route.get("selected_expert"):
                selected[str(row["id"])] = str(route["selected_expert"])
        result["human_router_prior"] = selected
    return result


def normalized_fingerprints(rows: list[dict], field: str) -> np.ndarray:
    fingerprints = []
    for row in rows:
        counts = np.asarray(row[field], dtype=np.float64)
        denom = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
        fingerprints.append(shares.reshape(-1))
    return np.asarray(fingerprints)


def permutation_baseline(assignments: np.ndarray, labels: list[str], seed: int, repeats: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    nmis = []
    aris = []
    shuffled = np.asarray(labels, dtype=object)
    for _ in range(repeats):
        shuffled = rng.permutation(shuffled)
        nmis.append(normalized_mutual_info_score(shuffled, assignments))
        aris.append(adjusted_rand_score(shuffled, assignments))
    return {
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
    }


def top_label_lifts(assignments: np.ndarray, labels: list[str], limit: int = 5) -> dict[str, list[dict]]:
    global_counts = Counter(labels)
    n = len(labels)
    result = {}
    for expert in range(4):
        indices = np.flatnonzero(assignments == expert)
        local = Counter(labels[index] for index in indices)
        items = []
        for label, count in local.most_common(limit):
            local_share = count / len(indices) if len(indices) else 0.0
            global_share = global_counts[label] / n
            items.append(
                {
                    "label": label,
                    "count": count,
                    "expert_share": local_share,
                    "dataset_share": global_share,
                    "lift": local_share / global_share if global_share else None,
                }
            )
        result[str(expert)] = items
    return result


def collapse_summary(rows: list[dict], field: str) -> dict:
    counts = np.asarray([row[field] for row in rows], dtype=np.int64).sum(axis=0)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return {
        "overall_expert_shares": (counts.sum(axis=0) / max(counts.sum(), 1)).tolist(),
        "collapsed_layers_ge_90pct": int((shares.max(axis=1) >= 0.90).sum()),
        "layer_expert_shares": shares.tolist(),
    }


def analyze_variant(
    rows: list[dict], labels_by_axis: dict[str, dict[str, str]], region: str, seed: int
) -> dict:
    field = f"{region}_route_counts"
    x = normalized_fingerprints(rows, field)
    global_assignments = x.reshape(len(rows), -1, 4).sum(axis=1).argmax(axis=1)
    kmeans_assignments = KMeans(n_clusters=4, random_state=seed, n_init=20).fit_predict(x)
    result = {
        "examples": len(rows),
        "fingerprint": "per-example concatenation of 32 layer-wise expert shares",
        "collapse": collapse_summary(rows, field),
        "axes": {},
    }
    ids = [str(row["id"]) for row in rows]
    for axis, mapping in labels_by_axis.items():
        keep = [index for index, row_id in enumerate(ids) if row_id in mapping]
        labels = [mapping[ids[index]] for index in keep]
        direct = global_assignments[keep]
        clustered = kmeans_assignments[keep]
        result["axes"][axis] = {
            "examples": len(keep),
            "classes": len(set(labels)),
            "global_dominant_expert": {
                "nmi": normalized_mutual_info_score(labels, direct),
                "ari": adjusted_rand_score(labels, direct),
                "permutation": permutation_baseline(direct, labels, seed),
                "top_label_lifts": top_label_lifts(direct, labels),
            },
            "routing_fingerprint_kmeans_k4": {
                "nmi": normalized_mutual_info_score(labels, clustered),
                "ari": adjusted_rand_score(labels, clustered),
                "permutation": permutation_baseline(clustered, labels, seed),
            },
            "layerwise_nmi": [
                normalized_mutual_info_score(
                    labels,
                    x[keep].reshape(len(keep), -1, 4)[:, layer].argmax(axis=1),
                )
                for layer in range(x.shape[1] // 4)
            ],
        }
    return result


def compare_variants(raw_rows: list[dict], sft_rows: list[dict], region: str) -> dict:
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    sft_by_id = {str(row["id"]): row for row in sft_rows}
    ids = sorted(raw_by_id.keys() & sft_by_id.keys())
    raw = [raw_by_id[key] for key in ids]
    sft = [sft_by_id[key] for key in ids]
    field = f"{region}_route_counts"
    raw_x = normalized_fingerprints(raw, field).reshape(len(ids), -1, 4)
    sft_x = normalized_fingerprints(sft, field).reshape(len(ids), -1, 4)
    raw_global = raw_x.sum(axis=1).argmax(axis=1)
    sft_global = sft_x.sum(axis=1).argmax(axis=1)
    layer_agreement = (raw_x.argmax(axis=2) == sft_x.argmax(axis=2)).mean(axis=0)
    return {
        "examples": len(ids),
        "global_dominant_expert_changed": float((raw_global != sft_global).mean()),
        "mean_absolute_fingerprint_change": float(np.abs(raw_x - sft_x).mean()),
        "layer_modal_expert_agreement": layer_agreement.tolist(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        f"# {summary['dataset'].upper()} 4x1 top-1 router clustering audit",
        "",
        "Raw and SFT models were evaluated on identical saved SFT-generated token sequences.",
        "NMI/ARI near their permutation baselines indicate no meaningful prior alignment.",
        "",
        "| Region | Variant | Prior axis | Classes | Dominant expert NMI | KMeans(k=4) NMI | Dominant expert ARI | Collapsed layers |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for region in ("prompt", "completion"):
        for variant in ("raw", "sft"):
            item = summary["variants"][variant][region]
            collapsed = item["collapse"]["collapsed_layers_ge_90pct"]
            for axis, metric in item["axes"].items():
                direct = metric["global_dominant_expert"]
                km = metric["routing_fingerprint_kmeans_k4"]
                lines.append(
                    f"| {region} | {variant} | {axis} | {metric['classes']} | "
                    f"{direct['nmi']:.4f} | {km['nmi']:.4f} | {direct['ari']:.4f} | {collapsed}/32 |"
                )
    lines.extend(["", "## Raw to SFT routing change", ""])
    for region, values in summary["raw_to_sft"].items():
        lines.append(
            f"- **{region}**: dominant expert changed for "
            f"{100 * values['global_dominant_expert_changed']:.2f}% of examples; "
            f"mean absolute fingerprint change {values['mean_absolute_fingerprint_change']:.4f}."
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lbox", "sni"), required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--category-labels", type=Path)
    parser.add_argument("--human-prior", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_trace(args.trace_dir, "raw")
    sft = load_trace(args.trace_dir, "sft")
    if not raw or not sft:
        raise RuntimeError(f"Missing raw or SFT traces under {args.trace_dir}")
    if raw.keys() != sft.keys():
        raise RuntimeError(f"Raw/SFT ID mismatch: raw={len(raw)} sft={len(sft)}")
    labels = label_maps(args)
    ordered_ids = sorted(raw)
    raw_rows = [raw[key] for key in ordered_ids]
    sft_rows = [sft[key] for key in ordered_ids]
    summary = {
        "dataset": args.dataset,
        "comparison": "raw 4x1 versus FFN-LoRA SFT 4x1 on identical saved SFT predictions",
        "variants": {
            "raw": {
                region: analyze_variant(raw_rows, labels, region, args.seed)
                for region in ("prompt", "completion")
            },
            "sft": {
                region: analyze_variant(sft_rows, labels, region, args.seed)
                for region in ("prompt", "completion")
            },
        },
        "raw_to_sft": {
            region: compare_variants(raw_rows, sft_rows, region)
            for region in ("prompt", "completion")
        },
    }
    summary_path = args.trace_dir / "summary.json"
    report_path = args.trace_dir / "report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
