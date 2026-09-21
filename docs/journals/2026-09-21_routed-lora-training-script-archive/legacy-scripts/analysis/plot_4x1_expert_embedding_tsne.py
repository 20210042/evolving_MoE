#!/usr/bin/env python3
"""Visualize LBox/SNI semantic embeddings colored by 4x1 routing experts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expert_embedding_tsne"
COLORS = {"E0": "#2a78d6", "E1": "#008300", "E2": "#e87ba4", "E3": "#eda100"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_traces(directory: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(directory.glob("sft_rank*.jsonl")):
        for row in read_jsonl(path):
            rows[str(row["id"])] = row
    return rows


def dominant(row: dict, region: str) -> str:
    counts = np.asarray(row[f"{region}_route_counts"], dtype=np.int64)
    return f"E{int(counts.sum(axis=0).argmax())}"


def lbox_prior(row: dict) -> str:
    return "statute" if row["task_type"] == "statute" else f"casename_{row['casetype']}"


def top_classes(labels: list[str], keep: int = 11) -> list[str]:
    common = {label for label, _ in Counter(labels).most_common(keep)}
    return [label if label in common else "Other" for label in labels]


def embed_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = normalize(vectors.astype(np.float32), norm="l2")
    pca = PCA(n_components=50, random_state=42).fit_transform(vectors)
    xy = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=42,
    ).fit_transform(pca)
    return pca, xy


def scatter(ax, xy: np.ndarray, labels: list[str], title: str, fixed_colors=None) -> None:
    counts = Counter(labels)
    order = [key for key, _ in counts.most_common()]
    palette = plt.get_cmap("tab20")
    cmap = fixed_colors or {key: palette(index % 20) for index, key in enumerate(order)}
    for label in reversed(order):
        mask = np.asarray(labels) == label
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=4, alpha=0.46,
            c=[cmap[label]], linewidths=0, rasterized=True,
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=cmap[label], markeredgewidth=0,
               label=f"{label} ({counts[label]:,})")
        for label in order
    ]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="best", ncol=1)


def metrics(dataset: str, pca: np.ndarray, prompt: list[str], completion: list[str], prior: list[str]) -> dict:
    # Silhouette is quadratic; a deterministic 3k sample is sufficient for comparison.
    result = {}
    for name, labels in (("prompt_expert", prompt), ("completion_expert", completion), ("prior", prior)):
        unique = len(set(labels))
        result[name] = {
            "classes": unique,
            "counts": dict(Counter(labels)),
            "silhouette_pca50_cosine": float(
                silhouette_score(pca, labels, metric="cosine", sample_size=min(3000, len(labels)), random_state=42)
            ) if unique > 1 else None,
        }
    semantic_cluster = KMeans(n_clusters=4, n_init=20, random_state=42).fit_predict(pca)
    result["embedding_kmeans_k4_alignment"] = {
        "prompt_expert_nmi": float(normalized_mutual_info_score(prompt, semantic_cluster)),
        "prompt_expert_ari": float(adjusted_rand_score(prompt, semantic_cluster)),
        "completion_expert_nmi": float(normalized_mutual_info_score(completion, semantic_cluster)),
        "completion_expert_ari": float(adjusted_rand_score(completion, semantic_cluster)),
    }
    return {dataset: result}


def prepare(dataset: str):
    emb_dir = ROOT / "results" / "embed_viz_test"
    vectors = np.load(emb_dir / f"{dataset}_test_hs_mean.npy")
    ids = json.loads((emb_dir / f"{dataset}_test_hs_ids.json").read_text())
    if dataset == "lbox":
        source_path = ROOT / "export/lbox/lbox_test.jsonl"
        trace_dir = ROOT / "results/lbox_4x1_router_clustering_audit"
        prior_fn = lbox_prior
    else:
        source_path = ROOT / "data/sni_split/sni_test.jsonl"
        trace_dir = ROOT / "results/sni_4x1_switch_router_groups/annealed_aux_best/traces"
        prior_fn = lambda row: str(row["category"])
    source = {str(row["id"]): row for row in read_jsonl(source_path)}
    traces = load_traces(trace_dir)
    keep = [index for index, row_id in enumerate(ids) if row_id in source and row_id in traces]
    ids = [ids[index] for index in keep]
    vectors = vectors[keep]
    prompt = [dominant(traces[row_id], "prompt") for row_id in ids]
    completion = [dominant(traces[row_id], "completion") for row_id in ids]
    prior = [prior_fn(source[row_id]) for row_id in ids]
    display_prior = prior if dataset == "lbox" else top_classes(prior)
    return ids, vectors, prompt, completion, prior, display_prior


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    all_metrics = {}
    for row_index, dataset in enumerate(("lbox", "sni")):
        ids, vectors, prompt, completion, prior, display_prior = prepare(dataset)
        pca, xy = embed_2d(vectors)
        scatter(axes[row_index, 0], xy, prompt, f"{dataset.upper()} · prompt dominant expert", COLORS)
        scatter(axes[row_index, 1], xy, completion, f"{dataset.upper()} · completion dominant expert", COLORS)
        prior_title = "task prior" if dataset == "lbox" else "SNI category (top 11 + Other)"
        scatter(axes[row_index, 2], xy, display_prior, f"{dataset.upper()} · {prior_title}")
        frame = pd.DataFrame({
            "id": ids, "tsne_x": xy[:, 0], "tsne_y": xy[:, 1],
            "prompt_dominant_expert": prompt,
            "completion_dominant_expert": completion,
            "prior_label": prior,
        })
        frame.to_csv(OUT / f"{dataset}_test_tsne_points.csv", index=False)
        all_metrics.update(metrics(dataset, pca, prompt, completion, prior))

    fig.suptitle(
        "4×1 MoE routing over dense-Llama semantic embedding space\n"
        "PCA(50) → t-SNE; coordinates are computed independently per dataset",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(OUT / "lbox_sni_4x1_expert_tsne.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "lbox_sni_4x1_expert_tsne.pdf", bbox_inches="tight")
    (OUT / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved plots, points, and metrics under {OUT}")


if __name__ == "__main__":
    main()
