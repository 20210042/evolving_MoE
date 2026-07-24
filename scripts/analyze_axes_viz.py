#!/usr/bin/env python3
"""analyze_axes.py의 두 분석을 그림으로.

fig1 (semantic routing): [프롬프트가 약속하는 배정 | 실제 best solver | solve-rate 막대]
fig2 (solvability):      [임베딩 기하 | human prior | solvability 클러스터 | ARI 히트맵]
결과: results/axes_analysis/
Usage: python scripts/analyze_axes_viz.py --dataset qasc|lbox
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
OUT = REPO / "results" / "axes_analysis"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GRAY = "#d0cfca"
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a",
           "#eb6834", "#4a3aa7", "#e34948"]
SEED = 42

CFG = {
    "qasc": dict(labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
                 mapping="export/qasc_binning_seed20210211/agent_mapping.json",
                 src="export/qasc/qasc_train.jsonl", emb="results/embed_viz/qasc_emb.npy",
                 prior_tags="results/embed_viz/qasc_llm_tags.json", prior_fn=None,
                 sub=None, title="QASC"),
    "lbox": dict(labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
                 mapping="export/lbox_binning_seed20210311/agent_mapping.json",
                 src="export/lbox/lbox_train.jsonl", emb="results/embed_viz/lbox_emb.npy",
                 prior_tags=None,
                 prior_fn=lambda r: f"{r.get('task_type')}·{r.get('casetype')}",
                 sub=15000, title="LBOX"),
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ccolor(i):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(PALETTE[i % len(PALETTE)])
    cyc = i // len(PALETTE)
    if cyc % 3 == 1:
        r, g, b = r * .6, g * .6, b * .6
    elif cyc % 3 == 2:
        r, g, b = min(1, .45 + r * .55), min(1, .45 + g * .55), min(1, .45 + b * .55)
    return mc.to_hex((r, g, b))


def scatter_by(ax, xy, labels, palette_map, title, ps, bg=None):
    for lab, col in palette_map.items():
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=ps, c=col, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=12, color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), default="qasc")
    a = ap.parse_args()
    c = CFG[a.dataset]
    OUT.mkdir(parents=True, exist_ok=True)

    labels = load_jsonl(REPO / c["labels"])
    id2lab = {str(r["id"]): r for r in labels}
    mapping = json.load(open(REPO / c["mapping"], encoding="utf-8"))
    experts = sorted({e for r in labels for e in r["per_expert"]})
    specialists = [e for e in experts if e != "luca"]

    src = load_jsonl(REPO / c["src"])
    src_ids = [str(r["id"]) for r in src]
    src_by_id = {str(r["id"]): r for r in src}
    emb = np.load(REPO / c["emb"])
    keep = [i for i, pid in enumerate(src_ids) if pid in id2lab]
    ids = [src_ids[i] for i in keep]
    P = emb[keep]
    if c["sub"] and len(ids) > c["sub"]:
        sel = np.sort(np.random.default_rng(SEED).choice(len(ids), c["sub"], replace=False))
        ids = [ids[i] for i in sel]; P = P[sel]
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    n = len(ids)

    prior = (np.array([json.load(open(REPO / c["prior_tags"], encoding="utf-8")).get(i, "?") for i in ids])
             if c["prior_tags"] else np.array([c["prior_fn"](src_by_id[i]) for i in ids]))
    S = np.array([[int(id2lab[i]["per_expert"].get(e, 0)) for e in specialists] for i in ids])
    n_solved = S.sum(1)
    per_rate = {k: S[:, k].mean() for k in range(len(specialists))}

    # prompt centroids
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
    kw = {"prompt_name": "Clustering"} if getattr(model, "prompts", None) and "Clustering" in model.prompts else {}
    proto = [mapping[e].get("system_prompt", "") or e for e in specialists]
    C = np.asarray(model.encode(proto, normalize_embeddings=True, **kw), dtype=np.float32)

    route = (P @ C.T).argmax(1)                       # 프롬프트가 약속하는 배정
    best = np.array([int(np.argmax([S[p, k] / (per_rate[k] + 1e-9) if S[p, k] else -1
                                    for k in range(len(specialists))])) if n_solved[p] else -1
                     for p in range(n)])               # 실제 best solver(-1=all fail)

    x50 = PCA(min(50, P.shape[1]), random_state=SEED).fit_transform(P)
    xy = TSNE(2, perplexity=min(30, max(5, n // 20)), init="pca",
              learning_rate="auto", random_state=SEED).fit_transform(x50)
    kside = min(len(specialists), 10)
    # 임베딩 클러스터 = train 패널과 동일한 HDBSCAN(t-SNE 평면) 재사용
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from embed_expert_viz import run_hdbscan
    emb_hdb, _ = run_hdbscan(xy)
    solve_km = KMeans(kside, n_init=5, random_state=SEED).fit_predict(S.astype(float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = 4 if n > 3000 else 9
    ecol = {k: ccolor(k) for k in range(len(specialists))}

    # ---------- fig1: semantic routing ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(SURFACE)
    scatter_by(ax[0], xy, route, ecol,
               "Prompt promises\n(nearest specialist-prompt)", ps)
    # best solver: all-fail 회색
    ax[1].scatter(xy[best == -1, 0], xy[best == -1, 1], s=ps, c=GRAY, alpha=.5, linewidths=0)
    for k in range(len(specialists)):
        m = best == k
        ax[1].scatter(xy[m, 0], xy[m, 1], s=ps, c=ecol[k], alpha=.8, linewidths=0)
    ax[1].set_title("Reality\n(specialist who actually solves it)", fontsize=12, color=INK)
    ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_facecolor(SURFACE)
    for s in ax[1].spines.values():
        s.set_visible(False)
    # bar
    sem = np.mean([S[p, route[p]] for p in range(n)]) * 100
    rnd = np.mean(list(per_rate.values())) * 100
    bst = max(per_rate.values()) * 100
    orc = (n_solved > 0).mean() * 100
    names = ["semantic\nrouting", "random\nrouting", "best single\nexpert", "oracle\n(union)"]
    vals = [sem, rnd, bst, orc]
    cols = ["#e34948" if sem < rnd else "#2a78d6", "#898781", "#4a3aa7", "#008300"]
    b = ax[2].bar(names, vals, color=cols, width=.66)
    ax[2].bar_label(b, fmt="%.1f%%", fontsize=10, color=INK)
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("solve-rate (%)", fontsize=10)
    ax[2].set_title(f"Does the promise solve?\nsemantic {sem:.1f} vs random {rnd:.1f}",
                    fontsize=12, color=INK)
    ax[2].spines[["top", "right"]].set_visible(False)
    ax[2].tick_params(labelsize=9)
    fig.suptitle(f"{c['title']} — Analysis 1: prompt semantics ≠ actual solving",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"analysis1_semantic_routing_{a.dataset}.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------- fig2: solvability ----------
    # 원본 train 패널과 동일한 팔레트 로직 재사용(연속성)
    from collections import Counter
    from embed_expert_viz import cluster_color_list, color_map, fold, SPECIAL_STYLE
    fig, ax = plt.subplots(1, 4, figsize=(19, 5))
    fig.patch.set_facecolor(SURFACE)
    # 패널0 HDBSCAN: 크기순 cluster_color_list (train HDBSCAN 패널과 동일)
    hcls = np.array([f"c{v}" for v in emb_hdb])
    hsorted = [c for c, _ in Counter(hcls).most_common()]
    hmap = dict(zip(hsorted, cluster_color_list(len(hsorted))))
    scatter_by(ax[0], xy, hcls, hmap, "Embedding geometry\n(HDBSCAN clusters)", ps)
    # 패널1 human prior: color_map (train prior 패널과 동일 → 같은 태그 같은 색)
    pmap = color_map(list(prior))
    pcls = np.array(fold(list(prior), pmap))
    pmap2 = {c: pmap.get(c, "#898781") for c in set(pcls)}
    scatter_by(ax[1], xy, pcls, pmap2, "Human prior", ps)
    # 패널2 solvability: 그대로(재훈 OK)
    scatter_by(ax[2], xy, solve_km, {k: ccolor(k) for k in range(kside)},
               "Solvability clusters (ours)\n(solve-signature kmeans)", ps)
    # ARI heatmap
    axes_lab = ["human\nprior", "embedding", "solvability"]
    parts = [prior, emb_hdb, solve_km]
    M = np.array([[ARI(parts[i], parts[j]) for j in range(3)] for i in range(3)])
    im = ax[3].imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax[3].set_xticks(range(3)); ax[3].set_yticks(range(3))
    ax[3].set_xticklabels(axes_lab, fontsize=9); ax[3].set_yticklabels(axes_lab, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax[3].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="#fff" if M[i, j] > .5 else INK, fontsize=11)
    ax[3].set_title("ARI between axes\n(1=aligned, 0=orthogonal)", fontsize=12, color=INK)
    fig.suptitle(f"{c['title']} — Analysis 2: solvability is a real axis, orthogonal to semantics",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"analysis2_solvability_{a.dataset}.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"done -> {OUT}  (semantic {sem:.1f} vs random {rnd:.1f}; "
          f"ARI prior-emb {M[0,1]:.2f}, solve-emb {M[1,2]:.2f}, solve-prior {M[0,2]:.2f})")


if __name__ == "__main__":
    main()
