#!/usr/bin/env python3
"""Solvability 공간(=누가 푸느냐)에 문제를 임베딩한 dual t-SNE.

텍스트 임베딩 공간에선 solvability가 소금후추(직교)였다. 여기선 반대로
solve-signature 자체를 좌표로 써서 임베딩 → 그 공간에서:
  (1) solvability 클러스터 = 깨끗 (native geometry)
  (2) n_solved(난이도) = 매끄러운 그라디언트
  (3) human prior = 무너짐(scramble)
  (4) 텍스트-임베딩 HDBSCAN 클러스터 = 무너짐(scramble)
→ solvability는 시맨틱과 별개로 실재하는 축임을 dual view로 증명.

Usage: python scripts/analyze_solvability_space.py --dataset qasc|lbox
결과: results/axes_analysis/solvability_space_<ds>.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
OUT = REPO / "results" / "axes_analysis"
sys.path.insert(0, str(REPO / "scripts"))
from embed_expert_viz import run_hdbscan, cluster_color_list, color_map, fold  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SEED = 42

CFG = {
    "qasc": dict(labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
                 src="export/qasc/qasc_train.jsonl", emb="results/embed_viz/qasc_emb.npy",
                 prior_tags="results/embed_viz/qasc_llm_tags.json", prior_fn=None,
                 sub=None, title="QASC"),
    "lbox": dict(labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
                 src="export/lbox/lbox_train.jsonl", emb="results/embed_viz/lbox_emb.npy",
                 prior_tags=None,
                 prior_fn=lambda r: f"{r.get('task_type')}·{r.get('casetype')}",
                 sub=15000, title="LBOX"),
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def scatter(ax, xy, labels, cmap, title, ps, order=None):
    for lab in (order or cmap):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=ps, c=cmap.get(lab, "#898781"), alpha=.8, linewidths=0)
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
    n = len(ids)

    prior = (np.array([json.load(open(REPO / c["prior_tags"], encoding="utf-8")).get(i, "?") for i in ids])
             if c["prior_tags"] else np.array([c["prior_fn"](src_by_id[i]) for i in ids]))
    S = np.array([[int(id2lab[i]["per_expert"].get(e, 0)) for e in specialists] for i in ids], float)
    n_solved = S.sum(1).astype(int)

    # 텍스트 임베딩 HDBSCAN(비교용 semantic 라벨)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    txt_xy = TSNE(2, perplexity=min(30, max(5, n // 20)), init="pca",
                  learning_rate="auto", random_state=SEED).fit_transform(
        PCA(min(50, Pn.shape[1]), random_state=SEED).fit_transform(Pn))
    emb_hdb, _ = run_hdbscan(txt_xy)

    kside = min(len(specialists), 10)
    solve_km = KMeans(kside, n_init=5, random_state=SEED).fit_predict(S)

    # ★ solve-signature 공간 t-SNE (동률 깨기 위해 미세 노이즈)
    rng = np.random.default_rng(SEED)
    Sj = S + rng.normal(0, 0.01, S.shape)
    sxy = TSNE(2, perplexity=min(30, max(5, n // 20)), init="pca",
               learning_rate="auto", random_state=SEED).fit_transform(Sj)

    nmi_prior = NMI(solve_km, prior)
    nmi_emb = NMI(solve_km, emb_hdb)
    print(f"{a.dataset}: NMI(solve-cluster, prior)={nmi_prior:.3f}, "
          f"NMI(solve-cluster, embedding)={nmi_emb:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = 4 if n > 3000 else 9

    fig, ax = plt.subplots(1, 4, figsize=(19, 5))
    fig.patch.set_facecolor(SURFACE)

    # (1) solvability 클러스터 = native, 깨끗
    skmap = {k: cluster_color_list(kside)[k] for k in range(kside)}
    scatter(ax[0], sxy, solve_km, skmap, "Solvability clusters\n(native — clean)", ps)

    # (2) n_solved 난이도 그라디언트
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(0, len(specialists))
    ax[1].scatter(sxy[:, 0], sxy[:, 1], s=ps, c=n_solved, cmap="viridis", norm=norm, linewidths=0)
    ax[1].set_title("Difficulty (n_solved)\nsmooth gradient", fontsize=12, color=INK)
    ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_facecolor(SURFACE)
    for s in ax[1].spines.values():
        s.set_visible(False)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax[1],
                 fraction=.046, pad=.02).set_label("# experts solving", fontsize=8)

    # (3) human prior = 무너짐 (train과 동일 색)
    pmap = color_map(list(prior))
    pcls = np.array(fold(list(prior), pmap))
    pmap2 = {cl: pmap.get(cl, "#898781") for cl in set(pcls)}
    scatter(ax[2], sxy, pcls, pmap2, f"Human prior — COLLAPSES\n(NMI {nmi_prior:.2f})", ps)

    # (4) 텍스트-임베딩 클러스터 = 무너짐
    hcls = np.array([f"c{v}" for v in emb_hdb])
    from collections import Counter
    hsorted = [cl for cl, _ in Counter(hcls).most_common()]
    hmap = dict(zip(hsorted, cluster_color_list(len(hsorted))))
    scatter(ax[3], sxy, hcls, hmap, f"Embedding clusters — COLLAPSE\n(NMI {nmi_emb:.2f})", ps)

    fig.suptitle(f"{c['title']} — problems embedded by WHO solves them "
                 f"(solvability space); semantic labels collapse here",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"solvability_space_{a.dataset}.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"done -> {OUT}/solvability_space_{a.dataset}.png")


if __name__ == "__main__":
    main()
