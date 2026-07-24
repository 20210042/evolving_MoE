#!/usr/bin/env python3
"""Embedding-space comparison of three problem partitions, per dataset.

Two stages (run separately: embedding is one-off on GPU, analysis iterates on CPU):

  --stage embed    embed FULL datasets (no subsampling) with
                   google/embeddinggemma-300m and save
                   results/embed_viz/<ds>_emb.npy + <ds>_ids.json.
                   Cache is keyed on the exact id list, so reruns skip cleanly.

  --stage analyze  load cached embeddings -> PCA(50) -> t-SNE(2D), colored 3 ways:
                     (1) HDBSCAN clusters on the embedding (granularity swept)
                     (2) human-prior metadata tags shipped with the dataset
                     (3) expert-solve classes from our binning labels
                         (all-fail = dark gray, solved-by-(almost)-everyone =
                          light gray, else the rarest solver among its solvers)
                   plus per-expert coverage facets and ARI/NMI -> metrics.json.
                   LBOX is subsampled here (seeded) for t-SNE readability only.

Notes
  - TACO expert labels exist only for the 500-problem test binning
    (full-train labeling pass unfinished); other points are unlabeled.
  - QASC ships no human-prior tag; its panel (2) is annotated as N/A.

Usage
  python scripts/embed_expert_viz.py --stage embed [--batch 512]
  python scripts/embed_expert_viz.py --stage analyze
Outputs under results/embed_viz/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "embed_viz"
EMBED_MODEL = "google/embeddinggemma-300m"
SEED = 42

# dataviz reference palette (light mode, fixed slot order — do not reorder).
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
GRAY_OTHER = "#898781"      # folded classes / HDBSCAN noise
GRAY_ALLFAIL = "#52514e"    # expert panel: nobody solved
GRAY_COMMON = "#c3c2b7"     # expert panel: (almost) everyone solved
GRAY_NOLABEL = "#ececea"    # expert panel: no label available (taco train)
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

_QASC_TAGS: dict | None = None


def qasc_llm_prior(r: dict) -> str | None:
    """scripts/tag_qasc_topics.py가 만든 LLM 태그(없으면 None → 패널 N/A)."""
    global _QASC_TAGS
    if _QASC_TAGS is None:
        p = OUT / "qasc_llm_tags.json"
        _QASC_TAGS = json.load(open(p, encoding="utf-8")) if p.is_file() else {}
    return _QASC_TAGS.get(str(r["id"]))


DATASETS = {
    "qasc": dict(
        source="export/qasc/qasc_train.jsonl",
        labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
        names="export/qasc_binning_seed20210211/agent_mapping.json",
        prior_fn=qasc_llm_prior,
        prior_title="LLM-tagged subject (gemma; no human prior exists)",
        tsne_subsample=None,
    ),
    "taco": dict(
        source="export/acc_selfconsistent/acc_train.jsonl",
        labels="results/acc/seed20210101/inference_test_binning_final.binned.jsonl",
        names="results/acc/seed20210111/roster_final.json",
        prior_fn=lambda r: r.get("main_critic_category") or None,
        tsne_subsample=None,
    ),
    "lbox": dict(
        source="export/lbox/lbox_train.jsonl",
        labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
        names="export/lbox_binning_seed20210311/agent_mapping.json",
        prior_fn=lambda r: f"{r.get('task_type')}·{r.get('casetype')}",
        tsne_subsample=15000,
    ),
}


def load_expert_names(rel_path: str | None) -> dict[str, str]:
    """codename -> 사람이 읽는 expert 이름. agent_mapping.json({id:{name}}) 또는
    roster_final.json([{id,name}]) 지원. 없으면 코드네임 그대로."""
    if not rel_path:
        return {}
    p = REPO / rel_path
    if not p.is_file():
        return {}
    data = json.load(open(p, encoding="utf-8"))
    if isinstance(data, dict):
        return {k: (v.get("name") or k) for k, v in data.items()}
    if isinstance(data, list):
        return {a["id"]: (a.get("name") or a["id"]) for a in data}
    return {}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def paths_for(name: str) -> tuple[Path, Path]:
    return OUT / f"{name}_emb.npy", OUT / f"{name}_ids.json"


# ---------------------------------------------------------------- embed stage

def encode_with_oom_fallback(model, texts: list[str], batch: int) -> np.ndarray:
    import torch

    kw = {}
    if getattr(model, "prompts", None) and "Clustering" in model.prompts:
        kw["prompt_name"] = "Clustering"
    while True:
        try:
            emb = model.encode(texts, batch_size=batch, show_progress_bar=True,
                               normalize_embeddings=True, **kw)
            return np.asarray(emb, dtype=np.float32)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 16:
                raise
            batch //= 2
            print(f"  CUDA OOM → batch {batch}로 재시도")


def embed_stage(names: list[str], batch: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    model = None
    for name in names:
        rows = load_jsonl(REPO / DATASETS[name]["source"])
        ids = [str(r["id"]) for r in rows]
        emb_p, ids_p = paths_for(name)
        if emb_p.is_file() and ids_p.is_file() and json.load(open(ids_p)) == ids:
            print(f"[{name}] 임베딩 캐시 유효 ({len(ids):,}) — 건너뜀")
            continue
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 드라이버 불일치 등으로 CUDA 초기화가 실패하면 조용히 CPU로 떨어져
            # 큰 배치의 어텐션이 호스트 램을 터뜨린다(n01 A6000에서 실측). 명시적으로 죽인다.
            if device == "cpu" and batch > 32:
                raise RuntimeError(
                    "CUDA 사용 불가(드라이버 확인) — GPU 노드에서 돌리거나 --batch 32 이하로 명시하세요.")
            model = SentenceTransformer(EMBED_MODEL, device=device)
            print(f"임베딩 모델 로드: {EMBED_MODEL} on {device}, batch={batch}")
        print(f"[{name}] {len(rows):,}개 임베딩 시작")
        emb = encode_with_oom_fallback(model, [r["instruction"][:6000] for r in rows], batch)
        np.save(emb_p, emb)
        json.dump(ids, open(ids_p, "w"))
        print(f"[{name}] 저장: {emb_p} {emb.shape}")


# -------------------------------------------------------------- analyze stage

def run_hdbscan(xy: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Sweep min_cluster_size; keep the run with 6..24 clusters and least noise.

    Clusters on the 2D t-SNE plane, not the PCA(50) space: high-dim density
    estimation left QASC/TACO mostly noise (46-79%), and the comparison is
    visual anyway — clusters should match the blobs the reader sees.
    """
    import hdbscan

    n = len(xy)
    sweep, results = [], {}
    grid = [m for m in (15, 25, 50, 100, 200, 400) if m < n // 10] or [max(15, n // 200)]
    # eom은 큰 덩어리를 선호해 매크로 블랍만 남기기도 함(TACO 3개) → leaf도 후보에 포함
    for method in ("eom", "leaf"):
        for mcs in grid:
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10,
                                  cluster_selection_method=method).fit_predict(xy)
            k = len(set(lab)) - (1 if -1 in lab else 0)
            noise = float((lab == -1).mean())
            sweep.append(dict(method=method, min_cluster_size=mcs,
                              n_clusters=k, noise_frac=round(noise, 3)))
            results[(method, mcs)] = lab
            print(f"  hdbscan {method} mcs={mcs}: {k} clusters, noise {noise:.1%}")
    ok = [s for s in sweep if 6 <= s["n_clusters"] <= 24 and s["noise_frac"] <= 0.35]
    if ok:
        best = min(ok, key=lambda s: s["noise_frac"])
    else:
        best = min(sweep, key=lambda s: (abs(s["n_clusters"] - 12), s["noise_frac"]))
    best["chosen"] = True
    lab = assign_noise_to_nearest(results[(best["method"], best["min_cluster_size"])], xy)
    return lab, sweep


def assign_noise_to_nearest(lab: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """HDBSCAN 노이즈(-1)를 최근접 클러스터 센트로이드로 배정해 노이즈를 없앤다."""
    lab = np.asarray(lab).copy()
    ks = sorted(set(lab.tolist()) - {-1})
    m = lab == -1
    if not ks or not m.any():
        return lab
    cents = np.stack([xy[lab == k].mean(axis=0) for k in ks])
    d = ((xy[m, None, :] - cents[None, :, :]) ** 2).sum(-1)
    lab[m] = np.asarray(ks)[d.argmin(1)]
    return lab


def expert_classes(rows: list[dict], labels_by_id: dict, n_experts: int,
                   solve_counts: dict) -> list[str]:
    out = []
    for r in rows:
        lab = labels_by_id.get(str(r["id"]))
        if lab is None:
            out.append("(no label)")
        elif lab["n_solved"] == 0:
            out.append("(all fail)")
        elif lab["n_solved"] >= n_experts - 1:
            out.append("(all solved)")
        else:
            out.append(min(lab["solved_by"], key=lambda a: solve_counts[a]))
    return out


# specials: (color, alpha) — train 그림에선 (all fail)/(all solved) 둘 다 배경(회색).
# all-pass/all-wrong 강조 구별은 test set 그림에서만(재훈 결정). 특화 클래스는 각자 색.
SPECIAL_STYLE = {"(all fail)": (GRAY_COMMON, 0.18), "(all solved)": (GRAY_COMMON, 0.18),
                 "(no label)": (GRAY_NOLABEL, 0.15), "(noise)": (GRAY_OTHER, 0.5),
                 "(other)": (GRAY_OTHER, 0.6), "(n/a)": (GRAY_NOLABEL, 0.15)}
SPECIAL_COLORS = {k: v[0] for k, v in SPECIAL_STYLE.items()}


def cluster_color_list(k: int) -> list[str]:
    """8색 팔레트를 명도 변형(×0.65, ×1.35)으로 확장해 k개 클러스터 전부에 색을 준다.
    식별은 센트로이드 직접 라벨이 담당하고, 색은 인접 영역 구분용."""
    import matplotlib.colors as mc
    out = []
    for i in range(k):
        r, g, b = mc.to_rgb(PALETTE[i % len(PALETTE)])
        cycle = i // len(PALETTE)
        if cycle % 3 == 1:
            r, g, b = r * 0.6, g * 0.6, b * 0.6
        elif cycle % 3 == 2:
            r, g, b = min(1, 0.45 + r * 0.55), min(1, 0.45 + g * 0.55), min(1, 0.45 + b * 0.55)
        out.append(mc.to_hex((r, g, b)))
    return out


def color_map(classes: list[str]) -> dict[str, str]:
    """Top-8 non-special classes by size take palette slots in size order; rest fold."""
    from collections import Counter
    counts = Counter(c for c in classes if c not in SPECIAL_COLORS)
    cmap = dict(SPECIAL_COLORS)
    for i, (cls, _) in enumerate(counts.most_common()):
        cmap[cls] = PALETTE[i] if i < len(PALETTE) else GRAY_OTHER
    return cmap


def fold(classes: list[str], cmap: dict[str, str]) -> list[str]:
    return [c if cmap.get(c) != GRAY_OTHER or c == "(noise)" else "(other)"
            for c in classes]


def draw_panel(ax, xy, classes, cmap, title, point_size, centroid_labels=True):
    import matplotlib.patheffects as pe
    from collections import Counter

    counts = Counter(classes)
    # specials first (background), colored classes on top
    order = sorted(counts, key=lambda c: (c not in SPECIAL_COLORS, -counts[c]))
    for cls in order:
        m = np.array([c == cls for c in classes])
        color, alpha = SPECIAL_STYLE.get(cls, (cmap.get(cls, GRAY_OTHER), 0.75))
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, linewidths=0,
                   c=color, alpha=alpha,
                   label=f"{cls} ({counts[cls]:,})")
    # direct labels at class medians (colored classes only)
    for cls in (counts if centroid_labels else ()):
        if cls in SPECIAL_COLORS or cmap.get(cls) == GRAY_OTHER:
            continue
        m = np.array([c == cls for c in classes])
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax.text(cx, cy, cls[:22], fontsize=7, color=INK, ha="center",
                path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])
    ax.set_title(title, fontsize=12, color=INK)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8,
              ncol=3 if len(counts) > 12 else 2, frameon=False,
              markerscale=2.2, labelcolor=INK_2)


def agreement(a: list[str], b: list[str], drop_a=(), drop_b=()) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    keep = [i for i in range(len(a)) if a[i] not in drop_a and b[i] not in drop_b]
    if len(keep) < 50:
        return dict(n=len(keep), ari=None, nmi=None)
    aa, bb = [a[i] for i in keep], [b[i] for i in keep]
    return dict(n=len(keep),
                ari=round(adjusted_rand_score(aa, bb), 4),
                nmi=round(normalized_mutual_info_score(aa, bb), 4))


def analyze_one(name: str, cfg: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"=== {name} ===")
    rows = load_jsonl(REPO / cfg["source"])
    emb_p, ids_p = paths_for(name)
    if not (emb_p.is_file() and ids_p.is_file()):
        raise FileNotFoundError(f"{emb_p} 없음 — 먼저 --stage embed를 돌리세요.")
    emb = np.load(emb_p)
    ids = json.load(open(ids_p))
    assert ids == [str(r["id"]) for r in rows], f"{name}: 캐시 id와 소스 불일치 — 재임베딩 필요"

    sub = cfg["tsne_subsample"]
    if sub and len(rows) > sub:
        keep = np.sort(np.random.default_rng(SEED).choice(len(rows), sub, replace=False))
        rows = [rows[i] for i in keep]
        emb = emb[keep]
        print(f"  t-SNE용 서브샘플 {len(rows):,}")

    x50 = PCA(n_components=min(50, len(rows) - 1, emb.shape[1]),
              random_state=SEED).fit_transform(emb)
    xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
              random_state=SEED, verbose=0).fit_transform(x50)

    # (1) hdbscan
    hlab, sweep = run_hdbscan(xy)
    hcls = [f"c{v}" if v >= 0 else "(noise)" for v in hlab]

    # (2) human prior
    if cfg["prior_fn"] is None:
        pcls = ["(n/a)"] * len(rows)
    else:
        pcls = [cfg["prior_fn"](r) or "(n/a)" for r in rows]

    # (3) expert-solve
    labels = load_jsonl(REPO / cfg["labels"])
    labels_by_id = {str(r["id"]): r for r in labels}
    experts = sorted({e for r in labels for e in r["per_expert"]})
    solve_counts = {e: sum(r["per_expert"].get(e, 0) for r in labels) for e in experts}
    id_to_name = load_expert_names(cfg.get("names"))
    # codename → 이름 표시(특수 클래스 (all fail) 등은 그대로 통과). 1:1 매핑이라 ARI 불변.
    ecls = [id_to_name.get(c, c) for c in
            expert_classes(rows, labels_by_id, len(experts), solve_counts)]
    # facet용 expert별 고유색(experts 정렬 고정) — all fail/solve 맵과 색 통일 기반.
    expert_color = dict(zip(experts, cluster_color_list(len(experts))))

    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ps = 8 if len(rows) <= 3000 else 4

    # (1) 클러스터 패널: 폴딩 없이 전 클러스터 색칠, 식별은 센트로이드 라벨
    csorted = [c for c, _ in Counter(hcls).most_common() if c not in SPECIAL_STYLE]
    ccmap = dict(zip(csorted, cluster_color_list(len(csorted))))
    draw_panel(axes[0], xy, hcls, ccmap, "HDBSCAN clusters (embedding)", ps)

    # (2) prior 패널: 태그가 없으면 명시적으로 표기
    prior_title = cfg.get("prior_title", "Human prior tags")
    if all(c == "(n/a)" for c in pcls):
        draw_panel(axes[1], xy, pcls, {}, prior_title, ps)
        axes[1].text(0.5, 0.5, "no prior tags available",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=13, color=INK_2)
    else:
        pmap = color_map(pcls)
        draw_panel(axes[1], xy, fold(pcls, pmap), pmap, prior_title, ps)

    # (3) expert 패널: all fail/common은 반투명 배경, 특화 클래스만 색
    emap = color_map(ecls)
    draw_panel(axes[2], xy, fold(ecls, emap), emap, "Expert-solve classes (ours)", ps,
               centroid_labels=False)
    fig.suptitle(f"{name.upper()} — {len(rows):,} problems · {EMBED_MODEL} + t-SNE",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{name}_panels.png", dpi=170,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # per-expert facets
    cols = 4
    rows_n = (len(experts) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 3.2 * rows_n))
    fig.patch.set_facecolor(SURFACE)
    # common(거의 전원이 푸는 쉬운 문제, n_solved>=N-1) 제외 → 각 expert의 고유 solve만 표시
    n_exp = len(experts)
    ns = np.array([labels_by_id.get(str(r["id"]), {}).get("n_solved", 0) for r in rows])
    not_common = ns < n_exp - 1
    for ax, e in zip(np.ravel(axes), experts):
        solved_e = np.array([bool(labels_by_id.get(str(r["id"]), {}).get("per_expert", {}).get(e, 0))
                             for r in rows])
        m = solved_e & not_common
        ax.scatter(xy[~m, 0], xy[~m, 1], s=3, c=GRAY_NOLABEL, linewidths=0)
        ax.scatter(xy[m, 0], xy[m, 1], s=4, c=expert_color[e], alpha=0.85, linewidths=0)
        ax.set_title(id_to_name.get(e, e)[:28], fontsize=11, color=INK)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in np.ravel(axes)[len(experts):]:
        ax.axis("off")
    fig.suptitle(f"{name.upper()} — per-expert solved coverage on the same t-SNE map",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{name}_expert_facets.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    drop_e_specialist = ("(no label)", "(all fail)", "(all solved)")
    metrics = dict(
        n=len(rows), n_experts=len(experts), hdbscan_sweep=sweep,
        cluster_vs_prior=agreement(hcls, pcls, ("(noise)",), ("(n/a)",)),
        cluster_vs_expert=agreement(hcls, ecls, ("(noise)",), ("(no label)",)),
        prior_vs_expert=agreement(pcls, ecls, ("(n/a)",), ("(no label)",)),
        cluster_vs_expert_specialist=agreement(hcls, ecls, ("(noise)",), drop_e_specialist),
        prior_vs_expert_specialist=agreement(pcls, ecls, ("(n/a)",), drop_e_specialist),
    )
    print(f"  metrics: {json.dumps(metrics, ensure_ascii=False)[:400]}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "analyze", "all"], default="all")
    ap.add_argument("--datasets", nargs="+", default=["qasc", "taco", "lbox"],
                    choices=list(DATASETS))
    ap.add_argument("--batch", type=int, default=256,
                    help="임베딩 배치 크기 (OOM 시 자동 반감)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("embed", "all"):
        embed_stage(a.datasets, a.batch)
    if a.stage in ("analyze", "all"):
        all_metrics = {name: analyze_one(name, DATASETS[name]) for name in a.datasets}
        json.dump(all_metrics, open(OUT / "metrics.json", "w"),
                  ensure_ascii=False, indent=2)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
