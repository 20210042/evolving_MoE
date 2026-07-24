#!/usr/bin/env python3
"""TEST-set tSNE (train과 별개 디렉토리 results/embed_viz_test/).

(a) coverage 4패널: LUCA / human prior roster / evolved roster / fine-tuned 가
    test 문제를 각각 얼마나 푸는지(검정=푼 문제) 나란히 비교.
(b) our roster expert facet: evolved roster가 test에서 뭘 푸나 (common 제외, 실명·고유색).
(c) human prior expert facet: prior binning(per_expert) 결과가 있으면 prior 3명 버전도 생성.

사용: python scripts/embed_test_viz.py --dataset lbox|qasc
  - lbox: test 8,203 (fine-tuned=종빈 ckpt-12000, prior=종빈 zip)
  - qasc: validation 926 (test 라벨 비공개라 validation이 test 역할; fine-tuned 대기)
결과 파일이 없는 solver는 pending 패널로 표시되고, 생기면 자동 포함.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
OUT = REPO / "results" / "embed_viz_test"
EMBED_MODEL = "google/embeddinggemma-300m"

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a",
           "#eb6834", "#4a3aa7", "#e34948"]

DS = {
    "lbox": dict(
        # fine-tuned 파일이 행 순서 기준(임베딩이 이 순서로 생성됨)
        order_file="lbox_sft_llama3_finetuned_lbox_baseline_eval500_full_eval_snapshot_checkpoint-12000_baseline_208278.jsonl",
        src="export/lbox/lbox_test.jsonl",
        emb="results/embed_viz/lbox_test_emb.npy",
        binning="results/lbox/seed20210311/binning_test_test.binned.jsonl",
        luca_file="results/lbox/seed20210311/lbox_gemma-4-26B-A4B-it_208891.jsonl",
        prior="results/lbox_human_prior/inference_test_human_prior.jsonl",
        prior_binning="results/lbox/seed20210311/binning_test_hp.binned.jsonl",
        prior_names="results/lbox_human_prior/lbox_human_prior_roster.json",
        ft="lbox_sft_llama3_finetuned_lbox_baseline_eval500_full_eval_snapshot_checkpoint-12000_baseline_208278.jsonl",
        names="export/lbox_binning_seed20210311/agent_mapping.json",
        prefix="lbox_test", title="LBOX test",
    ),
    "qasc": dict(
        order_file=None,  # validation 소스 순서 기준
        src="export/qasc/qasc_validation.jsonl",
        emb="results/embed_viz_test/qasc_val_emb.npy",
        binning="results/qasc/seed20210211/inference_validation_binning_final.binned.jsonl",
        luca_binning="luca",  # LUCA = binning per_expert['luca'] (baseline GEN 조건)
        prior="results/qasc_human_prior/inference_validation_human_prior.jsonl",
        prior_binning=None,
        prior_names=None,
        ft="qasc_sft_llama3_finetuned_qasc_baseline_eval300_ep9_baseline_208314.jsonl",
        names="export/qasc_binning_seed20210211/agent_mapping.json",
        prior_label="human prior roster (LLM-tagged)",
        prior_note="* QASC ships no human-authored prior — the 8-subject taxonomy was LLM-tagged (gemma), then used as persona roster.",
        prefix="qasc_val", title="QASC validation (test role)",
    ),
}


def pass_map(path: Path) -> dict:
    """id, pass_score 형식 → {id: bool}."""
    m = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        m[str(r["id"])] = float(r.get("pass_score", 0)) > 0
    return m


def prior_pass_map(path: Path, src_path: Path) -> dict:
    """routed 파이프라인 출력(final_output만) → 우리 채점기로 행별 pass 계산."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from evaluation.scorer import score_one
    gt = {str(r["id"]): r for r in
          (json.loads(l) for l in open(src_path, encoding="utf-8"))}
    m = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        item = gt.get(str(r["id"]))
        if item is not None:
            m[str(r["id"])] = float(score_one(item, str(r.get("final_output", "")))) > 0
    return m


def get_embeddings(texts: list[str], emb_p: Path) -> np.ndarray:
    if emb_p.is_file():
        emb = np.load(emb_p)
        if len(emb) == len(texts):
            return emb
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"임베딩 {len(texts):,}개 on {device}")
    emb = np.asarray(model.encode([t[:6000] for t in texts],
                                  batch_size=256 if device == "cuda" else 16,
                                  show_progress_bar=True, normalize_embeddings=True,
                                  prompt_name="Clustering"), dtype=np.float32)
    emb_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_p, emb)
    return emb


def cluster_color(i: int) -> str:
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(PALETTE[i % len(PALETTE)])
    cyc = i // len(PALETTE)
    if cyc % 3 == 1:
        r, g, b = r * 0.6, g * 0.6, b * 0.6
    elif cyc % 3 == 2:
        r, g, b = min(1, .45 + r * .55), min(1, .45 + g * .55), min(1, .45 + b * .55)
    return mc.to_hex((r, g, b))


def expert_facets(xy, ids, binning, names, title, out_png, exclude_common=True):
    import matplotlib.pyplot as plt
    experts = sorted({e for r in binning.values() for e in r["per_expert"]})
    n_exp = len(experts)
    ns = np.array([binning.get(i, {}).get("n_solved", 0) for i in ids])
    not_common = ns < (n_exp - 1 if exclude_common and n_exp > 2 else 10**9)
    cols = min(4, n_exp)
    rows_n = (n_exp + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 3.2 * rows_n))
    fig.patch.set_facecolor(SURFACE)
    for k, (ax, e) in enumerate(zip(np.ravel([axes]), experts)):
        solved = np.array([bool(binning.get(i, {}).get("per_expert", {}).get(e, 0)) for i in ids])
        m = solved & not_common
        ax.scatter(xy[~m, 0], xy[~m, 1], s=3, c="#ececea", linewidths=0)
        ax.scatter(xy[m, 0], xy[m, 1], s=4, c=cluster_color(k), alpha=0.85, linewidths=0)
        ax.set_title(names.get(e, e)[:28], fontsize=11, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in np.ravel([axes])[n_exp:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if rows_n == 1 else 0.95))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DS), default="lbox")
    a = ap.parse_args()
    cfg = DS[a.dataset]
    OUT.mkdir(parents=True, exist_ok=True)

    src_rows = [json.loads(l) for l in open(REPO / cfg["src"], encoding="utf-8")]
    src_by_id = {str(r["id"]): r for r in src_rows}
    if cfg["order_file"]:
        ids = [str(json.loads(l)["id"]) for l in open(REPO / cfg["order_file"], encoding="utf-8")]
    else:
        ids = [str(r["id"]) for r in src_rows]
    emb = get_embeddings([src_by_id[i]["instruction"] for i in ids], REPO / cfg["emb"])
    assert len(emb) == len(ids)

    binning = {str(json.loads(l)["id"]): json.loads(l)
               for l in open(REPO / cfg["binning"], encoding="utf-8")}

    # solver 4개 고정 순서. 없으면 None → pending 패널.
    SOLVERS = {}
    if cfg.get("luca_binning"):
        e = cfg["luca_binning"]
        SOLVERS["LUCA"] = {i: bool(r["per_expert"].get(e, 0)) for i, r in binning.items()}
    elif cfg.get("luca_file") and (REPO / cfg["luca_file"]).is_file():
        SOLVERS["LUCA"] = pass_map(REPO / cfg["luca_file"])
    else:
        SOLVERS["LUCA"] = None
    hp = REPO / cfg["prior"]
    prior_key = cfg.get("prior_label", "human prior roster")
    SOLVERS[prior_key] = prior_pass_map(hp, REPO / cfg["src"]) if hp.is_file() else None
    # evolved union (LUCA가 로스터 멤버면 제외 — LUCA는 첫 패널에 이미 있음)
    luca_member = cfg.get("luca_binning")
    SOLVERS["evolved roster"] = {
        i: any(v for e, v in r["per_expert"].items() if e != luca_member)
        for i, r in binning.items()}
    ft = (REPO / cfg["ft"]) if cfg.get("ft") else None
    SOLVERS["fine-tuned"] = pass_map(ft) if ft and ft.is_file() else None
    print("solvers:", [k for k, v in SOLVERS.items() if v is not None])

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x50 = PCA(min(50, len(ids) - 1), random_state=42).fit_transform(emb)
    xy = TSNE(2, perplexity=min(30, max(5, len(ids) // 20)), init="pca",
              learning_rate="auto", random_state=42).fit_transform(x50)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(ids)
    ps = 4 if N > 3000 else 9

    # (a) coverage 4패널
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, smap) in zip(axes, SOLVERS.items()):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_visible(False)
        if smap is None:
            ax.text(0.5, 0.5, f"{name}\n(pending)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color=INK2)
            ax.set_title(name, fontsize=11, color=INK2)
            continue
        cov = np.array([smap.get(i, False) for i in ids])
        ax.scatter(xy[~cov, 0], xy[~cov, 1], s=max(2, ps - 1), c="#c9c8c3", linewidths=0)
        ax.scatter(xy[cov, 0], xy[cov, 1], s=ps, c="#111111", alpha=0.85, linewidths=0)
        ax.set_title(f"{name} — {100*cov.mean():.1f}%\n({int(cov.sum()):,}/{N:,})",
                     fontsize=11, color=INK)
    fig.suptitle(f"{cfg['title']} — coverage per solver (black = solved)",
                 fontsize=13, color=INK)
    if cfg.get("prior_note"):
        fig.text(0.5, 0.005, cfg["prior_note"], ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03 if cfg.get("prior_note") else 0, 1, 0.92))
    fig.savefig(OUT / f"{cfg['prefix']}_coverage.png", dpi=170,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # (b) evolved roster expert facet
    names = {k: (v.get("name") or k) for k, v in
             json.load(open(REPO / cfg["names"], encoding="utf-8")).items()}
    expert_facets(xy, ids, binning, names,
                  f"{cfg['title']} — our roster per-expert solved (excl. common)",
                  OUT / f"{cfg['prefix']}_expert_facets.png")

    # (c) human prior expert facet (prior binning 결과가 있을 때)
    pb = REPO / cfg["prior_binning"] if cfg.get("prior_binning") else None
    if pb and pb.is_file():
        pbin = {str(json.loads(l)["id"]): json.loads(l)
                for l in open(pb, encoding="utf-8")}
        pnames = {}
        if cfg.get("prior_names") and (REPO / cfg["prior_names"]).is_file():
            pr = json.load(open(REPO / cfg["prior_names"], encoding="utf-8"))
            pnames = {x["id"]: (x.get("name") or x["id"]) for x in pr}
        expert_facets(xy, ids, pbin, pnames,
                      f"{cfg['title']} — human prior per-expert solved",
                      OUT / f"{cfg['prefix']}_prior_facets.png", exclude_common=False)
        print("prior facet 생성됨")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
