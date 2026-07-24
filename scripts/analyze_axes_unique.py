#!/usr/bin/env python3
"""analysis1의 Reality 패널을 '고유 solve만' 버전으로 다시 그린다.

기존 Reality 패널은 solver 중 가장 희귀한 한 명(1/pass_rate 가중)을 칠하므로
11명이 다 푼 문제에도 색이 찍힌다. 여기서는 n_solved == 1 인 문제,
즉 그 specialist만 유일하게 푼 문제만 칠하고 나머지는 전부 배경으로 뺀다.

  panel 1  Reality (기존 방식, 대조용)
  panel 2  고유 solve만 (n_solved == 1)
  panel 3  specialist별 고유 solve 수 (panel 2의 범례 겸 크기 비교)

t-SNE 좌표는 analyze_axes_viz.py와 동일한 경로/시드로 재현하고 캐시한다.
프롬프트 centroid는 쓰지 않으므로 임베딩 모델 로드가 없다(CPU 전용).

Usage: python scripts/analyze_axes_unique.py --dataset qasc|lbox
결과: results/axes_analysis/unique_solve_<ds>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
OUT = REPO / "results" / "axes_analysis"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GRAY = "#d0cfca"        # all-fail (기존 패널과 동일)
GRAY_BG = "#e6e5e1"     # 고유 패널의 배경 문제들
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a",
           "#eb6834", "#4a3aa7", "#e34948"]
SEED = 42

CFG = {
    "qasc": dict(labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
                 mapping="export/qasc_binning_seed20210211/agent_mapping.json",
                 src="export/qasc/qasc_train.jsonl", emb="results/embed_viz/qasc_emb.npy",
                 sub=None, title="QASC"),
    "lbox": dict(labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
                 mapping="export/lbox_binning_seed20210311/agent_mapping.json",
                 src="export/lbox/lbox_train.jsonl", emb="results/embed_viz/lbox_emb.npy",
                 sub=15000, title="LBOX"),
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ccolor(i):
    """analyze_axes_viz.py와 동일 — 기존 그림과 expert 색이 일대일 대응한다."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(PALETTE[i % len(PALETTE)])
    cyc = i // len(PALETTE)
    if cyc % 3 == 1:
        r, g, b = r * .6, g * .6, b * .6
    elif cyc % 3 == 2:
        r, g, b = min(1, .45 + r * .55), min(1, .45 + g * .55), min(1, .45 + b * .55)
    return mc.to_hex((r, g, b))


def bare(ax):
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

    src_ids = [str(r["id"]) for r in load_jsonl(REPO / c["src"])]
    emb = np.load(REPO / c["emb"])
    keep = [i for i, pid in enumerate(src_ids) if pid in id2lab]
    ids = [src_ids[i] for i in keep]
    P = emb[keep]
    if c["sub"] and len(ids) > c["sub"]:
        sel = np.sort(np.random.default_rng(SEED).choice(len(ids), c["sub"], replace=False))
        ids = [ids[i] for i in sel]; P = P[sel]
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    n = len(ids)

    S = np.array([[int(id2lab[i]["per_expert"].get(e, 0)) for e in specialists] for i in ids])
    n_solved = S.sum(1)
    per_rate = {k: S[:, k].mean() for k in range(len(specialists))}

    # t-SNE 좌표: analyze_axes_viz.py와 같은 경로/시드 → 같은 배치. 캐시해서 재실행 즉시.
    xy_p = OUT / f"{a.dataset}_xy.npy"
    if xy_p.is_file() and len(np.load(xy_p)) == n:
        xy = np.load(xy_p)
        print(f"t-SNE 캐시 사용: {xy_p.name}")
    else:
        print(f"t-SNE 계산 중 (n={n:,}) ...")
        x50 = PCA(min(50, P.shape[1]), random_state=SEED).fit_transform(P)
        xy = TSNE(2, perplexity=min(30, max(5, n // 20)), init="pca",
                  learning_rate="auto", random_state=SEED).fit_transform(x50)
        np.save(xy_p, xy)

    # 기존 Reality: solver 중 가장 희귀한 한 명 (-1 = all fail)
    best = np.array([int(np.argmax([S[p, k] / (per_rate[k] + 1e-9) if S[p, k] else -1
                                    for k in range(len(specialists))])) if n_solved[p] else -1
                     for p in range(n)])
    # 고유 solve: 정확히 한 명만 푼 문제
    uniq = np.where(n_solved == 1, S.argmax(1), -1)
    n_uniq = int((n_solved == 1).sum())
    per_uniq = np.array([int((uniq == k).sum()) for k in range(len(specialists))])

    print(f"{c['title']}: n={n:,} · specialists={len(specialists)} · "
          f"고유 solve {n_uniq:,} ({100*n_uniq/n:.1f}%) · "
          f"all-fail {int((n_solved==0).sum()):,} ({100*(n_solved==0).mean():.1f}%) · "
          f"전원 {int((n_solved==len(specialists)).sum()):,}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = 4 if n > 3000 else 9
    ecol = {k: ccolor(k) for k in range(len(specialists))}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4),
                           gridspec_kw={"width_ratios": [1, 1, 0.85]})
    fig.patch.set_facecolor(SURFACE)

    # panel 1 — 기존 Reality (대조군)
    ax[0].scatter(xy[best == -1, 0], xy[best == -1, 1], s=ps, c=GRAY, alpha=.5, linewidths=0)
    for k in range(len(specialists)):
        m = best == k
        ax[0].scatter(xy[m, 0], xy[m, 1], s=ps, c=ecol[k], alpha=.8, linewidths=0)
    ax[0].set_title("Reality (as published)\nrarest solver — overlapping solves still colored",
                    fontsize=12, color=INK)
    bare(ax[0])

    # panel 2 — 고유 solve만
    ax[1].scatter(xy[:, 0], xy[:, 1], s=2.5, c=GRAY_BG, alpha=.55, linewidths=0)
    for k in range(len(specialists)):
        m = uniq == k
        ax[1].scatter(xy[m, 0], xy[m, 1], s=17, c=ecol[k], alpha=.95,
                      linewidths=.6, edgecolors=SURFACE)
    ax[1].set_title(f"Unique solves only (n_solved = 1)\n{n_uniq:,} / {n:,} = {100*n_uniq/n:.1f}%",
                    fontsize=12, color=INK)
    bare(ax[1])

    # panel 3 — specialist별 고유 solve 수 (panel 2의 범례 역할)
    order = np.argsort(per_uniq)
    names = [mapping[specialists[k]].get("name", specialists[k]) for k in order]
    ax[2].barh(np.arange(len(order)), per_uniq[order],
               color=[ecol[k] for k in order], height=.68)
    ax[2].set_yticks(np.arange(len(order)))
    ax[2].set_yticklabels(names, fontsize=9, color=INK2)
    for i, k in enumerate(order):
        ax[2].text(per_uniq[k] + max(per_uniq) * .015, i, str(per_uniq[k]),
                   va="center", fontsize=9, color=INK2)
    ax[2].set_xlabel("# problems solved by this expert alone", fontsize=10, color=INK2)
    ax[2].set_title("Who owns the uniqueness", fontsize=12, color=INK)
    ax[2].set_facecolor(SURFACE)
    ax[2].set_xlim(0, max(per_uniq) * 1.14)
    ax[2].tick_params(axis="x", colors=INK2, labelsize=9)
    for s_ in ("top", "right", "left"):
        ax[2].spines[s_].set_visible(False)
    ax[2].spines["bottom"].set_color(GRAY)

    fig.suptitle(f"{c['title']} — strip the overlapping solves: what unique contribution is left",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = OUT / f"unique_solve_{a.dataset}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
