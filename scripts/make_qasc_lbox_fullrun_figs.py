"""Full-train QASC/LBox evolution figures for the handover doc.

Outputs:
  docs/fig_fullrun_qasc_seed20210211.png
  docs/fig_fullrun_lbox_seed20210311.png
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DECISION_COLORS = {
    "add": "#2ca02c",
    "delete": "#d62728",
    "swap": "#1f77b4",
    "noop": "#b0b0b0",
}
DECISION_MARKERS = {
    "add": "^",
    "delete": "v",
    "swap": "D",
    "noop": ".",
}


def moving_avg(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if window <= 1:
        return arr
    numerator = np.convolve(arr, np.ones(window), mode="same")
    denominator = np.convolve(np.ones(len(arr)), np.ones(window), mode="same")
    return numerator / denominator


def load_rows(dataset: str, seed: str) -> list[dict]:
    path = ROOT / "results" / dataset / f"seed{seed}" / dataset / f"seed{seed}" / "evolution_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_figure(dataset: str, seed: str, title: str, color: str) -> Path:
    rows = load_rows(dataset, seed)
    steps = [int(row.get("step", i + 1)) for i, row in enumerate(rows)]
    ubs = [float(row.get("upper_bound_pct", 0.0)) for row in rows]
    hards = [int(row.get("hard_error_n", 0)) for row in rows]
    decisions = [row.get("decision", "noop") for row in rows]
    sizes = [len(row.get("roster_after", [])) for row in rows]
    counts = Counter(decisions)

    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[2.2, 1.0])
    ax_ub = fig.add_subplot(grid[0, :])
    ax_size = fig.add_subplot(grid[1, 0])
    ax_count = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        f"{title} full-train evolution - seed {seed}",
        fontsize=14,
        fontweight="bold",
    )

    ax_ub.plot(steps, ubs, color=color, lw=1.1, alpha=0.55, label="batch UB")
    smooth_window = 5 if len(steps) < 250 else 25
    ax_ub.plot(
        steps,
        moving_avg(ubs, smooth_window),
        color=color,
        lw=2.3,
        label=f"{smooth_window}-step moving avg",
    )
    for step, ub, decision in zip(steps, ubs, decisions):
        if decision == "noop":
            continue
        ax_ub.scatter(
            step,
            ub,
            c=DECISION_COLORS[decision],
            marker=DECISION_MARKERS[decision],
            s=38,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
    ax_ub.set_ylabel("UB union (%)")
    ax_ub.set_xlim(min(steps) - 1, max(steps) + 1)
    ax_ub.set_ylim(max(0, min(ubs) - 8), min(100, max(ubs) + 8))
    ax_ub.grid(alpha=0.22)
    ax_ub.legend(loc="lower right", fontsize=8)

    ax_hard = ax_ub.twinx()
    ax_hard.plot(steps, hards, color="#555555", lw=0.8, alpha=0.28, label="hard errors")
    ax_hard.set_ylabel("hard errors")
    ax_hard.set_ylim(0, max(hards) + 5)

    ax_size.step(steps, sizes, where="post", color=color, lw=2.2)
    ax_size.fill_between(steps, sizes, step="post", color=color, alpha=0.18)
    for step, size, decision in zip(steps, sizes, decisions):
        if decision == "noop":
            continue
        ax_size.scatter(
            step,
            size,
            c=DECISION_COLORS[decision],
            marker=DECISION_MARKERS[decision],
            s=42,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
    ax_size.set_xlabel("evolution step")
    ax_size.set_ylabel("roster size")
    ax_size.set_xlim(min(steps) - 1, max(steps) + 1)
    ax_size.set_ylim(0, max(sizes) + 1.5)
    ax_size.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_size.grid(alpha=0.22)
    ax_size.text(
        0.015,
        0.93,
        f"post-step N {sizes[0]} -> {sizes[-1]} | steps {len(steps)}",
        transform=ax_size.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
    )

    order = ["add", "delete", "swap", "noop"]
    bars = ax_count.bar(
        order,
        [counts.get(name, 0) for name in order],
        color=[DECISION_COLORS[name] for name in order],
    )
    ax_count.set_title("Action gate choices", fontsize=10, fontweight="bold")
    ax_count.set_ylabel("count")
    ax_count.grid(axis="y", alpha=0.22)
    ax_count.bar_label(bars, padding=3, fontsize=8)
    patches = [mpatches.Patch(color=DECISION_COLORS[name], label=name) for name in order]
    ax_count.legend(handles=patches, fontsize=8, loc="upper right")

    out = DOCS / f"fig_fullrun_{dataset}_seed{seed}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    outputs = [
        make_figure("qasc", "20210211", "QASC", "#4c72b0"),
        make_figure("lbox", "20210311", "LBox Phase 1", "#c44e52"),
    ]
    for path in outputs:
        print(f"saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
