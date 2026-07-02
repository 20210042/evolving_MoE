"""Report figures for seed16 (topic) / seed17 (failure) evolution dynamics.

Outputs (docs/):
  fig_roster_periodicity_seed1617.png   roster size + smoothed envelope (limit cycle)
  fig_churn_seed1617.png                 cumulative add vs delete (revolving door)
  fig_persona_families_seed1617.png      top recurring added personas (proposal collapse)
  fig_ub_trajectory_seed1617.png         upper-bound % over steps (ceiling saturation)
Read-only over evolution logs.
"""
import json
import os
import statistics
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("docs", exist_ok=True)

# Korean glyphs (e.g. 풀이과정) — Noto Sans CJK is installed system-wide.
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RUNS = [
    ("20210016", "seed16 — topic scout (batch 100)", "#4878CF"),
    ("20210017", "seed17 — failure scout (batch 25)", "#D65F5F"),
]

# report-facing panel labels (replace the internal seed names)
AXIS_LABELS = {
    "20210016": "Hard error description only",
    "20210017": "Description + 풀이과정",
}


def load(seed):
    f = f"results/numina_cot/seed{seed}/numina_cot/seed{seed}/evolution_log.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()]


def pname(c):
    if isinstance(c, dict):
        return c.get("persona_name") or c.get("name") or "?"
    return str(c)[:40]


def moving_avg(y, w):
    y = np.array(y, float)
    if w < 2:
        return y
    # edge-corrected: divide by the real count in each window (no zero-pad dip at ends)
    num = np.convolve(y, np.ones(w), mode="same")
    cnt = np.convolve(np.ones(len(y)), np.ones(w), mode="same")
    return num / cnt


def period_of(sizes):
    m = statistics.mean(sizes)
    s = [1 if v > m else (-1 if v < m else 0) for v in sizes]
    cross = sum(1 for i in range(1, len(s)) if s[i] and s[i - 1] and s[i] != s[i - 1])
    return max(1, len(sizes) // max(1, cross // 2))


# ── Fig 1: periodicity (smoothed envelope) ──────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
for ax, (seed, title, color) in zip(axes, RUNS):
    rows = load(seed)
    sizes = [len(r.get("roster_after", [])) for r in rows]
    steps = list(range(len(sizes)))
    period = period_of(sizes)
    w = max(3, period // 3)                       # smoothing window < period → cycle survives
    sm = moving_avg(sizes, w)
    m = statistics.mean(sizes)

    ax.plot(steps, sizes, color=color, lw=0.5, alpha=0.18)     # raw, faint
    ax.plot(steps, sm, color=color, lw=1.8)                    # smoothed, bold
    ax.axhline(m, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.set_title(AXIS_LABELS[seed], fontsize=11, loc="left", fontweight="bold")
    ax.set_ylabel("roster size")
    ax.set_ylim(0, max(sizes) + 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.995, 0.05,
            f"mean {m:.1f}   range {min(sizes)}–{max(sizes)}   period ≈{period} steps   (smooth w={w})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85))
axes[-1].set_xlabel("evolution step")
fig.savefig("docs/fig_roster_periodicity_seed1617.png", dpi=150)
plt.close(fig)

# ── Fig 2: revolving door (cumulative add vs delete) ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
for ax, (seed, title, color) in zip(axes, RUNS):
    rows = load(seed)
    decs = [r.get("decision") for r in rows]
    steps = list(range(len(decs)))
    cum_add = np.cumsum([1 if d in ("add", "swap") else 0 for d in decs])
    cum_del = np.cumsum([1 if d in ("delete", "swap") else 0 for d in decs])
    ax.plot(steps, cum_add, color="#2ecc71", lw=2, label=f"cumulative add+swap ({cum_add[-1]})")
    ax.plot(steps, cum_del, color="#e74c3c", lw=2, label=f"cumulative delete+swap ({cum_del[-1]})")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.set_ylabel("cumulative count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
axes[-1].set_xlabel("evolution step")
fig.suptitle("Revolving door: adds and deletes accumulate in lockstep (membership churns)",
             fontsize=12.5, fontweight="bold")
fig.savefig("docs/fig_churn_seed1617.png", dpi=150)
plt.close(fig)

# ── Fig 3: proposal collapse (top recurring added personas) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
for ax, (seed, title, color) in zip(axes, RUNS):
    rows = load(seed)
    added = [pname(r.get("candidate_persona")) for r in rows if r.get("decision") in ("add", "swap")]
    top = Counter(added).most_common(12)[::-1]
    labels = [n for n, _ in top]
    counts = [c for _, c in top]
    ax.barh(range(len(top)), counts, color=color, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(AXIS_LABELS[seed], fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel("times added")
    ax.grid(True, axis="x", alpha=0.25)
fig.savefig("docs/fig_persona_families_seed1617.png", dpi=150)
plt.close(fig)

# ── Fig 4: upper-bound trajectory (ceiling saturation) ──────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
for ax, (seed, title, color) in zip(axes, RUNS):
    rows = load(seed)
    ub = [r.get("upper_bound_pct") for r in rows if r.get("upper_bound_pct") is not None]
    steps = list(range(len(ub)))
    sm = moving_avg(ub, 25)
    ax.plot(steps, ub, color=color, lw=0.5, alpha=0.2)
    ax.plot(steps, sm, color=color, lw=1.8, label="25-step moving avg")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.set_ylabel("UB union (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
axes[-1].set_xlabel("evolution step")
fig.suptitle("Upper-bound (oracle union) stays flat at ~77–78% — backbone ceiling, "
             "unmoved by roster churn", fontsize=12, fontweight="bold")
fig.savefig("docs/fig_ub_trajectory_seed1617.png", dpi=150)
plt.close(fig)

print("saved 4 figs to docs/: periodicity / churn / persona_families / ub_trajectory")

# ── Fig 5: gatefix comparison (OFF vs gatefix, topic + failure) ──────────
COMPARE = [
    ("topic (batch 100)",   ("20210016", "OFF (seed16)", "#999999"),
                            ("20210018", "gatefix (seed18)", "#4878CF")),
    ("failure (batch 25)",  ("20210017", "OFF (seed17)", "#999999"),
                            ("20210019", "gatefix (seed19)", "#C44E52")),
]
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for ax, (panel, off, on) in zip(axes, COMPARE):
    for seed, label, color in (off, on):
        try:
            rows = load(seed)
        except FileNotFoundError:
            continue
        sizes = [len(r.get("roster_after", [])) for r in rows]
        w = max(3, period_of(sizes) // 3)
        m = statistics.mean(sizes)
        dec = Counter(r.get("decision") for r in rows)
        churn = sum(dec.get(k, 0) for k in ("add", "delete", "swap"))
        ax.plot(range(len(sizes)), sizes, color=color, lw=0.4, alpha=0.13)
        ax.plot(range(len(sizes)), moving_avg(sizes, w), color=color, lw=2,
                label=f"{label} — mean {m:.1f}, churn {churn}")
        ax.axhline(m, color=color, ls="--", lw=1.0, alpha=0.5)
    ax.set_title(panel, fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("roster size")
    ax.set_ylim(0, 11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
axes[-1].set_xlabel("evolution step")
fig.suptitle("Gate-fix: churn drops sharply, but floor=0 inflates the roster near the cap "
             "— not yet convergence", fontsize=11.5, fontweight="bold")
fig.savefig("docs/fig_gatefix_compare_seed1618.png", dpi=150)
plt.close(fig)
print("saved: docs/fig_gatefix_compare_seed1618.png (topic + failure)")
