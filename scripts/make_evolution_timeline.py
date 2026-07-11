"""Evolution timeline visualization.

Fig4: Roster evolution timeline - 4 seeds (44, 45, 46, 48)
  - UB trajectory with decision markers (add/delete/swap/noop)
  - Roster size over steps
  - Epoch boundaries marked

Fig5: Agent lifetime Gantt chart (seed44 vs seed48)
  - Which agents were alive at each step
"""
import json
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("docs", exist_ok=True)

DECISION_COLORS = {
    "add":    "#2ecc71",
    "swap":   "#3498db",
    "delete": "#e74c3c",
    "noop":   "#bdc3c7",
}
DECISION_MARKERS = {
    "add":    "^",
    "swap":   "D",
    "delete": "v",
    "noop":   ".",
}
SEED_COLORS = {
    "20210044": "#4878CF",
    "20210045": "#D65F5F",
    "20210046": "#6ACC65",
    "20210048": "#C4AD66",
}
SEED_LABELS = {
    "20210044": "seed44  Llama8B / dedup ON",
    "20210045": "seed45  Llama8B / dedup OFF",
    "20210046": "seed46  gemma-31B / critic-refine",
    "20210048": "seed48  gemma-31B / one-step (clean)",
}

def build_id_to_name(seed, task="mbpp"):
    """Collect id→name from all roster_step_*.json files."""
    base = f"results/{task}/seed{seed}/{task}/seed{seed}"
    id2name = {}
    for f in glob.glob(f"{base}/roster_step_*.json"):
        for agent in json.load(open(f)):
            if isinstance(agent, dict):
                aid = agent.get("id", "")
                name = agent.get("name") or agent.get("persona_name") or aid
                if aid:
                    id2name[aid] = name
    # fallback: roster_final.json
    final = f"results/{task}/seed{seed}/roster_final.json"
    if os.path.exists(final):
        for agent in json.load(open(final)):
            if isinstance(agent, dict):
                aid = agent.get("id", "")
                name = agent.get("name") or agent.get("persona_name") or aid
                if aid:
                    id2name[aid] = name
    return id2name


def load_evo(seed, task="mbpp"):
    path = f"results/{task}/seed{seed}/{task}/seed{seed}/evolution_log.jsonl"
    id2name = build_id_to_name(seed, task)
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            roster_raw = d.get("roster_after", [])
            names = []
            for a in roster_raw:
                if isinstance(a, dict):
                    aid = a.get("id", "")
                    names.append(a.get("name") or a.get("persona_name") or aid)
                else:
                    # a is an ID string
                    names.append(id2name.get(str(a), str(a)))
            rows.append({
                "step":     d.get("step", len(rows) + 1),
                "epoch":    d.get("epoch", 0),
                "decision": d.get("decision", "noop"),
                "ub":       d.get("upper_bound_pct", 0),
                "roster":   names,
            })
    return rows


# ════════════════════════════════════════════════════════════
# FIG 4: UB + Roster-size + Decision overlay (2×4 grid)
# ════════════════════════════════════════════════════════════
seeds_mbpp = ["20210044", "20210045", "20210046", "20210048"]
titles = [
    "seed44  (Llama8B / dedup ON)",
    "seed45  (Llama8B / dedup OFF)",
    "seed46  (gemma-31B / critic-refine)",
    "seed48  (gemma-31B / one-step clean)",
]

fig, axes = plt.subplots(2, 4, figsize=(22, 9),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
fig.suptitle("Fig 4 · Evolution Process: UB Trajectory + Decision History + Roster Size  [MBPP]",
             fontsize=14, fontweight="bold", y=1.01)

for col, seed in enumerate(seeds_mbpp):
    rows = load_evo(seed)
    steps    = [r["step"] for r in rows]
    ubs      = [r["ub"]   for r in rows]
    sizes    = [len(r["roster"]) for r in rows]
    decisions= [r["decision"] for r in rows]
    epochs   = [r["epoch"]   for r in rows]

    # epoch boundaries
    epoch_steps = {}
    for r in rows:
        ep = r["epoch"]
        if ep not in epoch_steps:
            epoch_steps[ep] = r["step"]

    color = SEED_COLORS[seed]
    ax_ub   = axes[0][col]
    ax_size = axes[1][col]

    # ── top: UB trajectory ──────────────────────────────────
    ax_ub.plot(steps, ubs, color=color, lw=1.5, alpha=0.7, zorder=1)
    for s, ub, dec in zip(steps, ubs, decisions):
        ax_ub.scatter(s, ub,
                      c=DECISION_COLORS[dec],
                      marker=DECISION_MARKERS[dec],
                      s=55 if dec != "noop" else 18,
                      zorder=2,
                      edgecolors="white" if dec != "noop" else "none",
                      linewidths=0.5)

    # epoch boundaries
    for ep, es in epoch_steps.items():
        if ep > min(epoch_steps):
            ax_ub.axvline(es, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax_ub.text(es + 0.3, max(ubs) * 0.97, f"E{ep}",
                       fontsize=7, color="gray", va="top")

    ax_ub.set_title(titles[col], fontsize=9, pad=4)
    ax_ub.set_ylabel("UB (%)" if col == 0 else "")
    ax_ub.set_ylim(0, 105)
    ax_ub.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax_ub.grid(alpha=0.2)
    ax_ub.tick_params(labelbottom=False)

    # decision legend (only first column)
    if col == 0:
        handles = [mpatches.Patch(color=v, label=k) for k, v in DECISION_COLORS.items()]
        ax_ub.legend(handles=handles, fontsize=7, loc="lower left",
                     title="Decision", title_fontsize=7, framealpha=0.7)

    # ── bottom: roster size ─────────────────────────────────
    ax_size.step(steps, sizes, color=color, lw=2, where="post")
    ax_size.fill_between(steps, sizes, step="post", color=color, alpha=0.25)
    for ep, es in epoch_steps.items():
        if ep > min(epoch_steps):
            ax_size.axvline(es, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax_size.set_ylabel("# Agents" if col == 0 else "")
    ax_size.set_xlabel("Evolution Step")
    ax_size.set_ylim(0, max(sizes) + 1.5)
    ax_size.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax_size.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_size.grid(alpha=0.2)

# action count annotation per seed
for col, seed in enumerate(seeds_mbpp):
    rows = load_evo(seed)
    from collections import Counter
    cnt = Counter(r["decision"] for r in rows)
    txt = f"add:{cnt['add']} swap:{cnt['swap']} del:{cnt['delete']} noop:{cnt['noop']}"
    axes[1][col].text(0.5, -0.42, txt, transform=axes[1][col].transAxes,
                      ha="center", fontsize=7.5, color="#555555")

fig.tight_layout()
fig.savefig("docs/pres_fig4_evolution_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig4 saved.")


# ════════════════════════════════════════════════════════════
# FIG 5: Agent Lifetime Gantt  (seed44 vs seed48 side-by-side)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Fig 5 · Agent Lifetime in Roster  [MBPP]\n"
             "Green bar = alive in final roster  |  Grey = eventually deleted/replaced",
             fontsize=13, fontweight="bold", y=1.03)

for ax, seed, title in zip(axes,
                            ["20210044", "20210048"],
                            ["seed44  Llama8B / dedup ON",
                             "seed48  gemma-31B / one-step (clean)"]):
    rows = load_evo(seed)
    steps = [r["step"] for r in rows]

    # collect all unique agents in order of first appearance
    order = []
    seen = set()
    for r in rows:
        for name in r["roster"]:
            if name not in seen:
                order.append(name)
                seen.add(name)

    final_roster = set(rows[-1]["roster"])

    # short labels for display
    def shorten(name, maxlen=28):
        return name[:maxlen] + ".." if len(name) > maxlen else name

    y_pos = {name: i for i, name in enumerate(order)}
    n = len(order)

    for r in rows:
        s = r["step"]
        for name in r["roster"]:
            y = y_pos[name]
            color = "#2ecc71" if name in final_roster else "#95a5a6"
            ax.barh(y, 1, left=s - 0.5, height=0.7,
                    color=color, alpha=0.85, edgecolor="white", linewidth=0.4)

    # epoch boundaries
    epoch_steps = {}
    for r in rows:
        ep = r["epoch"]
        if ep not in epoch_steps:
            epoch_steps[ep] = r["step"]
    for ep, es in epoch_steps.items():
        if ep > min(epoch_steps):
            ax.axvline(es - 0.5, color="gray", ls="--", lw=1, alpha=0.6)
            ax.text(es - 0.5 + 0.2, n - 0.5, f"E{ep}", fontsize=8,
                    color="gray", va="top")

    # decision markers on top
    for r in rows:
        dec = r["decision"]
        if dec != "noop":
            ax.scatter(r["step"], n + 0.2,
                       c=DECISION_COLORS[dec],
                       marker=DECISION_MARKERS[dec],
                       s=60, zorder=5,
                       edgecolors="white", linewidths=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([shorten(name) for name in order], fontsize=8)
    ax.set_xlim(min(steps) - 0.7, max(steps) + 0.7)
    ax.set_ylim(-0.5, n + 0.8)
    ax.set_xlabel("Evolution Step")
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.2)

    green_patch = mpatches.Patch(color="#2ecc71", label="In final roster")
    grey_patch  = mpatches.Patch(color="#95a5a6", label="Eliminated")
    ax.legend(handles=[green_patch, grey_patch], fontsize=8,
              loc="lower right", framealpha=0.8)

fig.tight_layout()
fig.savefig("docs/pres_fig5_agent_lifetime.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig5 saved.")
print("Done.")
