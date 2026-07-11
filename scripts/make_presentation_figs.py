"""발표 사전자료 시각화 생성 스크립트.

Fig1: Dedup Ablation (seed44 vs seed45)
Fig2: Backbone Upgrade - misc (seed44 LCB vs seed47 LCB + UB trajectory)
Fig3: Architecture Comparison (seed46 vs seed48)
"""
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("docs", exist_ok=True)

# ── colour palette ──────────────────────────────────────────
C = {
    "44": "#4878CF",   # blue   – seed44 (dedup ON)
    "45": "#D65F5F",   # red    – seed45 (dedup OFF, Llama)
    "46": "#6ACC65",   # green  – seed46 (gemma / critic)
    "47": "#B47CC7",   # purple – seed47 (gemma / one-step / contaminated)
    "48": "#C4AD66",   # gold   – seed48 (gemma / one-step / clean)
    "lcb44": "#4878CF",
    "lcb47": "#C4AD66",
}

# ── data ────────────────────────────────────────────────────
mbpp = {
    "44": {1:31.6, 2:37.0, 3:33.0, 4:30.0, 5:30.8},
    "45": {1:33.8, 2:33.2, 3:32.4, 4:32.4},
    "46": {1:74.8, 2:74.6},
    "48": {1:74.0, 2:70.8, 3:76.0, 4:73.6},
}
he = {
    "44": {1:32.9, 2:37.2, 3:34.8, 4:34.8, 5:32.3},
    "45": {1:33.5, 2:37.2, 3:39.0, 4:41.5},
    "46": {1:84.1, 2:85.4},
    "47": {1:71.3, 2:79.3, 3:83.5, 4:72.6, 5:76.8},
    "48": {1:76.2, 2:73.8, 3:70.7, 4:70.7},
}
lcb_test = {
    "44": {1:8.6, 2:10.6, 3:10.6, 4:10.2, 5:9.2},
    "47": {1:46.2},
}

def get_ub(seed, task):
    path = f"results/{task}/seed2021{seed}/{task}/seed2021{seed}/evolution_log.jsonl"
    ubs = []
    try:
        for line in open(path):
            d = json.loads(line)
            v = d.get("upper_bound_pct")
            if v is not None:
                ubs.append(v)
    except Exception:
        pass
    return ubs

ub = {
    "44_mbpp": get_ub("0044", "mbpp"),
    "45_mbpp": get_ub("0045", "mbpp"),
    "46_mbpp": get_ub("0046", "mbpp"),
    "48_mbpp": get_ub("0048", "mbpp"),
    "44_lcb":  get_ub("0044", "lcb"),
    "47_lcb":  get_ub("0047", "lcb"),
}

# final rosters (domain tags)
roster = {
    "44": ["LUCA","Perf_Metric","Edge_Case","Error_Pattern","Maintainability","Input_Parsing"],
    "45": ["LUCA","Edge_Case","Edge_Case","Edge_Case","Edge_Case","Tuple"],
    "46": ["LUCA","Strict_TestCase"],
    "48": ["Array_Greedy","Regex_Math","Data_Spatial"],
}

runtime_h = {"46": 37.0, "48": 10.9}  # hours

# ════════════════════════════════════════════════════════════
# FIG 1: Dedup Ablation  (seed44 vs seed45)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fig 1 · Dedup Ablation: seed44 (ON) vs seed45 (OFF)  [Llama 8B / critic-refine]",
             fontsize=13, fontweight="bold", y=1.01)

# 1a: MBPP test scores
ax = axes[0]
for sid, color, ls in [("44", C["44"], "-"), ("45", C["45"], "--")]:
    d = mbpp[sid]
    epochs = sorted(d); vals = [d[e] for e in epochs]
    ax.plot(epochs, vals, marker="o", color=color, ls=ls, lw=2,
            label=f"seed{sid} (dedup {'ON' if sid=='44' else 'OFF'})")
ax.set_title("MBPP test pass@1 (by epoch)", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("pass@1 (%)")
ax.set_ylim(25, 45); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 1b: HumanEval scores
ax = axes[1]
for sid, color, ls in [("44", C["44"], "-"), ("45", C["45"], "--")]:
    d = he[sid]
    epochs = sorted(d); vals = [d[e] for e in epochs]
    ax.plot(epochs, vals, marker="s", color=color, ls=ls, lw=2,
            label=f"seed{sid}")
ax.set_title("HumanEval pass@1 (transfer)", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("pass@1 (%)")
ax.set_ylim(25, 50); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 1c: Final roster diversity
ax = axes[2]
domain_counts = {
    "seed44\n(dedup ON)\n5 domains": [1,1,1,1,1,1],   # 6 unique agents
    "seed45\n(dedup OFF)\nEdge_Case flood": [1,4,1],   # LUCA, Edge_Case×4, Tuple
}
labels44 = ["LUCA","Perf_Metric","Edge_Case","Error\nPattern","Maintain\nability","Input\nParsing"]
labels45 = ["LUCA","Edge_Case\n×4","Tuple"]
colors44 = ["#4878CF","#6ACC65","#D65F5F","#B47CC7","#C4AD66","#FF9F4A"]
colors45 = ["#4878CF","#D65F5F","#B47CC7"]

ax2_left = ax
ax2_left.axis("off")
ax2_left.set_title("Final Roster Composition", fontsize=11)

ax_left = fig.add_axes([0.675, 0.15, 0.12, 0.65])
ax_right = fig.add_axes([0.82, 0.15, 0.12, 0.65])

ax_left.pie([1]*6, labels=labels44, colors=colors44,
            autopct='%d%%', startangle=90, textprops={"fontsize":7})
ax_left.set_title("seed44\n(dedup ON)\n6 agents", fontsize=9, pad=4)

ax_right.pie([1,4,1], labels=labels45, colors=colors45,
             autopct='%d%%', startangle=90, textprops={"fontsize":7})
ax_right.set_title("seed45\n(dedup OFF)\n6 agents", fontsize=9, pad=4)

fig.tight_layout()
fig.savefig("docs/pres_fig1_dedup_ablation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig1 saved.")

# ════════════════════════════════════════════════════════════
# FIG 2: Backbone Upgrade misc (seed44 LCB vs seed47 LCB)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 2 · Backbone Upgrade: Llama 8B (seed44) → gemma-31B (seed47)  [LCB task]\n"
             "Warning: backbone + architecture changed simultaneously — reference only (not pure ablation)",
             fontsize=12, fontweight="bold", y=1.03)

# 2a: LCB train UB trajectory
ax = axes[0]
u44 = ub["44_lcb"]
u47 = ub["47_lcb"]
ax.plot(range(1, len(u44)+1), u44, color=C["44"], lw=2, label="seed44  Llama8B / critic-refine")
ax.plot(range(1, len(u47)+1), u47, color=C["47"], lw=2, ls="--", label="seed47  gemma-31B / one-step")
ax.set_title("LCB Training Upper Bound trajectory", fontsize=11)
ax.set_xlabel("Evolution Step"); ax.set_ylabel("Upper Bound (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 2b: LCB test pass@1
ax = axes[1]
for sid, color, label in [
    ("44", C["44"], "seed44  Llama8B\ncritic-refine"),
    ("47", C["47"], "seed47  gemma-31B\none-step (⚠MBPP contaminated\nbut LCB clean)"),
]:
    d = lcb_test[sid]
    epochs = sorted(d); vals = [d[e] for e in epochs]
    ax.plot(epochs, vals, marker="o", color=color, lw=2, label=label)
ax.set_title("LCB test pass@1 (by epoch)", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("pass@1 (%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.annotate("Only E1 scored\n(eval in progress)", xy=(1, 46.2), xytext=(1.2, 42),
            fontsize=8, color=C["47"],
            arrowprops=dict(arrowstyle="->", color=C["47"]))

fig.tight_layout()
fig.savefig("docs/pres_fig2_backbone_upgrade.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig2 saved.")

# ════════════════════════════════════════════════════════════
# FIG 3: Architecture Comparison (seed46 vs seed48, gemma-31B)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Fig 3 · Architecture: critic-refine (seed46) vs one-step MoE (seed48)  [gemma-31B / dedup OFF]",
             fontsize=13, fontweight="bold", y=1.01)

# 3a: MBPP test scores
ax = axes[0]
for sid, color, ls, label in [
    ("46", C["46"], "-",  "seed46  critic-refine"),
    ("48", C["48"], "--", "seed48  one-step MoE (clean)"),
]:
    d = mbpp[sid]
    epochs = sorted(d); vals = [d[e] for e in epochs]
    ax.plot(epochs, vals, marker="o", color=color, ls=ls, lw=2, label=label)
ax.set_title("MBPP test pass@1", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("pass@1 (%)")
ax.set_ylim(65, 82); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.annotate("E3-5\nnot yet\nscored", xy=(2, 74.6), xytext=(2.2, 78.5),
            fontsize=8, color=C["46"], arrowprops=dict(arrowstyle="->", color=C["46"]))

# 3b: Runtime comparison
ax = axes[1]
seeds_rt = ["seed46\ncritic-refine", "seed48\none-step MoE"]
runtimes = [37.0, 10.9]
colors_rt = [C["46"], C["48"]]
bars = ax.bar(seeds_rt, runtimes, color=colors_rt, width=0.5, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, runtimes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}h",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Total Runtime (5 Epochs)", fontsize=11)
ax.set_ylabel("Hours"); ax.set_ylim(0, 45)
ax.grid(axis="y", alpha=0.3)
speedup_text = f"3.4× speedup"
ax.text(0.5, 0.92, speedup_text, transform=ax.transAxes,
        ha="center", fontsize=12, color="#D65F5F", fontweight="bold")

# 3c: Roster diversity (final)
ax = axes[2]
ax.axis("off")
ax.set_title("Final Roster Composition", fontsize=11)

ax_r46 = fig.add_axes([0.69, 0.20, 0.12, 0.58])
ax_r48 = fig.add_axes([0.84, 0.20, 0.12, 0.58])

roster46_labels = ["LUCA","Strict\nTestCase"]
roster46_colors = ["#4878CF","#6ACC65"]
roster48_labels = ["Array\nGreedy","Regex\nMath","Data\nSpatial"]
roster48_colors = ["#D65F5F","#B47CC7","#C4AD66"]

ax_r46.pie([1,1], labels=roster46_labels, colors=roster46_colors,
           autopct='%d%%', startangle=90, textprops={"fontsize":8})
ax_r46.set_title("seed46\ncritic-refine\n2 agents", fontsize=9, pad=4)

ax_r48.pie([1,1,1], labels=roster48_labels, colors=roster48_colors,
           autopct='%d%%', startangle=90, textprops={"fontsize":8})
ax_r48.set_title("seed48\none-step MoE\n3 agents", fontsize=9, pad=4)

fig.tight_layout()
fig.savefig("docs/pres_fig3_arch_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig3 saved.")
print("Done. All figures in docs/")
