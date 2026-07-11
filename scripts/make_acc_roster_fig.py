"""Report figure for seed20210101 (coding/acc) evolution dynamics.

Output (docs/):
  fig_roster_acc_seed101.png   roster size (monotone → saturation) + UB per step
Read-only over the acc evolution log. Style matches make_roster_periodicity_fig.py.
"""
import json
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("docs", exist_ok=True)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR = "#55A868"      # coding = green (numina used blue/red)
LOG = "results/acc/seed20210101/acc/seed20210101/evolution_log.jsonl"
rows = [json.loads(l) for l in open(LOG) if l.strip()]

sizes = [len(r.get("roster_after", [])) for r in rows]
steps = list(range(1, len(sizes) + 1))
adds = [i + 1 for i, r in enumerate(rows) if r.get("decision") == "add"]
ub = [r.get("upper_bound_pct") for r in rows]
m = statistics.mean(sizes)


def moving_avg(y, w):
    y = np.array(y, float)
    if w < 2:
        return y
    num = np.convolve(y, np.ones(w), mode="same")
    cnt = np.convolve(np.ones(len(y)), np.ones(w), mode="same")
    return num / cnt


fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1]})

# ── Panel 1: roster size (monotone → saturation) ────────────────────────
ax = axes[0]
ax.step(steps, sizes, where="post", color=COLOR, lw=2.2)
ax.scatter(adds, [sizes[s - 1] for s in adds], color=COLOR, s=28, zorder=3,
           label=f"add event ({len(adds)})")
ax.axhline(9, color="black", ls="--", lw=1.0, alpha=0.5)
ax.set_title("코딩(acc) seed20210101 — 로스터 크기: 단조 증가 후 포화 (진동 없음)",
             fontsize=11.5, loc="left", fontweight="bold")
ax.set_ylabel("roster size")
ax.set_ylim(0, 10.5)
ax.set_yticks(range(0, 11, 2))
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="lower right", fontsize=9)
ax.text(0.015, 0.94,
        f"LUCA 단독(2) → N*=9 포화(step21~)   mean {m:.1f}   range {min(sizes)}–{max(sizes)}\n"
        f"※ 수학의 limit-cycle 진동과 대조",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=COLOR, alpha=0.9))

# ── Panel 2: UB per step ────────────────────────────────────────────────
ax = axes[1]
ax.plot(steps, ub, color=COLOR, lw=0.6, alpha=0.35)
ax.plot(steps, moving_avg(ub, 5), color=COLOR, lw=2, label="5-step moving avg")
ax.axhline(statistics.mean(ub), color="black", ls="--", lw=1.0, alpha=0.5,
           label=f"mean {statistics.mean(ub):.1f}%")
ax.set_title("배치별 UB union (%) — 고UB(하드에러 희소)라 add 고갈로 포화",
             fontsize=10.5, loc="left", fontweight="bold")
ax.set_ylabel("UB union (%)")
ax.set_xlabel("evolution step")
ax.set_ylim(70, 100)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", fontsize=9)

fig.savefig("docs/fig_roster_acc_seed101.png", dpi=150)
plt.close(fig)
print("saved: docs/fig_roster_acc_seed101.png")
