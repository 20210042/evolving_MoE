"""Report figure for seed20212001 (SNI) evolution dynamics.

Output (docs/):
  fig_roster_sni_seed2001.png   roster size (boom-bust) + UB per step
Read-only over the SNI evolution log. Style matches make_acc_roster_fig.py.

대조선: 같은 seed의 앞선 런(스카우트가 128토큰 캡에 막혀 81% 실패)은 스카우트가 성공한
스텝에만 기록이 남아 261점뿐이다. 로스터가 얼마나 덜 움직였는지를 같이 보인다.
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

COLOR = "#8172B2"      # SNI = purple (numina 파랑/빨강 · coding 초록)
GREY = "#999999"
LOG = "results/sni/seed20212001/sni/seed20212001/evolution_log.jsonl"
LOG_CAP = "results/sni/seed20212001_scoutcap128/sni/seed20212001/evolution_log.jsonl"
rows = [json.loads(l) for l in open(LOG) if l.strip()]

sizes = [len(r.get("roster_after", [])) for r in rows]
steps = [r["step"] for r in rows]
adds = [r["step"] for r in rows if r.get("decision") == "add"]
dels = [r["step"] for r in rows if r.get("decision") == "delete"]
size_at = dict(zip(steps, sizes))
ub = [r.get("upper_bound_pct") for r in rows]
m = statistics.mean(sizes)

# luca가 로스터에서 빠진 스텝
luca_out = next((r["step"] for r in rows if "luca" not in (r.get("roster_after") or [])), None)


def moving_avg(y, w):
    y = np.array(y, float)
    if w < 2:
        return y
    num = np.convolve(y, np.ones(w), mode="same")
    cnt = np.convolve(np.ones(len(y)), np.ones(w), mode="same")
    return num / cnt


fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1]})

# ── Panel 1: roster size ────────────────────────────────────────────────
ax = axes[0]
if os.path.exists(LOG_CAP):
    cap = [json.loads(l) for l in open(LOG_CAP) if l.strip()]
    ax.step([r["step"] for r in cap], [len(r.get("roster_after", [])) for r in cap],
            where="post", color=GREY, ls="--", lw=1.2, alpha=0.8,
            label=f"앞선 런: 스카우트 81% 막힘 ({len(cap)}점 기록)")
ax.step(steps, sizes, where="post", color=COLOR, lw=2.0)
ax.scatter(adds, [size_at[s] for s in adds], color=COLOR, s=16, zorder=3,
           label=f"add event ({len(adds)})")
ax.scatter(dels, [size_at[s] for s in dels], facecolors="none", edgecolors=COLOR,
           s=22, lw=1.0, zorder=3, label=f"delete event ({len(dels)})")
ax.axhline(sizes[-1], color="black", ls="--", lw=1.0, alpha=0.5)
ax.set_title("SNI seed20212001 — 로스터 크기: boom-bust 진동 (포화 없음)",
             fontsize=11.5, loc="left", fontweight="bold")
ax.set_ylabel("roster size")
ax.set_ylim(0, max(sizes) + 2)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="lower right", fontsize=9)
note = (f"LUCA 단독 시작 → 최종 {sizes[-1]}명   mean {m:.1f}   range {min(sizes)}–{max(sizes)}\n"
        f"등장 페르소나 82명 · add {len(adds)} / delete {len(dels)}"
        + (f"   ※ LUCA는 step {luca_out}에 도태" if luca_out else ""))
ax.text(0.015, 0.94, note, transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=COLOR, alpha=0.9))

# ── Panel 2: UB per step ────────────────────────────────────────────────
ax = axes[1]
ax.plot(steps, ub, color=COLOR, lw=0.5, alpha=0.25)
ax.plot(steps, moving_avg(ub, 25), color=COLOR, lw=2, label="25-step moving avg")
ax.axhline(statistics.mean(ub), color="black", ls="--", lw=1.0, alpha=0.5,
           label=f"mean {statistics.mean(ub):.1f}%")
ax.set_title("배치별 UB union (%) — 초반 상승 후 평탄 (동적평형)",
             fontsize=10.5, loc="left", fontweight="bold")
ax.set_ylabel("UB union (%)")
ax.set_xlabel("evolution step (batch 50)")
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", fontsize=9)

fig.savefig("docs/fig_roster_sni_seed2001.png", dpi=150)
plt.close(fig)
print("saved: docs/fig_roster_sni_seed2001.png")
