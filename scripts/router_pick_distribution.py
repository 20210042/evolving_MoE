#!/usr/bin/env python3
"""라우터가 실제로 각 expert를 얼마나/어떤 카테고리에서 고르는지 표로 뽑는다 (미팅용).

router_sweep_max_nsolved.py와 같은 라우터(soft-label CE MLP, 특정 τ 하나)를 학습시키고,
test751 전체(751문제)에 대해 pick을 계산 → main_critic_category(Quantitative Reasoning /
Constructive Implementation / Greedy Strategy / Structured Data / State-Space Reasoning)별로
어떤 expert가 몇 %씩 선택되는지 집계한다. LBOX의 "Low5/High6 Router"표와 같은 포맷.

Usage: python scripts/router_pick_distribution.py --dataset acc --feature hs_mean --tau 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ap = argparse.ArgumentParser()
rc.add_dataset_arg(ap, default="acc")
ap.add_argument("--feature", choices=["hs_mean", "hs_last", "emb"], default="hs_mean")
ap.add_argument("--tau", type=int, default=8)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--roster_path", default="results/acc/seed20210111/roster_final.json",
                help="expert 코드네임 -> 실제 persona 이름 매핑 소스")
ap.add_argument("--out", default=None)
A = ap.parse_args()

sp = rc.spec(A.dataset)
OUT = Path(A.out) if A.out else REPO / "results" / A.dataset / "router_pick_distribution.md"
np.random.seed(0)
torch.manual_seed(0)

EX = rc.experts(sp)
E = len(EX)

NAME = {e: e for e in EX}
if A.roster_path and Path(A.roster_path).is_file():
    for x in json.load(open(A.roster_path, encoding="utf-8")):
        if x["id"] in NAME:
            NAME[x["id"]] = x.get("name") or x["id"]


def load_features(which: str):
    if A.feature == "emb":
        return rc.emb_paths(sp, which)
    split = sp.train_split if which == "train" else sp.eval_split
    return rc.feat_path(sp, split, A.feature), rc.feat_path(sp, split, "hs_ids")


train_bin = rc.load_binning(REPO / sp.labels)
feat_npy, feat_ids = load_features("train")
train_ids, X_train, S_train = rc.align(feat_npy, feat_ids, train_bin, EX)
n_solved_train = S_train.sum(1)

test_bin = rc.load_binning(REPO / sp.eval_binned)
feat_npy, feat_ids = load_features("eval")
test_ids, X_test, S_test = rc.align(feat_npy, feat_ids, test_bin, EX)
N_test = len(test_ids)
best_single, oracle_union = rc.baselines(S_test)
print(f"train {len(train_ids)}문제 · test {N_test}문제 · feature={A.feature} · tau={A.tau}", flush=True)

MU = X_train.mean(0, keepdims=True)
SD = X_train.std(0, keepdims=True) + 1e-6


def mlp(d):
    return nn.Sequential(nn.Linear(d, A.hidden), nn.ReLU(), nn.Dropout(A.dropout), nn.Linear(A.hidden, E))


def soft_ce(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(1).mean()


mask = (n_solved_train >= 1) & (n_solved_train <= A.tau)
Xtr_raw, Str = X_train[mask], S_train[mask]
target = Str / Str.sum(1, keepdims=True)
Xtr = torch.tensor((Xtr_raw - MU) / SD, dtype=torch.float32).to(DEV)
Ttr = torch.tensor(target, dtype=torch.float32).to(DEV)
Xte = torch.tensor((X_test - MU) / SD, dtype=torch.float32).to(DEV)

probs = np.zeros((N_test, E), np.float32)
n = len(Xtr)
for s in A.seeds:
    torch.manual_seed(s)
    net = mlp(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), A.lr, weight_decay=A.wd)
    for _ in range(A.epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, A.batch):
            b = perm[i:i + A.batch]
            opt.zero_grad()
            soft_ce(net(Xtr[b]), Ttr[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        probs += torch.softmax(net(Xte), dim=1).cpu().numpy()
probs /= len(A.seeds)
pick = probs.argmax(1)
acc = 100 * np.mean([S_test[i, pick[i]] for i in range(N_test)])
print(f"top-1 정확도(전체751): {acc:.2f}%  (best-single {best_single:.2f}, oracle-union {oracle_union:.2f})",
      flush=True)

# ---- 문제별 main_critic_category 로드 ----
src_rows = [json.loads(l) for l in open(REPO / sp.src[sp.eval_split], encoding="utf-8")]
cat_by_id = {str(r["id"]): (r.get("main_critic_category") or "Unknown") for r in src_rows}
categories = sorted({cat_by_id.get(pid, "Unknown") for pid in test_ids})
print(f"카테고리: {categories}", flush=True)

# ---- 집계: expert별 전체 선택 수/비율, 카테고리별 선택 비율 ----
picked_expert = [EX[pick[i]] for i in range(N_test)]
picked_cat = [cat_by_id.get(test_ids[i], "Unknown") for i in range(N_test)]

overall_count = Counter(picked_expert)
cat_totals = Counter(picked_cat)
cat_expert_count = {c: Counter() for c in categories}
for e, c in zip(picked_expert, picked_cat):
    cat_expert_count[c][e] += 1

order = [e for e, _ in overall_count.most_common()]
for e in EX:
    if e not in order:
        order.append(e)

lines = [
    f"# {sp.name.upper()} 라우터 expert별 선택 비율 (feature={A.feature}, τ={A.tau})",
    "",
    f"- top-1 정확도(전체{N_test}): {acc:.2f}% · best-single {best_single:.2f}% · "
    f"oracle-union {oracle_union:.2f}%",
    f"- 카테고리(main_critic_category)별 문제수: " + ", ".join(f"{c}={cat_totals.get(c,0)}" for c in categories),
    "",
    "| Expert | 전체 선택 | " + " | ".join(categories) + " |",
    "|---|---:|" + "---:|" * len(categories),
]
for e in order:
    tot = overall_count.get(e, 0)
    row = [NAME.get(e, e), f"{tot} ({100*tot/N_test:.1f}%)"]
    for c in categories:
        denom = cat_totals.get(c, 0)
        n_e = cat_expert_count[c].get(e, 0)
        row.append(f"{100*n_e/denom:.1f}%" if denom else "-")
    lines.append("| " + " | ".join(row) + " |")

txt = "\n".join(lines) + "\n"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")
