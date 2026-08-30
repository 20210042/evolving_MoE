#!/usr/bin/env python3
"""top-1 라우터 재설계: soft-label softmax CE + max_n_solved 스윕.

§11-b의 MLP 라우터(moe_deploy_top1.py::cv_top1)는 두 가지가 top-1 배포와 안 맞았다.

1. BCEWithLogitsLoss(멀티라벨) — top-1은 결국 argmax 하나만 쓰는데, 학습은 "각 expert가
   풀지 못 풀지"를 서로 독립적으로 맞히도록만 시켰다. n_solved가 큰(여러 명이 동시에 푸는)
   문제일수록 "누가 더 나은가"를 구분하는 신호가 약해진다.
2. 5-fold CV를 test751(751문제) **안에서만** 돌렸다 — train split(7079문제, 이미
   per-expert solve 라벨이 있는 binning_labels.jsonl)은 hidden-state조차 뽑혀 있지 않아
   전혀 안 쓰였다.

이 스크립트는:
- 라벨: S(문제×expert, 0/1)를 행 정규화한 soft target(합=1)으로 바꾸고 CrossEntropy로 학습
  (n_solved==1이면 one-hot, n_solved가 클수록 평평해짐 — 랜덤 하드픽 불필요).
- 데이터: train split(binning_labels.jsonl, 11 persona expert — `shared`는 train 신호가
  없어 후보에서 제외)으로 학습하고 test751(11-expert 서브셋)에서 진짜 held-out 평가.
- 스윕: max_n_solved(τ)를 1..11로 늘려가며 "1<=n_solved<=τ" 문제만 학습에 포함 —
  τ가 작을수록 신호는 맑아지지만(most-solved 문제 배제) 데이터가 준다. 그 반대 트레이드오프의
  최적점을 찾는다.
- 입력 특징(--feature): base LLM hidden-state(hs_mean/hs_last, "문제를 이 LLM이 어떻게
  내부 표현하는가")과 encoder 임베딩(emb, google/embeddinggemma-300m — "문제가 의미적으로
  뭘 말하는가") 둘 다 시도 가능. 클러스터링에서 이미 나온 "설명 축 ≠ solvability 축" 진단이
  맞다면 어느 쪽을 넣어도 MLP가 그 축 전환 자체를 학습하는 데 실패할 수 있다.
- 학습곡선(--log_every): 매 seed의 20 epoch마다 (train loss, train top-1, test top-1)을
  찍어서 120 epoch가 부족한지(아직 내려가는 중) 과다한지(train-test 갭 벌어짐=암기)를 본다.

Usage:
  python scripts/router_sweep_max_nsolved.py --dataset acc --feature hs_mean
  python scripts/router_sweep_max_nsolved.py --dataset acc --feature emb
"""
from __future__ import annotations

import argparse
import sys
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
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--log_every", type=int, default=20, help="이 epoch 간격으로 학습곡선 출력(0=끄기)")
ap.add_argument("--taus", type=int, nargs="+", default=None, help="기본은 1..E 전체")
ap.add_argument("--out", default=None)
A = ap.parse_args()

sp = rc.spec(A.dataset)
OUT = Path(A.out) if A.out else REPO / "results" / A.dataset / f"router_sweep_max_nsolved_{A.feature}.md"
np.random.seed(0)
torch.manual_seed(0)

# ---- expert 후보: train binning 라벨에 있는 expert만 (shared 등 train 신호 없는 건 자동 제외) ----
EX = rc.experts(sp)
E = len(EX)
print(f"experts({E}, train-labeled only): {EX}", flush=True)


def load_features(which: str):
    if A.feature == "emb":
        npy, ids_p = rc.emb_paths(sp, which)
    else:
        split = sp.train_split if which == "train" else sp.eval_split
        npy, ids_p = rc.feat_path(sp, split, A.feature), rc.feat_path(sp, split, "hs_ids")
    return npy, ids_p


# ---- train: binning_labels 전체 + 입력 특징 ----
train_bin = rc.load_binning(REPO / sp.labels)
feat_npy, feat_ids = load_features("train")
train_ids, X_train, S_train = rc.align(feat_npy, feat_ids, train_bin, EX)
n_solved_train = S_train.sum(1)
print(f"train: {len(train_ids)}문제, n_solved 분포 {np.bincount(n_solved_train.astype(int)).tolist()}",
      flush=True)

# ---- test: eval_binned(shared 포함 12명일 수 있음) 중 EX(11명)만 사용 ----
test_bin = rc.load_binning(REPO / sp.eval_binned)
feat_npy, feat_ids = load_features("eval")
test_ids, X_test, S_test = rc.align(feat_npy, feat_ids, test_bin, EX)
N_test = len(test_ids)
best_single, oracle_union = rc.baselines(S_test)
rng = np.random.default_rng(0)
random_acc = 100 * np.mean([S_test[i, rng.integers(0, E)] for i in range(N_test)])
print(f"test: {N_test}문제  best-single={best_single:.2f}  oracle-union={oracle_union:.2f}  "
      f"random={random_acc:.2f}  feature={A.feature}(dim={X_train.shape[1]})", flush=True)

# ---- "결정 가능한" 서브셋: 아무도 못 풀거나(n_solved=0) 전원이 푸는(n_solved=E) 문제는
# 라우팅 선택과 무관하게 결과가 고정된다. 이걸 뺀 문제에서만 라우터의 실제 판별력을 본다. ----
n_solved_test = S_test.sum(1)
decidable = (n_solved_test >= 1) & (n_solved_test < E)
n_dec = int(decidable.sum())
cond_best_single = 100 * S_test[decidable].mean(0).max()
cond_random = 100 * np.mean([S_test[i, rng.integers(0, E)] for i in np.flatnonzero(decidable)])
print(f"decidable subset: {n_dec}/{N_test}문제(0<n_solved<{E})  "
      f"cond-best-single={cond_best_single:.2f}  cond-random={cond_random:.2f}  cond-oracle=100.00", flush=True)

# 정규화 통계는 τ와 무관하게 전체 train pool로 한 번만 추정한다.
MU = X_train.mean(0, keepdims=True)
SD = X_train.std(0, keepdims=True) + 1e-6


def mlp(d):
    return nn.Sequential(nn.Linear(d, A.hidden), nn.ReLU(), nn.Dropout(A.dropout), nn.Linear(A.hidden, E))


def soft_ce(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(1).mean()


def top1_acc(net, X, S, eval_mode=True):
    if eval_mode:
        net.eval()
    with torch.no_grad():
        pick = net(X).argmax(1).cpu().numpy()
    return 100 * np.mean([S[i, pick[i]] for i in range(len(S))])


def train_and_eval(tau: int, curve_tag: str | None = None):
    mask = (n_solved_train >= 1) & (n_solved_train <= tau)
    Xtr_raw, Str = X_train[mask], S_train[mask]
    target = Str / Str.sum(1, keepdims=True)
    Xtr = torch.tensor((Xtr_raw - MU) / SD, dtype=torch.float32).to(DEV)
    Ttr = torch.tensor(target, dtype=torch.float32).to(DEV)
    Xte = torch.tensor((X_test - MU) / SD, dtype=torch.float32).to(DEV)
    Str_t = torch.tensor(Str, dtype=torch.float32).to(DEV)

    probs = np.zeros((N_test, E), np.float32)
    n = len(Xtr)
    for s in A.seeds:
        torch.manual_seed(s)
        net = mlp(Xtr.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), A.lr, weight_decay=A.wd)
        for ep in range(A.epochs):
            net.train()
            perm = torch.randperm(n, device=DEV)
            ep_loss = 0.0
            for i in range(0, n, A.batch):
                b = perm[i:i + A.batch]
                opt.zero_grad()
                loss = soft_ce(net(Xtr[b]), Ttr[b])
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(b)
            if curve_tag and A.log_every and (ep + 1) % A.log_every == 0:
                tr_acc = top1_acc(net, Xtr, Str_t.cpu().numpy(), eval_mode=False)
                te_acc = top1_acc(net, Xte, S_test)
                print(f"    [{curve_tag} seed{s}] epoch {ep+1:3d}  loss={ep_loss/n:.4f}  "
                      f"train-top1={tr_acc:.2f}%  test-top1={te_acc:.2f}%", flush=True)
        net.eval()
        with torch.no_grad():
            probs += torch.softmax(net(Xte), dim=1).cpu().numpy()
    probs /= len(A.seeds)
    pick = probs.argmax(1)
    acc = 100 * np.mean([S_test[i, pick[i]] for i in range(N_test)])
    cond_acc = 100 * np.mean([S_test[i, pick[i]] for i in np.flatnonzero(decidable)])
    # 진단: 라우터가 문제별로 다른 expert를 고르는지, 그냥 한 expert로 collapse했는지 확인.
    pick_dec = pick[decidable]
    counts = np.bincount(pick_dec, minlength=E)
    n_unique = int((counts > 0).sum())
    dist = ", ".join(f"{EX[j]}:{counts[j]}" for j in np.argsort(-counts) if counts[j] > 0)
    print(f"    [tau{tau} pick분포(decidable {len(pick_dec)}개)] unique={n_unique}/{E}  {dist}", flush=True)
    return int(mask.sum()), acc, cond_acc


taus = A.taus or list(range(1, E + 1))
print(f"max_n_solved 스윕 (feature={A.feature})...", flush=True)
rows = []
for tau in taus:
    n_used, acc, cond_acc = train_and_eval(tau, curve_tag=f"tau{tau}")
    rows.append((tau, n_used, acc, cond_acc))
    print(f"  tau<={tau:2d}  train_n={n_used:5d}  top-1(전체751)={acc:.2f}%  "
          f"top-1(결정가능172)={cond_acc:.2f}%", flush=True)

lines = [
    f"# {sp.name.upper()} 라우터 재설계 — soft-label CE + max_n_solved 스윕 (feature={A.feature})",
    "",
    f"- expert({E}, train 라벨 있는 것만): {EX}",
    f"- train: {len(train_ids)}문제(binning_labels, n_solved>=1만 존재) · "
    f"test: {N_test}문제({sp.eval_split})",
    f"- 입력 특징: {A.feature} (dim={X_train.shape[1]})",
    f"- 기준선(test 751, {E}-expert 서브셋) — best-single(고정): **{best_single:.2f}%** · "
    f"oracle-union: **{oracle_union:.2f}%** · random: {random_acc:.2f}%",
    f"- 결정가능 서브셋({n_dec}/{N_test}, 0<n_solved<{E}) 기준선 — "
    f"cond-best-single: **{cond_best_single:.2f}%** · cond-random: {cond_random:.2f}% · cond-oracle: 100%",
    "",
    "| max_n_solved(τ) | train 문제수 | top-1(전체751) | top-1(결정가능172) |",
    "|---:|---:|---:|---:|",
]
for tau, n_used, acc, cond_acc in rows:
    lines.append(f"| {tau} | {n_used} | {acc:.2f}% | {cond_acc:.2f}% |")
txt = "\n".join(lines) + "\n"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")
