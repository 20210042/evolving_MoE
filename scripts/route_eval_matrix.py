#!/usr/bin/env python3
"""라우팅 방법론 평가 매트릭스 — 실제 926 llama solve 매트릭스로 top-k union coverage.

생성/병합 없이 라우팅 '품질'만 격리 평가(사용자: "942문제 결과로 라우팅 평가가 더 정확").
각 방법이 문제별 expert 상위 k를 고르면, 그 k 중 하나라도 실제로 푸는지(union coverage).
- 무학습: random-k / fixed best-single / fixed oracle-best-k / conf(raw,z,rank) / conf+prior / pred-agreement
- 학습: MLP(hs / emb / ansprob / conf / hs+conf) 5-fold CV (llama val solve 위에서만 — cross-backbone 라벨 없음)
- 상한: oracle union(문제별 solver 존재시 성공)
출력: 콘솔 표 + results/qasc/seed20210211/routing_eval_matrix.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

R = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
O = R / "results/embed_viz_test"
BINNED = R / "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl"
OUT = R / "results/qasc/seed20210211/routing_eval_matrix.md"
KS = [1, 2, 3]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)


def ids_json(p):
    return [str(x) for x in json.load(open(p))]


# --- 정답지: 실제 llama solve 매트릭스 S (N×E), 순서 = binned ---
binned = [json.loads(l) for l in open(BINNED)]
bids = [str(r["id"]) for r in binned]
EX = sorted(binned[0]["per_expert"])
S = np.array([[r["per_expert"].get(e, 0) for e in EX] for r in binned], np.float32)
N, E = S.shape
pos = {p: i for i, p in enumerate(bids)}
best_single = 100 * S.mean(0).max()
oracle = 100 * (S.sum(1) > 0).mean()


def align(fp, ip):
    """feature 파일을 binned id 순서로 정렬."""
    F = np.load(fp)
    fid = ids_json(ip)
    idx = {p: i for i, p in enumerate(fid)}
    return np.array([F[idx[p]] for p in bids], np.float32)


# --- feature 로드 (전부 binned 순서로 정렬) ---
conf_d = np.load(O / "qasc_validation_lora_conf.npz", allow_pickle=True)
c_ids = [str(x) for x in conf_d["ids"]]
c_ex = [str(x) for x in conf_d["experts"]]
# conf/pred를 EX(정렬된 expert 순서), bids(문제 순서)에 맞춤
ci = {p: i for i, p in enumerate(c_ids)}
cj = {e: j for j, e in enumerate(c_ex)}
CONF = np.array([[conf_d["conf"][ci[p], cj[e]] for e in EX] for p in bids], np.float32)  # (N,E) self-confidence
PRED = np.array([[conf_d["pred"][ci[p], cj[e]] for e in EX] for p in bids], np.int64)     # (N,E) 예측 letter idx

HS = align(O / "qasc_validation_hs_mean.npy", O / "qasc_validation_hs_ids.json")
ANS = align(O / "qasc_validation_ansprob.npy", O / "qasc_validation_ansprob_ids.json")
EMB_raw = np.load(O / "qasc_val_emb.npy")
emb_ids = [str(json.loads(l)["id"]) for l in open(R / "export/qasc/qasc_validation.jsonl")]
eidx = {p: i for i, p in enumerate(emb_ids)}
EMB = np.array([EMB_raw[eidx[p]] for p in bids], np.float32)


def cover(topk_idx):
    """topk_idx: (N,k) expert 인덱스 → union coverage %."""
    return 100 * np.mean([S[i, topk_idx[i]].max() for i in range(N)])


def rank_topk(score, k):
    """score (N,E) 내림차순 상위 k 인덱스 (N,k)."""
    return score.argsort(1)[:, -k:][:, ::-1]


def zscore_cols(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


def rank_cols(X):
    return np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float32)


# ============ 무학습 방법 ============
rows = {}  # name -> {k: cover%}

# 1) random-k (Monte Carlo)
rng = np.random.default_rng(0)
rand = {}
for k in KS:
    accs = []
    for _ in range(500):
        pick = np.array([rng.choice(E, k, replace=False) for _ in range(N)])
        accs.append(cover(pick))
    rand[k] = float(np.mean(accs))
rows["random-k"] = rand

# 2) fixed best-single set (global mean solve 상위 k 고정)
order = np.argsort(-S.mean(0))
rows["fixed best-by-mean"] = {k: cover(np.tile(order[:k], (N, 1))) for k in KS}

# 3) fixed oracle-best-k subset (val union 최대 고정 부분집합; val에 fit=낙관적)
from itertools import combinations
fbest = {}
for k in KS:
    best = -1
    for comb in combinations(range(E), k):
        c = 100 * np.mean(S[:, list(comb)].max(1))
        if c > best:
            best = c
    fbest[k] = best
rows["fixed oracle-best-set"] = fbest

# 4) confidence raw
rows["conf raw"] = {k: cover(rank_topk(CONF, k)) for k in KS}
# 5) confidence z-norm (per-expert)
rows["conf z-norm"] = {k: cover(rank_topk(zscore_cols(CONF), k)) for k in KS}
# 6) confidence rank-norm (per-expert)
rows["conf rank-norm"] = {k: cover(rank_topk(rank_cols(CONF), k)) for k in KS}
# 7) confidence z-norm + solve prior (전역 잘푸는 expert 가산)
prior = S.mean(0, keepdims=True)  # (1,E)
rows["conf z + prior"] = {k: cover(rank_topk(zscore_cols(CONF) + 3.0 * (prior - prior.mean()) / (prior.std() + 1e-6), k)) for k in KS}
# 8) pred-agreement (conf 가중 plurality letter에 동의하는 expert 우선, tie=conf)
agree = np.zeros((N, E), np.float32)
for i in range(N):
    votes = np.zeros(8, np.float32)
    for e in range(E):
        votes[PRED[i, e]] += CONF[i, e]
    plur = votes.argmax()
    for e in range(E):
        agree[i, e] = (1.0 if PRED[i, e] == plur else 0.0) + 0.01 * CONF[i, e]
rows["pred-agreement"] = {k: cover(rank_topk(agree, k)) for k in KS}


# ============ 학습 라우터 (5-fold CV, llama val solve 위) ============
def mlp(d, hid=512, nl=2, drop=0.3):
    L = [nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    L += [nn.Linear(hid, E)]
    return nn.Sequential(*L)


def cv_router(X, name, folds=5, ep=120, seeds=(0, 1, 2)):
    """5-fold CV로 held-out logit 산출 → top-k union coverage."""
    Xt = torch.tensor(X)
    St = torch.tensor(S)
    idx = np.arange(N)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    fold = np.array_split(idx, folds)
    logits = np.zeros((N, E), np.float32)
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xtr = torch.tensor(((X[tr] - mu) / sd).astype(np.float32)).to(DEV)
        Xte = torch.tensor(((X[te] - mu) / sd).astype(np.float32)).to(DEV)
        Str = St[tr].to(DEV)
        seed_logs = []
        for s in seeds:
            torch.manual_seed(s)
            net = mlp(X.shape[1]).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
            lf = nn.BCEWithLogitsLoss()
            for _ in range(ep):
                p = torch.randperm(len(Xtr), device=DEV)
                for i in range(0, len(Xtr), 256):
                    b = p[i:i + 256]
                    opt.zero_grad()
                    lf(net(Xtr[b]), Str[b]).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                seed_logs.append(net(Xte).cpu().numpy())
        logits[te] = np.mean(seed_logs, 0)
    return {k: cover(rank_topk(logits, k)) for k in KS}, name


for X, nm in [(HS, "MLP hidden-state"), (EMB, "MLP encoder-emb"),
              (ANS, "MLP answer-prob"), (CONF, "MLP confidence"),
              (np.concatenate([HS, CONF], 1), "MLP hs+conf")]:
    res, nm = cv_router(X, nm)
    rows[nm] = res
    print(f"  [{nm}] done: " + " ".join(f"k{k}={res[k]:.1f}" for k in KS), flush=True)

# oracle 상한
rows["oracle union"] = {k: oracle for k in KS}


# ============ 출력 ============
ordered = ["random-k", "fixed best-by-mean", "fixed oracle-best-set",
           "conf raw", "conf z-norm", "conf rank-norm", "conf z + prior", "pred-agreement",
           "MLP hidden-state", "MLP encoder-emb", "MLP answer-prob", "MLP confidence", "MLP hs+conf",
           "oracle union"]
hdr = f"QASC val {N} — routing 방법론 × top-k union coverage (실제 llama solve 매트릭스)\n"
hdr += f"기준선: best-single {best_single:.1f} / oracle-union {oracle:.1f}\n"
tbl = "| 방법 | top-1 | top-2 | top-3 | Δtop2−best |\n|---|---|---|---|---|\n"
for nm in ordered:
    r = rows[nm]
    d2 = r[2] - best_single
    tbl += f"| {nm} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {d2:+.1f} |\n"

print("\n" + hdr)
print(tbl)
OUT.write_text(hdr + "\n" + tbl, encoding="utf-8")
print(f"saved -> {OUT}")
