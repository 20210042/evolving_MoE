#!/usr/bin/env python3
"""seed20211004 로스터 top-1 라우팅 보고표 — 버킷별 성능.

라우터: soft-label softmax CE MLP (router_sweep_max_nsolved.py와 동일 설계).
  학습 = train binning(11,097 × 12) + encoder 임베딩(acc_trainfull_emb)
  평가 = test751 (held-out) — expert별 생성물이 이미 있으므로 "top-1로 고른다"는 것은
        그 expert의 실제 생성 결과를 그대로 쓰는 것과 동치다(재생성 불필요).

버킷(test751 solve 매트릭스 기준):
  all_failed  : 12명 전원 실패        → 어떤 라우터도 0%
  all_passed  : 12명 전원 성공        → 어떤 라우터도 100%
  contested   : 그 사이               → **라우팅이 실제로 값을 하는 유일한 구간**
  wo_all_failed = all_passed + contested
  overall       = 전체

⚠️ overall/wo_all_failed 숫자는 대부분 버킷 구성이 결정한다. 방법 간 차이는 contested에서만
생기므로, 보고 시 contested 열을 반드시 같이 봐야 한다.

대조군: random-1(무작위 1명) · best-single(전역 최고 1명 고정, 입력 무시) · oracle(상한).

Usage:
  python scripts/router_top1_report.py --feature emb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_matrix(path: Path, experts: list | None = None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if experts is None:
        experts = list(rows[0]["per_expert"].keys())
    ids = [str(r["id"]) for r in rows]
    S = np.array([[int(r["per_expert"].get(e, 0)) for e in experts] for r in rows], np.float32)
    return ids, S, experts


def align(ids: list, S: np.ndarray, npy: Path, ids_json: Path):
    X_all = np.load(npy)
    fid = [str(x) for x in json.load(open(ids_json, encoding="utf-8"))]
    pos = {p: i for i, p in enumerate(fid)}
    keep = [i for i, p in enumerate(ids) if p in pos]
    X = X_all[[pos[ids[i]] for i in keep]].astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X, S[keep], [ids[i] for i in keep]


def mlp(d: int, e: int, hidden: int, dropout: float):
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_binned", default="results/acc/seed20211004/binning_train_full.binned.jsonl")
    ap.add_argument("--test_binned", default="results/acc/seed20211004/binning_test_full.binned.jsonl")
    ap.add_argument("--train_feat", default="results/embed_viz_test/acc_trainfull_emb.npy")
    ap.add_argument("--train_ids", default="results/embed_viz_test/acc_trainfull_emb_ids.json")
    ap.add_argument("--test_feat", default="results/embed_viz_test/acc_test_emb.npy")
    ap.add_argument("--test_ids", default="results/embed_viz_test/acc_test_emb_ids.json")
    ap.add_argument("--tau", type=int, default=0, help=">0이면 1<=n_solved<=tau 문제만 학습에 사용")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/acc/seed20211004/router_top1_report.md")
    a = ap.parse_args()

    tr_ids, Str, experts = load_matrix(ROOT / a.train_binned)
    te_ids, Ste, _ = load_matrix(ROOT / a.test_binned, experts)
    Xtr, Str, _ = align(tr_ids, Str, ROOT / a.train_feat, ROOT / a.train_ids)
    Xte, Ste, te_ids = align(te_ids, Ste, ROOT / a.test_feat, ROOT / a.test_ids)
    E = len(experts)

    n = Ste.sum(1)
    buckets = {
        "all_failed": n == 0,
        "all_passed": n == E,
        "contested": (n > 0) & (n < E),
        "wo_all_failed": n > 0,
        "overall": np.ones(len(n), bool),
    }
    print("버킷 크기:", {k: int(v.sum()) for k, v in buckets.items()}, flush=True)

    # 학습 세트: 신호 있는 문제만(전원해결/전원실패는 어떤 선택도 결과가 같다)
    ntr = Str.sum(1)
    sel = (ntr > 0) & (ntr < E)
    if a.tau > 0:
        sel &= ntr <= a.tau
    Xf, Sf = Xtr[sel], Str[sel]
    print(f"학습 문제 {int(sel.sum()):,} / {len(ntr):,} (contested만)", flush=True)

    tgt = Sf / Sf.sum(1, keepdims=True)
    Xf_t = torch.tensor(Xf, device=DEV)
    tgt_t = torch.tensor(tgt, device=DEV)
    Xte_t = torch.tensor(Xte, device=DEV)

    picks = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        net = mlp(Xf.shape[1], E, a.hidden, a.dropout).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(len(Xf_t), device=DEV)
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]
                opt.zero_grad()
                logp = torch.log_softmax(net(Xf_t[idx]), -1)
                loss = -(tgt_t[idx] * logp).sum(1).mean()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            picks.append(net(Xte_t).argmax(1).cpu().numpy())

    def bucket_acc(correct: np.ndarray) -> dict:
        return {k: (100.0 * correct[m].mean() if m.sum() else float("nan"))
                for k, m in buckets.items()}

    rows = {}
    # 학습 라우터 (seed 평균)
    accs = [bucket_acc(Ste[np.arange(len(Ste)), p]) for p in picks]
    rows["MLP top-1 (학습 라우터)"] = {k: float(np.mean([d[k] for d in accs])) for k in buckets}
    # random-1
    rng = np.random.default_rng(0)
    r = [bucket_acc(Ste[np.arange(len(Ste)), rng.integers(0, E, len(Ste))]) for _ in range(50)]
    rows["random-1 (무작위 1명)"] = {k: float(np.mean([d[k] for d in r])) for k in buckets}
    # best-single (전역 최고 1명 고정)
    best = int(Ste.sum(0).argmax())
    rows[f"best-single 고정 (`{experts[best]}`)"] = bucket_acc(Ste[:, best])
    # oracle
    rows["oracle top-1 (상한)"] = bucket_acc(Ste.max(1))

    order = ["overall", "wo_all_failed", "all_failed", "all_passed", "contested"]
    head = {"overall": f"Overall [{int(buckets['overall'].sum())}]",
            "wo_all_failed": f"w/o All Failed [{int(buckets['wo_all_failed'].sum())}]",
            "all_failed": f"All Failed [{int(buckets['all_failed'].sum())}]",
            "all_passed": f"All Passed [{int(buckets['all_passed'].sum())}]",
            "contested": f"**Contested [{int(buckets['contested'].sum())}]**"}

    L = ["# seed20211004 로스터 top-1 라우팅 — 버킷별 성능 (test 751)", "",
         f"- 라우터: soft-label CE MLP · 특징 `{Path(a.test_feat).name}` · "
         f"학습 = train binning contested {int(sel.sum()):,}문제 · seed {a.seeds} 평균",
         f"- expert {E}명. top-1 선택 = 그 expert의 실제 생성 결과를 그대로 쓰는 것(재생성 없음).", "",
         "| Method | " + " | ".join(head[k] for k in order) + " |",
         "|---|" + "---:|" * len(order)]
    for name, d in rows.items():
        L.append(f"| {name} | " + " | ".join(f"{d[k]:.2f}" for k in order) + " |")
    L += ["", "**해석 주의**: All Failed는 정의상 어떤 방법도 0, All Passed는 100이다. 따라서",
          "Overall·w/o All Failed 숫자는 대부분 **버킷 구성**이 결정하며, 방법 간 실제 차이는",
          "**Contested 열에서만** 생긴다. 버킷을 그 로스터 자신의 solve 매트릭스로 정의했으므로",
          "다른 시스템과 이 표를 가로로 비교하려면 **공통 버킷 정의**가 필요하다.", ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
