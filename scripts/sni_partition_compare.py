#!/usr/bin/env python
"""같은 expert 수 k에서, 어느 분할이 더 많은 상보성을 담는가 — 재생성 0.

MoE는 expert 수가 예산이다(태스크 872개짜리 분할은 못 만든다). 그래서 사람 분할과 우리 분할을
**같은 k에서** 비교한다. 분할마다 두 수를 낸다:

  (a) 오라클 배정  — 그룹마다 train에서 최적 expert를 고르고 test에서 평가.
                     그룹 소속을 알고 있다고 가정 = 그 분할이 담은 상보성의 크기.
  (b) 입력 회수    — hs_last → 그룹 분류기를 train에서 학습해 test 문제의 그룹을 맞히고,
                     그 그룹의 expert로 답한다. 배포에서 실제로 나오는 값.

분할 후보:
  · ours     — 진화 로스터 **16명 그 자체**. 갈림 문제는 p̂ 최대인 사람에게(동점 무작위),
               만장일치 문제(전원 성공/실패)는 어느 쪽에 넣든 점수가 같으므로 분류기 학습에서 뺀다.
  · category — 상위 k-1개 카테고리 + 나머지(사람 상위 택소노미)
  · domain   — 상위 k-1개 도메인 + 나머지
  · random   — 같은 크기 분포의 무작위 분할(하한)

⚠️ '상위 k-1 + 나머지'는 내가 고른 규칙이다. 다른 묶는 규칙을 쓰면 사람 분할 쪽 수치가 달라진다.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def kmeans(X, k, rng, iters=50):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1) if len(X) < 20000 else None
        if d is None:
            g = np.concatenate([((X[i:i+20000, None, :] - C[None]) ** 2).sum(-1).argmin(1)
                                for i in range(0, len(X), 20000)])
        else:
            g = d.argmin(1)
        for j in range(k):
            m = g == j
            if m.any():
                C[j] = X[m].mean(0)
    return g, C


def topk_partition(lab, k):
    u, c = np.unique(lab, return_counts=True)
    top = set(u[np.argsort(-c)[:k - 1]])
    return np.array([x if x in top else "__rest__" for x in lab])


def classify(Xtr, gtr, Xte, k, epochs=60):
    """hs_last → 그룹 분류기(라우터가 감당해야 할 일)."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32).to(DEV)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32).to(DEV)
    y = torch.tensor(gtr, dtype=torch.long).to(DEV)
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(A.shape[1], 512), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(512, k)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        p = torch.randperm(len(A), device=DEV)
        for i in range(0, len(A), 256):
            b = p[i:i + 256]
            opt.zero_grad(); lf(net(A[b]), y[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(B).argmax(1).cpu().numpy(), net(A).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="16")
    ap.add_argument("--out", default="results/sni/partition_compare.md")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sp = rc.spec("sni")
    trb, teb = rc.labels(sp)
    ex = rc.experts(sp)

    ids_tr, Xtr, Str = rc.align(rc.feat_path(sp, "train", "hs_last"),
                                rc.feat_path(sp, "train", "hs_ids"), trb, ex)
    ids_te, Xte, Ste = rc.align(rc.feat_path(sp, "test", "hs_last"),
                                rc.feat_path(sp, "test", "hs_ids"), teb, ex)
    meta = {}
    for split in ("train", "test"):
        for l in open(f"export/sni_v4/sni_{split}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            meta[r["id"]] = (r.get("category") or "?", r.get("sni_domain") or "?")
    cat_tr = np.array([meta[i][0] for i in ids_tr]); cat_te = np.array([meta[i][0] for i in ids_te])
    dom_tr = np.array([meta[i][1] for i in ids_tr]); dom_te = np.array([meta[i][1] for i in ids_te])
    Dtr = (Str - Str.mean(1, keepdims=True)).astype(np.float32)
    Dte = (Ste - Ste.mean(1, keepdims=True)).astype(np.float32)

    bs = 100 * Ste[:, int(Str.mean(0).argmax())].mean()
    orc = 100 * Ste.max(1).mean()
    L = ["# 같은 expert 수에서 어느 분할이 나은가 (재생성 0)", "",
         f"- train {len(Xtr):,} / test {len(Xte):,} · 라벨 p̂(K=3)",
         f"- best-single(train 선택) **{bs:.2f}** · 문제별 오라클 top-1 {orc:.2f} · 폭 {orc-bs:.2f}pp",
         "- (a) 오라클 배정 = 그룹 소속을 안다고 가정 · (b) 입력 회수 = hs_last로 그룹을 맞혀 배정", ""]

    for k in [int(x) for x in a.ks.split(",")]:
        L += [f"## k = {k}", "",
              "사다리: `입력 회수` ≤ `그룹 앎(train 배정)` ≤ `그 분할의 오라클(test 배정)` ≤ `문제별 오라클`",
              "칸 사이 = 라우팅 손실 / 일반화 손실 / 라벨 해상도 한계", "",
              "| 분할 | 입력 회수 | 그룹 앎(train) | 분할 오라클(test) | 라우팅 손실 | 일반화 손실 | 해상도 한계 | 분류 정확도 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        parts = {}
        if k == len(ex):     # 우리 분할은 로스터 크기에서만 정의된다
            tie = rng.random(Dtr.shape) * 1e-9
            parts["ours (진화 16명 배정)"] = ((Dtr + tie).argmax(1),
                                              (Dte + rng.random(Dte.shape) * 1e-9).argmax(1))
        for tag, ltr, lte in (("category 상위", cat_tr, cat_te), ("domain 상위", dom_tr, dom_te)):
            ptr, pte = topk_partition(ltr, k), topk_partition(lte, k)
            u = {v: i for i, v in enumerate(sorted(set(ptr) | set(pte)))}
            parts[tag] = (np.array([u[x] for x in ptr]), np.array([u[x] for x in pte]))
        parts["random (균등)"] = (rng.integers(0, k, len(Xtr)), rng.integers(0, k, len(Xte)))
        # 우리가 실제로 만든 대조군 분할(export/sni_split_random)은 expert별 총량이 고르지 않다
        # (6,845 ~ 2,573). 그 크기 분포를 그대로 쓴 무작위 배정도 같이 잰다.
        sf = Path("export/sni_split_seed20212003/split.jsonl")
        if k == len(ex) and sf.exists():
            tot = np.zeros(k)
            for line in open(sf, encoding="utf-8"):
                d = json.loads(line)
                for c in d["experts"]:
                    if c in ex:
                        tot[ex.index(c)] += 1
            pr = tot / tot.sum()
            parts["random (우리 분할 크기 맞춤)"] = (
                rng.choice(k, len(Xtr), p=pr), rng.choice(k, len(Xte), p=pr))

        for tag, (gtr, gte_) in parts.items():
            # 만장일치 문제는 배정이 점수를 못 바꾸므로 분류기 학습에서 제외한다
            ns = (Str > 0.5).sum(1)
            con = (ns > 0) & (ns < Str.shape[1])
            pick = {}
            for j in np.unique(gtr):
                m = gtr == j
                pick[j] = int(Str[m].mean(0).argmax())
            known = 100 * float(np.mean([Ste[i, pick.get(gte_[i], 0)] for i in range(len(Ste))]))
            # 그 분할의 오라클: 그룹별 최적을 **test에서** 고른다(= 라벨 해상도 천장)
            pick_te = {}
            for j in np.unique(gte_):
                m = gte_ == j
                pick_te[j] = int(Ste[m].mean(0).argmax())
            oracle = 100 * float(np.mean([Ste[i, pick_te[gte_[i]]] for i in range(len(Ste))]))
            pred, _ = classify(Xtr[con], gtr[con], Xte, int(max(gtr.max(), gte_.max())) + 1)
            got = 100 * float(np.mean([Ste[i, pick.get(pred[i], 0)] for i in range(len(Ste))]))
            acc = 100 * float((pred == gte_).mean())
            L.append(f"| {tag} | {got:.2f} | {known:.2f} | {oracle:.2f} | {known-got:.2f}pp | "
                     f"{oracle-known:.2f}pp | {orc-oracle:.2f}pp | {acc:.1f}% |")
        L.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
