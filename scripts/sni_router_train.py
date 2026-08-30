#!/usr/bin/env python
"""SNI 라우터 — multi-label(expert별 독립 이진분류) MLP 두 형태 비교.

형태 A `head`     : 입력=문제 벡터, 출력=expert 16개 로짓(`Linear(hid, E)`).
                    전문가 표현이 **라벨에서** 학습된다. 기존 도메인이 쓰던 형태.
형태 B `profile`  : 입력에 **전문가 프로파일 벡터**(각자의 system prompt를 같은 모델로 뽑은 것)를
                    결합한 two-tower. logits = P(문제) · Q(프로파일)ᵀ.
                    로스터가 바뀌어도 프로파일만 넣으면 점수가 나온다.
대조군 `profile_random` : 프로파일 자리에 같은 노름의 **난수 벡터**를 넣는다.
    ⚠️ 프로파일을 '섞는' 건 대조군이 못 된다 — 순열도 전단사라 모델이 그냥 다시 외운다.
    텍스트 정보가 실제로 쓰이는지 보려면 정보를 없애야 한다. 성능이 그대로면
    프로파일은 "서로 다른 ID" 이상의 역할을 안 한 것이다.

프로토콜: train을 fit 90% / dev 10%로 갈라 **dev에서 하이퍼파라미터를 고르고**,
고른 것 하나만 test에 적용한다(test에서 고르지 않는다).
손실은 BCE, 타깃은 p̂ 소프트 라벨(K=3 → {0,⅓,⅔,1}).

Usage: python3 scripts/sni_router_train.py --out results/sni/router_2003.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# README_router.md의 기존 스윕 격자 (hid, layers, epoch, dropout, wd)
GRID = [(256, 2, 100, .2, 1e-3), (512, 2, 120, .3, 1e-3), (1024, 2, 150, .3, 1e-3),
        (2048, 2, 150, .3, 1e-2), (1024, 3, 150, .3, 1e-2), (2048, 3, 200, .4, 1e-2)]
SEEDS = [42, 1, 7]


def mlp(d_in, hid, nl, drop, d_out):
    L = [nn.Linear(d_in, hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    L += [nn.Linear(hid, d_out)]
    return nn.Sequential(*L)


class Head(nn.Module):
    """형태 A — 전문가 표현이 마지막 층 가중치."""
    def __init__(self, d, hid, nl, drop, E, _P=None):
        super().__init__()
        self.net = mlp(d, hid, nl, drop, E)

    def forward(self, x):
        return self.net(x)


class Profile(nn.Module):
    """형태 B — 전문가 프로파일 벡터를 사영해 내적."""
    def __init__(self, d, hid, nl, drop, E, P=None):
        super().__init__()
        dim = 256
        self.p = mlp(d, hid, nl, drop, dim)
        self.register_buffer("P", P)                     # (E, d_e) 고정
        self.q = nn.Sequential(nn.Linear(P.shape[1], hid), nn.ReLU(), nn.Linear(hid, dim))
        self.b = nn.Parameter(torch.zeros(E))

    def forward(self, x):
        return self.p(x) @ self.q(self.P).T + self.b


def routed(logit, S, k=1):
    """라우팅 정확도. k=1은 top-1의 p̂ 평균, k=2는 두 명을 다 굴렸을 때의 기대 성공률."""
    idx = torch.topk(torch.as_tensor(logit), k, dim=1).indices.numpy()
    if k == 1:
        return 100 * float(np.mean(S[np.arange(len(S)), idx[:, 0]]))
    miss = np.ones(len(S))
    for j in range(k):
        miss = miss * (1 - S[np.arange(len(S)), idx[:, j]])
    return 100 * float(np.mean(1 - miss))


def train_once(cls, Xf, Sf, Xd, Xt, P, cfg, seed):
    hid, nl, ep, drop, wd = cfg
    torch.manual_seed(seed)
    net = cls(Xf.shape[1], hid, nl, drop, Sf.shape[1], P).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    n = len(Xf)
    for _ in range(ep):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(net(Xf[b]), Sf[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return (net(Xd).cpu().numpy(), net(Xt).cpu().numpy(),
                net(Xf[:20000]).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", nargs="*", default=["hs_mean", "hs_last"])
    ap.add_argument("--out", default="results/sni/router_2003.md")
    ap.add_argument("--train_on", choices=["all", "contested"], default="all",
                    help="contested = 만장일치 문제(아무도 못 품 26.2% + 전원 15.7%)를 "
                         "학습에서 뺀다. 그 구간은 누구를 골라도 결과가 같아 손실만 "
                         "난이도 예측 쪽으로 끌고 간다. 평가는 항상 전체 + 갈림 두 열.")
    a = ap.parse_args()
    sp = rc.spec("sni")
    trb, teb = rc.labels(sp)
    ex = rc.experts(sp)
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}

    Pall = np.load(rc.FEAT_DIR / "sni_experts_hs_mean.npy")
    pid = json.load(open(rc.FEAT_DIR / "sni_experts_ids.json"))
    Praw = np.array([Pall[pid.index(e)] for e in ex], dtype=np.float32)
    # ⚠️ 프로파일도 문제 벡터와 **같은 처리**를 해야 한다. raw로 넣으면 gemma hidden state의
    # 공통 성분(노름 158.5 중 중심화 후 26.0만 남음 = 6배가 공통 방향)이 16명을 거의 같은
    # 입력으로 만든다. 실측: 원본 코사인 중앙값 0.977 → 전문가 평균 제거 후 -0.079.
    # (문제끼리도 원본 0.984이므로 0.977은 프롬프트 유사성이 아니라 표현 공간의 성질이다.)
    Pc = (Praw - Praw.mean(0)) / (Praw.std(0) + 1e-6)
    P = torch.tensor(Pc, dtype=torch.float32)
    rng = np.random.default_rng(0)
    # 대조군: 같은 분포(z-정규화된 좌표)의 난수. 텍스트 정보만 없앤다.
    Prand = torch.tensor(rng.normal(size=P.shape).astype(np.float32))

    L = ["# SNI 라우터 — multi-label MLP 두 형태", "",
         f"- 로스터 {len(ex)}명 · 라벨 p̂ 소프트(K=3) · 손실 BCE",
         "- train 90%로 학습 / 10%(dev)로 하이퍼파라미터 선택 / test는 고른 것 한 번만", ""]

    for feat in a.feat:
        ids_tr, Xtr, Str = rc.align(rc.feat_path(sp, "train", feat),
                                    rc.feat_path(sp, "train", "hs_ids"), trb, ex)
        ids_te, Xte, Ste = rc.align(rc.feat_path(sp, "test", feat),
                                    rc.feat_path(sp, "test", "hs_ids"), teb, ex)
        idx = np.random.default_rng(0).permutation(len(Xtr))
        cut = int(len(idx) * 0.9)
        fi, di = idx[:cut], idx[cut:]
        n_all = len(fi)
        if a.train_on == "contested":
            h = (Str[fi] > 0.5).sum(1)
            fi = fi[(h > 0) & (h < Str.shape[1])]   # dev/test는 손대지 않는다
        mu, sd = Xtr[fi].mean(0, keepdims=True), Xtr[fi].std(0, keepdims=True) + 1e-6
        Z = lambda M: torch.tensor((M - mu) / sd, dtype=torch.float32).to(DEV)
        Xf, Xd, Xt = Z(Xtr[fi]), Z(Xtr[di]), Z(Xte)
        Sf = torch.tensor(Str[fi], dtype=torch.float32).to(DEV)
        Sd = Str[di]

        bs_te, or_te = rc.baselines(Ste)
        bs_d, or_d = rc.baselines(Sd)
        # 갈림 구간(0 < 푼 사람 < 전원) — 라우터 선택이 결과를 바꾸는 유일한 구간
        ct = ((Ste > 0.5).sum(1) > 0) & ((Ste > 0.5).sum(1) < Ste.shape[1])
        cd = ((Sd > 0.5).sum(1) > 0) & ((Sd > 0.5).sum(1) < Sd.shape[1])
        bs_ct, or_ct = rc.baselines(Ste[ct])
        L += [f"## 특징 `{feat}`", "",
              f"- 학습 대상: {a.train_on} ({len(fi):,}/{n_all:,} 문제)",
              f"- test 전체 {len(Ste):,}: best-single {bs_te:.2f} · oracle {or_te:.2f}",
              f"- test 갈림 {int(ct.sum()):,}: best-single {bs_ct:.2f} · oracle {or_ct:.2f}",
              f"- dev  전체 {len(Sd):,}: best-single {bs_d:.2f} · oracle {or_d:.2f}", "",
              "### 격자 전체",  "",
              "| 형태 | 설정 (hid,nl,ep,drop,wd) | dev top-1 | dev 갈림 |",
              "|---|---|---:|---:|"]
        picks = []
        for tag, cls, Pv in (("head (출력 E개)", Head, P),
                             ("profile (프로파일 결합)", Profile, P),
                             ("profile_random (대조군)", Profile, Prand)):
            best = None
            for cfg in GRID:
                ld, lt, lf_ = [], [], []
                for s in SEEDS:
                    d_, t_, f_ = train_once(cls, Xf, Sf, Xd, Xt,
                                            Pv.to(DEV) if Pv is not None else None, cfg, s)
                    ld.append(d_); lt.append(t_); lf_.append(f_)
                Ld, Lt, Lf = np.mean(ld, 0), np.mean(lt, 0), np.mean(lf_, 0)
                dv, dvc = routed(Ld, Sd), routed(Ld[cd], Sd[cd])
                L.append(f"| {tag} | {cfg} | {dv:.2f} | {dvc:.2f} |")
                if best is None or dv > best[0]:      # 선택은 dev 전체 top-1(배포 지표)
                    best = (dv, Lt, Lf, cfg)
                print(f"  [{feat}|{tag}] {cfg} dev {dv:.2f} (갈림 {dvc:.2f})", flush=True)
            picks.append((tag, best))
        L += ["", "### 고른 설정으로 test", "",
              "| 형태 | dev top-1 | **test top-1** | test 갈림 top-1 | test top-2 | "
              "train top-1 (외웠나) | 설정 |", "|---|---:|---:|---:|---:|---:|---|"]
        Sfit = Str[fi]
        for tag, (dv, Lt, Lf, cfg) in picks:
            L.append(f"| {tag} | {dv:.2f} | **{routed(Lt, Ste):.2f}** | "
                     f"{routed(Lt[ct], Ste[ct]):.2f} | {routed(Lt, Ste, 2):.2f} | "
                     f"{routed(Lf, Sfit[:len(Lf)]):.2f} | {cfg} |")
        lt = picks[1][1][1]
        # 라우터가 누구를 고르는가
        L += ["", "선택 분포(profile 형태, test):", ""]
        pick = np.bincount(lt.argmax(1), minlength=len(ex))
        L += ["| 비율 | expert |", "|---:|---|"] + [
            f"| {pick[i]/len(Ste)*100:5.1f}% | {names.get(ex[i], ex[i])} |"
            for i in np.argsort(-pick)[:5]] + [""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
