#!/usr/bin/env python3
"""expert×문제 상호작용이 '문제 내용'으로 예측 가능한가 — 원핫 ANOVA를 저랭크 이중선형으로 교체.

왜 바꾸나:
  원핫 요인 ANOVA에서 문제 주효과는 N-1 자유도로 이미 **포화**라 임베딩(저랭크)이 더 설명할 수
  없다. 임베딩이 실제로 바꾸는 건 **상호작용 항**이다. 원핫 상호작용은 (N-1)(E-1) 자유도로 셀
  수와 맞먹어 식별이 안 되고 잔차를 통째로 흡수한다 — 그래서 기존 분석은 이항 노이즈를
  '모수적으로 가정해서 빼는' 방식으로 3.7%를 얻었다(측정이 아니라 가정).
  저랭크로 바꾸면 파라미터가 줄고 **held-out 셀 예측**으로 검정할 수 있다. 분산 몫 해석 논쟁
  대신 "본 적 없는 (문제,expert) 조합을 주효과 모델보다 잘 맞히는가"라는 반증가능한 질문이 된다.

모형 (셀 y_ij ∈ {0,1}):
  main     : logit = mu + a_i + b_j                     (a_i,b_j 자유 = 원핫 주효과, 포화)
  bilinear : logit = mu + a_i + b_j + (W e_i)·v_j       (e_i = 문제 임베딩, v_j = expert별 자유 벡터)

  v_j를 '자유 벡터'로 두는 게 핵심이다. 이건 어떤 **페르소나 표현(텍스트 임베딩 포함)으로도 넘을 수
  없는 상한**이다. 즉 자유 v_j로도 이득이 없으면 페르소나 임베딩은 볼 것도 없다. 페르소나
  임베딩이 따로 의미를 갖는 건 '새 페르소나로의 일반화'뿐인데 E=11로는 검정력이 없다.

두 가지 split (다른 질문):
  cell     : 셀 무작위 홀드아웃. 문제 i가 학습에 남아 a_i가 추정되므로 **순수 상호작용 검정**.
  problem  : 문제 통째로 홀드아웃. a_i를 못 쓰니 난이도까지 임베딩으로 예측해야 함
             = **라우터 배포 상황**과 동일.

귀무 대조: 문제별로 expert 라벨을 셔플(행 내 permutation) → 주효과는 보존, 상호작용만 파괴.
같은 파이프라인을 돌려 ΔAUC의 귀무 분포를 만든다.

Usage:
  python scripts/interaction_lowrank_test.py --feature emb --ranks 1,2,4,8
  python scripts/interaction_lowrank_test.py --feature hs_last --split problem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FEATS = {
    "emb": ("results/embed_viz_test/acc_train_emb.npy", "results/embed_viz_test/acc_train_emb_ids.json"),
    "hs_last": ("results/embed_viz_test/acc_train_hs_last.npy", "results/embed_viz_test/acc_train_hs_ids.json"),
    "hs_mean": ("results/embed_viz_test/acc_train_hs_mean.npy", "results/embed_viz_test/acc_train_hs_ids.json"),
    # 원핫: 문제마다 자유 벡터(포화). 같은 문제의 다른 expert 칸에서 배워 그 문제의 빈칸을 맞힌다.
    # 임베딩과 **같은 데이터·같은 홀드아웃**에서 돌려 "특징만" 바꾼 대조를 만들기 위한 것.
    # 문제 집합을 임베딩 실행과 똑같이 맞추려고 emb의 id 목록으로 교집합을 잡는다.
    "onehot": ("results/embed_viz_test/acc_train_emb.npy", "results/embed_viz_test/acc_train_emb_ids.json"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binned", default="results/acc/seed20210111/binning_train_full.binned.jsonl")
    ap.add_argument("--feature", default="emb", choices=list(FEATS))
    ap.add_argument("--feat_npy", default="", help="FEATS 기본 경로 대신 쓸 특징 npy(repo 상대)")
    ap.add_argument("--feat_ids", default="", help="그 npy에 대응하는 id json. onehot도 이 id 목록으로 문제집합을 잡는다")
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--split", default="cell", choices=["cell", "problem"])
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-3, help="W,V에 거는 L2")
    ap.add_argument("--wd_main", type=float, default=1e-3, help="a_i에 거는 L2(문제당 관측이 ~E개뿐)")
    ap.add_argument("--n_perm", type=int, default=5, help="귀무 대조 반복 수")
    ap.add_argument("--positive_control", type=float, default=0.0,
                    help=">0이면 그 세기로 인위적 저랭크 상호작용을 라벨에 심고 검출되는지 본다. "
                         "ΔAUC≈0이 '신호 없음'인지 '파이프라인이 못 찾음'인지 가르는 검증.")
    ap.add_argument("--band", default="",
                    help="난이도 구간 제한. 'lo,hi' 형태로 '11명 중 몇 명이 풀었나'를 걸러낸다. "
                         "예: '1,5' = 11명 중 1~5명만 푼 어려운 문제만. 빈 값이면 전체.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/acc/interaction_lowrank.md")
    return ap.parse_args()


# ------------------------------------------------------------------ 데이터
def feat_paths(a: argparse.Namespace) -> tuple:
    """--feat_npy/--feat_ids가 있으면 그걸, 없으면 FEATS 기본값을 쓴다."""
    if a.feat_npy or a.feat_ids:
        if not (a.feat_npy and a.feat_ids):
            raise SystemExit("--feat_npy와 --feat_ids는 함께 줘야 한다")
        return a.feat_npy, a.feat_ids
    return FEATS[a.feature]


def load(a: argparse.Namespace):
    rows = [json.loads(l) for l in open(ROOT / a.binned, encoding="utf-8")]
    experts = list(rows[0]["per_expert"].keys())
    npy, idsj = feat_paths(a)
    X_all = np.load(ROOT / npy)
    ids_all = json.load(open(ROOT / idsj, encoding="utf-8"))
    pos = {pid: k for k, pid in enumerate(ids_all)}
    keep = [r for r in rows if r["id"] in pos]
    if a.band:
        lo, hi = (int(v) for v in a.band.split(","))
        keep = [r for r in keep if lo <= sum(r["per_expert"].values()) <= hi]
        print(f"난이도 구간 {lo}~{hi}명이 푼 문제로 제한 → {len(keep)}문제", flush=True)
    Y = np.array([[r["per_expert"].get(e, 0) for e in experts] for r in keep], np.float32)
    if a.feature == "onehot":
        # 문제 하나에 좌표 하나. 이미 L2 노름 1이라 표준화하지 않는다(표준화하면 원핫이 깨진다).
        X = np.eye(len(keep), dtype=np.float32)
    else:
        X = X_all[[pos[r["id"]] for r in keep]].astype(np.float32)
        # 표준화 후 L2 정규화 — 스케일이 다른 hs/emb를 같은 wd로 다룰 수 있게
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return Y, X, experts, [r["id"] for r in keep]


def make_mask(Y: np.ndarray, a: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """True = 학습 셀."""
    N, E = Y.shape
    if a.split == "cell":
        return rng.random((N, E)) >= a.holdout
    hold = rng.random(N) < a.holdout
    return np.repeat(~hold[:, None], E, axis=1)


# ------------------------------------------------------------------ 모형
def fit(Y, X, train_m, rank, a, dev, use_problem_effect: bool, use_expert_effect: bool = True):
    """rank=0 이면 주효과만. use_problem_effect=False면 a_i를 쓰지 않는다(문제 홀드아웃용)."""
    N, E = Y.shape
    d = X.shape[1]
    t = lambda z: torch.as_tensor(z, device=dev)
    Yt, Xt, M = t(Y), t(X), t(train_m.astype(np.float32))

    mu = torch.zeros(1, device=dev, requires_grad=True)
    params = [mu]
    b = None
    if use_expert_effect:
        b = torch.zeros(E, device=dev, requires_grad=True)
        params.append(b)
    aa = None
    if use_problem_effect:
        aa = torch.zeros(N, device=dev, requires_grad=True)
        params.append(aa)
    W = V = None
    if rank > 0:
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        W = (torch.randn(d, rank, generator=g) * 0.01).to(dev).requires_grad_(True)
        V = (torch.randn(E, rank, generator=g) * 0.01).to(dev).requires_grad_(True)
        params += [W, V]
    # 문제 홀드아웃일 때는 난이도도 임베딩에서 나와야 하므로 선형 항을 하나 준다
    w0 = None
    if not use_problem_effect:
        w0 = torch.zeros(d, device=dev, requires_grad=True)
        params.append(w0)

    opt = torch.optim.Adam(params, lr=a.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    n_tr = M.sum()
    for _ in range(a.epochs):
        opt.zero_grad()
        logit = mu.expand(N, E) if b is None else mu + b[None, :]
        if aa is not None:
            logit = logit + aa[:, None]
        if w0 is not None:
            logit = logit + (Xt @ w0)[:, None]
        if rank > 0:
            logit = logit + (Xt @ W) @ V.T
        loss = (bce(logit, Yt, reduction="none") * M).sum() / n_tr
        if rank > 0:
            loss = loss + a.wd * (W.pow(2).sum() + V.pow(2).sum())
        if aa is not None:
            loss = loss + a.wd_main * aa.pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logit = mu.expand(N, E) if b is None else mu + b[None, :]
        if aa is not None:
            logit = logit + aa[:, None]
        if w0 is not None:
            logit = logit + (Xt @ w0)[:, None]
        if rank > 0:
            logit = logit + (Xt @ W) @ V.T
        te = ~train_m
        p = torch.sigmoid(logit).cpu().numpy()[te]
        y = Y[te]
        ll = -float(np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        # 원핫 ANOVA와 같은 단위(전체 분산 대비 %)로 읽기 위한 홀드아웃 잔차분산
        resid = float(np.mean((y - p) ** 2))
        tot = float(np.var(y))
    return ll, auc(y, p), int(te.sum()), resid, tot


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 동점 평균순위
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y > 0.5].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def permute_rows(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """문제별로 expert 라벨 셔플 — 문제 난이도(행 합)는 보존, 상호작용만 파괴."""
    out = Y.copy()
    for i in range(out.shape[0]):
        rng.shuffle(out[i])
    return out


def main() -> None:
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Y, X, experts, pids = load(a)
    N, E = Y.shape
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    rng = np.random.default_rng(a.seed)
    train_m = make_mask(Y, a, rng)
    use_pe = a.split == "cell"
    print(f"문제 {N} × expert {E} = {N*E:,}셀 · feature={a.feature}({X.shape[1]}d) · "
          f"split={a.split} · 학습셀 {int(train_m.sum()):,} · device={dev}", flush=True)

    if a.positive_control > 0:
        # 실제 주효과(문제 난이도·expert 평균)는 보존하고, 그 위에 알려진 저랭크 상호작용만 얹는다.
        pc = np.random.default_rng(7)
        W0 = pc.normal(size=(X.shape[1], 2))
        V0 = pc.normal(size=(E, 2))
        inter = (X @ W0) @ V0.T
        inter /= inter.std() + 1e-9
        eps = 1e-6
        logit0 = np.log(np.clip(Y.mean(0), eps, 1 - eps) / (1 - np.clip(Y.mean(0), eps, 1 - eps)))[None, :]
        rowb = (Y.mean(1) - Y.mean())[:, None] * 4.0
        p = 1 / (1 + np.exp(-(logit0 + rowb + a.positive_control * inter)))
        Y = (pc.random(p.shape) < p).astype(np.float32)
        print(f"[양성 대조] 세기 {a.positive_control} 인위 상호작용 주입 · "
              f"평균 pass율 {Y.mean():.3f}", flush=True)

    # 원핫 ANOVA와 동일한 컬럼을 만들기 위해 중첩 모형을 순서대로 적합한다.
    #   ① 난이도만 → ② +실력 → ③ +궁합.  각 단계의 설명분산 증가분 = 그 성분의 몫.
    _, _, n_te, res_d, tot_var = fit(Y, X, train_m, 0, a, dev, use_pe, use_expert_effect=False)
    base_ll, base_auc, _, base_res, _ = fit(Y, X, train_m, 0, a, dev, use_pe)
    share_diff = 100 * (tot_var - res_d) / tot_var
    share_skill = 100 * (res_d - base_res) / tot_var
    print(f"[난이도만]  설명분산 {share_diff:.1f}%", flush=True)
    print(f"[+실력]     설명분산 {100*(tot_var-base_res)/tot_var:.1f}%  (실력 몫 {share_skill:+.2f}%)", flush=True)

    res = []
    for r in ranks:
        ll, au, _, rr, _ = fit(Y, X, train_m, r, a, dev, use_pe)
        gain = 100 * (base_res - rr) / tot_var
        res.append((r, ll, au, base_ll - ll, au - base_auc, gain))
        print(f"[+궁합 rank {r:2d}] 궁합 몫 {gain:+.2f}%   AUC {au:.4f} (ΔAUC {au-base_auc:+.4f})", flush=True)

    best_r = max(res, key=lambda x: x[4])[0]
    best_gain = max(r[5] for r in res)
    print(f"\n귀무 대조 {a.n_perm}회 (문제별 expert 라벨 셔플, rank={best_r})", flush=True)
    null, null_gain = [], []
    for s in range(a.n_perm):
        pr = np.random.default_rng(1000 + s)
        Yp = permute_rows(Y, pr)
        b_ll, b_au, _, b_res, b_tot = fit(Yp, X, train_m, 0, a, dev, use_pe)
        _, p_au, _, p_res, _ = fit(Yp, X, train_m, best_r, a, dev, use_pe)
        null_gain.append(100 * (b_res - p_res) / b_tot)
        null.append(p_au - b_au)
        print(f"  perm {s+1}: ΔAUC {p_au-b_au:+.4f}", flush=True)
    nm, ns = float(np.mean(null)), float(np.std(null) + 1e-9)
    obs = max(r[4] for r in res)
    z = (obs - nm) / ns

    L = ["# expert×문제 상호작용이 문제 내용으로 예측 가능한가 (저랭크 이중선형)", "",
         f"- 데이터: `{a.binned}` ∩ `{feat_paths(a)[0]}` → 문제 {N} × expert {E} = **{N*E:,}셀** (K=1)",
         f"- feature: `{a.feature}` ({X.shape[1]}차원) · split: `{a.split}` "
         f"({'셀 무작위 홀드아웃 = 순수 상호작용 검정' if use_pe else '문제 통째 홀드아웃 = 라우터 배포 상황'})",
         f"- 홀드아웃 {n_te:,}셀 · epochs {a.epochs} · wd {a.wd}", "",
         "모형: `logit = mu + a_i + b_j` (주효과, 원핫·포화) vs `+ (W e_i)·v_j` (저랭크 상호작용).",
         "`v_j`는 expert별 **자유 벡터** — 어떤 페르소나 표현으로도 넘을 수 없는 상한이다.", "",
         "| 문제 난이도 | 전문가 실력 | **궁합** | 나머지(운) |", "|---:|---:|---:|---:|",
         f"| {share_diff:.1f}% | {share_skill:.1f}% | **{best_gain:+.2f}%** | {100-share_diff-share_skill-best_gain:.1f}% |",
         "", "rank별 궁합 몫:", "",
         "| rank | 궁합 몫 | AUC |", "|---|---:|---:|"]
    for r, ll, au, dll, dau, gain in res:
        L.append(f"| {r} | {gain:+.2f}% | {au:.4f} |")
    ng_m, ng_s = float(np.mean(null_gain)), float(np.std(null_gain) + 1e-9)
    L += ["", f"**귀무 대조**(문제별 expert 라벨 셔플 → 주효과 보존·상호작용 파괴, rank={best_r}, {a.n_perm}회): "
          f"ΔAUC = {nm:+.4f} ± {ns:.4f} · 추가 설명 분산 = {ng_m:+.2f}% ± {ng_s:.2f}%", "",
          f"**관측 최대 ΔAUC = {obs:+.4f} → z = {z:+.2f}**", "",
          ("> z > 2: 상호작용이 실재하고 문제 내용으로 예측된다 — 라우터가 먹을 구조가 있다."
           if z > 2 else
           "> z ≈ 0: 주효과(난이도·expert 평균실력) 위에 문제 내용으로 예측 가능한 상호작용이 없다."),
          "",
          "## 해석 주의", "",
          "- 문제 주효과는 원핫이 이미 **포화**(N-1 df)라 임베딩이 더 설명할 수 없다. 여기서 임베딩이",
          "  바꾸는 건 오직 상호작용 항이며, 그래서 주효과 분산 몫은 이 표에 나오지 않는다.",
          "- `v_j` 자유 벡터가 상한이므로, ΔAUC≈0이면 **페르소나 텍스트 임베딩으로 바꿔도 개선 없다**.",
          "  페르소나 임베딩이 별도 가치를 갖는 건 '새 페르소나로의 일반화'뿐인데 E=11로는 검정력이 없다.",
          f"- 라벨은 K=1(단일 드로우)이라 셀마다 이항 노이즈가 있다. 그 노이즈는 달성 가능한 AUC의",
          "  천장을 낮추지만, 귀무 대조가 같은 노이즈를 겪으므로 z 비교는 유효하다.", ""]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-8:]))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
