#!/usr/bin/env python
"""진화 분할 → per-expert 학습 데이터 배정 (student MoE용). 재생성 0.

규칙 (2026-08-30 결정, τ=10):
  · 1 ≤ n_solved ≤ τ   → 푼 사람(들)에게 **개별 배정**. 한 문제가 여러 명에게 갈 수 있다.
  · n_solved > τ        → **shared expert(짬통)**. 전원 성공(15.7%)도 여기로 간다.
      τ는 `results/sni/tau_sweep.md` 곡선에서 고름 — E=16에서 expert당 ~2,844건, 평균 쌍 Jaccard 0.234.
      (acc의 τ=8은 E=11 기준이라 그대로 쓰지 않았다.)
  · n_solved = 0(26.2%) → **각 expert 센트로이드에 최근접인 쪽으로 강제 배정**.
      센트로이드 = 그 expert가 개별 배정받은 문제들의 임베딩 **가중** 평균,
      가중치 w=(E-n)/(E-1) (단독 솔브 1.0 → 10명 공동 0.4). 학습 분할의 τ와는 별개 손잡이다.
      k-NN 투표는 폐기했다 — 이웃이 여러 명에게 풀린 중복을 그대로 물려받는다.
      ROUGE 최근접도 못 쓴다 — 전원 실패의 17.8%가 닫힌 태스크(gold 평균 1.1단어)라 전부 0에 붙는다.
      ⚠️ 중심화 필수: 원본 코사인은 문제끼리도 0.984(anisotropy), 전체 평균을 빼야 0.002로 흩어진다.
  · 학습 타깃은 **gold**(통과분의 90.07%가 EM 통과라 페르소나 출력과 사실상 동일).

⚠️ 임베딩으로 배정한 26.2%는 **정의상 입력에서 예측 가능**해진다. 나중에 라우터가 이 분할을
얼마나 되찾는지 잴 때는 갈림 몫과 이 몫을 **나눠서** 보고할 것.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--tau_c", type=int, default=5,
                    help="센트로이드를 뽑을 밴드 (학습 분할의 --tau와 별개)")
    ap.add_argument("--feat", default="hs_mean")
    ap.add_argument("--out", default="export/sni_split_seed20212003/split.jsonl")
    ap.add_argument("--report", default="results/sni/split_build.md")
    a = ap.parse_args()
    sp = rc.spec("sni")
    trb, _ = rc.labels(sp)
    ex = rc.experts(sp)
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}

    ids, X, S = rc.align(rc.feat_path(sp, "train", a.feat),
                         rc.feat_path(sp, "train", "hs_ids"), trb, ex)
    E, N = len(ex), len(ids)
    solved = S > 0.5
    ns = solved.sum(1)
    indiv = (ns >= 1) & (ns <= a.tau)      # 개별 배정
    shared = ns > a.tau                     # 짬통(전원 성공 포함)
    allfail = ns == 0

    # 전처리는 라우터(sni_router_train.py)와 동일하게 맞춘다:
    #   차원별 z-정규화(train 통계) → L2 → 코사인.
    # ⚠️ 중심화 필수: 원본 코사인은 문제끼리도 0.984(anisotropy).
    mu, sd = rc.zscore(X)
    Z = (X - mu) / sd
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    # expert별 센트로이드 = 개별 배정받은 문제들의 (정규화된) 임베딩 **가중** 평균.
    # 가중치 w = (E-n)/(E-1) — 단독 솔브 1.0, τ=10명이 함께 푼 문제 0.4.
    # WAR soft_linear가 쓰는 바로 그 가중치라 축의 정의와 일관된다.
    # 하드 컷(n<=1 등)을 안 쓴 이유: 단독 솔브만 모으면 표본 최소 32건(2,816차원)이라
    # 방향이 노이즈이고, 표본이 반전 페르소나 쪽으로 12배 쏠린다.
    w_all = (E - ns).astype(np.float32) / (E - 1)
    cband = (ns >= 1) & (ns <= a.tau_c)    # 센트로이드 표본 밴드 (학습 분할 τ와 별개)
    Craw = np.zeros((E, Z.shape[1]), np.float32)
    ess = np.zeros(E, np.float32)          # 유효표본 (sum w)^2 / sum w^2
    for j in range(E):
        m = cband & solved[:, j]
        w = w_all[m]
        Craw[j] = (Z[m] * w[:, None]).sum(0) / (w.sum() + 1e-9)
        ess[j] = w.sum() ** 2 / (np.square(w).sum() + 1e-9)
    # 정규화 전 norm = 그 expert가 맡은 문제들의 응집도(흩어지면 상쇄돼 작아진다).
    # L2로 나누면 이 정보가 사라지므로 보고서에 남긴다.
    cnorm = np.linalg.norm(Craw, axis=1)
    C = Craw / (cnorm[:, None] + 1e-9)
    ccos = C @ C.T
    qry = torch.tensor(Z[allfail], dtype=torch.float32).to(DEV)
    sim = qry @ torch.tensor(C, dtype=torch.float32).to(DEV).T
    near = sim.argmax(1).cpu().numpy()
    top2 = sim.topk(2, dim=1).values
    tie = ((top2[:, 0] - top2[:, 1]) < 1e-3).cpu().numpy()   # 1·2위 차가 거의 없는 경우

    assign = [[] for _ in range(N)]
    for i in np.where(indiv)[0]:
        assign[i] = [ex[j] for j in np.where(solved[i])[0]]
    for i in np.where(shared)[0]:
        assign[i] = ["__shared__"]
    for q, i in enumerate(np.where(allfail)[0]):
        assign[i] = [ex[int(near[q])]]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, pid in enumerate(ids):
            kind = "indiv" if indiv[i] else ("shared" if shared[i] else "all_fail")
            f.write(json.dumps({"id": pid, "kind": kind, "experts": assign[i],
                                "n_solved": int(ns[i])}, ensure_ascii=False) + "\n")

    cnt = {c: [0, 0] for c in list(ex) + ["__shared__"]}   # [개별, 전원실패]
    for i in range(N):
        col = 1 if allfail[i] else 0
        for c in assign[i]:
            cnt[c][col] += 1
    # 최종 학습셋 쌍별 Jaccard
    fin = {c: set() for c in ex}
    for i in range(N):
        for c in assign[i]:
            if c in fin:
                fin[c].add(i)
    import itertools
    J = [len(fin[x] & fin[y]) / max(1, len(fin[x] | fin[y]))
         for x, y in itertools.combinations(ex, 2)]
    L = [f"# 분할 빌드 — 진화 16명 (학습 τ={a.tau}, 센트로이드 tc={a.tau_c} 가중, feature={a.feat})", "",
         f"- train {N:,}문제 · 개별 {int(indiv.sum()):,}({100*indiv.mean():.1f}%) · "
         f"shared {int(shared.sum()):,}({100*shared.mean():.1f}%) · "
         f"전원실패 {int(allfail.sum()):,}({100*allfail.mean():.1f}%)",
         f"- 최종 학습셋 평균 쌍 Jaccard **{np.mean(J):.3f}** (최대 {max(J):.3f})",
         f"- 전원실패 센트로이드 1·2위 차 < 1e-3: {int(tie.sum()):,}건 ({100*tie.mean():.1f}%)",
         "- ⚠️ 전원실패 몫은 임베딩으로 배정했으므로 정의상 입력에서 예측 가능하다. "
         "라우터 평가 시 갈림 몫과 분리해 보고할 것.", "",
         "| expert | 합계 | 개별(n_solved≤τ) | 전원실패(센트로이드) |", "|---|---:|---:|---:|"]
    for c in sorted(ex, key=lambda c: -sum(cnt[c])):
        v = cnt[c]
        L.append(f"| {names.get(c, c)} | **{sum(v):,}** | {v[0]:,} | {v[1]:,} |")
    L.append(f"| **shared expert** | **{cnt['__shared__'][0]:,}** | {cnt['__shared__'][0]:,} | — |")
    L += ["", f"전원실패 배정 쏠림: 최다 {max(cnt[c][1] for c in ex):,} / "
              f"최소 {min(cnt[c][1] for c in ex):,} (균등이면 {int(allfail.sum())//E:,})", "",
          "## 센트로이드 진단", "",
          "`응집도` = L2 정규화 **전** 센트로이드 norm — 그 expert가 맡은 문제들이 "
          "임베딩 공간에서 얼마나 뭉쳐 있나(흩어지면 상쇄돼 0에 가까워진다).",
          "L2가 이 값을 1로 만들어 버리므로, 흩어진 expert의 잔차 방향이 "
          "뾰족한 expert와 동등하게 경쟁하게 된다.", "",
          "| expert | 응집도(정규화 전 norm) | 유효표본(ESS) | 다른 센트로이드와의 평균 코사인 | 전원실패 배정 |",
          "|---|---:|---:|---:|---:|"]
    off = ~np.eye(E, dtype=bool)
    for j in sorted(range(E), key=lambda j: -cnorm[j]):
        L.append(f"| {names.get(ex[j], ex[j])} | {cnorm[j]:.4f} | {ess[j]:,.0f} | "
                 f"{ccos[j][off[j]].mean():+.4f} | {cnt[ex[j]][1]:,} |")
    r = np.corrcoef(cnorm, [cnt[c][1] for c in ex])[0, 1]
    L += ["", f"응집도 ↔ 전원실패 배정 수 상관 **r = {r:+.3f}** "
              f"(음수면 '흩어진 expert가 어려운 문제를 빨아들인다'는 가설이 맞다)",
          f"센트로이드 쌍 코사인: 평균 {ccos[off].mean():+.4f} · "
          f"최소 {ccos[off].min():+.4f} · 최대 {ccos[off].max():+.4f}", "",
          f"산출: `{a.out}`"]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
