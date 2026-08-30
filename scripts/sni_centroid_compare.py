#!/usr/bin/env python
"""센트로이드 위치만 비교 (배정·분할 산출 없음). 재생성 0.

학습 분할의 τ=10은 건드리지 않는다. 여기서 바꾸는 건 **센트로이드를 어느 문제로 뽑느냐**뿐이다.
  · tc=10 가중  : 현행 (export/sni_split_seed20212003/split.jsonl 을 만든 규칙)
  · tc=5  가중  : 더 정제된 밴드
  · tc=5  무가중: 가중치 효과 분리용
가중치 w=(E-n)/(E-1) (단독 솔브 1.0).
전처리는 라우터와 동일: 차원별 z-정규화(train 통계) → L2 → 코사인.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPORT = "results/sni/centroid_compare.md"


def centroids(Z, indiv, solved, w_all, E, weighted):
    C = np.zeros((E, Z.shape[1]), np.float32)
    ess = np.zeros(E, np.float32)
    for j in range(E):
        m = indiv & solved[:, j]
        w = w_all[m] if weighted else np.ones(int(m.sum()), np.float32)
        C[j] = (Z[m] * w[:, None]).sum(0) / (w.sum() + 1e-9)
        ess[j] = w.sum() ** 2 / (np.square(w).sum() + 1e-9)
    nrm = np.linalg.norm(C, axis=1)
    return C / (nrm[:, None] + 1e-9), nrm, ess


def main():
    sp = rc.spec("sni")
    trb, _ = rc.labels(sp)
    ex = rc.experts(sp)
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}
    ids, X, S = rc.align(rc.feat_path(sp, "train", "hs_mean"),
                         rc.feat_path(sp, "train", "hs_ids"), trb, ex)
    E = len(ex)
    solved = S > 0.5
    ns = solved.sum(1)
    w_all = (E - ns).astype(np.float32) / (E - 1)
    mu, sd = rc.zscore(X)
    Z = (X - mu) / sd
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    allfail = ns == 0
    Q = Z[allfail]
    off = ~np.eye(E, dtype=bool)

    variants = [("tc=10 가중 (현행)", 10, True),
                ("tc=5 가중", 5, True),
                ("tc=5 무가중", 5, False)]
    out, tab = {}, []
    for tag, tc, wt in variants:
        band = (ns >= 1) & (ns <= tc)
        C, nrm, ess = centroids(Z, band, solved, w_all, E, wt)
        cc = C @ C.T
        cnt = np.bincount((Q @ C.T).argmax(1), minlength=E)
        out[tag] = (C, nrm, ess, cc, cnt)
        tab.append((tag, int(band.sum()), cc[off].mean(), cc[off].max(),
                    cc[off].min(), cnt.max(), cnt.min(), float(np.std(cnt))))

    L = ["# 센트로이드 위치 비교 — 진화 16명 (배정 산출 없음, 재생성 0)", "",
         "학습 분할의 τ=10은 고정. 바꾼 것은 **센트로이드를 어느 문제로 뽑는가**뿐이다.",
         f"전원실패 {int(allfail.sum()):,}건에 대한 argmax 배정 수는 **참고용**으로만 싣는다"
         " (split.jsonl 은 갱신하지 않았다).", "",
         "## A. 센트로이드끼리 얼마나 떨어져 있나", "",
         "| 변형 | 센트로이드 표본 문제 | 쌍 코사인 평균 | 최대 | 최소 | (참고) 배정 최다 | 최소 | 표준편차 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tab:
        L.append(f"| {t[0]} | {t[1]:,} | **{t[2]:+.4f}** | {t[3]:+.4f} | {t[4]:+.4f} "
                 f"| {t[5]:,} | {t[6]:,} | {t[7]:,.0f} |")

    L += ["", "## B. 컷을 바꾸면 센트로이드가 실제로 움직이나", "",
          "같은 expert의 tc=10 가중 센트로이드와의 코사인. 1에 가까우면 컷을 바꿔도 방향이 그대로다.", "",
          "| expert | tc=5 가중과의 코사인 | tc=5 무가중과의 코사인 | ESS(tc=5 가중) |",
          "|---|---:|---:|---:|"]
    C10 = out["tc=10 가중 (현행)"][0]
    C5w, _, ess5, _, _ = out["tc=5 가중"]
    C5u = out["tc=5 무가중"][0]
    mv = (C10 * C5w).sum(1)
    for j in np.argsort(mv):
        L.append(f"| {names.get(ex[j], ex[j])} | {mv[j]:+.4f} | "
                 f"{(C10[j] * C5u[j]).sum():+.4f} | {ess5[j]:,.0f} |")
    L += ["", f"평균 이동 코사인: tc=5 가중 {mv.mean():+.4f} · "
              f"tc=5 무가중 {(C10 * C5u).sum(1).mean():+.4f}"]

    L += ["", "## C. 가장 붙어 있는 센트로이드 쌍 (tc=5 가중)", "",
          "| 쌍 | 코사인 |", "|---|---:|"]
    cc5 = out["tc=5 가중"][3]
    pr = sorted(itertools.combinations(range(E), 2), key=lambda p: -cc5[p])[:8]
    for i, j in pr:
        L.append(f"| {names.get(ex[i], ex[i])} ↔ {names.get(ex[j], ex[j])} | {cc5[i, j]:+.4f} |")

    L += ["", "## D. 변형별 expert 상세 (tc=5 가중)", "",
          "| expert | 응집도(정규화 전 norm) | ESS | 다른 센트로이드와의 평균 코사인 | (참고) 배정 |",
          "|---|---:|---:|---:|---:|"]
    _, nrm5, _, _, cnt5 = out["tc=5 가중"]
    for j in np.argsort(-nrm5):
        L.append(f"| {names.get(ex[j], ex[j])} | {nrm5[j]:.4f} | {ess5[j]:,.0f} | "
                 f"{cc5[j][off[j]].mean():+.4f} | {cnt5[j]:,} |")
    r = np.corrcoef(nrm5, cnt5)[0, 1]
    L += ["", f"응집도 ↔ 배정 수 상관 r = {r:+.3f}"]

    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
