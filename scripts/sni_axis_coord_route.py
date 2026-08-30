#!/usr/bin/env python
"""우리 로스터(진화 16명)의 축 좌표로 라우팅이 되는가 — 재생성 0.

문제 임베딩으로는 승자가 안 잡혔다(`results/sni/interaction_lowrank_hs_mean_cell.md`, z=-7.36).
여기서는 임베딩 대신 **데이터에 실재하는 연속량**을 좌표로 쓴다. 진화가 만든 축이
"짧게·원문 그대로 vs 의미를 옮겨라"라면 그 축은 출력 요구량·정답 종수 같은 양에 실려야 한다.

규칙(이전 판이 여기서 무너졌다):
  · 연속량은 **자르지 않는다**. 구간을 나누면 결론이 절단점의 함수가 된다.
  · 라벨은 데이터에 있는 것만: gold_len_median · n_gold_types · 입력/정의 길이 · task_closed.
  · 난이도는 **문제 내 짝지은 차이**로 소거한다: d_j(p) = p̂_j(p) − 그 문제 16명 평균.
  · 계수 신뢰구간은 **태스크 클러스터 부트스트랩**(한 태스크의 문제들이 담당자를 공유).
  · 배정도 대조군도 **train에서만** 정하고 test는 한 번만 본다.
  · 대조군 둘: 좌표 셔플(문제 간) · expert 정체성 셔플(문제 내).

Usage: python3 scripts/sni_axis_coord_route.py --out results/sni/axis_coord_route.md
"""
import argparse
import json
import numpy as np

R = "export/sni_binning_seed20212003"
FEATS = ["log gold길이", "log 정답종수", "log 입력길이", "log 정의길이", "닫힘"]


def load(labels, data):
    lab = {}
    for l in open(labels, encoding="utf-8"):
        r = json.loads(l)
        lab[r["id"]] = r["per_expert"]
    rows, X, Y, task = [], [], [], []
    ex = None
    for l in open(data, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in lab:
            continue
        pe = lab[r["id"]]
        if ex is None:
            ex = sorted(pe)
        X.append([
            np.log1p(float(r.get("gold_len_median") or 0)),
            np.log1p(float(r.get("n_gold_types") or 0)),
            np.log1p(len((r.get("instruction") or "").split())),
            np.log1p(len((r.get("definition") or "").split())),
            1.0 if r.get("task_closed") else 0.0,
        ])
        Y.append([pe[e] for e in ex])
        task.append(r["id"].split("-")[0])
        rows.append(r["id"])
    return np.array(X, np.float64), np.array(Y, np.float64), np.array(task), ex


def fit(X, D):
    """열별 z-정규화 + 절편. D = 문제 내 짝지은 이득(문제 × expert)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    B = np.linalg.lstsq(Z, D, rcond=None)[0]          # (1+F, E)
    return B, mu, sd


def predict(X, B, mu, sd):
    return np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ B


def routed(P, S):
    return 100 * float(np.mean(S[np.arange(len(S)), P.argmax(1)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sni/axis_coord_route.md")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    Xtr, Str, Ttr, ex = load(f"{R}/binning_labels.jsonl", "export/sni_v4/sni_train.jsonl")
    Xte, Ste, Tte, _ = load(f"{R}/test_binned.jsonl", "export/sni_v4/sni_test.jsonl")
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}
    E = Str.shape[1]
    Dtr = Str - Str.mean(1, keepdims=True)            # 난이도 소거
    B, mu, sd = fit(Xtr, Dtr)

    # 기준선은 전부 train에서 정한다
    best_j = int(Str.mean(0).argmax())
    L = ["# 우리 축 좌표로 라우팅되는가 (진화 16명, 재생성 0)", "",
         f"- train {len(Xtr):,}문제 / test {len(Xte):,}문제 · expert {E}명 · 라벨 p̂(K=3)",
         "- 좌표: " + " · ".join(FEATS) + " (전부 연속, 자르지 않음)",
         "- 이득 d_j(p) = p̂_j(p) − 그 문제 16명 평균 (문제 내 짝지은 차이 = 난이도 소거)", "",
         "## 1. 라우팅 실현치 (test, 배정·대조군 모두 train에서 결정)", "",
         "| 방식 | test |", "|---|---:|"]
    Pte = predict(Xte, B, mu, sd)
    L.append(f"| best-single (train 선택: {names.get(ex[best_j], ex[best_j])}) | {100*Ste[:, best_j].mean():.2f} |")
    L.append(f"| **축 좌표 라우팅 (top-1)** | **{routed(Pte, Ste):.2f}** |")
    # 대조군 1: 좌표를 문제 간 셔플 → 좌표 정보 파괴
    sh = [routed(predict(Xte[rng.permutation(len(Xte))], B, mu, sd), Ste) for _ in range(20)]
    L.append(f"| 대조군: 좌표 셔플 | {np.mean(sh):.2f} ± {np.std(sh):.2f} |")
    L.append(f"| 무작위 배정 | {np.mean([100*Ste[np.arange(len(Ste)), rng.integers(0, E, len(Ste))].mean() for _ in range(20)]):.2f} |")
    L.append(f"| 문제별 오라클 top-1 | {100*Ste.max(1).mean():.2f} |")
    L.append(f"| union (16명 전원) | {100*(Ste.max(1) > 0).mean():.2f} |")
    head = 100*Ste.max(1).mean() - 100*Ste[:, best_j].mean()
    got = routed(Pte, Ste) - 100*Ste[:, best_j].mean()
    L += ["", f"회수: 전체 폭 {head:.2f}pp 중 **{got:+.2f}pp** ({100*got/head:.1f}%)", ""]

    # 2. 축이 무엇인가 — 계수 (태스크 클러스터 부트스트랩)
    tasks = np.unique(Ttr)
    idx_by_task = {t: np.where(Ttr == t)[0] for t in tasks}
    boots = []
    for _ in range(a.boot):
        pick = rng.choice(tasks, len(tasks), replace=True)
        ii = np.concatenate([idx_by_task[t] for t in pick])
        boots.append(fit(Xtr[ii], Dtr[ii])[0])
    Bb = np.stack(boots)                                   # (boot, 1+F, E)
    lo, hi = np.percentile(Bb, [2.5, 97.5], axis=0)
    L += ["## 2. 축의 정체 — 좌표별 기울기 (태스크 클러스터 부트스트랩 95% CI)", "",
          "0을 안 무는 것만 유의하다. 값은 '좌표가 1SD 커질 때 그 expert의 이득 변화(확률)'.", "",
          "| expert | " + " | ".join(FEATS) + " |", "|---|" + "---:|" * len(FEATS)]
    for j in range(E):
        cells = []
        for f in range(len(FEATS)):
            v, l_, h_ = B[1 + f, j], lo[1 + f, j], hi[1 + f, j]
            star = "**" if (l_ > 0) == (h_ > 0) else ""
            cells.append(f"{star}{v:+.4f}{star}")
        L.append(f"| {names.get(ex[j], ex[j])} | " + " | ".join(cells) + " |")
    sig = int(sum(1 for j in range(E) for f in range(len(FEATS))
                  if (lo[1+f, j] > 0) == (hi[1+f, j] > 0)))
    L += ["", f"유의한 칸 {sig}/{E*len(FEATS)}", ""]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
