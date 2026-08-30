#!/usr/bin/env python
"""진화가 찾은 축이 사람 택소노미와 같은 것인가 — 재생성 0.

주장의 핵심: "사람 분류(도메인/카테고리)로 쪼갠 것과 달리, 모델이 스스로 풀어보며 찾은 분할은
다른 것을 본다." 정렬도가 높으면 그건 도메인 재발견이고, 낮아야 주장이 선다.

**배정 규칙을 만들지 않는다.** 다 틀린/다 맞춘 문제를 누구에게 줄지는 미정 사항이고, 그걸 여기서
정하면 결과가 그 규칙의 함수가 된다. 대신 규칙 없이 되는 두 가지로 잰다:

  A. 설명력 R² — 문제마다 '누가 푸는가' 프로필 d(p) = p̂(p) − 그 문제 평균(난이도 제거).
     사람 라벨로 묶었을 때 그 프로필 변이가 얼마나 줄어드는가.
     ⚠️ 군집 수가 많으면 우연히도 설명력이 오르므로 **같은 크기 분포의 무작위 라벨**을 대조군으로 낸다.
  B. 하드 라벨 일치도 — d(p)의 argmax(=그 문제에서 상대적으로 가장 잘한 사람)를 16-way 라벨로 보고
     사람 라벨과의 NMI. 역시 무작위 대조군과 함께 낸다.

쓰는 라벨은 데이터에 실재하는 것뿐: category · sni_domain · task_name.
"""
import argparse
import json
import numpy as np

R = "export/sni_binning_seed20212003"


def load(labels, data):
    lab = {}
    for l in open(labels, encoding="utf-8"):
        r = json.loads(l)
        lab[r["id"]] = r["per_expert"]
    Y, meta = [], []
    ex = None
    for l in open(data, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in lab:
            continue
        pe = lab[r["id"]]
        if ex is None:
            ex = sorted(pe)
        Y.append([pe[e] for e in ex])
        meta.append((r.get("category"), r.get("sni_domain"), r["id"].split("-")[0]))
    return np.array(Y, np.float64), meta, ex


def r2(D, g):
    """군집 내 분산이 전체 대비 얼마나 줄었나(다변량)."""
    tot = ((D - D.mean(0)) ** 2).sum()
    within = 0.0
    for u in np.unique(g):
        m = g == u
        within += ((D[m] - D[m].mean(0)) ** 2).sum()
    return 100 * (1 - within / tot)


def nmi(a, b):
    ua, ub = np.unique(a), np.unique(b)
    n = len(a)
    P = np.zeros((len(ua), len(ub)))
    ia = {v: i for i, v in enumerate(ua)}; ib = {v: i for i, v in enumerate(ub)}
    for x, y in zip(a, b):
        P[ia[x], ib[y]] += 1
    P /= n
    px, py = P.sum(1), P.sum(0)
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / np.outer(px, py)[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 2 * mi / (hx + hy) if (hx + hy) > 0 else 0.0


def shuffled(g, rng):
    """같은 크기 분포를 유지한 무작위 라벨."""
    p = rng.permutation(len(g))
    return g[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sni/axis_vs_taxonomy.md")
    ap.add_argument("--rep", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    Y, meta, ex = load(f"{R}/binning_labels.jsonl", "export/sni_v4/sni_train.jsonl")
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}
    D = Y - Y.mean(1, keepdims=True)        # 난이도 제거 = '누가' 성분만
    cat = np.array([m[0] or "?" for m in meta])
    dom = np.array([m[1] or "?" for m in meta])
    tsk = np.array([m[2] for m in meta])
    # ⚠️ argmax는 동점일 때 첫 인덱스를 고른다. 전원 성공/전원 실패 문제는 d가 전부 0이라
    # 정렬상 첫 expert로 몰린다(첫 판에서 c_10299가 88%를 먹은 원인). 갈림 문제만 쓰고
    # 남은 동점은 무작위로 깬다.
    ns = (Y > 0.5).sum(1)
    con = (ns > 0) & (ns < Y.shape[1])
    tie = rng.random(D.shape) * 1e-9
    hard_all = np.array([ex[j] for j in (D + tie).argmax(1)])
    hard, catH, domH, tskH = hard_all[con], cat[con], dom[con], tsk[con]

    L = ["# 진화가 찾은 축 vs 사람 택소노미 (진화 16명, 재생성 0)", "",
         f"- train {len(Y):,}문제 · expert {len(ex)}명 · 라벨 p̂(K=3)",
         "- d(p) = p̂(p) − 그 문제 16명 평균 → 난이도를 뺀 '누가' 성분",
         "- 배정 규칙을 만들지 않았다(다 틀린/다 맞춘 문제 처리는 미정 사항)", "",
         "## A. 사람 라벨이 '누가' 성분을 얼마나 설명하나", "",
         "| 라벨 | 군집 수 | 설명력 R² | 같은 크기 무작위 대조군 | 차이 |",
         "|---|---:|---:|---:|---:|"]
    for tag, g in (("category", cat), ("sni_domain", dom), ("task_name", tsk)):
        obs = r2(D, g)
        null = [r2(D, shuffled(g, rng)) for _ in range(a.rep)]
        L.append(f"| {tag} | {len(np.unique(g))} | {obs:.2f}% | "
                 f"{np.mean(null):.2f}% ± {np.std(null):.2f} | **{obs-np.mean(null):+.2f}%p** |")

    L += ["", "## B. '그 문제에서 상대적으로 가장 잘한 사람' 라벨과의 일치도(NMI)", "",
          f"갈림 문제 {int(con.sum()):,}/{len(Y):,}만 사용(만장일치는 승자가 정의되지 않는다). 동점은 무작위.", "",
          "| 사람 라벨 | NMI | 무작위 대조군 | 차이 |", "|---|---:|---:|---:|"]
    for tag, g in (("category", catH), ("sni_domain", domH), ("task_name", tskH)):
        obs = nmi(hard, g)
        null = [nmi(hard, shuffled(g, rng)) for _ in range(a.rep)]
        L.append(f"| {tag} | {obs:.4f} | {np.mean(null):.4f} ± {np.std(null):.4f} | "
                 f"**{obs-np.mean(null):+.4f}** |")

    # 참고: 카테고리별로 누가 이기나 (해석용)
    L += ["", "## C. 카테고리별 최다 승자 (상위 12개 카테고리, 해석용)", "",
          "| category | 문제수 | 1위(비율) | 2위(비율) |", "|---|---:|---|---|"]
    order = sorted(np.unique(catH), key=lambda c: -(catH == c).sum())[:12]
    for c in order:
        m = catH == c
        cnt = {}
        for w in hard[m]:
            cnt[w] = cnt.get(w, 0) + 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:2]
        cells = [f"{names.get(e, e)} ({100*n/m.sum():.0f}%)" for e, n in top]
        L.append(f"| {c} | {int(m.sum()):,} | " + " | ".join(cells + [""] * (2 - len(cells))) + " |")

    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
