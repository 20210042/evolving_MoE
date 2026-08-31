#!/usr/bin/env python
"""대조군 분할 — Human prior split (사람 택소노미 상위 16 + 나머지는 임베딩). 재생성 0.

사람이 만든 라벨 축 하나를 골라 **상위 16개 값이 각각 expert 하나**가 된다.
그 16개에 안 걸리는 문제는 **Ours와 같은 방식으로 센트로이드 최근접 배정**한다.

  · 축은 둘 중 하나: `category`(태스크 유형 72개) 또는 `sni_domain`(주제·출처 74개)
  · 한 문제는 정확히 **1명**에게 간다 (중복 없음 → 학습 row = train 문제 수)
  · shared expert **없음** (16 routed only)
  · 진화 쪽 n_solved 구간은 쓰지 않는다 — teacher가 만든 사실이라 사람 사전지식 조건에
    넣으면 그 축이 새어 들어온다

⚠️ 상위 16개가 전수를 덮지 못한다(실측: category 68.0% · domain 76.1%). 남는 몫을 버리면
   다른 조건과 데이터량이 달라지므로, `sni_build_split.py`의 전원실패 배정과 **똑같은 절차**로
   임베딩 최근접 배정한다:
     차원별 z-정규화(train 통계) → L2 → 코사인 → argmax.
     ⚠️ 중심화 필수 — 원본 코사인은 문제끼리도 0.984(anisotropy)라 축이 안 잡힌다.
   센트로이드는 그 expert가 **택소노미로 직접 받은** 문제들의 (정규화된) 임베딩 평균이다.
   Ours의 w=(E-n)/(E-1) 가중은 여기 쓰지 않는다 — n_solved를 안 쓰는 조건이라 가중이 없다.

⚠️ 임베딩으로 배정한 몫은 정의상 입력에서 예측 가능하다. 라우터가 이 분할을 얼마나
   되찾는지 잴 때 택소노미 몫과 반드시 나눠서 보고할 것.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

TRAIN = "export/sni_v4/sni_train.jsonl"
ROSTER = "results/sni/seed20212003/roster_final.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="category", choices=["category", "sni_domain"])
    ap.add_argument("--n_experts", type=int, default=16)
    ap.add_argument("--feat", default="hs_mean")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    tag = "cat" if a.axis == "category" else "dom"
    out = a.out or f"export/sni_split_human_{tag}/split.jsonl"
    report = a.report or f"results/sni/split_build_human_{tag}.md"
    E = a.n_experts

    axis = {}
    for l in open(TRAIN):
        d = json.loads(l)
        axis[d["id"]] = d.get(a.axis)

    sp = rc.spec("sni")
    ids = json.load(open(rc.feat_path(sp, "train", "hs_ids")))
    X = np.load(rc.feat_path(sp, "train", a.feat))
    assert len(ids) == len(X), (len(ids), len(X))
    lab = np.array([axis.get(p) for p in ids], dtype=object)

    size = Counter(lab.tolist())
    top = [v for v, _ in sorted(size.items(), key=lambda kv: (-kv[1], str(kv[0])))[:E]]
    rank = {v: j for j, v in enumerate(top)}
    direct = np.array([rank.get(v, -1) for v in lab])       # 택소노미로 직접 잡힌 expert
    rest = direct < 0

    # 전처리는 라우터·Ours 분할과 동일하게: 차원별 z-정규화 → L2 → 코사인.
    mu, sd = rc.zscore(X)
    Z = (X - mu) / sd
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    C = np.zeros((E, Z.shape[1]), np.float32)
    for j in range(E):
        C[j] = Z[direct == j].mean(0)
    cnorm = np.linalg.norm(C, axis=1)        # 정규화 전 norm = 그 expert가 맡은 문제들의 응집도
    C = C / (cnorm[:, None] + 1e-9)
    sim = (torch.tensor(Z[rest], dtype=torch.float32).to(DEV)
           @ torch.tensor(C, dtype=torch.float32).to(DEV).T)
    near = sim.argmax(1).cpu().numpy()
    top2 = sim.topk(2, dim=1).values
    tie = ((top2[:, 0] - top2[:, 1]) < 1e-3).cpu().numpy()

    assign = direct.copy()
    assign[np.where(rest)[0]] = near

    ex = [p["id"] for p in json.load(open(ROSTER))][:E]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cnt = {c: [0, 0] for c in ex}            # [택소노미 직접, 임베딩 배정]
    with open(out, "w", encoding="utf-8") as f:
        for i, pid in enumerate(ids):
            j = int(assign[i])
            kind = "taxonomy" if direct[i] >= 0 else "embed"
            cnt[ex[j]][0 if direct[i] >= 0 else 1] += 1
            f.write(json.dumps({"id": pid, "kind": kind, "experts": [ex[j]],
                                "axis_value": lab[i], "n_solved": -1},
                               ensure_ascii=False) + "\n")

    N = len(ids)
    nd = int((~rest).sum())
    L = [f"# 대조군 분할 — Human prior split (축={a.axis}, 상위 {E} + 나머지 임베딩)", "",
         f"사람 택소노미 `{a.axis}` 상위 {E}개 값이 각각 expert 하나가 되고, "
         "거기 안 걸리는 문제는 Ours와 같은 절차(z-정규화→L2→코사인 argmax)로 "
         "센트로이드 최근접 배정했다.",
         "**shared expert 없음 · 문제당 1명 · 중복 없음 · n_solved 미사용.**", "",
         f"- train {N:,}문제 전수 → 학습 row {N:,}",
         f"- 택소노미로 직접 {nd:,} ({100*nd/N:.1f}%) · 임베딩 배정 {int(rest.sum()):,} "
         f"({100*rest.mean():.1f}%)",
         f"- 축값 {len(size)}개 중 상위 {E}개 사용 (나머지 {len(size)-E}개는 임베딩 몫으로)",
         f"- 임베딩 배정 1·2위 차 < 1e-3: {int(tie.sum()):,}건 ({100*tie.mean():.1f}%)",
         "- ⚠️ 임베딩 몫은 정의상 입력에서 예측 가능하다. 라우터 평가 시 택소노미 몫과 "
         "분리해 보고할 것.", "",
         "| # | expert 슬롯 | 축값 | 학습 row | 택소노미 직접 | 임베딩 배정 | 응집도 |",
         "|---:|---|---|---:|---:|---:|---:|"]
    for j, c in enumerate(ex):
        v = cnt[c]
        L.append(f"| {j} | {c} | {top[j]} | **{sum(v):,}** | {v[0]:,} | {v[1]:,} | "
                 f"{cnorm[j]:.4f} |")
    tot_e = [cnt[c][1] for c in ex]
    L += ["", f"임베딩 배정 쏠림: 최다 {max(tot_e):,} / 최소 {min(tot_e):,} "
              f"(균등이면 {int(rest.sum())//E:,})",
          "", "`응집도` = L2 정규화 전 센트로이드 norm — 그 축값의 문제들이 임베딩 공간에서 "
              "얼마나 뭉쳐 있나. 낮을수록 흩어져 있다.", "",
          "expert 슬롯 id는 파일 형식을 다른 조건과 맞추려고 로스터 id를 재사용한 것뿐이고 "
          "**페르소나와 아무 관계가 없다**.", "", f"산출: `{out}`"]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    open(report, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
