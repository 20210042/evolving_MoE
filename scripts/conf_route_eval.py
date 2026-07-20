#!/usr/bin/env python3
"""Expert 자기확신 라우팅 평가: 확신도 argmax로 top-1 → oracle 근접하나?
best-single / oracle 대비 + confidence 캘리브레이션(확신 vs 정답 상관).

⚠️ 확신도 특징이 MCQA 전용이라 오픈 QA 데이터셋에서는 실행되지 않는다.
Usage: python scripts/conf_route_eval.py --dataset qasc
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

ap = argparse.ArgumentParser()
rc.add_dataset_arg(ap)
_a = ap.parse_args()
sp = rc.spec(_a.dataset)

d = np.load(rc.feat_path(sp, sp.eval_split, "conf"), allow_pickle=True)
conf, ids, experts = d["conf"], [str(x) for x in d["ids"]], [str(x) for x in d["experts"]]

vb = rc.load_binning(rc.REPO / sp.eval_binned)

keep = [i for i, pid in enumerate(ids) if pid in vb]
conf = conf[keep]
S = np.array([[vb[ids[i]]["per_expert"].get(e, 0) for e in experts] for i in keep], np.float32)
N = len(keep)
best, oracle = rc.baselines(S)

from scipy.stats import rankdata
route = conf.argmax(1)
routed = 100 * np.mean([S[i, route[i]] for i in range(N)])

def eval_route(scoremat, name):
    r = scoremat.argmax(1)
    acc = 100 * np.mean([S[i, r[i]] for i in range(N)])
    print(f"  [{name:16}] routed {acc:.1f}%  (best-single {best:.1f}, oracle {oracle:.1f})")
    return acc

print("\n=== expert 간 확신 정규화 변형 ===")
eval_route(conf, "raw")
# z-score per expert (expert별 확신 평균/편차 제거 → 스케일 통일)
z = (conf - conf.mean(0, keepdims=True)) / (conf.std(0, keepdims=True) + 1e-6)
eval_route(z, "z-norm/expert")
# rank per expert (분포모양 무시, 순위만)
rk = np.stack([rankdata(conf[:, e]) for e in range(conf.shape[1])], 1) / N
eval_route(rk, "rank/expert")
# expert별 확신을 그 expert의 전체 pass율로 보정(강한 expert 우대)
prior = S.mean(0, keepdims=True)
eval_route(z + 3 * prior, "z-norm + pass-prior")
# 캘리브레이션: 각 expert의 (확신 상위 50% vs 하위 50%) 정답률
cal = []
for e in range(len(experts)):
    hi = conf[:, e] >= np.median(conf[:, e])
    cal.append((100 * S[hi, e].mean(), 100 * S[~hi, e].mean()))
avg_hi = np.mean([c[0] for c in cal]); avg_lo = np.mean([c[1] for c in cal])

print(f"=== {sp.name.upper()} {sp.eval_split} {N} — expert 자기확신 라우팅 ===")
print(f"  best-single      : {best:.1f}%")
print(f"  confidence route : {routed:.1f}%   (top-1, argmax 자기확신)")
print(f"  oracle (union)   : {oracle:.1f}%")
print(f"  → routed − best  : {routed-best:+.1f}pp")
print(f"  → oracle 회수율   : {100*(routed-best)/(oracle-best+1e-9):.0f}% of headroom")
print(f"\n  캘리브레이션: 확신 상위50% 정답률 {avg_hi:.1f}% vs 하위50% {avg_lo:.1f}% "
      f"(gap {avg_hi-avg_lo:+.1f}pp; 클수록 확신이 정답과 정렬 = 라우팅 신호 강함)")
# 확신 top-1이 실패했는데 union엔 solver 있던 문제 비율(놓친 headroom)
miss = np.mean([(S[i].sum() > 0) and (S[i, route[i]] == 0) for i in range(N)]) * 100
print(f"  놓친 문제(solver 있는데 확신라우팅 실패): {miss:.1f}%")
