#!/usr/bin/env python
"""로스터 16명을 협업자와 같은 지표(EM · ROUGE-L)로 재채점. 재생성 0.

협업자 기준선 (SNI test 8,699, 단일 생성):
  Dense Llama ckpt-4350        EM 55.63 · ROUGE-L 68.93
  4x1 MoE     ckpt-4000        EM 56.24 · ROUGE-L 69.47

우리 쪽은 gemma teacher 16명 × K=3이 이미 있으므로 저장된 출력 문자열만 다시 잰다.
주의: 우리 UB는 **16명 오라클**이라 단일 모델 수치와 같은 층이 아니다. 그래서
  · per-expert (rep 0 단일 생성) — 같은 층에서 직접 비교 가능
  · best-single / mean(무작위 배정)
  · UB K=1(16명 중 최선) · UB K=3(48회 중 최선)
을 전부 병기한다. EM 오라클과 ROUGE 오라클은 서로 다른 선택이므로 각각 따로 잡는다.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.scorer import sni_metrics  # noqa: E402

RAW = "results/sni/binning_seed20212003/test_raw.jsonl"
SRC = "export/sni_v4/sni_test.jsonl"
ROSTER = "results/sni/seed20212003/roster_final.json"
REPORT = "results/sni/official_metrics_test.md"
BASE = [("Dense Llama (ckpt-4350)", 55.63, 68.93), ("4x1 MoE (ckpt-4000)", 56.24, 69.47)]


def main():
    src = {}
    for l in open(SRC):
        r = json.loads(l)
        src[r["id"]] = r
    roster = json.load(open(ROSTER))
    ex = [p["id"] for p in roster]
    names = {p["id"]: p["name"] for p in roster}
    ei = {c: i for i, c in enumerate(ex)}
    pids = sorted(src)
    pi = {p: i for i, p in enumerate(pids)}
    P, E, K = len(pids), len(ex), 3
    EM = np.full((P, E, K), np.nan, np.float32)
    RL = np.full((P, E, K), np.nan, np.float32)

    cache = {}
    n = 0
    for l in open(RAW):
        d = json.loads(l)
        p, c, k = d["pid"], d["cid"], int(d["rep"])
        if p not in pi or c not in ei or k >= K:
            continue
        key = (p, d.get("code") or "")
        m = cache.get(key)
        if m is None:
            m = sni_metrics(src[p], key[1])
            cache[key] = m
        EM[pi[p], ei[c], k] = m["exact_match"]
        RL[pi[p], ei[c], k] = m["rougeL"]
        n += 1
        if n % 50000 == 0:
            print(f"  {n:,} 채점", flush=True)
    miss = int(np.isnan(EM).sum())
    EM0, RL0 = np.nan_to_num(EM), np.nan_to_num(RL)

    per = [(names[c], EM0[:, j, 0].mean(), RL0[:, j, 0].mean(),
            EM0[:, j].mean(), RL0[:, j].mean()) for j, c in enumerate(ex)]
    bs_em = max(per, key=lambda t: t[1])
    bs_rl = max(per, key=lambda t: t[2])

    L = ["# 로스터 16명 공식 지표 재채점 — SNI test 8,699 (재생성 0)", "",
         f"저장된 출력 {n:,}건({P:,}문제 × {E}명 × K={K})을 `sni_metrics`로 다시 쟀다. "
         f"결측 {miss:,}칸은 0으로 뒀다.",
         "협업자 수치는 **단일 모델 · 단일 생성**이다. 아래에서 같은 층은 `rep 0` 열뿐이고, "
         "UB는 16명을 다 돌려본 뒤 최선을 고른 오라클이므로 같은 층이 아니다.", "",
         "## 0. 협업자 기준선과 나란히", "",
         "| 모델 | EM | ROUGE-L | 층 |", "|---|---:|---:|---|"]
    for nm, e, r in BASE:
        L.append(f"| {nm} | {e:.2f} | {r:.2f} | 단일 모델 · 단일 생성 (llama student) |")
    L += [f"| gemma teacher — best-single (EM 기준: {bs_em[0]}) | **{bs_em[1]:.2f}** | "
          f"{bs_em[2]:.2f} | 단일 모델 · 단일 생성 |",
          f"| gemma teacher — best-single (ROUGE 기준: {bs_rl[0]}) | {bs_rl[1]:.2f} | "
          f"**{bs_rl[2]:.2f}** | 단일 모델 · 단일 생성 |",
          f"| gemma teacher — 16명 평균 (무작위 배정) | "
          f"{np.mean([t[1] for t in per]):.2f} | {np.mean([t[2] for t in per]):.2f} | 단일 생성 |",
          f"| **UB (16명 오라클, K=1)** | **{np.max(EM0[:, :, 0], 1).mean():.2f}** | "
          f"**{np.max(RL0[:, :, 0], 1).mean():.2f}** | ⚠️ 오라클 — 같은 층 아님 |",
          f"| **UB (16명 × K=3 오라클, 48회)** | "
          f"**{np.max(EM0.reshape(P, -1), 1).mean():.2f}** | "
          f"**{np.max(RL0.reshape(P, -1), 1).mean():.2f}** | ⚠️ 오라클 — 같은 층 아님 |", "",
          "## 1. expert별", "",
          "| expert | EM (rep 0) | ROUGE-L (rep 0) | EM (K=3 평균) | ROUGE-L (K=3 평균) |",
          "|---|---:|---:|---:|---:|"]
    for t in sorted(per, key=lambda t: -t[1]):
        L.append(f"| {t[0]} | {t[1]:.2f} | {t[2]:.2f} | {t[3]:.2f} | {t[4]:.2f} |")

    ubk1_em, ubk1_rl = np.max(EM0[:, :, 0], 1).mean(), np.max(RL0[:, :, 0], 1).mean()
    L += ["", "## 2. 헤드룸", "",
          "| | EM | ROUGE-L |", "|---|---:|---:|",
          f"| best-single (각 지표 기준 최고) | {bs_em[1]:.2f} | {bs_rl[2]:.2f} |",
          f"| UB K=1 (16명 오라클) | {ubk1_em:.2f} | {ubk1_rl:.2f} |",
          f"| **헤드룸** | **+{ubk1_em - bs_em[1]:.2f}pp** | **+{ubk1_rl - bs_rl[2]:.2f}pp** |",
          "", "## 3. 우리 이진 판정과의 관계", "",
          "우리 진화·비닝이 쓴 통과 기준은 **EM==100 or ROUGE-L>70**이다. 위 표의 EM·ROUGE는 "
          "그 임계를 적용하지 않은 공식 지표 원값이라 서로 다른 숫자다.",
          f"참고: rep 0에서 이진 통과율 best-single은 "
          f"{max((100*((EM0[:, j, 0] >= 100) | (RL0[:, j, 0] > 70)).mean()) for j in range(E)):.2f}, "
          f"16명 오라클은 "
          f"{100*(((EM0[:, :, 0] >= 100) | (RL0[:, :, 0] > 70)).any(1)).mean():.2f}.", ""]
    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
