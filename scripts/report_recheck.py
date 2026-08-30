#!/usr/bin/env python3
"""REPORT_task3_task4.md 본문 수치 재검산 — 요약문서 재인용을 원본에서 다시 계산한다.

대상(전부 기존 산출물 재사용, 신규 생성 0):
  ① 전체 train 선택압    <- results/acc/seed20210111/binning_train_full.binned.jsonl
  ② WAR 배분 χ²          <- 같은 행렬 (현행 코퍼스 교집합도 함께)
  ③ 재현성 검정 재설계   <- results/acc/evo_repro_exclusive128.raw.jsonl
       기존 z는 귀무 SE를 이산균등 (E^2-1)/12로 고정했는데, 128문제 중 다수가 전원 동점(0점)이라
       그 문제의 순위 분산은 실제로 0이다. 동점을 반영한 **순열 정확 귀무**로 다시 계산하고,
       검정력(최소검출효과)까지 낸다.
  ⑤ K=5 분산분해         <- results/acc/router_self_consistency_full.md 표 파싱(11 x 40)
  부수  진화 실효 온도    <- configs/base.yaml + 도메인 config + load_merged_config 재현

scipy 미사용(공유 env 오염 금지) — χ² 상측확률은 정칙 불완전감마로 직접 구현.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acc/report_recheck.md"
RNG = np.random.default_rng(0)
L: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    L.append(s)


# ---------------------------------------------------------------- χ² 상측확률
def _gammq(a: float, x: float) -> float:
    """Q(a,x) = 1 - P(a,x). Numerical Recipes 급수/연분수."""
    if x < a + 1.0:  # 급수
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c = x + 1.0 - a, 1e300  # 연분수
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(stat: float, df: int) -> float:
    return _gammq(df / 2.0, stat / 2.0)


# ---------------------------------------------------------------- ①②
def sec_binning() -> None:
    say("## ① 전체 train 선택압 (원본 행렬 재계산)")
    say()
    path = ROOT / "results/acc/seed20210111/binning_train_full.binned.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    experts = sorted(rows[0]["per_expert"])
    cur = {json.loads(l)["id"] for l in open(ROOT / "export/acc_v2/acc_train.jsonl", encoding="utf-8")}

    def stats(sub: list[dict]) -> dict:
        S = np.array([[r["per_expert"].get(e, 0) for e in experts] for r in sub], float)
        n = S.shape[0]
        ns = S.sum(1)
        war = np.zeros(len(experts))
        for i in np.where(ns == 1)[0]:
            war[int(np.argmax(S[i]))] += 1
        return dict(n=n, mean=100 * S.mean(), union=100 * (ns > 0).mean(),
                    unani=100 * ((ns == 0) | (ns == len(experts))).mean(),
                    excl=100 * (ns == 1).mean(), war=war, per50=50 * (ns == 1).mean())

    full, inter = stats(rows), stats([r for r in rows if r["id"] in cur])
    say(f"- 행렬: `{path.relative_to(ROOT)}` · {full['n']:,}문제 × expert {len(experts)}명 · 단일 드로우")
    say(f"- 현행 코퍼스(`export/acc_v2/acc_train.jsonl`)와 겹치는 문제: **{inter['n']:,}개**")
    say()
    say("| | 전체 | 현행 코퍼스 교집합 |")
    say("|---|---:|---:|")
    say(f"| 문제 수 | {full['n']:,} | {inter['n']:,} |")
    for k, lab in [("mean", "평균 pass율"), ("union", "union"), ("unani", "만장일치(0명+전원)"),
                   ("excl", "**n_solved == 1**")]:
        say(f"| {lab} | {full[k]:.2f}% | {inter[k]:.2f}% |")
    say(f"| WAR 총합 | {int(full['war'].sum())} | {int(inter['war'].sum())} |")
    say(f"| **batch 50당 기대 단독해결** | **{full['per50']:.2f}개** | **{inter['per50']:.2f}개** |")
    say()

    say("## ② WAR 배분이 균등 무작위와 구별되는가 (χ² 재계산)")
    say()
    for lab, st in [("전체", full), ("현행 코퍼스", inter)]:
        w = st["war"]
        tot, E = w.sum(), len(experts)
        exp = tot / E
        chi = float(((w - exp) ** 2 / exp).sum())
        p = chi2_sf(chi, E - 1)
        sd = math.sqrt(tot * (1 / E) * (1 - 1 / E))
        say(f"**{lab}** (WAR 총합 {int(tot)}) — 균등 기대 {exp:.1f}/명, SD {sd:.1f}, "
            f"관측 {int(w.min())}~{int(w.max())}")
        say(f"- **χ²({E-1}) = {chi:.2f}, p = {p:.3f}**")
        order = np.argsort(-w)
        say("- " + " · ".join(f"{experts[i]} {int(w[i])}" for i in order))
        say()


# ---------------------------------------------------------------- ③
def sec_repro() -> None:
    say("## ③ 단독해결 재현성 — 동점을 반영한 순열 귀무로 재검정")
    say()
    raw = ROOT / "results/acc/evo_repro_exclusive128.raw.jsonl"
    prior = json.loads((ROOT / "results/acc/exclusive128.problem_ids.json")
                       .read_text(encoding="utf-8"))["prior_exclusive_solver"]
    cnt: dict = defaultdict(lambda: defaultdict(list))
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != "persona":
            continue
        cnt[r["pid"]][r["cid"]].append(int(r["pass"]))
    pids = [p for p in cnt if p in prior]
    experts = sorted({c for p in pids for c in cnt[p]})
    E = len(experts)
    K = max(len(v) for p in pids for v in cnt[p].values())
    P = np.array([[np.mean(cnt[p][e]) if cnt[p].get(e) else np.nan for e in experts] for p in pids])
    say(f"- 원본 `{raw.name}` · 문제 {len(pids)} × expert {E} × K={K} "
        f"(생성 {int(np.isfinite(P).sum()) * K:,}건)")

    def midranks(v: np.ndarray) -> np.ndarray:
        return np.array([1 + (v > x).sum() + ((v == x).sum() - 1) / 2 for x in v])

    def run(mask: np.ndarray, lab: str) -> None:
        idx = np.where(mask)[0]
        obs, mu, var = [], 0.0, 0.0
        for i in idx:
            v = P[i]
            R = midranks(v)
            obs.append(R[experts.index(prior[pids[i]])])
            mu += R.mean()          # 순열 귀무: 원단독해결자가 11명 중 균등 선택
            var += R.var()          # 문제 내 순위 분산 (전원 동점이면 0)
        obs = np.array(obs)
        n = len(obs)
        se_perm = math.sqrt(var) / n
        se_unif = math.sqrt((E ** 2 - 1) / 12) / math.sqrt(n)   # 기존 스크립트의 SE
        say(f"**{lab}** (n={n})")
        say(f"- 관측 평균순위 **{obs.mean():.3f}** · 순열 귀무 평균 **{mu / n:.3f}**")
        say(f"- 기존 방식(이산균등 SE {se_unif:.3f}): z = **{(mu / n - obs.mean()) / se_unif:+.2f}**")
        say(f"- **동점 반영 순열 SE {se_perm:.3f}: z = {(mu / n - obs.mean()) / se_perm:+.2f}**"
            f"  (양측 p = {chi2_sf(((mu / n - obs.mean()) / se_perm) ** 2, 1):.3f})")
        return idx

    ns = np.nansum(np.where(P > 0, 1, 0), 1)          # p̂>0 인 expert 수
    contested = (ns > 0) & (ns < E)
    run(np.ones(len(pids), bool), "전체 128문제")
    say()
    run(contested, "갈리는 문제만 (0 < 푼 사람 < 11)")
    say()

    # --- 순위 대신 정답률 차이로 직접 검정 (동점 82문제 때문에 순위는 해석이 안 된다)
    say("**정답률 차이로 직접 검정** — 문제마다 (원단독해결자 p̂) − (나머지 10명 평균 p̂)")
    say()
    for mask, lab in [(np.ones(len(pids), bool), "전체 128문제"),
                      (contested, "한 번이라도 푼 사람이 있는 문제")]:
        idx = np.where(mask)[0]
        d, perm_mu, perm_var = [], 0.0, 0.0
        for i in idx:
            v = P[i]
            j = experts.index(prior[pids[i]])
            own = v[j]
            oth = (v.sum() - own) / (E - 1)
            d.append(own - oth)
            # 순열 귀무: 그 문제에서 '원단독해결자'가 11명 중 누구였든 동등
            alld = np.array([(v[k] - (v.sum() - v[k]) / (E - 1)) for k in range(E)])
            perm_mu += alld.mean()
            perm_var += alld.var()
        d = np.array(d)
        n = len(d)
        se = math.sqrt(perm_var) / n
        z = (d.mean() - perm_mu / n) / se if se > 0 else float("nan")
        say(f"- **{lab}** (n={n}): 원단독해결자 {np.mean([P[i][experts.index(prior[pids[i]])] for i in idx]):.3f} · "
            f"나머지 {np.mean([(P[i].sum() - P[i][experts.index(prior[pids[i]])]) / (E-1) for i in idx]):.3f} · "
            f"차이 **{d.mean():+.4f}**")
        say(f"  순열 귀무 평균 {perm_mu / n:+.4f} · SE {se:.4f} · **z = {z:+.2f}** "
            f"(양측 p = {chi2_sf(z * z, 1):.3f})")
    say()

    # --- 교란 제거: 전문가 전체 실력(이 문제집합 평균)을 빼고 같은 검정
    say("**전문가 전체 실력을 제거하고 다시** — 전반적으로 잘하는 사람이 원단독해결자로 뽑히기도 "
        "쉬우므로, 각 전문가의 이 128문제 평균을 뺀 뒤 같은 대비를 본다.")
    say()
    gmean = np.nanmean(P, 0)                      # expert별 전체 평균 (128문제)
    say("- expert별 평균 정답률: " + " · ".join(
        f"{e} {100 * g:.1f}%" for e, g in sorted(zip(experts, gmean), key=lambda t: -t[1])))
    say()
    Pc = P - gmean[None, :]                       # 전문가 실력 제거
    for M, lab in [(np.ones(len(pids), bool), "전체 128문제"),
                   (contested, "한 번이라도 푼 사람이 있는 문제")]:
        idx = np.where(M)[0]
        d, mu, var = [], 0.0, 0.0
        for i in idx:
            v = Pc[i]
            alld = np.array([v[k] - (v.sum() - v[k]) / (E - 1) for k in range(E)])
            d.append(alld[experts.index(prior[pids[i]])])
            mu += alld.mean()
            var += alld.var()
        d = np.array(d)
        n = len(d)
        se = math.sqrt(var) / n
        z = (d.mean() - mu / n) / se if se > 0 else float("nan")
        say(f"- **{lab}** (n={n}): 실력 제거 후 차이 **{d.mean():+.4f}** · SE {se:.4f} · "
            f"**z = {z:+.2f}** (양측 p = {chi2_sf(z * z, 1):.3f})")
    say()
    war = np.array([sum(1 for p in pids if prior[p] == e) for e in experts], float)
    r = float(np.corrcoef(war, gmean)[0, 1])
    say(f"- 참고: 원단독해결 횟수와 전체 실력의 상관 **r = {r:+.2f}** "
        f"(높으면 위 원시 차이가 실력 교란으로 설명된다는 뜻)")
    say()

    # --- 검정력: 원단독해결자만 pass 확률이 delta만큼 높을 때 검출 확률
    say("**검정력 — 이 표본이 잡을 수 있는 최소 효과**")
    say()
    q = np.nanmean(P, 1)                              # 문제별 기저 난이도
    n = len(pids)
    se_perm_full = None
    say("| 원단독해결자 pass확률 가산 δ | 검출력(α=0.05 양측) |")
    say("|---:|---:|")
    for d in (0.05, 0.10, 0.15, 0.20, 0.30):
        hit = 0
        for _ in range(2000):
            zs = []
            mu = var = 0.0
            for qi in q:
                v = RNG.binomial(5, np.clip([qi] * E, 0, 1)) / 5.0
                v[0] = RNG.binomial(5, np.clip(qi + d, 0, 1)) / 5.0
                R = midranks(v)
                zs.append(R[0])
                mu += R.mean()
                var += R.var()
            zs = np.array(zs)
            se = math.sqrt(var) / n
            if se > 0 and abs((mu / n - zs.mean()) / se) > 1.96:
                hit += 1
        say(f"| +{d:.2f} | {100 * hit / 2000:.1f}% |")
    say()


# ---------------------------------------------------------------- ⑤ 첫 행
def sec_selfcons() -> None:
    say("## ⑤ 첫 행 — SFT 11명 × 40문제 × K=5 분산분해 (표 파싱 후 재계산)")
    say()
    md = (ROOT / "results/acc/router_self_consistency_full.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*(c_\d+|luca)\s*\|\s*(\S.*?)\s*\|\s*([01])\s*\|\s*(\d+)/(\d+)\s*\|",
                      md, re.M)
    by_e: dict = defaultdict(list)
    for e, pid, _g, k, K in rows:
        by_e[e].append((pid, int(k), int(K)))
    experts = sorted(by_e)
    order = [p for p, _, _ in by_e[experts[0]]]
    ok = all([p for p, _, _ in by_e[e]] == order for e in experts)
    K = rows[0][4]
    say(f"- 파싱: expert {len(experts)}명 × {len(order)}문제 = {sum(len(v) for v in by_e.values())}셀 · K={K}")
    say(f"- 문제 순서가 expert 블록 간 동일한가(행렬 복원 가능 조건): **{'예' if ok else '아니오 — 중단'}**")
    if not ok:
        say("- ⚠️ md의 문제 id가 40자로 잘려 있어 위치로 매칭한다. 순서가 다르면 복원 불가.")
        say()
        return
    K = int(K)
    P = np.array([[k / K for _, k, _ in by_e[e]] for e in experts])   # (E, N)
    E, N = P.shape
    gm = P.mean()
    ss_e = N * ((P.mean(1) - gm) ** 2).sum()
    ss_p = E * ((P.mean(0) - gm) ** 2).sum()
    ss_t = ((P - gm) ** 2).sum()
    ss_r = ss_t - ss_e - ss_p
    noise = (P * (1 - P) / (K - 1)).sum()      # 이항 샘플링 분산의 불편추정 합
    say()
    say("| 분산 성분 | 비중 | (요약문서 기재값) |")
    say("|---|---:|---:|")
    say(f"| 문제(난이도) | {100 * ss_p / ss_t:.1f}% | 69.0% |")
    say(f"| expert 주효과 | {100 * ss_e / ss_t:.1f}% | 0.3% |")
    say(f"| 잔차 | {100 * ss_r / ss_t:.1f}% | 30.6% |")
    say(f"| └ 이항 샘플링 노이즈 | {100 * noise / ss_t:.1f}%p | 26.9%p |")
    say(f"| └ **expert×문제 상호작용** | **{100 * (ss_r - noise) / ss_t:.1f}%p** | 3.7%p |")
    say()
    say(f"- MIXED(0 < k < {K}) 비율: **{100 * ((P > 0) & (P < 1)).mean():.1f}%**")
    say()


# ---------------------------------------------------------------- 부수: 실효 온도
def sec_config() -> None:
    say("## 부수 — 진화의 실효 샘플링 파라미터 (머지 재현)")
    say()
    base = yaml.safe_load(open(ROOT / "configs/base.yaml", encoding="utf-8"))
    dom = yaml.safe_load(open(ROOT / "configs/acc_train_seed20210101.yaml", encoding="utf-8"))
    cfg = dict(base)
    cfg.update({k: v for k, v in dom.items() if v is not None})   # run_evolution.load_merged_config
    say(f"- `configs/base.yaml` llm.sampling: `{base['llm'].get('sampling')}`")
    say(f"- `configs/acc_train_seed20210101.yaml` llm: `{dom['llm']}`")
    say(f"- shallow `cfg.update` 머지 후 llm: `{cfg['llm']}`")
    say(f"- sampling 키 생존 여부: **{'있음' if 'sampling' in cfg['llm'] else '사라짐 → LLMService 기본값'}**")
    say("- `src/utils/llm.py:58-60` 기본값: temperature 0.7 / top_p 0.8 / top_k 20")
    say()


if __name__ == "__main__":
    say("# REPORT_task3_task4 본문 재검산")
    say()
    sec_binning()
    sec_repro()
    sec_selfcons()
    sec_config()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"saved -> {OUT}")
