#!/usr/bin/env python3
"""진화 레짐에서의 다회시행(K회) 파일럿 — solvability가 선택압으로 쓸 만한 신호인가.

동기: results/acc/router_self_consistency_full.md(11×40×5, SFT LoRA expert, T=0.7)에서
p̂=k/5로 분산분해하면 문제(난이도) 69.0% / expert 주효과 0.3% / 잔차 30.6%, 그 잔차의
26.9%p가 이항 샘플링 노이즈라 진짜 expert×문제 상호작용은 ~3.7%p뿐이었다. 같은 데이터로
라벨 규칙만 바꾸면 단독해결(WAR) 총합이 greedy 15 → majority 3/5에서 4로 무너진다.
즉 진화가 선택압으로 쓰는 단독해결의 대부분이 디코딩 복권일 수 있다.

그런데 그 측정은 '배포 레짐'(SFT된 LoRA expert, T=0.7)이었다. 진화 루프는 다른 레짐이다
— gemma 베이스 + 프롬프트 persona, T=1.0(configs/base.yaml), enable_thinking=false.
이 스크립트는 **진화 자기 조건에서 실제 한 배치(batch_size개 학습 문제)를 K회 재생성**해
같은 붕괴가 일어나는지 직접 확인한다. 전체 진화를 5회 시행으로 바꾸기 전의 사전 검정.

두 arm을 같은 문제·같은 K로 동시에 돌린다:
  persona          : 현행 진화 프롬프트 (build_expert_prompt, approach=None)
  persona_fewshot  : SFT/배포에서 쓴 persona+few-shot 레시피를 진화 프롬프트에 그대로 적용
                     (build_fewshot_block → approach=). few-shot 예시는 라벨 패키지에서
                     select_expert_rows/pick_fewshot_examples로 뽑아 배포와 동일하게 재현.
                     ⚠️ 라벨 패키지는 진화가 끝난 뒤에 만들어진 사후 산물이라, 이 arm은
                     '진화 중 few-shot이 낼 수 있는 최선'의 낙관적 상한이다. 여기서도
                     상호작용이 안 늘면 in-loop few-shot은 볼 것 없다.

산출: arm별 p̂ 행렬(expert×문제) → MIXED 비율, 노이즈 보정 분산분해, 그리고 라벨 규칙
(single-draw / majority / soft)에 따른 WAR·축출 순위 변화.

Usage:
  # 프롬프트만 만들어 눈으로 확인 (모델 로드 없음, 로그인 노드에서 실행 가능)
  python scripts/evo_multisample_pilot.py --dry_run

  # 스모크: 배선·처리량 확인
  python scripts/evo_multisample_pilot.py --n_problems 5 --k 2 \
      --out results/acc/evo_multisample_pilot_smoke.md

  # 본 파일럿: 진화 한 배치(50문제) × K=5 × 2 arm
  python scripts/evo_multisample_pilot.py --n_problems 50 --k 5 \
      --out results/acc/evo_multisample_pilot.md
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from data.loader import get_dataset  # noqa: E402
from evaluation.scorer import pass_at_threshold  # noqa: E402
from orchestrator import _score_pair  # noqa: E402  (진화와 동일한 채점 경로)
from prompts.coding import build_expert_prompt, build_fewshot_block  # noqa: E402
from utils.helpers import finalize_generation_output  # noqa: E402

ARMS = ("persona", "persona_fewshot")


# ----------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/acc_train_seed20210101.yaml",
                    help="base.yaml 위에 머지할 진화 config (샘플링·모델·thinking을 여기서 가져온다)")
    ap.add_argument("--dataset", default="acc")
    ap.add_argument("--split", default="train", help="진화가 순회하는 split")
    ap.add_argument("--data_dir", default="export/acc_v2",
                    help="재빌드된 현행 코퍼스(run_acc_pipeline_v2.sh의 CORPUS와 동일)")
    ap.add_argument("--roster_path", default="results/acc/seed20210111/roster_final.json")
    ap.add_argument("--n_problems", type=int, default=50, help="진화 한 배치 크기와 맞춘다")
    ap.add_argument("--k", type=int, default=5, help="같은 (expert,문제)를 몇 번 재생성할지")
    ap.add_argument("--arms", default=",".join(ARMS))
    # persona_fewshot arm 재현용 (scripts/router_self_consistency.py와 동일 인자)
    ap.add_argument("--label_package", default="export/acc_binning_seed20210111_v2")
    ap.add_argument("--max_n_solved", type=int, default=9)
    ap.add_argument("--n_fewshot", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="문제 표본 추출용")
    ap.add_argument("--prior_solver", default=None,
                    help="{'prior_exclusive_solver': {pid: expert}} json. 주면 '단일드로우에서 그 문제를 "
                         "단독으로 푼 expert가 K회에서도 재현되는가'를 문제내 순위로 검정한다(귀무=6.0/11).")
    ap.add_argument("--problem_ids", default=None,
                    help="문제 id 리스트 json. 주면 무작위 표본 대신 이 문제들만 쓴다 "
                         "(스크리닝에서 고른 '갈리는 문제' 집합을 그대로 넘길 때)")
    ap.add_argument("--score_workers", type=int, default=4)
    ap.add_argument("--exclude_ids", default=None,
                    help="제외할 문제 id 목록 json(scripts/sni_context_filter.py 산출). "
                         "표집 이전에 빼므로 '전수 - 제외'가 대상 모집단이 된다.")
    ap.add_argument("--gen_chunk", type=int, default=0,
                    help="한 번에 생성할 문제 수(0=전부 한 번에). 전수 런에서는 반드시 지정한다 — "
                         "문제×expert 프롬프트를 통째로 쌓으면 생성 전에 메모리가 터진다.")
    ap.add_argument("--out", default="results/acc/evo_multisample_pilot.md")
    ap.add_argument("--raw_out", default=None,
                    help="생성물 원본 jsonl 경로 (기본: --out과 같은 이름의 .raw.jsonl)")
    ap.add_argument("--resume_raw", default=None,
                    help="기존 raw jsonl에서 '완결된 (arm,rep) 패스'를 읽어 그 패스는 건너뛴다. "
                         "인프라 장애로 중단된 잡을 이어 돌릴 때 (완결=문제수×expert수 다 있음).")
    ap.add_argument("--dry_run", action="store_true",
                    help="프롬프트만 조립해 층별 대표문제 × 로스터 전원을 파일로 덤프하고 종료 "
                         "(모델 로드 없음). 잡 제출 전 승인용.")
    ap.add_argument("--preview_out", default="results/sni/prompts_preview.md",
                    help="--dry_run 덤프 경로")
    return ap.parse_args()


def load_cfg(config: str | None) -> dict:
    with open(ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if config:
        p = Path(config)
        if not p.is_absolute():
            p = ROOT / p
        with open(p, "r", encoding="utf-8") as f:
            over = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


# ------------------------------------------------------------------- prompt 조립
def build_fewshot_approach(
    roster: List[Dict[str, Any]], a: argparse.Namespace
) -> Dict[str, str | None]:
    """expert별 few-shot 프리앰블. 배포(generate_lora_binning/router_self_consistency)와
    똑같이 select_expert_rows + pick_fewshot_examples를 써서 같은 예시가 나오게 한다."""
    from train_sft import pick_fewshot_examples, select_expert_rows  # noqa: E402

    out: Dict[str, str | None] = {}
    for p in roster:
        cid = p["id"]
        try:
            chosen, selected, _ds, is_shared = select_expert_rows(
                package_dir=str(ROOT / a.label_package), expert_id=cid, source_jsonl=None,
                seed=42, data_ratio=1.0, max_n_solved=a.max_n_solved, min_n_solved=None,
            )
        except Exception as exc:  # 라벨 패키지에 없는 expert는 few-shot 없이 간다
            logging.warning("few-shot 풀 없음 (%s): %s", cid, exc)
            out[cid] = None
            continue
        if is_shared or not selected:
            out[cid] = None
            continue
        out[cid] = build_fewshot_block(pick_fewshot_examples(chosen, selected, a.n_fewshot)) or None
    return out


def make_msgs(
    item: Dict[str, Any],
    player: Dict[str, Any],
    *,
    arm: str,
    model_name: str,
    dataset_name: str,
    fewshot: Dict[str, str | None],
):
    """진화 run_batch(src/orchestrator.py)와 동일한 프롬프트. arm이 fewshot이면
    approach 슬롯에 few-shot 블록을 넣는다 (acc 로스터는 approach가 전부 None이라 충돌 없음)."""
    approach = player.get("approach")
    if arm == "persona_fewshot":
        approach = fewshot.get(player["id"]) or approach
    return build_expert_prompt(
        item.get("instruction", ""),
        player.get("system_prompt", "You are an expert coder."),
        dataset=item.get("dataset") or dataset_name,
        model_name=model_name,
        starter_code=item.get("starter_code"),
        approach=approach,
        domain=item.get("domain"),
        answer_line=item.get("answer_line"),
        definition=item.get("definition"),
        positive_examples=item.get("positive_examples"),
        negative_examples=item.get("negative_examples"),
    )


# ----------------------------------------------------------------------- 채점
def score_chunk(
    batch_data: List[Dict[str, Any]],
    codes: Dict[Tuple[str, str], str],
    *,
    cfg: dict,
    workers: int,
) -> Dict[Tuple[str, str], float]:
    """진화의 _score_pairs_parallel과 같은 구조(같은 _score_pair, 같은 wall-clock 가드)."""
    by_id = {b["id"]: b for b in batch_data}
    args = [
        (by_id[pid], code, int(cfg.get("lcb_timeout", 10)), float(cfg.get("code_exec_timeout", 3)),
         cfg.get("lcb_release_version", "release_v5"), tag)
        for (pid, tag), code in codes.items() if pid in by_id
    ]
    results: Dict[Tuple[str, str], float] = {}
    if not args:
        return results
    workers = max(1, workers)
    if workers == 1 or len(args) == 1:
        for x in args:
            pid, tag, sc = _score_pair(x)
            results[(pid, tag)] = sc
        return results

    timeout_s = max(180.0, 5.0 * len(args) / workers)
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = {pool.submit(_score_pair, x): (x[0]["id"], x[5]) for x in args}
    try:
        for fut in as_completed(futures, timeout=timeout_s):
            pid, tag, sc = fut.result()
            results[(pid, tag)] = sc
    except FuturesTimeoutError:
        unfinished = [futures[f] for f in futures if not f.done()]
        logging.error("채점 wall-clock 초과(%.0fs): %d/%d 미완 → 0.0", timeout_s, len(unfinished), len(futures))
    finally:
        for proc in list(getattr(pool, "_processes", {}).values()):
            if proc.is_alive():
                proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    for pair in futures.values():
        results.setdefault(pair, 0.0)
    return results


# ------------------------------------------------------------------------ 분석
def analyze(P: List[List[float]], experts: List[str], K: int) -> Dict[str, Any]:
    """p̂ 행렬(expert × 문제) → 분산분해 + 라벨 규칙별 WAR."""
    E, N = len(P), len(P[0])
    cells = [(j, i) for j in range(E) for i in range(N)]
    grand = sum(P[j][i] for j, i in cells) / (E * N)
    tot = sum((P[j][i] - grand) ** 2 for j, i in cells) / (E * N)
    prob_m = [sum(P[j][i] for j in range(E)) / E for i in range(N)]
    exp_m = [sum(P[j]) / N for j in range(E)]
    v_prob = sum((m - grand) ** 2 for m in prob_m) / N
    v_exp = sum((m - grand) ** 2 for m in exp_m) / E
    v_res = tot - v_prob - v_exp
    # 이항 샘플링 분산의 불편추정: E[p̂(1-p̂)] = p(1-p)(K-1)/K → noise = p̂(1-p̂)/(K-1)
    noise = (sum(P[j][i] * (1 - P[j][i]) for j, i in cells) / (E * N) / (K - 1)) if K > 1 else 0.0

    mixed = sum(1 for j, i in cells if 0 < P[j][i] < 1)

    def war(lab: List[List[int]]) -> Tuple[Dict[str, int], int]:
        solved = [set(i for i in range(N) if lab[j][i]) for j in range(E)]
        union = set().union(*solved) if solved else set()
        out = {}
        for j in range(E):
            others = set().union(*[solved[k] for k in range(E) if k != j]) if E > 1 else set()
            out[experts[j]] = len(union) - len(others)
        return out, len(union)

    def soft_war() -> Dict[str, float]:
        out = {}
        for j in range(E):
            s = 0.0
            for i in range(N):
                others = 1.0
                for k in range(E):
                    if k != j:
                        others *= (1 - P[k][i])
                s += P[j][i] * others
            out[experts[j]] = s
        return out

    rules = {}
    for name, thr in (("majority", 0.5), ("any", 0.0), ("strict", 1.0 - 1e-9)):
        lab = [[1 if P[j][i] > thr else 0 for i in range(N)] for j in range(E)]
        w, u = war(lab)
        rules[name] = {"war": w, "union": u}
    sw = soft_war()
    soft_union = sum(1 - math.prod(1 - P[j][i] for j in range(E)) for i in range(N))
    # 신호 밀도: 문제별 몇 명이 푸는가. 0명/전원(만장일치)은 WAR에 아무 기여도 못 한다.
    n_solved = [sum(1 for j in range(E) if P[j][i] > 0.5) for i in range(N)]
    hist = [0] * (E + 1)
    for v in n_solved:
        hist[v] += 1
    contested = sum(1 for v in n_solved if 0 < v < E)
    return {
        "E": E, "N": N, "grand": grand, "tot_var": tot,
        "v_prob": v_prob, "v_exp": v_exp, "v_res": v_res, "noise": noise,
        "mixed": mixed, "cells": E * N,
        "rules": rules, "soft_war": sw, "soft_union": soft_union,
        "expert_mean": {experts[j]: exp_m[j] for j in range(E)},
        "hist": hist, "contested": contested,
    }


def reproducibility(
    P: List[List[float]], experts: List[str], batch: List[Dict[str, Any]],
    prior: Dict[str, str], K: int,
) -> Dict[str, Any] | None:
    """단일드로우에서 그 문제를 '단독으로' 푼 expert가 K회 재측정에서도 앞서는가.

    문제 안에서 11명을 비교하므로 대조군 문제가 따로 필요 없다(self-controlled).
    귀무가설(단독해결이 순전한 복권) 아래 원단독해결자의 문제내 평균순위 = (E+1)/2.
    """
    E = len(experts)
    idx = {e: j for j, e in enumerate(experts)}
    ranks, own, oth, top1, still_excl = [], [], [], 0, 0
    for i, it in enumerate(batch):
        s = prior.get(it["id"])
        if s is None or s not in idx:
            continue
        js = idx[s]
        col = [P[j][i] for j in range(E)]
        v = col[js]
        # 동점은 평균순위 (1 = 가장 높은 p̂)
        ranks.append(1 + sum(1 for x in col if x > v) + (sum(1 for x in col if x == v) - 1) / 2)
        own.append(v)
        oth.append((sum(col) - v) / (E - 1))
        if v > max(x for j, x in enumerate(col) if j != js):
            top1 += 1
        if sum(1 for x in col if x > 0.5) == 1 and v > 0.5:
            still_excl += 1
    if not ranks:
        return None
    n = len(ranks)
    null_mean = (E + 1) / 2
    # 순위 평균의 귀무 표준오차: 이산균등(1..E)의 분산 = (E^2-1)/12
    se = math.sqrt((E * E - 1) / 12 / n)
    z = (null_mean - sum(ranks) / n) / se
    return {
        "n": n, "mean_rank": sum(ranks) / n, "null_rank": null_mean, "z": z,
        "own": sum(own) / n, "other": sum(oth) / n,
        "top1": top1, "still_excl": still_excl,
    }


def pct(x: float, tot: float) -> str:
    return f"{100 * x / tot:5.1f}%" if tot > 0 else "  n/a"


def render(arm_stats: Dict[str, Dict[str, Any]], first_rep_war: Dict[str, Dict[str, int]],
           experts: List[str], a: argparse.Namespace, cfg: dict, elapsed: Dict[str, float],
           repro: Dict[str, Dict[str, Any] | None] | None = None, k_done: int | None = None) -> str:
    L: List[str] = []
    L.append("# 진화 레짐 다회시행 파일럿 — solvability가 선택압으로 쓸 만한가")
    L.append("")
    eff = cfg.get("_eff_sampling", {})
    L.append(f"- 레짐: `{cfg.get('model')}` · T={eff.get('temperature', '?')} top_p={eff.get('top_p', '?')} "
             f"top_k={eff.get('top_k', '?')} · enable_thinking={cfg.get('enable_thinking')} "
             f"· 진화 프롬프트(build_expert_prompt) — run_evolution.py와 동일한 config 머지 경로")
    L.append(f"- 문제: {a.data_dir} `{a.dataset}_{a.split}` 에서 무작위 {a.n_problems}개 "
             f"(seed={a.seed}, 진화 batch_size와 동일 구성 · 사전 선별 없음)")
    K = a.k if k_done is None else k_done
    L.append(f"- 로스터: {a.roster_path} ({len(experts)}명) · K={K}회 재생성"
             + ("" if K == a.k else f" ⚠️ 계획 {a.k}회 중 {K}회까지만 완료"))
    L.append("")

    for arm, st in arm_stats.items():
        L.append(f"## arm: `{arm}`  (생성 {elapsed.get(arm, 0)/60:.1f}분)")
        L.append("")
        tot = st["tot_var"]
        inter = st["v_res"] - st["noise"]
        L.append(f"- 평균 pass율 {100*st['grand']:.1f}% · MIXED(0<p̂<1) **{st['mixed']}/{st['cells']} "
                 f"({100*st['mixed']/st['cells']:.1f}%)**")
        L.append("")
        # 신호 밀도 — 만장일치(0명/전원) 문제는 WAR에 기여 못 하므로 진화가 실제로 쓰는 문제 수
        L.append(f"- **신호 밀도: 갈리는 문제(0<n_solved<{st['E']}) {st['contested']}/{st['N']} "
                 f"({100*st['contested']/st['N']:.1f}%)** — 나머지는 만장일치라 WAR 기여 0")
        L.append("")
        L.append("| n_solved | " + " | ".join(str(v) for v in range(st["E"] + 1)) + " |")
        L.append("|---" * (st["E"] + 2) + "|")
        L.append("| 문제 수 | " + " | ".join(str(v) for v in st["hist"]) + " |")
        L.append("")
        L.append("| 분산 성분 | 비중 |")
        L.append("|---|---:|")
        L.append(f"| 문제(난이도) | {pct(st['v_prob'], tot)} |")
        L.append(f"| expert 주효과 | {pct(st['v_exp'], tot)} |")
        L.append(f"| 잔차(상호작용+노이즈) | {pct(st['v_res'], tot)} |")
        if K > 1:
            L.append(f"| └ 이항 샘플링 노이즈(불편추정) | {pct(st['noise'], tot)} |")
            L.append(f"| └ **진짜 expert×문제 상호작용** | **{pct(inter, tot)}** |")
        L.append("")
        if K == 1:
            L.append("> ⚠️ K=1이라 잔차에서 노이즈와 상호작용을 **분리할 수 없다**. 위 잔차는 둘의 합이다.")
            L.append("")
        L.append("| 라벨 규칙 | union | WAR 총합 | WAR=0 인원 |")
        L.append("|---|---:|---:|---:|")
        fw = first_rep_war.get(arm, {})
        L.append(f"| single-draw (현행 진화 1회) | {fw.get('_union', 0)} "
                 f"| {sum(v for k, v in fw.items() if not k.startswith('_'))} "
                 f"| {sum(1 for k, v in fw.items() if not k.startswith('_') and v == 0)} |")
        for name in ("majority", "any", "strict"):
            r = st["rules"][name]
            L.append(f"| {name} | {r['union']} | {sum(r['war'].values())} | "
                     f"{sum(1 for v in r['war'].values() if v == 0)} |")
        L.append(f"| soft (p̂ 그대로) | {st['soft_union']:.1f} | {sum(st['soft_war'].values()):.2f} | 0 |")
        L.append("")
        order = sorted(st["soft_war"], key=lambda k: -st["soft_war"][k])
        L.append("| expert | 평균 pass율 | single-draw WAR | majority WAR | soft-WAR |")
        L.append("|---|---:|---:|---:|---:|")
        for e in order:
            L.append(f"| {e} | {100*st['expert_mean'][e]:.1f}% | {fw.get(e, 0)} | "
                     f"{st['rules']['majority']['war'][e]} | {st['soft_war'][e]:.3f} |")
        L.append("")

    if repro and any(repro.values()):
        L.append("## 단독해결 재현성 — 단일드로우의 단독해결자가 K회에서도 앞서는가")
        L.append("")
        L.append(f"문제 안에서 {len(experts)}명을 비교(self-controlled). 귀무가설(단독해결=복권) 아래 "
                 f"원단독해결자의 문제내 평균순위 = **{(len(experts)+1)/2:.1f}**.")
        L.append("")
        L.append("| arm | n | 원단독해결자 평균 p̂ | 나머지 평균 p̂ | 평균순위 (귀무 {:.1f}) | z | 여전히 최상위 | 여전히 단독 |".format((len(experts)+1)/2))
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm, r in repro.items():
            if not r:
                continue
            L.append(f"| {arm} | {r['n']} | {r['own']:.3f} | {r['other']:.3f} | {r['mean_rank']:.2f} | "
                     f"{r['z']:+.2f} | {r['top1']}/{r['n']} | {r['still_excl']}/{r['n']} |")
        L.append("")
        L.append("> z > 2 이면 원단독해결자가 우연 이상으로 앞선다(= 문제×페르소나 구조가 실재).")
        L.append("> z ≈ 0 이면 단독해결은 재현되지 않는 복권 — K를 늘려도 WAR은 못 살린다.")
        L.append("")

    if len(arm_stats) == 2:
        a1, a2 = list(arm_stats)
        i1 = arm_stats[a1]["v_res"] - arm_stats[a1]["noise"]
        i2 = arm_stats[a2]["v_res"] - arm_stats[a2]["noise"]
        L.append("## arm 비교 — few-shot이 expert×문제 상호작용을 키우는가")
        L.append("")
        L.append(f"- `{a1}` 상호작용 {pct(i1, arm_stats[a1]['tot_var'])} → "
                 f"`{a2}` {pct(i2, arm_stats[a2]['tot_var'])}")
        L.append(f"- soft-WAR 총합: {sum(arm_stats[a1]['soft_war'].values()):.2f} → "
                 f"{sum(arm_stats[a2]['soft_war'].values()):.2f}")
        L.append(f"- union(soft): {arm_stats[a1]['soft_union']:.1f} → {arm_stats[a2]['soft_union']:.1f}")
        L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"알 수 없는 arm: {arm} (가능: {ARMS})")

    cfg = load_cfg(a.config)
    model_name = cfg["model"]
    enable_thinking = bool(cfg.get("enable_thinking", True))

    roster = json.loads((ROOT / a.roster_path).read_text(encoding="utf-8"))
    experts = [p["id"] for p in roster]

    data = get_dataset(a.dataset, split=a.split, local_dir=str(ROOT / a.data_dir))
    if a.exclude_ids:
        ex = json.loads(Path(a.exclude_ids).read_text(encoding="utf-8"))
        ex = set(ex.get("ids", []) if isinstance(ex, dict) else ex)
        before = len(data)
        data = [r for r in data if r["id"] not in ex]
        logging.info("컨텍스트 초과 제외: %d건 (%d → %d) [%s]",
                     before - len(data), before, len(data), a.exclude_ids)
    if a.problem_ids:
        wanted = json.loads(Path(a.problem_ids).read_text(encoding="utf-8"))
        wanted = wanted["ids"] if isinstance(wanted, dict) else wanted
        by_id = {r["id"]: r for r in data}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"{a.data_dir}에 없는 문제 id {len(missing)}개: {missing[:3]}")
        batch = [by_id[i] for i in wanted]
        logging.info("문제 %d개 (--problem_ids %s) · expert %d명 · K=%d · arms=%s",
                     len(batch), a.problem_ids, len(experts), a.k, arms)
    else:
        rng = random.Random(a.seed)
        batch = rng.sample(data, min(a.n_problems, len(data)))
        logging.info("문제 %d개 표본 (전체 %d) · expert %d명 · K=%d · arms=%s",
                     len(batch), len(data), len(experts), a.k, arms)

    fewshot: Dict[str, str | None] = {}
    if "persona_fewshot" in arms:
        fewshot = build_fewshot_approach(roster, a)
        logging.info("few-shot 확보: %s", {e: bool(v) for e, v in fewshot.items()})

    if a.dry_run:
        # 대표 문제를 층별로 고른다. 잡을 올리기 전에 조립된 실물을 보이기 위한 것 —
        # 원 프로브(job 229352)는 이 단계를 건너뛰어 41,400 생성이 무효가 됐다.
        picks: List[Tuple[str, Dict[str, Any]]] = []
        for tag, pred in (
            ("객관식(닫힘)", lambda r: r.get("task_closed")),
            ("서술형(gold>=15단어)", lambda r: not r.get("task_closed")
             and (r.get("gold_len_median") or 0) >= 15),
            ("열림/단답", lambda r: not r.get("task_closed")
             and (r.get("gold_len_median") or 0) < 15),
            ("비영어 출력", lambda r: r.get("output_language") not in (None, "English", "unknown")),
        ):
            hit = next((r for r in batch if pred(r)), None) or next(
                (r for r in data if pred(r)), None)
            if hit is not None:
                picks.append((tag, hit))
        if not picks:
            picks = [("표본", batch[0])]

        L: List[str] = ["# 조립된 프롬프트 실물 — 실행 전 검수용", "",
                        f"- 로스터: `{a.roster_path}` ({len(experts)}명)",
                        f"- 데이터: `{a.data_dir}` `{a.dataset}_{a.split}`",
                        "- **검수 포인트: user 턴에 태스크 정의가 들어가 있지 않은가 "
                        "· system(정체성)이 user에 묻히지 않는가**", ""]
        for tag, item in picks:
            L += [f"## {tag} — `{item.get('task_name', item['id'])}`", "",
                  f"- category `{item.get('category')}` · closed={item.get('task_closed')} "
                  f"· gold중앙 {item.get('gold_len_median')}단어",
                  f"- gold: `{str(item.get('ground_truth', [''])[0])[:160]}`", ""]
            for arm in arms:
                for p in roster:
                    msgs = make_msgs(item, p, arm=arm, model_name=model_name,
                                     dataset_name=a.dataset, fewshot=fewshot)
                    L.append(f"### `{p['id']}` (arm={arm})")
                    L.append("")
                    for m in msgs:
                        body = m["content"]
                        clipped = body[:1200] + (
                            f"\n... [+{len(body)-1200} chars]" if len(body) > 1200 else "")
                        L += [f"**[{m['role']}]** ({len(body)} chars)", "", "```", clipped, "```", ""]
        out_path = Path(a.preview_out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"프롬프트 실물 {len(picks)}문제 × {len(experts)}명 × {len(arms)}arm -> {out_path}")
        for m in make_msgs(picks[0][1], roster[-1], arm=arms[0], model_name=model_name,
                           dataset_name=a.dataset, fewshot=fewshot):
            print(f"--- [{m['role']}] ({len(m['content'])} chars) ---\n{m['content'][:1200]}")
        return

    from agents.base import Agent  # noqa: E402
    from utils.llm import llm_service_from_yaml_config  # noqa: E402

    llm = llm_service_from_yaml_config(model_name, cfg)
    agent = Agent(llm)
    # 실제로 적용된 샘플링을 기록한다. config 머지가 shallow update라 진화 config의 llm 블록이
    # base.yaml의 llm.sampling을 통째로 덮어쓸 수 있어서, YAML만 봐선 실제 T를 알 수 없다.
    cfg["_eff_sampling"] = {"temperature": llm.temperature, "top_p": llm.top_p, "top_k": llm.top_k}
    logging.info("모델 로드 완료: %s (실효 T=%s top_p=%s top_k=%s)",
                 model_name, llm.temperature, llm.top_p, llm.top_k)

    raw_path = Path(a.raw_out) if a.raw_out else Path(str(ROOT / a.out).replace(".md", ".raw.jsonl"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # passes[(arm, cid, pid)] = K회 중 통과 횟수
    passes: Dict[Tuple[str, str, str], int] = {
        (arm, p["id"], it["id"]): 0 for arm in arms for p in roster for it in batch
    }
    first_rep: Dict[Tuple[str, str, str], int] = {}
    elapsed: Dict[str, float] = {arm: 0.0 for arm in arms}

    # 이어 돌리기: 완결된 (arm, rep) 패스만 채택한다(부분 패스는 버리고 다시 생성).
    done: set = set()
    expect = len(batch) * len(experts)
    if a.resume_raw:
        seen: Dict[Tuple[str, int], List[dict]] = collections.defaultdict(list)
        with open(a.resume_raw, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                seen[(d["arm"], d["rep"])].append(d)
        for (arm, rep), recs in sorted(seen.items()):
            if arm not in arms or rep >= a.k:
                continue
            if len({(r["pid"], r["cid"]) for r in recs}) != expect:
                logging.warning("[%s rep %d] 부분 패스(%d/%d) — 버리고 재생성", arm, rep, len(recs), expect)
                continue
            for r in recs:
                key = (arm, r["cid"], r["pid"])
                if key not in passes:
                    continue
                passes[key] += int(r["pass"])
                if rep == 0:
                    first_rep[key] = int(r["pass"])
            done.add((arm, rep))
        logging.info("이어 돌리기: 완결 패스 %d개 재사용 %s", len(done), sorted(done))

    same_file = a.resume_raw and Path(a.resume_raw).resolve() == raw_path.resolve()
    raw_f = raw_path.open("a" if same_file else "w", encoding="utf-8")
    if a.resume_raw and not same_file:
        with open(a.resume_raw, encoding="utf-8") as f:
            for line in f:
                if (json.loads(line)["arm"], json.loads(line)["rep"]) in done:
                    raw_f.write(line)
        raw_f.flush()

    for arm in arms:
        for rep in range(a.k):
            if (arm, rep) in done:
                logging.info("[%s rep %d/%d] 재사용 — 건너뜀", arm, rep + 1, a.k)
                continue
            t0, gen_s, n_pass, n_done = time.time(), 0.0, 0, 0
            # 문제를 청크로 흘려보낸다. 전수(87k문제 × 25명 = 2.18M 프롬프트)를 한 리스트에
            # 쌓으면 생성 전에 메모리가 터진다 — 이 스크립트는 원래 600문제용이었다.
            step = a.gen_chunk if a.gen_chunk > 0 else len(batch)
            for s0 in range(0, len(batch), step):
                sub = batch[s0:s0 + step]
                g0 = time.time()
                msgs, order = [], []
                for it in sub:
                    for p in roster:
                        msgs.append(make_msgs(it, p, arm=arm, model_name=model_name,
                                              dataset_name=a.dataset, fewshot=fewshot))
                        order.append((it["id"], p["id"]))
                outs = agent.chat_batch(msgs, enable_thinking=enable_thinking)
                by_id = {b["id"]: b for b in sub}
                codes: Dict[Tuple[str, str], str] = {}
                for (pid, cid), raw in zip(order, outs):
                    item = by_id[pid]
                    codes[(pid, cid)] = finalize_generation_output(
                        raw, dataset=item.get("dataset") or a.dataset, domain=item.get("domain"))
                gen_s += time.time() - g0

                scores = score_chunk(sub, codes, cfg=cfg, workers=a.score_workers)
                for (pid, cid), sc in scores.items():
                    ok = 1 if pass_at_threshold(sc) else 0
                    passes[(arm, cid, pid)] += ok
                    if rep == 0:
                        first_rep[(arm, cid, pid)] = ok
                    n_pass += ok
                    raw_f.write(json.dumps({"arm": arm, "rep": rep, "pid": pid, "cid": cid,
                                            "score": sc, "pass": ok, "code": codes[(pid, cid)]},
                                           ensure_ascii=False) + "\n")
                raw_f.flush()
                n_done += len(scores)
                del msgs, order, outs, codes, scores
                if step < len(batch):
                    logging.info("[%s rep %d/%d] 문제 %d/%d · 생성 %.0f분 · pass %.1f%%",
                                 arm, rep + 1, a.k, min(s0 + step, len(batch)), len(batch),
                                 gen_s / 60, 100 * n_pass / max(1, n_done))
            elapsed[arm] += time.time() - t0
            logging.info("[%s rep %d/%d] 생성 %.0fs · 채점 %.0fs · pass %d/%d (%.1f%%)",
                         arm, rep + 1, a.k, gen_s, time.time() - t0 - gen_s,
                         n_pass, n_done, 100 * n_pass / max(1, n_done))
            # 중단돼도 결과가 남도록 rep마다 중간 저장(현재 arm만, 완료 rep = rep+1).
            # rep을 순서대로 도니 rep 미만은 전부 채워져 있다(재사용분 포함).
            _aggregate_and_write(arms=[arm], experts=experts, batch=batch, passes=passes,
                                 first_rep=first_rep, a=a, cfg=cfg, elapsed=elapsed, k_done=rep + 1)
def _aggregate_and_write(
    *, arms: List[str], experts: List[str], batch: List[Dict[str, Any]],
    passes: Dict[Tuple[str, str, str], int], first_rep: Dict[Tuple[str, str, str], int],
    a: argparse.Namespace, cfg: dict, elapsed: Dict[str, float], k_done: int,
) -> None:
    """지금까지 채운 패스만으로 집계·렌더·저장. rep마다 호출해 중단돼도 결과가 남게 한다."""
    prior: Dict[str, str] = {}
    if a.prior_solver:
        d = json.loads(Path(a.prior_solver).read_text(encoding="utf-8"))
        prior = d.get("prior_exclusive_solver", d) if isinstance(d, dict) else {}

    arm_stats, first_rep_war, repro = {}, {}, {}
    for arm in arms:
        P = [[passes[(arm, cid, it["id"])] / k_done for it in batch] for cid in experts]
        arm_stats[arm] = analyze(P, experts, k_done)
        repro[arm] = reproducibility(P, experts, batch, prior, k_done) if prior else None
        # single-draw(rep 0) = 현행 진화가 실제로 쓰는 라벨
        solved = [set(it["id"] for it in batch if first_rep.get((arm, cid, it["id"]), 0)) for cid in experts]
        union = set().union(*solved) if solved else set()
        w: Dict[str, Any] = {}
        for j, cid in enumerate(experts):
            others = set().union(*[solved[k] for k in range(len(experts)) if k != j])
            w[cid] = len(union) - len(others)
        w["_union"] = len(union)
        first_rep_war[arm] = w

    # 다음 단계(갈리는 문제만 K회)로 바로 넘길 수 있게 n_solved 구간별 id를 떨궈둔다.
    ids_path = Path(str(ROOT / a.out).replace(".md", ".problem_ids.json"))
    E = len(experts)
    buckets: Dict[str, List[str]] = {"contested": [], "all_solved": [], "none_solved": []}
    for i, it in enumerate(batch):
        n = sum(1 for cid in experts if passes[(arms[0], cid, it["id"])] / k_done > 0.5)
        key = "none_solved" if n == 0 else ("all_solved" if n == E else "contested")
        buckets[key].append(it["id"])
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps(buckets, ensure_ascii=False, indent=1), encoding="utf-8")
    logging.info("문제 id 구간 저장 → %s (contested=%d, all=%d, none=%d)", ids_path,
                 len(buckets["contested"]), len(buckets["all_solved"]), len(buckets["none_solved"]))

    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = render(arm_stats, first_rep_war, experts, a, cfg, elapsed, repro, k_done)
    out_path.write_text(txt, encoding="utf-8")
    print(txt)
    logging.info("saved -> %s", out_path)


if __name__ == "__main__":
    main()
