#!/usr/bin/env python3
"""조건부 개입(critic)이 solvability를 움직이는가 — 실패 셀 수정 확률 측정.

배경: 이 도메인에서 상수 개입(정체성 페르소나, 고정 절차)은 정오답을 못 바꿨다
(docs/REPORT_persona_axis_diagnosis_20260820.md). 남은 후보는 **시도 결과에 조건부인 개입**이다.
critic은 이미 존재하는 산출물 위의 탐지 문제라, 실패가 코드에 드러나 있다는 점에서 상수 개입과
성질이 다르다. 그게 실제로 정오답을 뒤집는지를 진화에 손대기 전에 먼저 잰다.

세 arm (같은 실패 셀, 같은 모델·온도, 수정 단계 프롬프트 동일):
  redraw         : 원래 프롬프트로 그냥 한 번 더 생성.  ← **필수 대조군**
                   재시행만으로도 결과가 자주 뒤집히므로(MIXED 54.5%), 이걸 넘지 못하는
                   수정률은 critic 효과가 아니다.
  critic_luca    : LUCA(일반 프롬프트)가 비평 → 그 지적을 받아 수정. (2회 생성)
  critic_persona : 로스터의 **다른** expert(무작위)가 자기 페르소나로 비평 → 수정. (2회 생성)

arm 2와 3의 차이는 **비평자가 누구냐**뿐이다. 진화된 페르소나가 solver로서는 값이 없었는데
(정오답·실패유형 모두 평평) **critic으로서는 값이 있는가**를 직접 잰다. 자기 코드를 자기가
비평하면 맹점을 공유하므로 타인 비평으로 간다.

⚠️ 세 arm 모두 **실행 결과(테스트 통과 여부·stderr)를 프롬프트에 넣지 않는다.** 실행 피드백을
넣으면 코딩 전용 기법이 되고, 재는 대상이 "critic이 결함을 읽어내는가"에서 "채점기를 봤는가"로
바뀐다. 채점은 사후에만 한다.

Usage:
  python scripts/critic_repair_probe.py --dry_run          # 프롬프트만 확인(모델 로드 없음)
  python scripts/critic_repair_probe.py --n_cells 500
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from data.loader import get_dataset  # noqa: E402
from evaluation.scorer import pass_at_threshold  # noqa: E402
from orchestrator import _score_pair  # noqa: E402
from prompts.coding import build_expert_prompt  # noqa: E402
from utils.helpers import finalize_generation_output  # noqa: E402

ARMS = ("redraw", "critic_luca", "critic_persona")

CRITIC_USER = """Problem:
{instruction}

Candidate solution (known to be incorrect):
```python
{code}
```

Identify what is actually wrong with this solution. Be concrete and specific: point at the exact
step, condition, or case that fails. Do not rewrite the code. At most 5 short bullet points."""

CRITIC_REPAIR_USER = """Your previous solution to this problem was incorrect.

Problem:
{instruction}

Your previous solution:
```python
{code}
```

A reviewer examined it and reported:
{critique}

Write a corrected complete solution."""


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/acc_eval_a4b_v2_sampled.yaml")
    ap.add_argument("--binning", default="results/acc/seed20211004/binning_test_full.jsonl")
    ap.add_argument("--cells", default="results/acc/seed20211004/binning_test_full.failure_modes.cells.json")
    ap.add_argument("--roster", default="results/acc/seed20211004/roster_final.json")
    ap.add_argument("--dataset", default="acc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="export/acc_v2")
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default="results/acc/seed20211004/critic_repair_probe.md")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    cells = json.load(open(ROOT / a.cells, encoding="utf-8"))
    gens = {str(json.loads(l)["id"]): json.loads(l)["expert_outputs"]
            for l in open(ROOT / a.binning, encoding="utf-8") if l.strip()}
    items = {str(it["id"]): it for it in get_dataset(a.dataset, split=a.split, local_dir=a.data_dir)}
    roster = {p["id"]: p for p in json.load(open(ROOT / a.roster, encoding="utf-8"))}

    # 실패 셀만 (accepted 제외), status 비율을 유지해 표본 추출
    failing = []
    for key, v in cells.items():
        pid, eid = key.split("\t")
        if v["status"] == "accepted" or pid not in items or pid not in gens:
            continue
        if eid not in gens[pid] or eid not in roster:
            continue
        failing.append((pid, eid, v["status"]))
    by_status = collections.defaultdict(list)
    for x in failing:
        by_status[x[2]].append(x)
    rng = random.Random(a.seed)
    sample = []
    for st, lst in by_status.items():
        k = max(1, round(a.n_cells * len(lst) / len(failing)))
        rng.shuffle(lst)
        sample += lst[:k]
    rng.shuffle(sample)
    sample = sample[:a.n_cells]
    logging.info("실패 셀 %d개 중 %d개 표본 (status 비율 유지: %s)",
                 len(failing), len(sample),
                 dict(collections.Counter(s for _, _, s in sample)))

    def orig_msgs(pid, eid):
        it = items[pid]
        return build_expert_prompt(
            it.get("instruction", ""), roster[eid].get("system_prompt", ""),
            dataset=it.get("dataset") or a.dataset, model_name=cfg_model,
            starter_code=it.get("starter_code"), approach=None, domain=it.get("domain"))

    cfg = load_cfg(a.config)
    cfg_model = cfg.get("model")
    enable_thinking = bool(cfg.get("enable_thinking", False))

    if a.dry_run:
        pid, eid, st = sample[0]
        code = gens[pid][eid]
        others = [x for x in roster if x != eid and x != "luca"]
        print("=== redraw ===\n", json.dumps(orig_msgs(pid, eid), ensure_ascii=False)[:800])
        print("\n=== critic_luca system ===\n", roster["luca"].get("system_prompt"))
        print("\n=== critic_persona system (예: %s) ===\n" % others[0],
              roster[others[0]].get("system_prompt"))
        print("\n=== critic user (arm 공통) ===\n", CRITIC_USER.format(
            instruction=items[pid].get("instruction", "")[:600], code=code[:600]))
        print("\n=== 수정 단계 user (arm 공통) ===\n", CRITIC_REPAIR_USER.format(
            instruction=items[pid].get("instruction", "")[:300], code=code[:300],
            critique="<비평자 출력>"))
        return

    from agents.base import Agent  # noqa: E402
    from utils.llm import llm_service_from_yaml_config  # noqa: E402

    llm = llm_service_from_yaml_config(cfg_model, cfg)
    agent = Agent(llm)
    logging.info("모델 로드 완료: %s (실효 T=%s top_p=%s top_k=%s)",
                 cfg_model, llm.temperature, llm.top_p, llm.top_k)

    sys_of = lambda eid: roster[eid].get("system_prompt", "You are an expert coder.")
    codes: dict = {}

    # --- arm 1: redraw (원래 프롬프트 그대로)
    msgs = [orig_msgs(pid, eid) for pid, eid, _ in sample]
    outs = agent.chat_batch(msgs, enable_thinking=enable_thinking)
    for (pid, eid, _), raw in zip(sample, outs):
        codes[("redraw", pid, eid)] = finalize_generation_output(
            raw, dataset=items[pid].get("dataset") or a.dataset, domain=items[pid].get("domain"))

    # --- arm 2·3: 비평자만 다르고 수정 단계는 동일
    #   critic_luca    = LUCA의 일반 프롬프트
    #   critic_persona = 로스터의 **다른** expert(셀마다 고정 시드로 무작위)
    crit_rng = random.Random(a.seed + 1)
    other = {}
    for pid, eid, _ in sample:
        # 자기 자신 제외(맹점 공유) + luca 제외(그건 arm 2 자체다)
        cands = [x for x in roster if x != eid and x != "luca"]
        other[(pid, eid)] = crit_rng.choice(cands) if cands else eid
    critiques: dict = {}
    for arm in ("critic_luca", "critic_persona"):
        msgs = [[{"role": "system",
                  "content": sys_of("luca") if arm == "critic_luca" else sys_of(other[(pid, eid)])},
                 {"role": "user", "content": CRITIC_USER.format(
                     instruction=items[pid].get("instruction", ""), code=gens[pid][eid])}]
                for pid, eid, _ in sample]
        outs = agent.chat_batch(msgs, enable_thinking=enable_thinking)
        for (pid, eid, _), raw in zip(sample, outs):
            critiques[(arm, pid, eid)] = raw or ""
        # 수정 단계: system은 원래 expert 자신, user 템플릿도 arm 간 동일
        msgs = [[{"role": "system", "content": sys_of(eid)},
                 {"role": "user", "content": CRITIC_REPAIR_USER.format(
                     instruction=items[pid].get("instruction", ""), code=gens[pid][eid],
                     critique=critiques[(arm, pid, eid)].strip()[:3000])}]
                for pid, eid, _ in sample]
        outs = agent.chat_batch(msgs, enable_thinking=enable_thinking)
        for (pid, eid, _), raw in zip(sample, outs):
            codes[(arm, pid, eid)] = finalize_generation_output(
                raw, dataset=items[pid].get("dataset") or a.dataset, domain=items[pid].get("domain"))

    # --- 채점 (실행은 여기서만)
    args = [(items[pid], codes[(arm, pid, eid)], 10, 3.0, "release_v5", (arm, pid, eid))
            for arm in ARMS for pid, eid, _ in sample]
    scores: dict = {}
    pool = ProcessPoolExecutor(max_workers=2)
    futs = {pool.submit(_score_pair, x): x[5] for x in args}
    try:
        for f in as_completed(futs, timeout=max(1800.0, 2.0 * len(args))):
            _pid, key, sc = f.result()
            scores[key] = sc
    except FuturesTimeoutError:
        logging.error("채점 wall-clock timeout: %d 미완료 -> 0.0",
                      sum(1 for f in futs if not f.done()))
    finally:
        for p in list(getattr(pool, "_processes", {}).values()):
            if p.is_alive():
                p.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    for key in futs.values():
        scores.setdefault(key, 0.0)

    fixed = {arm: [pass_at_threshold(scores[(arm, pid, eid)]) for pid, eid, _ in sample]
             for arm in ARMS}
    n = len(sample)

    def rate(arm):
        return 100.0 * sum(fixed[arm]) / n

    # McNemar (redraw 대비 짝지은 비교)
    def mcnemar(a1, a2):
        b = sum(1 for x, y in zip(fixed[a1], fixed[a2]) if x and not y)
        c = sum(1 for x, y in zip(fixed[a1], fixed[a2]) if y and not x)
        if b + c == 0:
            return b, c, 1.0
        # 정규근사 (연속성 보정)
        z = (abs(b - c) - 1) / ((b + c) ** 0.5)
        from math import erfc
        return b, c, float(erfc(abs(z) / (2 ** 0.5)))

    L = [f"# 조건부 개입(critic)이 실패를 뒤집는가 — `{a.binning}`", "",
         f"- 실패 셀 {len(failing):,}개 중 {n}개 표본 · 모델 {cfg_model} · "
         f"실효 T={llm.temperature}/top_p={llm.top_p}/top_k={llm.top_k}",
         "- 세 arm 모두 실행 결과(테스트 통과 여부·stderr)를 프롬프트에 넣지 않는다.",
         "- 비평자만 다르고 수정 단계 프롬프트는 arm 간 동일(system = 원래 expert).", "",
         "| arm | 수정률 | vs redraw (McNemar) |", "|---|---:|---|",
         f"| redraw (재시행만) | **{rate('redraw'):.1f}%** | — |"]
    for arm in ("critic_luca", "critic_persona"):
        b, c, p = mcnemar("redraw", arm)
        L.append(f"| {arm} | **{rate(arm):.1f}%** | redraw만 {b} / {arm}만 {c} · p = {p:.4f} |")
    b, c, p = mcnemar("critic_luca", "critic_persona")
    L += ["", f"**LUCA vs 페르소나 비평자**: LUCA만 고침 {b} / 페르소나만 고침 {c} · "
          f"McNemar p = **{p:.4f}**"]
    L += ["", "status별 수정률:", "", "| status | n | " + " | ".join(ARMS) + " |",
          "|---|---:|" + "---:|" * len(ARMS)]
    for st in sorted({s for _, _, s in sample}):
        idx = [i for i, (_, _, s) in enumerate(sample) if s == st]
        L.append(f"| {st} | {len(idx)} | " +
                 " | ".join(f"{100*sum(fixed[arm][i] for i in idx)/len(idx):.1f}%" for arm in ARMS) + " |")
    L += ["", "판정: critic이 redraw를 유의하게 넘지 못하면, 조건부 개입도 이 백본에서",
          "solvability를 움직이지 못한다 — 진화의 대상으로 올릴 자격이 없다.", ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    raw = str(out).replace(".md", ".raw.jsonl")
    with open(raw, "w", encoding="utf-8") as f:
        for i, (pid, eid, st) in enumerate(sample):
            f.write(json.dumps({"id": pid, "expert": eid, "status": st,
                                "critic_persona_id": other[(pid, eid)],
                                **{f"{arm}_critique": critiques[(arm, pid, eid)][:4000]
                                   for arm in ("critic_luca", "critic_persona")},
                                **{f"{arm}_fixed": bool(fixed[arm][i]) for arm in ARMS},
                                **{f"{arm}_code": codes[(arm, pid, eid)] for arm in ARMS}},
                               ensure_ascii=False) + "\n")
    print("\n".join(L))
    print(f"saved -> {out}\nraw -> {raw}")


if __name__ == "__main__":
    main()
