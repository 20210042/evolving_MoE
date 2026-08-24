"""GM evolution orchestrator (batch training + roster updates)."""

from __future__ import annotations

import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set, Tuple

from action_selector import ActionDecision, ActionGateConfig, select_action
from agents.base import Agent
from evaluation.scorer import (
    pass_at_threshold,
    score_acc_item_partial,
    score_one,
    score_sni_item_partial,
)
from prompts.coding import build_expert_prompt
from roster import assign_candidate_id, ensure_roster, normalize_persona_fields, save_roster
from scout import scout_new_persona
from step_logger import StepLogContext, StepLogger
from utils.domains import task_family
from utils.helpers import finalize_generation_output
from war import compute_war_scores, pick_worst_agent

_SCORE_WORKERS = min(4, os.cpu_count() or 1)
# Wall-clock guard for parallel scoring. A single pathological math_verify comparison
# can hang a worker past the library's own timeout; without this the run freezes
# forever (as_completed waits on the dead future). Budget per pair + a floor.
_SCORE_PER_PAIR_TIMEOUT = float(os.environ.get("SCORE_PER_PAIR_TIMEOUT", "5.0"))
_SCORE_MIN_TIMEOUT = float(os.environ.get("SCORE_MIN_TIMEOUT", "180.0"))


def _persona_label(persona: Dict[str, Any]) -> str:
    """스카우트 출력의 이름 키는 프롬프트마다 다르다(prompt_name / persona_name / name)."""
    for k in ("name", "prompt_name", "persona_name"):
        v = persona.get(k)
        if v:
            return str(v)
    return "(unnamed)"


def _score_pair(
    args: Tuple[Dict[str, Any], str, int, float, str, str],
) -> Tuple[str, str, float]:
    """Top-level scorer for ProcessPoolExecutor (picklable)."""
    item, code, lcb_timeout, code_timeout, lcb_release_version, cid = args
    pid = item["id"]
    sc = score_one(
        item,
        code,
        lcb_timeout=lcb_timeout,
        code_timeout=code_timeout,
        lcb_release_version=lcb_release_version,
    )
    return pid, cid, sc


def _score_pair_partial(args: Tuple[Dict[str, Any], str, str]) -> Tuple[str, str, float]:
    """Top-level 부분점수 스코어러(ProcessPoolExecutor용). war_mode='soft_partial' 전용 —
    squad_results(0/1)는 기존 _score_pair 결과로 그대로 만들고, 이건 WAR 계산에만 쓴다."""
    item, code, cid = args
    pid = item["id"]
    # dataset/domain 어느 쪽으로 들어와도 같게 판정한다(원본 jsonl엔 dataset·scoring_kind가
    # 없고 domain만 있다 — 붙이는 건 loader.annotate_items다).
    if task_family(dataset=item.get("dataset"), domain=item.get("domain")) == "sni":
        # SNI엔 테스트케이스가 없다 — 부분점수는 레퍼런스 최대 ROUGE-L이다.
        sc = score_sni_item_partial(item, code)
    else:
        sc = score_acc_item_partial(item, code)
    return pid, cid, sc


class GMEvolutionOrchestrator:
    def __init__(
        self,
        agent: Agent,
        roster_path: str,
        *,
        action_cfg: Optional[ActionGateConfig] = None,
        max_refine_iters: int = 4,  # deprecated: kept for API compat; unused in one-step evolution
        lcb_timeout: int = 10,
        lcb_release_version: str = "release_v5",
        code_exec_timeout: float = 3.0,
        war_tiebreak: str = "random",
        max_lives: int = 3,
        results_dir: str = "results",
        run_id: str = "default",
        dataset_name: str = "livecodebench",
        seed: int = 42,
        score_workers: int = _SCORE_WORKERS,
        enable_thinking: bool = True,
        use_exclusive_solves: bool = False,
        use_approach_persona: bool = False,
        shared_contribution_exemption: bool = True,
        failure_mode_scout: bool = False,
        deletion_window: int = 0,
        deletion_floor: float = 0.0,
        delete_cooldown: int = 0,
        add_only: bool = False,
        war_mode: str = "hard",
        lives_mode: str = "legacy",
    ):
        self.agent = agent
        self.roster_path = roster_path
        self.roster = ensure_roster(roster_path)
        self.enable_thinking = enable_thinking
        self.use_exclusive_solves = use_exclusive_solves
        self.use_approach_persona = use_approach_persona
        # When True (default = legacy): an agent that solved ANY problem (even shared)
        # keeps its life even with WAR=0. When False: only an exclusive contribution
        # (WAR>0) or an all-zero batch protects the life → redundant agents decay and
        # become eviction-eligible. Data: shared exemption alone froze eviction to 0
        # candidates across seed05–12; turning it off restores the seed04 regime.
        self.shared_contribution_exemption = shared_contribution_exemption
        # failure-mode scout: attach one random failed attempt to each hard error so
        # the scout types the recurring mistake instead of the topic (seed17+).
        self.failure_mode_scout = failure_mode_scout
        self._fm_rng = random.Random(seed)
        self.max_lives = max_lives
        # Windowed deletion (deletion_window>0): the whole removal path — eviction
        # eligibility (lives), worst ordering, and the delete trigger — keys off one
        # accumulated unique-solve rate over the last `deletion_window` batches,
        # instead of instantaneous single-batch WAR/mcl + the two lives exemptions.
        # 0 = OFF = legacy behavior, byte-identical. delete_cooldown rate-limits
        # deletes (≤1 per K steps) to damp the overshoot-top cascade.
        self.deletion_window = deletion_window
        self.deletion_floor = deletion_floor
        self.delete_cooldown = delete_cooldown
        self._last_delete_step = -(10**9)
        # Saturated run: never delete/swap → roster grows to the add-penalty fixed point.
        self.add_only = add_only
        for p in self.roster:
            p.setdefault("total_war", 0)
            p.setdefault("active_steps", 0)
            p.setdefault("average_war", 0.0)
            p.setdefault("lives", self.max_lives)
            p.setdefault("routing_history", [])
            p.setdefault("exclusive_solves", [])
            p.setdefault("recent_unique", [])
        self.action_cfg = action_cfg or ActionGateConfig()
        self.max_refine_iters = max_refine_iters
        self.lcb_timeout = lcb_timeout
        self.lcb_release_version = lcb_release_version
        self.code_exec_timeout = code_exec_timeout
        self.war_tiebreak = war_tiebreak
        # war_mode: "hard"(기존) | "soft_linear"((E-n)/(E-1) 배점) | "soft_partial"
        #   (soft_linear + 통과테스트비율 부분점수, war.py 참고).
        # lives_mode="rank": 목숨 게이트를 순위 기반으로 — 매 배치 최하위 1명 -1, 그 외 +1(상한
        #   max_lives), 신규 멤버는 deletion_window스텝 유예. 기존 게이트는 전부
        #   `current_score > 0`("단독해결 했나")에 걸려 있어 soft 점수에서는 항상 참이 되어
        #   아무도 안 죽는다. "legacy" = OFF = 기존 동작과 완전 동일.
        self.war_mode = war_mode
        self.lives_mode = lives_mode
        self.score_workers = score_workers
        self.step_logger = StepLogger(results_dir)
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.seed = seed
        self._step_for_log = 0
        self._epoch_for_log = 1
        self._batch_for_log = 1

    def set_log_coords(self, step: int, epoch: int, batch_idx: int) -> None:
        self._step_for_log = step
        self._epoch_for_log = epoch
        self._batch_for_log = batch_idx

    def _score(self, item: dict, code: str) -> float:
        return score_one(
            item,
            code,
            lcb_timeout=self.lcb_timeout,
            code_timeout=self.code_exec_timeout,
            lcb_release_version=self.lcb_release_version,
        )

    def _score_pairs_parallel(
        self,
        batch_data: List[Dict[str, Any]],
        codes: Dict[Tuple[str, str], str],
    ) -> Dict[Tuple[str, str], float]:
        """Score all (problem_id, critic_id) pairs in parallel."""
        pairs = list(codes.keys())
        if not pairs:
            return {}

        by_id = {b["id"]: b for b in batch_data}
        score_args = [
            (
                by_id[pid],
                codes[(pid, cid)],
                self.lcb_timeout,
                self.code_exec_timeout,
                self.lcb_release_version,
                cid,
            )
            for pid, cid in pairs
            if pid in by_id
        ]

        results: Dict[Tuple[str, str], float] = {}
        workers = max(1, min(self.score_workers, _SCORE_WORKERS))
        if workers <= 1 or len(score_args) <= 1:
            for args in score_args:
                pid, cid, sc = _score_pair(args)
                results[(pid, cid)] = sc
            return results

        # Hard wall-clock cap: if a worker hangs (e.g. math_verify comparison stuck in
        # sympy past its own timeout), kill it and default the unfinished pairs to 0.0
        # (== the math_verify timeout outcome) instead of freezing the whole run.
        timeout_s = max(_SCORE_MIN_TIMEOUT, _SCORE_PER_PAIR_TIMEOUT * len(score_args) / workers)
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(_score_pair, a): (a[0]["id"], a[5]) for a in score_args}
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                pid, cid, sc = fut.result()
                results[(pid, cid)] = sc
        except FuturesTimeoutError:
            unfinished = [futures[f] for f in futures if not f.done()]
            logging.error(
                "Scoring wall-clock timeout (%.0fs): %d/%d pairs unfinished -> scored 0.0 "
                "(likely hung math_verify comparison)",
                timeout_s, len(unfinished), len(futures),
            )
        finally:
            # never block on a hung worker: force-kill live processes, don't wait.
            for proc in list(getattr(pool, "_processes", {}).values()):
                if proc.is_alive():
                    proc.kill()
            pool.shutdown(wait=False, cancel_futures=True)
        for pair in futures.values():
            results.setdefault(pair, 0.0)
        return results

    def _score_pairs_parallel_partial(
        self,
        batch_data: List[Dict[str, Any]],
        codes: Dict[Tuple[str, str], str],
    ) -> Dict[Tuple[str, str], float]:
        """_score_pairs_parallel과 같은 워커 패턴이지만 score_acc_item_partial을 쓴다
        (war_mode='soft_partial' 전용, 같은 코드 문자열을 재실행만 한다 — 재생성 없음)."""
        pairs = list(codes.keys())
        if not pairs:
            return {}
        by_id = {b["id"]: b for b in batch_data}
        score_args = [
            (by_id[pid], codes[(pid, cid)], cid)
            for pid, cid in pairs
            if pid in by_id
        ]
        results: Dict[Tuple[str, str], float] = {}
        workers = max(1, min(self.score_workers, _SCORE_WORKERS))
        if workers <= 1 or len(score_args) <= 1:
            for args in score_args:
                pid, cid, sc = _score_pair_partial(args)
                results[(pid, cid)] = sc
            return results
        timeout_s = max(_SCORE_MIN_TIMEOUT, _SCORE_PER_PAIR_TIMEOUT * len(score_args) / workers)
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(_score_pair_partial, a): (a[0]["id"], a[2]) for a in score_args}
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                pid, cid, sc = fut.result()
                results[(pid, cid)] = sc
        except FuturesTimeoutError:
            unfinished = [futures[f] for f in futures if not f.done()]
            logging.error(
                "Partial-credit scoring wall-clock timeout (%.0fs): %d/%d pairs unfinished -> 0.0",
                timeout_s, len(unfinished), len(futures),
            )
        finally:
            for proc in list(getattr(pool, "_processes", {}).values()):
                if proc.is_alive():
                    proc.kill()
            pool.shutdown(wait=False, cancel_futures=True)
        for pair in futures.values():
            results.setdefault(pair, 0.0)
        return results

    def run_batch(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, Dict[str, float]] | None]:
        """One-step raw generation per (problem × roster member); parallel scoring.

        third return value: war_mode="soft_partial"일 때만 {agent_id: {problem_id: 0..1}}
        (score_acc_item_partial, 재생성 없이 같은 코드를 재채점). 그 외엔 None.
        """
        squad_results: Dict[str, Set[str]] = {p["id"]: set() for p in self.roster}
        hard_errors_texts: Dict[str, str] = {}

        llm_svc = self.agent.llm
        model_name = llm_svc.model_name

        gen_msgs = []
        pair_order: List[Tuple[str, str]] = []
        for item in batch_data:
            pid = item["id"]
            instruction = item.get("instruction", "")
            ds = item.get("dataset") or self.dataset_name
            dom = item.get("domain")
            starter = item.get("starter_code")
            for player in self.roster:
                cid = player["id"]
                sys_prompt = player.get("system_prompt", "You are an expert coder.")
                gen_msgs.append(
                    build_expert_prompt(
                        instruction,
                        sys_prompt,
                        dataset=ds,
                        model_name=model_name,
                        starter_code=starter,
                        approach=player.get("approach"),
                        domain=dom,
                        answer_line=item.get("answer_line"),
                        definition=item.get("definition"),
                        positive_examples=item.get("positive_examples"),
                        negative_examples=item.get("negative_examples"),
                    )
                )
                pair_order.append((pid, cid))

        gen_out = self.agent.chat_batch(gen_msgs, enable_thinking=self.enable_thinking)
        by_id = {b["id"]: b for b in batch_data}
        codes: Dict[Tuple[str, str], str] = {}
        for pair, raw in zip(pair_order, gen_out):
            pid, cid = pair
            item = by_id[pid]
            codes[pair] = finalize_generation_output(
                raw,
                dataset=item.get("dataset") or self.dataset_name,
                domain=item.get("domain"),
            )

        scores = self._score_pairs_parallel(batch_data, codes)

        partial_credit: Dict[str, Dict[str, float]] | None = None
        if self.war_mode == "soft_partial":
            partial_scores = self._score_pairs_parallel_partial(batch_data, codes)
            partial_credit = {p["id"]: {} for p in self.roster}
            for (pid, cid), sc in partial_scores.items():
                partial_credit.setdefault(cid, {})[pid] = sc / 100.0

        for item in batch_data:
            problem_id = item["id"]
            instruction = item.get("instruction", "")
            any_solved = False

            for player in self.roster:
                cid = player["id"]
                pair = (problem_id, cid)
                sc = scores.get(pair, 0.0)
                if pass_at_threshold(sc):
                    squad_results[cid].add(problem_id)
                    any_solved = True

            if not any_solved:
                clean_desc = item.get("prompt_text") or instruction
                family = task_family(dataset=item.get("dataset") or self.dataset_name, domain=item.get("domain"))
                if family == "math":
                    if self.failure_mode_scout:
                        attempts = [
                            codes[(problem_id, p["id"])]
                            for p in self.roster
                            if (problem_id, p["id"]) in codes
                        ]
                        if attempts:
                            clean_desc = (
                                f"{clean_desc}\n--- A failed attempt ---\n"
                                f"{self._fm_rng.choice(attempts)}"
                            )
                    hard_errors_texts[problem_id] = clean_desc
                elif family == "coding":
                    tests_str = "\n".join(item.get("test_list", []))
                    hard_errors_texts[problem_id] = (
                        f"{clean_desc}\n"
                        f"Tests:\n{tests_str}"
                    )
                elif family == "sni":
                    # SNI는 instruction이 Input 원문뿐이다(정의는 별도 필드). 그대로 주면
                    # 스카우트는 무슨 조작을 요구하는 태스크인지 알 방법이 없어 소재(주제)로만
                    # 분화한다 — 프로브에서 죽은 것으로 판정된 축이다. 정의·기대·실제를 같이 준다.
                    gts = item.get("ground_truth") or []
                    expected = str(gts[0]) if gts else ""
                    attempts = [
                        codes[(problem_id, p["id"])]
                        for p in self.roster
                        if (problem_id, p["id"]) in codes
                    ]
                    produced = self._fm_rng.choice(attempts) if attempts else ""
                    hard_errors_texts[problem_id] = (
                        f"Task: {(item.get('definition') or '').strip()[:600]}\n"
                        f"Input: {clean_desc[:600]}\n"
                        f"Expected: {expected[:300]}\n"
                        f"Produced: {produced[:300]}"
                    )
                else:
                    hard_errors_texts[problem_id] = clean_desc

        return squad_results, hard_errors_texts, partial_credit

    def _run_candidate_on_item(
        self,
        item: Dict[str, Any],
        new_persona: Dict[str, Any],
    ) -> str:
        new_sys = new_persona["system_prompt"]
        instruction = item.get("instruction", "")
        llm_svc = self.agent.llm
        model_name = llm_svc.model_name
        ds_probe = item.get("dataset") or self.dataset_name

        msg = build_expert_prompt(
            instruction,
            new_sys,
            dataset=ds_probe,
            model_name=model_name,
            starter_code=item.get("starter_code"),
            domain=item.get("domain"),
            answer_line=item.get("answer_line"),
            definition=item.get("definition"),
            positive_examples=item.get("positive_examples"),
            negative_examples=item.get("negative_examples"),
        )
        raw = self.agent.chat(msg, enable_thinking=self.enable_thinking)
        return finalize_generation_output(raw, dataset=ds_probe, domain=item.get("domain"))

    def _update_routing_memory(self, batch_data: List[Dict], squad_results: Dict[str, Set[str]]) -> None:
        for item in batch_data:
            pid = item["id"]
            solvers = [a for a, s in squad_results.items() if pid in s]
            if len(solvers) == 1:
                solver_id = solvers[0]
                for p in self.roster:
                    if p["id"] == solver_id:
                        p.setdefault("routing_history", [])
                        entry = {
                            "instruction": (
                                item.get("prompt_text") or item.get("instruction", "")
                            )[:500]
                        }
                        if entry not in p["routing_history"]:
                            p["routing_history"].append(entry)
                        p["routing_history"] = p["routing_history"][-10:]
                        break
        save_roster(self.roster_path, self.roster)

    def run_epoch(self, batch_data: List[Dict[str, Any]]) -> None:
        logging.info("Starting Epoch with %s problems.", len(batch_data))
        rng = random.Random(self.seed + self._step_for_log)

        squad_results, hard_errors_texts, partial_credit = self.run_batch(batch_data)
        war_scores, _ub_count, ub_rate = compute_war_scores(
            squad_results,
            len(batch_data),
            tiebreak=self.war_tiebreak,
            rng=rng,
            mode=self.war_mode,
            partial_credit=partial_credit,
        )
        logging.info("WAR Scores: %s", war_scores)

        all_zero_war = war_scores and all(v == 0 for v in war_scores.values())
        if all_zero_war:
            logging.info("All-zero WAR batch — skipping lives penalty (collective failure).")

        win = self.deletion_window
        batch_n = len(batch_data)
        unique_rate_map: Dict[str, float] = {}

        # lives_mode="rank": 이번 배치 단발 점수로 최하위 1명만 -1, 나머지 +1.
        # 2026-08-04 seed20211001(회복 없음)이 더 크게 진동해서(방향전환 37 vs 3회) 회복을
        # 다시 넣었더니(seed20211002) 이번엔 삭제가 0번 — 배치 하나(100문제)의 soft_partial
        # 점수는 재현성 7%(거의 무작위)라 노이즈로 매 배치 최하위가 계속 바뀌고, 회복이 그걸
        # 즉시 지웠다. 실제 로그로 재보니 16배치 누적 평균은 재현성 80%(오프라인 5,000문제
        # 사전검증 79%와 일치) — 신호는 있는데 배치 하나론 노이즈에 묻힌다.
        # → lives_mode="rank_windowed": 단발 대신 win배치 누적 평균으로 최하위를 정한다.
        #
        # ⚠️ 2026-08-13 seed20211003에서 발견/수정: 이 풀(pool)·unique_rate_map 값을 batch_n으로
        # 안 나눠서(원점수 절대값 2.6~6.4를 그대로) select_action에 mcl로 넘겼다.
        # action_selector.py:70 docstring이 "mcl = 배치 전체 대비 비율(fraction)"이라고 명시하는데,
        # lambda_del(~0.04)과 비교되는 값이라 mcl이 4~6이면 u_delete가 항상 -4 근방으로 나와
        # 삭제가 영원히 안 나온다(seed20211003: 축출후보 40스텝, 실제 delete 0). legacy 윈도
        # 경로의 `rate = sum(ru)/(len(ru)*batch_n)`과 같은 정규화를 여기도 적용한다.
        rank_loser: str | None = None
        if self.lives_mode in ("rank", "rank_windowed") and war_scores and not all_zero_war:
            grace = win if win > 0 else 0
            eligible = {
                p["id"] for p in self.roster
                if p.get("id") in war_scores and p.get("active_steps", 0) >= grace
            }
            if self.lives_mode == "rank_windowed":
                # 판단은 누적 평균으로 하되, 버퍼 업데이트는 아래 for-루프에서 한 번만 한다
                # (여기선 판정 전용으로 "만약 이번 점수까지 넣으면"을 미리 계산해 쓴다).
                pool = {}
                for p in self.roster:
                    pid = p["id"]
                    if pid not in eligible:
                        continue
                    ru = list(p.get("recent_unique", [])) + [war_scores[pid]]
                    ru = ru[-win:] if win > 0 else ru
                    pool[pid] = sum(ru) / (len(ru) * batch_n) if batch_n > 0 else 0.0
            else:
                pool = {a: v for a, v in war_scores.items() if a in eligible}
            pool = pool or war_scores
            worst_val = min(pool.values())
            tied = sorted(a for a, v in pool.items() if v == worst_val)
            rank_loser = tied[0] if len(tied) == 1 else tied[int(rng.random() * len(tied))]
            logging.info("Lives penalty (%s mode) -> %s (score %.4f, %d/%d eligible)",
                         self.lives_mode, rank_loser, worst_val, len(pool), len(war_scores))
        elif self.lives_mode in ("rank", "rank_windowed") and all_zero_war:
            logging.info("All-zero WAR batch — skipping lives penalty (collective failure), "
                         "%s mode.", self.lives_mode)

        for p in self.roster:
            p_id = p["id"]
            if p_id in war_scores:
                current_score = war_scores[p_id]
                p["total_war"] = p.get("total_war", 0) + current_score
                p["active_steps"] = p.get("active_steps", 0) + 1
                p["average_war"] = p["total_war"] / p["active_steps"]
                if self.lives_mode in ("rank", "rank_windowed"):
                    if self.lives_mode == "rank_windowed":
                        # 판정에 쓴 것과 같은 버퍼를 여기서 실제로 갱신(한 번만) — 다음 스텝의
                        # "판정 전 미리계산"이 이 갱신을 이어받는다. batch_n으로 정규화해
                        # select_action의 mcl(비율 계약)과 스케일을 맞춘다.
                        ru = p.setdefault("recent_unique", [])
                        ru.append(current_score)
                        if win > 0:
                            del ru[:-win]
                        unique_rate_map[p_id] = sum(ru) / (len(ru) * batch_n) if batch_n > 0 else 0.0
                    if all_zero_war:
                        pass  # 집단 실패 배치 — 개인 페널티/회복 없음(legacy와 동일한 면제)
                    elif p_id == rank_loser:
                        p["lives"] = max(0, p.get("lives", self.max_lives) - 1)
                    else:
                        p["lives"] = min(self.max_lives, p.get("lives", self.max_lives) + 1)
                elif win > 0:
                    # Window mode: accumulate the per-batch unique-solve count (== WAR
                    # score) over a sliding window and key lives off the sustained
                    # unique-rate, not one unlucky batch. Newborns get a full-window
                    # grace period; exemptions are unnecessary (a collective-fail or
                    # shared-only batch simply lowers everyone's rate equally).
                    ru = p.setdefault("recent_unique", [])
                    ru.append(current_score)
                    del ru[:-win]
                    rate = sum(ru) / (len(ru) * batch_n) if ru and batch_n > 0 else 0.0
                    unique_rate_map[p_id] = rate
                    if len(ru) < win:
                        pass  # grace: not enough history to judge redundancy
                    elif rate > self.deletion_floor:
                        p["lives"] = min(self.max_lives, p.get("lives", self.max_lives) + 1)
                    else:
                        p["lives"] = max(0, p.get("lives", self.max_lives) - 1)
                else:
                    agent_solved_any = bool(squad_results.get(p_id, set()))
                    if current_score > 0:
                        p["lives"] = self.max_lives
                    elif all_zero_war:
                        pass  # batch-level collective failure — no individual penalty
                    elif self.shared_contribution_exemption and agent_solved_any:
                        pass  # solved shared problems — not an individual failure (toggleable)
                    else:
                        p["lives"] = max(0, p.get("lives", self.max_lives) - 1)

        # Accumulate exclusive solve history per agent (problems only that agent solved).
        roster_by_id = {p["id"]: p for p in self.roster}
        for item in batch_data:
            pid = item["id"]
            solvers = [cid for cid, solved_set in squad_results.items() if pid in solved_set]
            if len(solvers) == 1:
                p = roster_by_id.get(solvers[0])
                if p is not None:
                    text = (item.get("prompt_text") or item.get("instruction", ""))[:200]
                    p.setdefault("exclusive_solves", [])
                    if text not in p["exclusive_solves"]:
                        p["exclusive_solves"].append(text)
                    p["exclusive_solves"] = p["exclusive_solves"][-10:]

        save_roster(self.roster_path, self.roster)

        worst_agent = pick_worst_agent(
            war_scores,
            self.roster,
            tiebreak=self.war_tiebreak,
            rng=rng,
            # unique_rate_map은 legacy+windowed(win>0)와 rank_windowed에서만 채워진다.
            # "rank"(단발)는 이 딕셔너리를 안 채우므로, 잘못 넘기면 pick_worst_agent가 전원
            # average_war=0.0 취급해 최종 축출 후보 정렬이 깨진다 — None을 넘겨 누적
            # average_war로 폴백시킨다.
            unique_rate_map=(
                unique_rate_map
                if (self.lives_mode == "rank_windowed" or (win > 0 and self.lives_mode == "legacy"))
                else None
            ),
        )
        logging.info("Worst Agent for Eviction: %s", worst_agent)
        logging.info("Total Hard Errors: %s", len(hard_errors_texts))

        noop_decision = ActionDecision(
            "noop",
            {"u_add": 0.0, "u_delete": 0.0},
            0.0,
            0.0,
        )

        if not hard_errors_texts:
            logging.info("No Hard Errors found. Roster stays the same.")
            self._log_step(
                squad_results,
                war_scores,
                worst_agent,
                hard_errors_texts,
                noop_decision,
                None,
                [],
                set(),
                ub_rate,
            )
            return

        self._update_routing_memory(batch_data, squad_results)

        hard_errors_combined = "\n".join(
            f"{i+1}. {txt}" for i, txt in enumerate(hard_errors_texts.values())
        )

        exclusive_solves_map: Optional[Dict[str, List[str]]] = None
        if self.use_exclusive_solves:
            exclusive_solves_map = {
                p.get("name", p.get("persona_name", p["id"])): p.get("exclusive_solves", [])
                for p in self.roster
            }

        new_persona = scout_new_persona(
            self.agent, self.roster, hard_errors_combined,
            dataset_name=self.dataset_name,
            enable_thinking=self.enable_thinking,
            exclusive_solves_map=exclusive_solves_map,
            use_approach_persona=self.use_approach_persona,
            domain=(batch_data[0].get("domain") if batch_data else None),
            failure_mode=self.failure_mode_scout,
        )
        if not new_persona or "system_prompt" not in new_persona:
            logging.warning("Failed to scout new persona. Skipping.")
            return

        logging.info("Scouted New Persona: %s", _persona_label(new_persona))

        roster_ids = [p["id"] for p in self.roster]
        probe_hard = list(hard_errors_texts.keys())
        by_id = {b["id"]: b for b in batch_data}
        new_pass_ids: Set[str] = set()

        # Hole-aware swap (prof feedback #3): also probe the newface on the worst
        # agent's NICHE (problems only the worst solved). If a swap evicts the worst,
        # that niche becomes a hole; select_action only allows the swap if the newface
        # recovers it. These ids are disjoint from probe_hard (worst solved them), so
        # adding them to the probe doesn't change gh_add (which intersects probe_hard).
        worst_unique_ids: List[str] = []
        if worst_agent:
            worst_solves = set(squad_results.get(worst_agent, set()))
            other_solves: Set[str] = set()
            for rid, solves in squad_results.items():
                if rid != worst_agent:
                    other_solves.update(solves)
            worst_unique_ids = [q for q in (worst_solves - other_solves) if q in by_id]

        probe_ids = probe_hard + [q for q in worst_unique_ids if q not in hard_errors_texts]
        probe_items = [by_id[q] for q in probe_ids if q in by_id]
        if probe_items:
            probe_msgs = []
            for item in probe_items:
                probe_msgs.append(
                    build_expert_prompt(
                        item.get("instruction", ""),
                        new_persona["system_prompt"],
                        dataset=item.get("dataset") or self.dataset_name,
                        model_name=self.agent.llm.model_name,
                        starter_code=item.get("starter_code"),
                        approach=new_persona.get("approach"),
                        domain=item.get("domain"),
                        answer_line=item.get("answer_line"),
                        definition=item.get("definition"),
                        positive_examples=item.get("positive_examples"),
                        negative_examples=item.get("negative_examples"),
                    )
                )
            probe_out = self.agent.chat_batch(probe_msgs, enable_thinking=self.enable_thinking)
            for item, raw in zip(probe_items, probe_out):
                code = finalize_generation_output(
                    raw,
                    dataset=item.get("dataset") or self.dataset_name,
                    domain=item.get("domain"),
                )
                if pass_at_threshold(self._score(item, code)):
                    new_pass_ids.add(item["id"])

        decision = select_action(
            roster_ids=roster_ids,
            worst_id=worst_agent,
            squad_results={k: set(v) for k, v in squad_results.items()},
            hard_errors=probe_hard,
            new_pass_ids=new_pass_ids,
            batch_size=len(batch_data),
            cfg=self.action_cfg,
            worst_unique_rate=(
                unique_rate_map.get(worst_agent) if (win > 0 and worst_agent) else None
            ),
            add_only=self.add_only,
        )

        # Down-stroke damping: rate-limit deletes (≤1 per delete_cooldown steps) so the
        # overshoot top doesn't cascade-collapse. Swaps keep size, so they're exempt.
        if (
            self.delete_cooldown > 0
            and decision.action == "delete"
            and (self._step_for_log - self._last_delete_step) < self.delete_cooldown
        ):
            logging.info(
                "Delete suppressed by cooldown (since last=%d < %d).",
                self._step_for_log - self._last_delete_step,
                self.delete_cooldown,
            )
            decision.action = "noop"

        logging.info(
            "Action gate: %s | U=%s | MCL(worst)≈%.3f | new_pass=%s/%s | niche_recover=%s/%s%s",
            decision.action,
            decision.utility,
            decision.mcl_worst,
            len(new_pass_ids),
            len(probe_hard),
            decision.recovered_count,
            decision.worst_unique_count,
            " | SWAP→ADD demote: niche not recovered" if decision.demoted_swap else "",
        )

        if decision.action == "noop":
            logging.info("Roster unchanged (action gate).")
        elif decision.action == "add":
            new_id = assign_candidate_id(self.roster)
            persona = normalize_persona_fields(new_persona, new_id)
            persona["lives"] = self.max_lives

            new_history = []
            for q in new_pass_ids:
                if q in by_id:
                    new_history.append(
                        {
                            "instruction": (
                                by_id[q].get("prompt_text") or by_id[q].get("instruction", "")
                            )[:500]
                        }
                    )
            persona["routing_history"] = new_history[-10:]

            self.roster.append(persona)
            save_roster(self.roster_path, self.roster)
            logging.info("Commit ADD: new persona %s", new_id)
        elif decision.action == "delete":
            self.roster = [r for r in self.roster if r["id"] != worst_agent]
            save_roster(self.roster_path, self.roster)
            self._last_delete_step = self._step_for_log
            logging.info("Commit DELETE: removed worst agent %s", worst_agent)
        elif decision.action == "swap":
            self.roster = [r for r in self.roster if r["id"] != worst_agent]
            new_id = assign_candidate_id(self.roster)
            persona = normalize_persona_fields(new_persona, new_id)
            persona["lives"] = self.max_lives

            new_history = []
            for q in new_pass_ids:
                if q in by_id:
                    new_history.append(
                        {
                            "instruction": (
                                by_id[q].get("prompt_text") or by_id[q].get("instruction", "")
                            )[:500]
                        }
                    )
            persona["routing_history"] = new_history[-10:]

            self.roster.append(persona)
            save_roster(self.roster_path, self.roster)
            logging.info("Commit SWAP: replaced %s with %s", worst_agent, new_id)

        self._log_step(
            squad_results,
            war_scores,
            worst_agent,
            hard_errors_texts,
            decision,
            new_persona,
            probe_hard,
            new_pass_ids,
            ub_rate,
        )

    def _log_step(
        self,
        squad_results: Dict[str, Set[str]],
        war_scores: Dict[str, int],
        worst_agent: str,
        hard_errors_texts: Dict[str, str],
        decision: ActionDecision,
        new_persona: Optional[Dict[str, Any]],
        probe_hard: List[str],
        new_pass_ids: Set[str],
        upper_bound_rate: float,
    ) -> None:
        ctx = StepLogContext(
            run_id=self.run_id,
            step=self._step_for_log,
            epoch=self._epoch_for_log,
            batch_idx=self._batch_for_log,
            dataset=self.dataset_name,
            seed=self.seed,
        )
        record = {
            "upper_bound_pct": upper_bound_rate,
            "war": war_scores,
            "squad_solves": {cid: sorted(pids) for cid, pids in squad_results.items()},
            "worst_eject_candidate": worst_agent,
            "hard_error_n": len(hard_errors_texts),
            "probe": {"hard_n": len(probe_hard), "stab_n": 0},
            "new_pass_on_union": sorted(new_pass_ids),
            "utility": decision.utility,
            "marginal_hard_gain_add": decision.marginal_hard_gain_add,
            "marginal_hard_gain_swap_extra": 0.0,
            "mcl_worst_est": decision.mcl_worst,
            "worst_unique_n": decision.worst_unique_count,
            "niche_recovered_n": decision.recovered_count,
            "swap_demoted_to_add": decision.demoted_swap,
            "decision": decision.action,
            "roster_after": [p.get("id") for p in self.roster],
        }
        if decision.action in ("add", "swap") and self.roster:
            record["added_id"] = self.roster[-1]["id"]
        if new_persona:
            record["candidate_persona"] = _persona_label(new_persona)

        self.step_logger.append(ctx, record)
        self.step_logger.save_roster_snapshot(self.run_id, self._step_for_log, list(self.roster))
