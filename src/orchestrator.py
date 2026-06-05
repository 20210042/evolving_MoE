"""GM evolution orchestrator (batch training + roster updates)."""

from __future__ import annotations

import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from action_selector import ActionDecision, ActionGateConfig, select_action
from agents.base import Agent
from evaluation.scorer import pass_at_threshold, score_one
from prompts.coding import build_expert_prompt
from roster import assign_candidate_id, ensure_roster, normalize_persona_fields, save_roster
from scout import scout_new_persona
from step_logger import StepLogContext, StepLogger
from utils.helpers import extract_code_block
from war import compute_war_scores, pick_worst_agent

_SCORE_WORKERS = min(4, os.cpu_count() or 1)


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
    ):
        self.agent = agent
        self.roster_path = roster_path
        self.roster = ensure_roster(roster_path)
        self.max_lives = max_lives
        for p in self.roster:
            p.setdefault("total_war", 0)
            p.setdefault("active_steps", 0)
            p.setdefault("average_war", 0.0)
            p.setdefault("lives", self.max_lives)
            p.setdefault("routing_history", [])
        self.action_cfg = action_cfg or ActionGateConfig()
        self.max_refine_iters = max_refine_iters
        self.lcb_timeout = lcb_timeout
        self.lcb_release_version = lcb_release_version
        self.code_exec_timeout = code_exec_timeout
        self.war_tiebreak = war_tiebreak
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

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_pair, a): (a[0]["id"], a[5]) for a in score_args}
            for fut in as_completed(futures):
                pid, cid, sc = fut.result()
                results[(pid, cid)] = sc
        return results

    def run_batch(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
        """One-step raw generation per (problem × roster member); parallel scoring."""
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
                    )
                )
                pair_order.append((pid, cid))

        gen_out = self.agent.chat_batch(gen_msgs, enable_thinking=True)
        by_id = {b["id"]: b for b in batch_data}
        codes: Dict[Tuple[str, str], str] = {}
        for pair, raw in zip(pair_order, gen_out):
            pid, cid = pair
            item = by_id[pid]
            if item.get("domain") == "math":
                codes[pair] = raw
            else:
                codes[pair] = extract_code_block(raw) or raw

        scores = self._score_pairs_parallel(batch_data, codes)

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
                if item.get("domain") == "math":
                    hard_errors_texts[problem_id] = f"Problem: {clean_desc}\n"
                else:
                    tests_str = "\n".join(item.get("test_list", []))
                    hard_errors_texts[problem_id] = (
                        f"Problem: {clean_desc}\n"
                        f"Tests:\n{tests_str}\n"
                    )

        return squad_results, hard_errors_texts

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
        )
        raw = self.agent.chat(msg, enable_thinking=True)
        domain = item.get("domain", "coding")
        return raw if domain == "math" else (extract_code_block(raw) or raw)

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

        squad_results, hard_errors_texts = self.run_batch(batch_data)
        war_scores, _ub_count, ub_rate = compute_war_scores(
            squad_results,
            len(batch_data),
            tiebreak=self.war_tiebreak,
            rng=rng,
        )
        logging.info("WAR Scores: %s", war_scores)

        all_zero_war = war_scores and all(v == 0 for v in war_scores.values())
        if all_zero_war:
            logging.info("All-zero WAR batch — skipping lives penalty (collective failure).")

        for p in self.roster:
            p_id = p["id"]
            if p_id in war_scores:
                current_score = war_scores[p_id]
                p["total_war"] = p.get("total_war", 0) + current_score
                p["active_steps"] = p.get("active_steps", 0) + 1
                p["average_war"] = p["total_war"] / p["active_steps"]
                if current_score > 0:
                    p["lives"] = self.max_lives
                elif all_zero_war:
                    pass  # batch-level failure — no individual penalty
                else:
                    p["lives"] = max(0, p.get("lives", self.max_lives) - 1)

        save_roster(self.roster_path, self.roster)

        worst_agent = pick_worst_agent(
            war_scores,
            self.roster,
            tiebreak=self.war_tiebreak,
            rng=rng,
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

        hard_errors_combined = "\n\n---\n\n".join(hard_errors_texts.values())
        new_persona = scout_new_persona(self.agent, self.roster, hard_errors_combined, dataset_name=self.dataset_name)
        if not new_persona or "system_prompt" not in new_persona:
            logging.warning("Failed to scout new persona. Skipping.")
            return

        logging.info("Scouted New Persona: %s", new_persona.get("persona_name"))

        roster_ids = [p["id"] for p in self.roster]
        probe_hard = list(hard_errors_texts.keys())
        by_id = {b["id"]: b for b in batch_data}
        new_pass_ids: Set[str] = set()

        probe_items = [by_id[q] for q in probe_hard if q in by_id]
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
                    )
                )
            probe_out = self.agent.chat_batch(probe_msgs, enable_thinking=True)
            for item, raw in zip(probe_items, probe_out):
                if item.get("domain") == "math":
                    code = raw
                else:
                    code = extract_code_block(raw) or raw
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
        )

        logging.info(
            "Action gate: %s | U=%s | MCL(worst)≈%.3f | new_pass=%s/%s",
            decision.action,
            decision.utility,
            decision.mcl_worst,
            len(new_pass_ids),
            len(probe_hard),
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
            "worst_eject_candidate": worst_agent,
            "hard_error_n": len(hard_errors_texts),
            "probe": {"hard_n": len(probe_hard), "stab_n": 0},
            "new_pass_on_union": sorted(new_pass_ids),
            "utility": decision.utility,
            "marginal_hard_gain_add": decision.marginal_hard_gain_add,
            "marginal_hard_gain_swap_extra": 0.0,
            "mcl_worst_est": decision.mcl_worst,
            "decision": decision.action,
            "roster_after": [p.get("id") for p in self.roster],
        }
        if decision.action in ("add", "swap") and self.roster:
            record["added_id"] = self.roster[-1]["id"]
        if new_persona:
            record["candidate_persona"] = new_persona.get("persona_name")

        self.step_logger.append(ctx, record)
        self.step_logger.save_roster_snapshot(self.run_id, self._step_for_log, list(self.roster))
