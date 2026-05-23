"""GM evolution orchestrator (batch training + roster updates)."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from meta_agent_evo.action_selector import ActionDecision, ActionGateConfig, select_action
from meta_agent_evo.agents.base import Agent
from meta_agent_evo.evaluation.scorer import pass_at_threshold, score_one
from meta_agent_evo.prompts.coding import build_baseline_prompt, build_critic_prompt, build_refine_prompt
from meta_agent_evo.roster import assign_candidate_id, ensure_roster, normalize_persona_fields, save_roster
from meta_agent_evo.scout import scout_new_persona
from meta_agent_evo.step_logger import StepLogContext, StepLogger
from meta_agent_evo.utils.helpers import check_stop_condition, extract_code_block
from meta_agent_evo.war import compute_war_scores, pick_worst_agent


class GMEvolutionOrchestrator:
    def __init__(
        self,
        agent: Agent,
        roster_path: str,
        *,
        action_cfg: Optional[ActionGateConfig] = None,
        max_refine_iters: int = 4,
        lcb_timeout: int = 10,
        lcb_release_version: str = "release_v5",
        code_exec_timeout: float = 3.0,
        war_tiebreak: str = "random",
        results_dir: str = "results",
        run_id: str = "default",
        dataset_name: str = "livecodebench",
        seed: int = 42,
    ):
        self.agent = agent
        self.roster_path = roster_path
        self.roster = ensure_roster(roster_path)
        for p in self.roster:
            p.setdefault("total_war", 0)
            p.setdefault("active_steps", 0)
            p.setdefault("average_war", 0.0)
            p.setdefault("routing_history", [])
        self.action_cfg = action_cfg or ActionGateConfig()
        self.max_refine_iters = max_refine_iters
        self.lcb_timeout = lcb_timeout
        self.lcb_release_version = lcb_release_version
        self.code_exec_timeout = code_exec_timeout
        self.war_tiebreak = war_tiebreak
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

    def run_batch(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
        squad_results: Dict[str, Set[str]] = {p["id"]: set() for p in self.roster}
        hard_errors_texts: Dict[str, str] = {}
        baselines: Dict[str, str] = {}

        llm_svc = self.agent.llm
        model_name = llm_svc.model_name

        baseline_msgs = [
            build_baseline_prompt(
                item.get("instruction", ""),
                dataset=item.get("dataset") or self.dataset_name,
                model_name=model_name,
                starter_code=item.get("starter_code"),
            )
            for item in batch_data
        ]
        baseline_raws = self.agent.chat_batch(baseline_msgs, temperature=0.0)
        for item, baseline_raw in zip(batch_data, baseline_raws):
            pid = item["id"]
            baselines[pid] = extract_code_block(baseline_raw) or baseline_raw

        pairs: List[Tuple[str, str]] = [
            (item["id"], player["id"]) for item in batch_data for player in self.roster
        ]
        codes: Dict[Tuple[str, str], str] = {(pid, cid): baselines[pid] for pid, cid in pairs}
        active_mask: Dict[Tuple[str, str], bool] = {(pid, cid): True for pid, cid in pairs}
        pid_to_instruction = {item["id"]: item.get("instruction", "") for item in batch_data}
        pid_to_ds = {item["id"]: (item.get("dataset") or self.dataset_name) for item in batch_data}

        sys_by_cid = {player["id"]: player.get("system_prompt", "You are an expert coder.") for player in self.roster}

        for _ in range(self.max_refine_iters):
            active_pairs = [pair for pair in pairs if active_mask[pair]]
            if not active_pairs:
                break

            critic_msgs = [
                build_critic_prompt(
                    pid_to_instruction[pid],
                    codes[(pid, cid)],
                    sys_by_cid[cid],
                    dataset=pid_to_ds[pid],
                    model_name=model_name,
                )
                for pid, cid in active_pairs
            ]

            critic_out = self.agent.chat_batch(critic_msgs)

            refinement_tasks: List[Tuple[Tuple[str, str], str]] = []
            for idx, pair in enumerate(active_pairs):
                feedback = critic_out[idx]
                if check_stop_condition(feedback):
                    active_mask[pair] = False
                    continue
                refinement_tasks.append((pair, feedback))

            if refinement_tasks:
                refine_msgs = [
                    build_refine_prompt(
                        pid_to_instruction[pair[0]],
                        fb,
                        codes[pair],
                        dataset=pid_to_ds[pair[0]],
                        model_name=model_name,
                    )
                    for pair, fb in refinement_tasks
                ]
                refined_out = self.agent.chat_batch(refine_msgs, temperature=0.0)

                for (pair, _fb), ref_raw in zip(refinement_tasks, refined_out):
                    codes[pair] = extract_code_block(ref_raw) or ref_raw

        for item in batch_data:
            problem_id = item["id"]
            instruction = item.get("instruction", "")

            any_solved = False
            failed_attempts: List[str] = []

            for player in self.roster:
                cid = player["id"]
                pair = (problem_id, cid)
                current_code = codes.get(pair, baselines[problem_id])
                sc = self._score(item, current_code)
                if pass_at_threshold(sc):
                    squad_results[cid].add(problem_id)
                    any_solved = True
                else:
                    failed_attempts.append(f"Critic {cid} Failure:\n{current_code}")

            if not any_solved:
                hard_errors_texts[problem_id] = (
                    f"### Problem ID: {problem_id}\n"
                    f"Instruction: {instruction[:1000]}\n"
                    f"Failed Code Sample:\n"
                    f"{failed_attempts[0] if failed_attempts else baselines.get(problem_id, '')}\n"
                )

        return squad_results, hard_errors_texts, baselines

    def _run_candidate_on_item(
        self,
        item: Dict[str, Any],
        baseline_code: str,
        new_persona: Dict[str, Any],
    ) -> str:
        new_sys = new_persona["system_prompt"]
        custom = new_persona.get(
            "custom_refine_prompt_template",
            "Fix the bugs. Output ONLY valid Python code inside a single ```python ... ``` block.",
        )
        instruction = item.get("instruction", "")
        current_code = baseline_code
        llm_svc = self.agent.llm
        model_name = llm_svc.model_name
        ds_probe = item.get("dataset") or self.dataset_name

        for _ in range(self.max_refine_iters):
            critic_user = (
                f"Review the following code.\n\n"
                f"Problem:\n{instruction}\n\n"
                f"Code:\n{current_code}\n\n"
                f"Focus: {custom}\n\n"
                "1. Identify any syntax errors or bugs.\n"
                "2. Check if it solves the problem correctly.\n"
                "3. Assess the efficiency.\n\n"
                "Output your review in the following format:\n"
                "Feedback: [Your detailed feedback]"
            )
            critic_msg = [
                {"role": "system", "content": new_sys},
                {"role": "user", "content": critic_user},
            ]
            feedback = self.agent.chat(critic_msg)
            if check_stop_condition(feedback):
                break

            ref_msg = build_refine_prompt(
                instruction,
                feedback,
                current_code,
                dataset=ds_probe,
                model_name=model_name,
            )
            ref_raw = self.agent.chat(ref_msg, temperature=0.0)
            current_code = extract_code_block(ref_raw) or ref_raw

        return current_code

    def _update_routing_memory(self, batch_data: List[Dict], squad_results: Dict[str, Set[str]]) -> None:
        for item in batch_data:
            pid = item["id"]
            solvers = [a for a, s in squad_results.items() if pid in s]
            if len(solvers) == 1:
                solver_id = solvers[0]
                for p in self.roster:
                    if p["id"] == solver_id:
                        p.setdefault("routing_history", [])
                        entry = {"instruction": item.get("instruction", "")[:500]}
                        if entry not in p["routing_history"]:
                            p["routing_history"].append(entry)
                        p["routing_history"] = p["routing_history"][-10:]
                        break
        save_roster(self.roster_path, self.roster)

    def run_epoch(self, batch_data: List[Dict[str, Any]]) -> None:
        logging.info("Starting Epoch with %s problems.", len(batch_data))
        rng = random.Random(self.seed + self._step_for_log)

        squad_results, hard_errors_texts, baselines = self.run_batch(batch_data)
        war_scores, _ub_count, ub_rate = compute_war_scores(
            squad_results,
            len(batch_data),
            tiebreak=self.war_tiebreak,
            rng=rng,
        )
        logging.info("WAR Scores: %s", war_scores)

        # Update cumulative WAR metrics for all agents in the current roster
        for p in self.roster:
            p_id = p["id"]
            if p_id in war_scores:
                current_score = war_scores[p_id]
                p["total_war"] = p.get("total_war", 0) + current_score
                p["active_steps"] = p.get("active_steps", 0) + 1
                p["average_war"] = p["total_war"] / p["active_steps"]

        # Save cumulative metrics updates
        save_roster(self.roster_path, self.roster)

        worst_agent = pick_worst_agent(war_scores, self.roster, tiebreak=self.war_tiebreak, rng=rng)
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
        new_persona = scout_new_persona(self.agent, self.roster, hard_errors_combined)
        if not new_persona or "system_prompt" not in new_persona:
            logging.warning("Failed to scout new persona. Skipping.")
            return

        logging.info("Scouted New Persona: %s", new_persona.get("persona_name"))

        roster_ids = [p["id"] for p in self.roster]
        probe_hard = list(hard_errors_texts.keys())
        by_id = {b["id"]: b for b in batch_data}
        new_pass_ids: Set[str] = set()

        for q in probe_hard:
            item = by_id[q]
            code = self._run_candidate_on_item(item, baselines[q], new_persona)
            if pass_at_threshold(self._score(item, code)):
                new_pass_ids.add(q)

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

            # Pre-populate routing history from newly solved hard errors
            new_history = []
            for q in new_pass_ids:
                if q in by_id:
                    new_history.append({"instruction": by_id[q].get("instruction", "")[:500]})
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

            # Pre-populate routing history from newly solved hard errors
            new_history = []
            for q in new_pass_ids:
                if q in by_id:
                    new_history.append({"instruction": by_id[q].get("instruction", "")[:500]})
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
