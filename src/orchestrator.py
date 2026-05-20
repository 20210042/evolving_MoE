"""GM evolution orchestrator (batch training + roster updates)."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from action_selector import ActionDecision, ActionGateConfig, select_action
from llm import LLMService
from evaluation.scorer import pass_at_threshold, score_one
from prompts import baseline_prompts
from prompts.coding import build_baseline_prompt, build_critic_prompt, build_refine_prompt
from roster import assign_candidate_id, ensure_roster, normalize_persona_fields, save_roster
from scout import scout_new_persona
from step_logger import StepLogContext, StepLogger
from utils import extract_code_block
from utils import check_stop_condition
from war import compute_war_scores, pick_worst_agent


class GMEvolutionOrchestrator:
    def __init__(
        self,
        llm: LLMService,
        roster_path: str,
        *,
        action_cfg: Optional[ActionGateConfig] = None,
        max_refine_iters: int = 4,
        lcb_timeout: int = 10,
        lcb_release_version: str = "release_v5",
        code_exec_timeout: float = 3.0,
        war_tiebreak: str = "random",
        probe_stability_k: int = 8,
        results_dir: str = "results",
        run_id: str = "default",
        dataset_name: str = "livecodebench",
        seed: int = 42,
    ):
        self.llm = llm
        self.roster_path = roster_path
        self.roster = ensure_roster(roster_path)
        self.action_cfg = action_cfg or ActionGateConfig()
        self.max_refine_iters = max_refine_iters
        self.lcb_timeout = lcb_timeout
        self.lcb_release_version = lcb_release_version
        self.code_exec_timeout = code_exec_timeout
        self.war_tiebreak = war_tiebreak
        self.probe_stability_k = probe_stability_k
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

        model_name = self.llm.model_name

        for item in batch_data:
            problem_id = item["id"]
            instruction = item.get("instruction", "")
            ds = item.get("dataset") or self.dataset_name
            starter = item.get("starter_code")

            baseline_msg = build_baseline_prompt(
                instruction,
                dataset=ds,
                model_name=model_name,
                starter_code=starter,
            )
            baseline_raw = self.llm.chat(baseline_msg, temperature=0.0)
            baseline_code = extract_code_block(baseline_raw) or baseline_raw
            baselines[problem_id] = baseline_code

            any_solved = False
            failed_attempts: List[str] = []

            for player in self.roster:
                cid = player["id"]
                sys_prompt = player.get("system_prompt", "You are an expert coder.")
                current_code = baseline_code

                for _ in range(self.max_refine_iters):
                    critic_msg = build_critic_prompt(
                        instruction,
                        current_code,
                        sys_prompt,
                        dataset=ds,
                        model_name=model_name,
                    )
                    feedback = self.llm.chat(critic_msg)
                    if check_stop_condition(feedback):
                        break

                    refine_msg = build_refine_prompt(
                        instruction,
                        feedback,
                        current_code,
                        dataset=ds,
                        model_name=model_name,
                    )
                    ref_raw = self.llm.chat(refine_msg, temperature=0.0)
                    current_code = extract_code_block(ref_raw) or ref_raw

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
                    f"Failed Code Sample:\n{failed_attempts[0] if failed_attempts else baseline_code}\n"
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
            feedback = self.llm.chat(critic_msg)
            if check_stop_condition(feedback):
                break

            refine_user = (
                f"Refine the code based on the feedback.\n\n"
                f"Problem:\n{instruction}\n\n"
                f"Previous Code:\n{current_code}\n\n"
                f"Feedback:\n{feedback}\n\n"
                "Return the improved code in the following format:\n"
                "```python\n[CODE]\n```"
            )
            ref_msg = [
                {"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM},
                {"role": "user", "content": refine_user},
            ]
            ref_raw = self.llm.chat(ref_msg, temperature=0.0)
            current_code = extract_code_block(ref_raw) or ref_raw

        return current_code

    def _update_routing_memory(self, batch_data: List[Dict], squad_results: Dict[str, Set[str]]) -> None:
        routing_memory: List[Dict[str, Any]] = []
        mem_path = os.path.join(self.step_logger.results_dir, "routing_memory.json")
        if os.path.exists(mem_path):
            try:
                with open(mem_path, "r", encoding="utf-8") as f:
                    routing_memory = json.load(f)
            except Exception:
                pass

        for item in batch_data:
            pid = item["id"]
            solvers = [a for a, s in squad_results.items() if pid in s]
            if len(solvers) == 1:
                routing_memory.append(
                    {"instruction": item.get("instruction", "")[:500], "best_critic_id": solvers[0]}
                )

        routing_memory = routing_memory[-50:]
        os.makedirs(self.step_logger.results_dir, exist_ok=True)
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(routing_memory, f, indent=4)

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

        worst_agent = pick_worst_agent(war_scores, squad_results, tiebreak=self.war_tiebreak, rng=rng)
        logging.info("Worst Agent for Eviction: %s", worst_agent)
        logging.info("Total Hard Errors: %s", len(hard_errors_texts))

        noop_decision = ActionDecision(
            "noop",
            {"A1": 0.0, "A2": 0.0, "A3": 0.0},
            0.0,
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
                [],
                set(),
                ub_rate,
            )
            return

        self._update_routing_memory(batch_data, squad_results)

        hard_errors_combined = "\n\n---\n\n".join(hard_errors_texts.values())
        new_persona = scout_new_persona(self.llm, self.roster, hard_errors_combined)
        if not new_persona or "system_prompt" not in new_persona:
            logging.warning("Failed to scout new persona. Skipping.")
            return

        logging.info("Scouted New Persona: %s", new_persona.get("persona_name"))

        roster_ids = [p["id"] for p in self.roster]
        probe_hard = list(hard_errors_texts.keys())

        covered_in_batch = [
            item["id"]
            for item in batch_data
            if any(item["id"] in squad_results[r] for r in roster_ids)
        ]
        k = min(self.probe_stability_k, len(covered_in_batch))
        probe_stab = rng.sample(covered_in_batch, k) if k > 0 else []

        union_probe = list(dict.fromkeys(probe_hard + probe_stab))
        by_id = {b["id"]: b for b in batch_data}
        new_pass_ids: Set[str] = set()

        for q in union_probe:
            item = by_id[q]
            code = self._run_candidate_on_item(item, baselines[q], new_persona)
            if pass_at_threshold(self._score(item, code)):
                new_pass_ids.add(q)

        decision = select_action(
            roster_ids=roster_ids,
            worst_id=worst_agent,
            squad_results={k: set(v) for k, v in squad_results.items()},
            probe_hard=probe_hard,
            probe_stability=probe_stab,
            new_pass_ids=new_pass_ids,
            cfg=self.action_cfg,
        )

        logging.info(
            "Action gate: %s | U=%s | MCL(worst)≈%.3f | new_pass=%s/%s",
            decision.action,
            decision.utility,
            decision.mcl_worst,
            len(new_pass_ids),
            len(union_probe),
        )

        if decision.action == "noop":
            logging.info("Roster unchanged (action gate).")
        elif decision.action == "add":
            new_id = assign_candidate_id(self.roster)
            persona = normalize_persona_fields(new_persona, new_id)
            self.roster.append(persona)
            save_roster(self.roster_path, self.roster)
            logging.info("Commit ADD: new persona %s", new_id)
        else:
            self.roster = [r for r in self.roster if r["id"] != worst_agent]
            new_id = assign_candidate_id(self.roster)
            persona = normalize_persona_fields(new_persona, new_id)
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
            probe_stab,
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
        probe_stab: List[str],
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
            "probe": {"hard_n": len(probe_hard), "stab_n": len(probe_stab)},
            "new_pass_on_union": sorted(new_pass_ids),
            "utility": decision.utility,
            "marginal_hard_gain_add": decision.marginal_hard_gain_add,
            "marginal_hard_gain_swap_extra": decision.marginal_hard_gain_swap_extra,
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
