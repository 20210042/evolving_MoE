"""Phase-2 routing + single-critic refinement (formerly ``GMRoutingPipeline``)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from llm import LLMService
from pipelines.base_pipeline import BasePipeline
from prompts import baseline_prompts
from prompts.coding import build_baseline_prompt, build_critic_prompt, build_refine_prompt
from prompts.meta import MANAGER_PROMPT
from utils import extract_code_block
from utils import check_stop_condition


class GMRoutingPipeline(BasePipeline):
    """Zero-shot manager routes to one critic; backbone refines (temp=0)."""

    def __init__(
        self,
        llm: LLMService,
        scouting_report_path: str,
        domain: str = "coding",
        routing_memory_path: str = "results/routing_memory.json",
        max_refine_iters: int = 2,
    ):
        super().__init__(agent, domain)
        self.max_refine_iters = max_refine_iters
        self.scouting_report_path = scouting_report_path

        try:
            with open(self.scouting_report_path, "r", encoding="utf-8") as f:
                self.roster = json.load(f)
        except Exception as e:
            logging.error("Failed to load scouting report at %s: %s", scouting_report_path, e)
            self.roster = []

        self.routing_memory: list = []
        mem = Path(routing_memory_path)
        if mem.is_file():
            try:
                self.routing_memory = json.loads(mem.read_text(encoding="utf-8"))
            except Exception:
                pass



    def _route_and_generate(self, prompt: str, dataset: str) -> tuple[str, str]:
        model_name = self.llm.model_name

        roster_str = json.dumps(
            [
                {
                    "id": p.get("id", "default_id"),
                    "name": p.get("name", p.get("persona_name", "Expert")),
                    "strengths": p.get("strengths", "Specialized coding expert"),
                }
                for p in self.roster
            ],
            indent=2,
        )

        few_shot_str = ""
        if self.routing_memory:
            import random

            few_shot_str = "\n\n### Past Successful Routing Examples:\n"
            sample_mem = random.sample(self.routing_memory, min(5, len(self.routing_memory)))
            for i, mem in enumerate(sample_mem):
                few_shot_str += (
                    f"Example {i+1}:\nProblem: {mem['instruction']}\nOptimal Critic ID: {mem['best_critic_id']}\n\n"
                )

        manager_prompt = MANAGER_PROMPT.substitute(scouting_report=roster_str, problem_description=prompt) + few_shot_str
        manager_msg = [
            {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
            {"role": "user", "content": manager_prompt},
        ]
        router_res = self.llm.chat(manager_msg)

        baseline_msg = build_baseline_prompt(
            prompt, dataset=dataset, model_name=model_name, starter_code=None
        )
        baseline_res = self.llm.chat(baseline_msg, temperature=0.0)

        selected_id = self.roster[0]["id"] if self.roster else "default"
        try:
            clean_json = extract_code_block(router_res) or router_res
            data = json.loads(clean_json)
            if "selected_critic_id" in data:
                if any(p["id"] == data["selected_critic_id"] for p in self.roster):
                    selected_id = data["selected_critic_id"]
        except Exception as e:
            logging.warning("Failed to parse Manager JSON routing output: %s", e)

        return selected_id, (extract_code_block(baseline_res) or baseline_res)

    def run(self, input_item: dict) -> Dict[str, Any]:
        prompt = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            prompt = f"{prompt}\n\nStarter Code:\n```python\n{starter_code}\n```"

        ds = input_item.get("dataset") or "mbpp"
        history: list = [{"role": "user", "content": prompt}]

        selected_critic_id, baseline_code = self._route_and_generate(prompt, ds)
        history.append(
            {"stage": "baseline_and_routing", "selected_critic": selected_critic_id, "baseline_code": baseline_code}
        )

        selected_player = next((p for p in self.roster if p["id"] == selected_critic_id), None)
        if not selected_player:
            return {
                "id": input_item.get("id"),
                "initial_output": baseline_code,
                "final_output": baseline_code,
                "history": history,
            }

        critic_sys = selected_player.get("system_prompt", "You are a specialized code critic.")
        current_code = baseline_code
        model_name = self.llm.model_name

        for i in range(self.max_refine_iters):
            crit_msg = build_critic_prompt(
                prompt, current_code, critic_sys, dataset=ds, model_name=model_name
            )
            feedback = self.llm.chat(crit_msg)
            history.append({"iteration": i + 1, "stage": "critique", "feedback": feedback})

            if check_stop_condition(feedback):
                break

            if self.domain != "coding":
                gen_sys_prompt = baseline_prompts.MATH_GEN_SYSTEM
                refine_user_prompt = (
                    f"Refine the solution based on the feedback.\n\n"
                    f"Problem:\n{prompt}\n\n"
                    f"Previous Solution:\n{current_code}\n\n"
                    f"Feedback:\n{feedback}\n\n"
                    "Provide the corrected solution.\n"
                    "Answer format:\n"
                    "Final Answer: [ANSWER]"
                )
                ref_msg = [
                    {"role": "system", "content": gen_sys_prompt},
                    {"role": "user", "content": refine_user_prompt},
                ]
            else:
                ref_msg = build_refine_prompt(
                    prompt, feedback, current_code, dataset=ds, model_name=model_name
                )

            final_code_raw = self.llm.chat(ref_msg, temperature=0.0)
            current_code = extract_code_block(final_code_raw) or final_code_raw
            history.append({"iteration": i + 1, "stage": "revision", "code": current_code})

        return {
            "id": input_item.get("id"),
            "initial_output": baseline_code,
            "final_output": current_code,
            "history": history,
        }
