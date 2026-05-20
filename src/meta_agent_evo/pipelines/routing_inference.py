"""Phase-2 routing + single-critic refinement."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.pipelines.base_pipeline import BasePipeline
from meta_agent_evo.prompts import baseline_prompts
from meta_agent_evo.prompts.coding import build_baseline_prompt, build_critic_prompt, build_refine_prompt
from meta_agent_evo.prompts.meta import MANAGER_PROMPT
from meta_agent_evo.utils.helpers import check_stop_condition, extract_code_block


class GMRoutingPipeline(BasePipeline):
    """Zero-shot manager routes to one critic; backbone refines (temp=0)."""

    def __init__(
        self,
        agent: Agent,
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
        except Exception as exc:
            logging.error("Failed to load scouting report at %s: %s", scouting_report_path, exc)
            self.roster = []

        self.routing_memory: list = []
        mem = Path(routing_memory_path)
        if mem.is_file():
            try:
                self.routing_memory = json.loads(mem.read_text(encoding="utf-8"))
            except Exception:
                pass

    def run(self, input_item: dict) -> Dict[str, Any]:
        prompt = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            prompt = f"{prompt}\n\nStarter Code:\n```python\n{starter_code}\n```"

        ds = input_item.get("dataset") or "mbpp"
        model_name = self.agent.llm.model_name
        history: list = [{"role": "user", "content": prompt}]

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
            few_shot_str = "\n\n### Past Successful Routing Examples:\n"
            for i, mem in enumerate(random.sample(self.routing_memory, min(5, len(self.routing_memory)))):
                few_shot_str += (
                    f"Example {i+1}:\nProblem: {mem['instruction']}"
                    f"\nOptimal Critic ID: {mem['best_critic_id']}\n\n"
                )

        manager_prompt = MANAGER_PROMPT.substitute(scouting_report=roster_str, problem_description=prompt) + few_shot_str
        router_res = self.agent.chat(
            [
                {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
                {"role": "user", "content": manager_prompt},
            ]
        )

        baseline_res = self.agent.chat(
            build_baseline_prompt(prompt, dataset=ds, model_name=model_name),
            temperature=0.0,
        )
        baseline_code = extract_code_block(baseline_res) or baseline_res

        selected_id = self.roster[0]["id"] if self.roster else "default"
        try:
            data = json.loads(extract_code_block(router_res) or router_res)
            if "selected_critic_id" in data and any(p["id"] == data["selected_critic_id"] for p in self.roster):
                selected_id = data["selected_critic_id"]
        except Exception as exc:
            logging.warning("Failed to parse Manager JSON routing output: %s", exc)

        history.append(
            {"stage": "baseline_and_routing", "selected_critic": selected_id, "baseline_code": baseline_code}
        )

        selected_player = next((p for p in self.roster if p["id"] == selected_id), None)
        if not selected_player:
            return {
                "id": input_item.get("id"),
                "initial_output": baseline_code,
                "final_output": baseline_code,
                "history": history,
            }

        critic_sys = selected_player.get("system_prompt", "You are a specialized code critic.")
        current_code = baseline_code

        for i in range(self.max_refine_iters):
            feedback = self.agent.chat(
                build_critic_prompt(prompt, current_code, critic_sys, dataset=ds, model_name=model_name)
            )
            history.append({"iteration": i + 1, "stage": "critique", "feedback": feedback})
            if check_stop_condition(feedback):
                break

            if self.domain != "coding":
                ref_msg = [
                    {"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Refine the solution based on the feedback.\n\n"
                            f"Problem:\n{prompt}\n\n"
                            f"Previous Solution:\n{current_code}\n\n"
                            f"Feedback:\n{feedback}\n\n"
                            "Final Answer: [ANSWER]"
                        ),
                    },
                ]
            else:
                ref_msg = build_refine_prompt(
                    prompt, feedback, current_code, dataset=ds, model_name=model_name
                )

            current_code = extract_code_block(self.agent.chat(ref_msg, temperature=0.0)) or current_code
            history.append({"iteration": i + 1, "stage": "revision", "code": current_code})

        return {
            "id": input_item.get("id"),
            "initial_output": baseline_code,
            "final_output": current_code,
            "history": history,
        }
