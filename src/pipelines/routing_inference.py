"""Phase-2 routing + one-step expert raw generation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from agents.base import Agent
from pipelines.base_pipeline import BasePipeline
from prompts.coding import build_expert_prompt
from prompts.meta import MANAGER_PROMPT, MANAGER_MATH_PROMPT
from utils.helpers import extract_code_block, extract_json_object


class GMRoutingPipeline(BasePipeline):
    """Manager routes to one expert; expert generates code in a single turn."""

    def __init__(
        self,
        agent: Agent,
        scouting_report_path: str,
        domain: str = "coding",
        routing_memory_path: str = "results/routing_memory.json",
        max_refine_iters: int = 2,  # deprecated: kept for CLI compat
        gen_enable_thinking: bool = True,
    ):
        super().__init__(agent, domain)
        self.max_refine_iters = max_refine_iters
        self.gen_enable_thinking = gen_enable_thinking
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

    def _roster_json(self) -> str:
        return json.dumps(
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

    def _few_shot_suffix(self) -> str:
        examples: list = []
        roster_size = len(self.roster)
        agent_history_pools: Dict[str, list] = {}
        for p in self.roster:
            hist = p.get("routing_history", [])
            if hist:
                agent_history_pools[p["id"]] = list(hist)
                chosen = random.choice(agent_history_pools[p["id"]])
                examples.append({"best_expert_id": p["id"], "instruction": chosen["instruction"]})
                agent_history_pools[p["id"]].remove(chosen)

        if len(examples) < roster_size:
            extra_needed = roster_size - len(examples)
            available_agent_ids = [aid for aid, pool in agent_history_pools.items() if pool]
            while extra_needed > 0 and available_agent_ids:
                aid = random.choice(available_agent_ids)
                chosen = random.choice(agent_history_pools[aid])
                examples.append({"best_expert_id": aid, "instruction": chosen["instruction"]})
                agent_history_pools[aid].remove(chosen)
                if not agent_history_pools[aid]:
                    available_agent_ids.remove(aid)
                extra_needed -= 1

        if not examples:
            return ""
        random.shuffle(examples)
        few_shot_str = "\n\n### Past Successful Routing Examples:\n"
        for i, mem in enumerate(examples):
            few_shot_str += (
                f"Example {i+1}:\nProblem: {mem['instruction']}"
                f"\nOptimal Expert ID: {mem['best_expert_id']}\n\n"
            )
        return few_shot_str

    def _parse_expert_id(self, router_res: str) -> str:
        default_id = self.roster[0]["id"] if self.roster else "default"
        data = extract_json_object(router_res)
        if not data:
            logging.warning("Failed to parse Manager JSON routing output.")
            return default_id
        expert_id = data.get("selected_expert_id") or data.get("selected_critic_id")
        if expert_id and any(p["id"] == expert_id for p in self.roster):
            return expert_id
        logging.warning("Router selected unknown expert id: %s", expert_id)
        return default_id

    def _prepare_prompt(self, input_item: dict) -> str:
        prompt = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            prompt = f"{prompt}\n\nStarter Code:\n```python\n{starter_code}\n```"
        return prompt or ""

    def _route_one(self, prompt: str, few_shot_str: str) -> str:
        roster_str = self._roster_json()
        mgr_template = MANAGER_MATH_PROMPT if self.domain == "math" else MANAGER_PROMPT
        manager_prompt = (
            mgr_template.substitute(scouting_report=roster_str, problem_description=prompt)
            + few_shot_str
        )
        router_res = self.agent.chat(
            [
                {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
                {"role": "user", "content": manager_prompt},
            ],
            enable_thinking=False,
        )
        return self._parse_expert_id(router_res)

    def run(self, input_item: dict) -> Dict[str, Any]:
        prompt = self._prepare_prompt(input_item)
        ds = input_item.get("dataset") or "mbpp"
        model_name = self.agent.llm.model_name
        few_shot_str = self._few_shot_suffix()

        selected_id = self._route_one(prompt, few_shot_str)
        history: list = [{"stage": "routing", "selected_expert": selected_id}]

        selected_player = next((p for p in self.roster if p["id"] == selected_id), None)
        if not selected_player:
            return {
                "id": input_item.get("id"),
                "initial_output": "",
                "final_output": "",
                "history": history,
            }

        expert_sys = selected_player.get("system_prompt", "You are an expert coder.")
        gen_msg = build_expert_prompt(
            prompt,
            expert_sys,
            dataset=ds,
            model_name=model_name,
            starter_code=None,
        )
        raw = self.agent.chat(gen_msg, enable_thinking=self.gen_enable_thinking)
        code = raw if self.domain == "math" else (extract_code_block(raw) or raw)
        history.append({"stage": "generation", "code": code})

        return {
            "id": input_item.get("id"),
            "initial_output": code,
            "final_output": code,
            "history": history,
        }

    def run_batch(self, items: List[dict]) -> List[Dict[str, Any]]:
        if not items:
            return []

        model_name = self.agent.llm.model_name
        few_shot_str = self._few_shot_suffix()
        roster_str = self._roster_json()

        route_msgs = []
        prompts: List[str] = []
        mgr_template = MANAGER_MATH_PROMPT if self.domain == "math" else MANAGER_PROMPT
        for item in items:
            prompt = self._prepare_prompt(item)
            prompts.append(prompt)
            manager_prompt = (
                mgr_template.substitute(scouting_report=roster_str, problem_description=prompt)
                + few_shot_str
            )
            route_msgs.append(
                [
                    {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
                    {"role": "user", "content": manager_prompt},
                ]
            )

        route_out = self.agent.chat_batch(route_msgs, enable_thinking=False)
        selected_ids = [self._parse_expert_id(r) for r in route_out]

        gen_msgs = []
        for item, prompt, sid in zip(items, prompts, selected_ids):
            ds = item.get("dataset") or "mbpp"
            player = next((p for p in self.roster if p["id"] == sid), None)
            expert_sys = (
                player.get("system_prompt", "You are an expert coder.") if player else "You are an expert coder."
            )
            gen_msgs.append(
                build_expert_prompt(
                    prompt,
                    expert_sys,
                    dataset=ds,
                    model_name=model_name,
                    starter_code=None,
                )
            )

        gen_out = self.agent.chat_batch(gen_msgs, enable_thinking=self.gen_enable_thinking)

        results: List[Dict[str, Any]] = []
        for item, sid, raw in zip(items, selected_ids, gen_out):
            code = raw if self.domain == "math" else (extract_code_block(raw) or raw)
            results.append(
                {
                    "id": item.get("id"),
                    "initial_output": code,
                    "final_output": code,
                    "history": [
                        {"stage": "routing", "selected_expert": sid},
                        {"stage": "generation", "code": code},
                    ],
                }
            )
        return results
