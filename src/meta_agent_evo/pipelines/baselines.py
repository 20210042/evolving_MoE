"""Baseline pipelines: Raw (1-pass) and Self-Refine (N-turn, persona-free)."""

from __future__ import annotations

import logging
from typing import List

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.pipelines.base_pipeline import BasePipeline
from meta_agent_evo.prompts.coding import (
    build_baseline_prompt,
    build_critic_prompt,
    build_refine_prompt,
)
from meta_agent_evo.utils.helpers import check_stop_condition, extract_code_block


class RawPipeline(BasePipeline):
    """Single-pass generation, no persona, no refinement."""

    def _instruction(self, input_item: dict) -> str:
        instruction = (
            input_item.get("instruction")
            or input_item.get("prompt")
            or input_item.get("problem")
            or ""
        )
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        return instruction

    def run(self, input_item: dict) -> dict:
        instruction = self._instruction(input_item)
        ds = (input_item.get("dataset") or "mbpp").lower()
        model_name = self.agent.llm.model_name

        raw_output = self.agent.chat(
            build_baseline_prompt(instruction, dataset=ds, model_name=model_name),
        )
        code = extract_code_block(raw_output) or raw_output

        return {
            "id": input_item.get("id"),
            "initial_output": code,
            "final_output": code,
        }

    def run_batch(self, items: List[dict]) -> List[dict]:
        if not items:
            return []
        model_name = self.agent.llm.model_name
        msgs = []
        for item in items:
            instruction = self._instruction(item)
            ds = (item.get("dataset") or "mbpp").lower()
            msgs.append(build_baseline_prompt(instruction, dataset=ds, model_name=model_name))
        outs = self.agent.chat_batch(msgs)
        results = []
        for item, raw in zip(items, outs):
            code = extract_code_block(raw) or raw
            results.append(
                {
                    "id": item.get("id"),
                    "initial_output": code,
                    "final_output": code,
                }
            )
        return results


class SelfRefinePipeline(BasePipeline):
    """Generate → Critic → Refine loop, no persona."""

    def __init__(self, agent: Agent, domain: str = "coding", max_refine_iters: int = 2):
        super().__init__(agent, domain)
        self.max_refine_iters = max_refine_iters

    def _instruction(self, input_item: dict) -> str:
        instruction = (
            input_item.get("instruction")
            or input_item.get("prompt")
            or input_item.get("problem")
            or ""
        )
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        return instruction

    def run(self, input_item: dict) -> dict:
        return self.run_batch([input_item])[0]

    def run_batch(self, items: List[dict]) -> List[dict]:
        if not items:
            return []

        from meta_agent_evo.prompts import baseline_prompts

        model_name = self.agent.llm.model_name
        neutral_critic_sys = baseline_prompts.CODING_CRITIC_SYSTEM

        instructions = [self._instruction(it) for it in items]
        datasets = [(it.get("dataset") or "mbpp").lower() for it in items]

        init_msgs = [
            build_baseline_prompt(instr, dataset=ds, model_name=model_name)
            for instr, ds in zip(instructions, datasets)
        ]
        codes = [extract_code_block(r) or r for r in self.agent.chat_batch(init_msgs)]
        histories = [[{"step": "initial", "output": c}] for c in codes]
        active = [True] * len(items)

        for i in range(self.max_refine_iters):
            active_idx = [j for j, ok in enumerate(active) if ok]
            if not active_idx:
                break

            critic_msgs = [
                build_critic_prompt(
                    instructions[j],
                    codes[j],
                    neutral_critic_sys,
                    dataset=datasets[j],
                    model_name=model_name,
                )
                for j in active_idx
            ]
            feedbacks = self.agent.chat_batch(critic_msgs, enable_thinking=True)

            refine_tasks: list = []
            for j, fb in zip(active_idx, feedbacks):
                histories[j].append({"step": f"critic_{i}", "feedback": fb})
                if check_stop_condition(fb):
                    active[j] = False
                    continue
                refine_tasks.append(j)

            if not refine_tasks:
                continue

            refine_msgs = [
                build_refine_prompt(
                    instructions[j],
                    histories[j][-1]["feedback"],
                    codes[j],
                    dataset=datasets[j],
                    model_name=model_name,
                )
                for j in refine_tasks
            ]
            refined = self.agent.chat_batch(refine_msgs)
            for j, ref_raw in zip(refine_tasks, refined):
                codes[j] = extract_code_block(ref_raw) or ref_raw
                histories[j].append({"step": f"refine_{i}", "output": codes[j]})

        results = []
        for item, code, hist in zip(items, codes, histories):
            results.append(
                {
                    "id": item.get("id"),
                    "initial_output": hist[0]["output"],
                    "final_output": code,
                    "history": hist,
                }
            )
        return results
