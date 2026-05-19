"""Baseline pipelines: Raw (1-pass) and Self-Refine (N-turn, persona-free).

Ported from MultiAgent/src/pipelines/baselines.py but prompt construction
delegates to `meta_agent_evo.prompts.coding` (build_baseline_prompt,
build_critic_prompt, build_refine_prompt) — the exact same path used by the
evolved GMRoutingPipeline in routing_inference.py.  This ensures identical
Qwen3 prompt formatting across all conditions.
"""

from __future__ import annotations

import logging

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

    def run(self, input_item: dict) -> dict:
        instruction = (
            input_item.get("instruction")
            or input_item.get("prompt")
            or input_item.get("problem")
            or ""
        )
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

        ds = (input_item.get("dataset") or "mbpp").lower()
        model_name = self.agent.llm.model_name

        msgs = build_baseline_prompt(instruction, dataset=ds, model_name=model_name)
        raw_output = self.agent.chat(msgs, temperature=0.0)
        code = extract_code_block(raw_output) or raw_output

        return {
            "id": input_item.get("id"),
            "initial_output": code,
            "final_output": code,
        }


class SelfRefinePipeline(BasePipeline):
    """Generate → Critic → Refine loop, no persona.

    Uses a neutral critic (CODING_CRITIC_SYSTEM from baseline_prompts) and
    the same build_* helpers as the evolved pipeline so Qwen3 prompts are
    formatted identically.
    """

    def __init__(self, agent: Agent, domain: str = "coding", max_refine_iters: int = 2):
        super().__init__(agent, domain)
        self.max_refine_iters = max_refine_iters

    def run(self, input_item: dict) -> dict:
        instruction = (
            input_item.get("instruction")
            or input_item.get("prompt")
            or input_item.get("problem")
            or ""
        )
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

        ds = (input_item.get("dataset") or "mbpp").lower()
        model_name = self.agent.llm.model_name

        # ── 1. Initial generation (same as RawPipeline) ─────────────────────
        gen_msgs = build_baseline_prompt(instruction, dataset=ds, model_name=model_name)
        raw_init = self.agent.chat(gen_msgs, temperature=0.0)
        current_code = extract_code_block(raw_init) or raw_init
        history = [{"step": "initial", "output": current_code}]

        # ── 2. Refinement loop ───────────────────────────────────────────────
        # Use a neutral critic system prompt (no persona).
        from meta_agent_evo.prompts import baseline_prompts
        neutral_critic_sys = baseline_prompts.CODING_CRITIC_SYSTEM

        for i in range(self.max_refine_iters):
            # Critic step
            crit_msgs = build_critic_prompt(
                instruction,
                current_code,
                neutral_critic_sys,
                dataset=ds,
                model_name=model_name,
            )
            feedback = self.agent.chat(crit_msgs)
            history.append({"step": f"critic_{i}", "feedback": feedback})

            if check_stop_condition(feedback):
                logging.info("Self-refine: early stop at iteration %d", i)
                break

            # Refine step
            ref_msgs = build_refine_prompt(
                instruction,
                feedback,
                current_code,
                dataset=ds,
                model_name=model_name,
            )
            refined_raw = self.agent.chat(ref_msgs, temperature=0.0)
            current_code = extract_code_block(refined_raw) or refined_raw
            history.append({"step": f"refine_{i}", "output": current_code})

        return {
            "id": input_item.get("id"),
            "initial_output": history[0]["output"],
            "final_output": current_code,
            "history": history,
        }
