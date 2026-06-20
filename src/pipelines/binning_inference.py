"""Binning inference — every roster expert solves every problem, no routing.

Labels the full training set with each expert's solve outcome (the training
signal for the weight-level MoE). Unlike GMRoutingPipeline there is *no*
manager/routing generation: for each problem we fan out one expert generation
per roster member. The roster size N is read from the roster file — nothing is
hardcoded.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agents.base import Agent
from pipelines.base_pipeline import BasePipeline
from prompts.coding import build_expert_prompt
from utils.helpers import extract_code_block


class BinningPipeline(BasePipeline):
    """Run all N roster experts on each problem in a single batched vLLM pass."""

    def __init__(
        self,
        agent: Agent,
        roster_path: str,
        domain: str = "coding",
        gen_enable_thinking: bool = False,
    ):
        super().__init__(agent, domain)
        self.gen_enable_thinking = gen_enable_thinking
        with open(roster_path, "r", encoding="utf-8") as f:
            self.roster = json.load(f)
        if not self.roster:
            raise ValueError(f"Empty roster at {roster_path}")
        self.expert_ids = [e["id"] for e in self.roster]
        logging.info(
            "BinningPipeline: %d experts × all problems, no routing [%s]",
            len(self.roster),
            ", ".join(self.expert_ids),
        )

    def _prompt(self, item: dict) -> str:
        text = item.get("instruction") or item.get("prompt") or item.get("problem") or ""
        starter = item.get("starter_code")
        if starter:
            text = f"{text}\n\nStarter Code:\n```python\n{starter}\n```"
        return text

    def run(self, input_item: dict) -> Dict[str, Any]:
        return self.run_batch([input_item])[0]

    def run_batch(self, items: List[dict]) -> List[Dict[str, Any]]:
        if not items:
            return []
        model_name = self.agent.llm.model_name
        n = len(self.roster)

        # Fan out to a flat (item × expert) message list so a single chat_batch
        # lets vLLM continuous-batch every generation in one pass.
        msgs: List[Any] = []
        for item in items:
            prompt = self._prompt(item)
            ds = item.get("dataset") or "mbpp"
            for expert in self.roster:
                msgs.append(
                    build_expert_prompt(
                        prompt,
                        expert.get("system_prompt", ""),
                        dataset=ds,
                        model_name=model_name,
                        starter_code=None,  # already folded into prompt above
                        approach=expert.get("approach"),
                        domain=self.domain,
                    )
                )

        outs = self.agent.chat_batch(msgs, enable_thinking=self.gen_enable_thinking)

        results: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            expert_outputs: Dict[str, str] = {}
            for j, expert in enumerate(self.roster):
                raw = outs[i * n + j]
                code = raw if self.domain == "math" else (extract_code_block(raw) or raw)
                expert_outputs[expert["id"]] = code
            results.append({"id": item.get("id"), "expert_outputs": expert_outputs})
        return results
