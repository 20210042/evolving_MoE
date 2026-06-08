"""Meta-agent (LLM) scouting for a new expert coder persona."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base import Agent
from prompts.meta import META_AGENT_PROMPT
from utils.helpers import extract_json_object


def _format_roster_table(roster: List[Dict[str, Any]]) -> str:
    lines = [
        "| name | strengths |",
        "|------|-----------|",
    ]
    for p in roster:
        name = p.get("name", p.get("persona_name", ""))
        strengths = (p.get("strengths") or "").replace("|", "/")
        lines.append(f"| {name} | {strengths} |")
    return "\n".join(lines)


def scout_new_persona(
    agent: Agent,
    roster: List[Dict[str, Any]],
    hard_errors_text: str,
    dataset_name: str = "livecodebench",
) -> Dict[str, Any]:
    roster_str = _format_roster_table(roster)

    ds = (dataset_name or "").lower()
    if ds in ("bigmath", "math"):
        from prompts.meta import META_AGENT_MATH_PROMPT
        prompt = META_AGENT_MATH_PROMPT.substitute(
            hard_errors=hard_errors_text[:4000],
            current_roster=roster_str,
        )
    else:
        prompt = META_AGENT_PROMPT.substitute(
            hard_errors=hard_errors_text[:4000],
            current_roster=roster_str,
        )

    msg = [
        {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
        {"role": "user", "content": prompt},
    ]
    response = agent.chat(msg, enable_thinking=True)
    data = extract_json_object(response)
    if data:
        return data
    logging.error("Failed to parse new persona JSON from scout response.")
    return {}
