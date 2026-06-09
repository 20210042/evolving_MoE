"""Meta-agent (LLM) scouting for a new expert coder persona."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.base import Agent
from prompts.meta import META_AGENT_PROMPT
from utils.helpers import extract_json_object


def _format_roster_table(roster: List[Dict[str, Any]]) -> str:
    # approach 모드(strengths 없이 identity+approach)면 그 둘을 보여준다.
    if any(p.get("approach") for p in roster):
        lines = ["| name | identity | approach |", "|------|----------|----------|"]
        for p in roster:
            name = p.get("name", p.get("persona_name", ""))
            ident = (p.get("system_prompt") or "").replace("|", "/")
            appr = (p.get("approach") or "").replace("|", "/")
            lines.append(f"| {name} | {ident} | {appr} |")
        return "\n".join(lines)
    lines = [
        "| name | strengths |",
        "|------|-----------|",
    ]
    for p in roster:
        name = p.get("name", p.get("persona_name", ""))
        strengths = (p.get("strengths") or "").replace("|", "/")
        lines.append(f"| {name} | {strengths} |")
    return "\n".join(lines)


def _format_exclusive_solves(exclusive_solves_map: Dict[str, List[str]]) -> str:
    if not exclusive_solves_map:
        return "(none yet — early stage)"
    lines = []
    for name, solves in exclusive_solves_map.items():
        if solves:
            snippets = "\n".join(f"  • {s[:150]}" for s in solves[-3:])
            lines.append(f"{name}:\n{snippets}")
        else:
            lines.append(f"{name}:\n  • (none yet)")
    return "\n\n".join(lines)


def scout_new_persona(
    agent: Agent,
    roster: List[Dict[str, Any]],
    hard_errors_text: str,
    dataset_name: str = "livecodebench",
    enable_thinking: bool = True,
    exclusive_solves_map: Optional[Dict[str, List[str]]] = None,
    use_approach_persona: bool = False,
) -> Dict[str, Any]:
    roster_str = _format_roster_table(roster)

    ds = (dataset_name or "").lower()
    if ds in ("bigmath", "math"):
        if exclusive_solves_map is not None and use_approach_persona:
            from prompts.meta import META_AGENT_MATH_PROMPT_V3
            prompt = META_AGENT_MATH_PROMPT_V3.substitute(
                hard_errors=hard_errors_text[:4000],
                current_roster=roster_str,
                exclusive_solves=_format_exclusive_solves(exclusive_solves_map),
            )
        elif exclusive_solves_map is not None:
            from prompts.meta import META_AGENT_MATH_PROMPT_V2
            prompt = META_AGENT_MATH_PROMPT_V2.substitute(
                hard_errors=hard_errors_text[:4000],
                current_roster=roster_str,
                exclusive_solves=_format_exclusive_solves(exclusive_solves_map),
            )
        else:
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
    response = agent.chat(msg, enable_thinking=enable_thinking)
    data = extract_json_object(response)
    if data:
        return data
    logging.error("Failed to parse new persona JSON from scout response.")
    return {}
