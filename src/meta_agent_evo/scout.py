"""Meta-agent (LLM) scouting for a new critic persona."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.prompts.meta import META_AGENT_PROMPT


def scout_new_persona(
    agent: Agent,
    roster: List[Dict[str, Any]],
    hard_errors_text: str,
) -> Dict[str, Any]:
    roster_str = json.dumps(
        [
            {"id": p.get("id"), "name": p.get("name", p.get("persona_name")), "strengths": p.get("strengths")}
            for p in roster
        ],
        indent=2,
    )

    prompt = META_AGENT_PROMPT.substitute(
        hard_errors=hard_errors_text[:4000],
        current_roster=roster_str,
    )

    msg = [
        {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
        {"role": "user", "content": prompt},
    ]
    response = agent.chat(msg, temperature=0.7)
    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        logging.error("Failed to parse new persona JSON: %s", e)
    return {}
