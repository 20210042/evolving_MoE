"""Meta-agent (LLM) scouting for a new critic persona."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.prompts.meta import META_AGENT_PROMPT


def _format_roster_table(roster: List[Dict[str, Any]]) -> str:
    lines = [
        "| id | name | strengths |",
        "|----|------|-----------|",
    ]
    for p in roster:
        pid = p.get("id", "")
        name = p.get("name", p.get("persona_name", ""))
        strengths = (p.get("strengths") or "").replace("|", "/")
        lines.append(f"| {pid} | {name} | {strengths} |")
    lines.append("")
    lines.append("Do NOT duplicate domains already covered above.")
    return "\n".join(lines)


def _extract_forbidden_keywords(roster: List[Dict[str, Any]], max_kws_per_agent: int = 3) -> List[str]:
    """Option B: Extract the most representative domain keywords per agent to ensure orthogonality.

    For each agent in the roster, extracts the top `max_kws_per_agent` content words.
    This prevents domain overlap immediately from the first addition without creating
    an excessively long forbidden word list that restricts LLM's creativity.
    """
    _STOPWORDS = {
        "and", "the", "for", "with", "that", "this", "from", "are", "have",
        "such", "each", "when", "also", "both", "than", "they", "their",
        "code", "data", "into", "case", "cases", "based", "using", "large",
        "small", "given", "type", "types", "list", "lists", "value", "values",
        "input", "output", "ensure", "correct", "including", "correct",
        "potential", "improve", "identify", "analysis", "handling",
        "performance", "implementation", "optimization", "operations",
        "datasets", "solutions", "techniques", "problems", "issues",
        "complex", "handling", "reduction", "algorithms", "specialist",
        "critic", "help", "formatting", "place", "expert", "specializes",
    }
    forbidden_kws = set()
    for p in roster:
        text = " ".join([
            p.get("persona_name", ""),
            p.get("name", ""),
            p.get("strengths", ""),
        ]).lower()
        words = re.findall(r"[a-z]{4,}", text)
        content_words = [w for w in words if w not in _STOPWORDS]
        counts = Counter(content_words)
        top_words = [w for w, _ in counts.most_common(max_kws_per_agent)]
        forbidden_kws.update(top_words)

    return sorted(list(forbidden_kws))


def scout_new_persona(
    agent: Agent,
    roster: List[Dict[str, Any]],
    hard_errors_text: str,
) -> Dict[str, Any]:
    roster_str = _format_roster_table(roster)

    # Option B: inject dynamic forbidden keyword list into the prompt
    forbidden_keywords = _extract_forbidden_keywords(roster)
    forbidden_note = ""
    if forbidden_keywords:
        kw_str = ", ".join(forbidden_keywords)
        forbidden_note = (
            f"\n\n⚠️ STRICTLY FORBIDDEN DOMAINS (already covered by current roster):\n"
            f"  Keywords: [{kw_str}]\n"
            f"Any proposed persona whose strengths substantially overlap with these keywords "
            f"will be REJECTED. You MUST propose expertise in a genuinely different domain."
        )

    prompt = META_AGENT_PROMPT.substitute(
        hard_errors=hard_errors_text[:4000],
        current_roster=roster_str + forbidden_note,
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
