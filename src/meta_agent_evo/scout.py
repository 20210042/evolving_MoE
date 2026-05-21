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


def _calculate_overlap(set1: set[str], set2: set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))


def scout_new_persona(
    agent: Agent,
    roster: List[Dict[str, Any]],
    hard_errors_text: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    roster_str = _format_roster_table(roster)
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

    def get_word_set(persona: Dict[str, Any]) -> set[str]:
        text = " ".join([
            persona.get("persona_name", ""),
            persona.get("name", persona.get("persona_name", "")),
            persona.get("strengths", ""),
        ]).lower()
        words = re.findall(r"[a-z]{4,}", text)
        return {w for w in words if w not in _STOPWORDS}

    # Pre-calculate word sets for existing roster
    roster_word_sets = [(p, get_word_set(p)) for p in roster]

    msg = [
        {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, max_retries + 1):
        response = agent.chat(msg, temperature=0.7)
        try:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                continue
            persona = json.loads(m.group(0))

            # Validate domain overlap using Overlap Coefficient
            new_set = get_word_set(persona)
            overlap_detected = False
            overlap_details = []

            for old_p, old_set in roster_word_sets:
                o_ratio = _calculate_overlap(new_set, old_set)
                if o_ratio >= 0.35:  # Overlap Threshold 35%
                    overlap_detected = True
                    old_name = old_p.get("name", old_p.get("persona_name", ""))
                    overlap_details.append(f"'{old_name}' (Overlap: {o_ratio:.2%})")

            if not overlap_detected:
                return persona

            logging.warning(
                f"[Attempt {attempt}] Proposed domain overlaps with: {overlap_details}. Retrying..."
            )

            # Append assistant response and feedback prompt for auto-correction
            msg.append({"role": "assistant", "content": response})
            feedback_str = (
                f"Your proposal overlaps significantly with existing domains: {', '.join(overlap_details)}. "
                f"Please propose a completely orthogonal domain. Do NOT duplicate their expertise."
            )
            msg.append({"role": "user", "content": feedback_str})

        except Exception as e:
            logging.error("Failed to parse or validate new persona JSON: %s", e)

    return {}
