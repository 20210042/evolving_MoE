"""Roster JSON persistence and defaults."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List


DEFAULT_ROSTER: List[Dict[str, Any]] = [
    {
        "id": "array",
        "name": "Array Specialist",
        "strengths": "Fixed-size and dynamic arrays: index bounds, off-by-one, two-pointer, sliding window, prefix/suffix sums, in-place updates",
        "system_prompt": "You are an array algorithms critic. Help improve solutions involving indexing, loops, subarrays, and in-place updates. Point out likely off-by-one or boundary mistakes when you see them.",
    },
    {
        "id": "string",
        "name": "String Specialist",
        "strengths": "String parsing, building, comparison, regex, formatting rules, empty and whitespace edge cases",
        "system_prompt": "You are a string processing critic. Help improve parsing, formatting, and character-level logic. Mention empty strings or formatting issues when relevant.",
    },
    {
        "id": "dp",
        "name": "Dynamic Programming Specialist",
        "strengths": "State design, recurrence relations, base cases, memoization, tabulation, overlapping subproblems",
        "system_prompt": "You are a dynamic programming critic. Help check state definitions, transitions, and base cases. Note missing states or table bounds if the approach looks incomplete.",
    },
    {
        "id": "graph",
        "name": "Tree & Graph Specialist",
        "strengths": "Graph representation, BFS/DFS, trees, shortest paths, connectivity, cycle detection, visited tracking",
        "system_prompt": "You are a tree and graph algorithms critic. Help review traversals, graph representation, and visited handling. Flag obvious connectivity or traversal issues when you notice them.",
    },
    {
        "id": "math_geom",
        "name": "Math & Geometry Specialist",
        "strengths": "Number theory, modular arithmetic, primes and divisibility, combinatorics, coordinate geometry, distances and angles",
        "system_prompt": "You are a math and geometry algorithms critic. Help check formulas, modular arithmetic, and basic geometric reasoning. Mention overflow or edge values if they seem risky.",
    },
]


def load_roster(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning("Failed to load roster %s: %s", path, e)
            return []
    return []


def save_roster(path: str, roster: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(roster, f, indent=4)


def ensure_roster(path: str) -> List[Dict[str, Any]]:
    roster = load_roster(path)
    if not roster:
        roster = [dict(x) for x in DEFAULT_ROSTER]
        save_roster(path, roster)
    return roster


def assign_candidate_id(roster: List[Dict[str, Any]]) -> str:
    return f"c_{len(roster) + int(os.urandom(2).hex(), 16)}"


def _as_text(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_text(x)}" for k, x in v.items())
    return str(v) if v is not None else ""


def normalize_persona_fields(persona: Dict[str, Any], new_id: str) -> Dict[str, Any]:
    persona = dict(persona)
    persona["id"] = new_id
    # scout LLM may return these as list/dict → coerce to str (used in prompts & formatting)
    for k in ("system_prompt", "strengths", "approach", "name", "persona_name"):
        if k in persona:
            persona[k] = _as_text(persona[k])
    if "name" not in persona and "persona_name" in persona:
        persona["name"] = persona["persona_name"]
    persona.setdefault("total_war", 0)
    persona.setdefault("active_steps", 0)
    persona.setdefault("average_war", 0.0)
    return persona
