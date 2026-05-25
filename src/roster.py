"""Roster JSON persistence and defaults."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List


DEFAULT_ROSTER: List[Dict[str, Any]] = [
    {
        "id": "senior_dev",
        "name": "Senior Software Engineer",
        "strengths": "Python mastery, system architecture, standard library usage",
        "system_prompt": "You are a senior software engineer. Focus on clean, modular, and robust code for competitive programming.",
    },
    {
        "id": "code_grader",
        "name": "Code Grader & Optimizer",
        "strengths": "Complexity analysis, TLE/MLE prevention, edge case scoring",
        "system_prompt": "You are a strict code grader. Focus on time/space complexity and identifying potential logical flaws that lead to incorrect answers.",
    },
    {
        "id": "qa_red_team",
        "name": "QA Red Team",
        "strengths": "Contradiction finding, edge cases, large scale input testing",
        "system_prompt": "You are a member of the QA Red Team. Your goal is to break the code by finding unhandled edge cases, infinite loops, or memory limits.",
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


def normalize_persona_fields(persona: Dict[str, Any], new_id: str) -> Dict[str, Any]:
    persona = dict(persona)
    persona["id"] = new_id
    if "name" not in persona and "persona_name" in persona:
        persona["name"] = persona["persona_name"]
    return persona
