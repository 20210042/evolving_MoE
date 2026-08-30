#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/jaehoonjeong/data/MetaAgentEvolution_Release")

RUNS = {
    "hard-WAR": ROOT / "results/acc/seed20210111/roster_final.json",
    "latest": ROOT / "results/acc/seed20211004/roster_final.json",
}

OUT = ROOT / "results/acc/final_roster_prompts_compare.md"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_agents(obj):
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in (
            "roster",
            "agents",
            "experts",
            "members",
            "final_roster",
        ):
            if key in obj and isinstance(obj[key], list):
                return obj[key]

        if all(isinstance(v, dict) for v in obj.values()):
            result = []

            for agent_id, value in obj.items():
                item = dict(value)
                item.setdefault("id", agent_id)
                result.append(item)

            return result

    raise ValueError(
        f"Could not identify agent list. "
        f"Top-level type={type(obj).__name__}"
    )


def agent_id(agent: dict, index: int) -> str:
    for key in (
        "id",
        "agent_id",
        "name",
        "uid",
    ):
        value = agent.get(key)

        if value:
            return str(value)

    return f"agent_{index}"


def title(agent: dict) -> str:
    for key in (
        "persona",
        "title",
        "role",
        "specialty",
        "expertise",
        "name",
    ):
        value = agent.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, dict):
            for subkey in (
                "name",
                "title",
                "role",
                "specialty",
            ):
                subvalue = value.get(subkey)

                if isinstance(subvalue, str) and subvalue.strip():
                    return subvalue.strip()

    return "(untitled)"


def main():
    lines = []

    lines.append(
        "# Hard-WAR vs Latest WAR — Final Roster Expert Definitions"
    )
    lines.append("")

    for run_name, path in RUNS.items():
        data = load_json(path)
        agents = find_agents(data)

        lines.append(f"## {run_name}")
        lines.append("")
        lines.append(f"- Source: `{path}`")
        lines.append(f"- Experts: **{len(agents)}**")
        lines.append("")

        for i, agent in enumerate(agents, 1):
            aid = agent_id(agent, i)
            persona = title(agent)

            lines.append(
                f"### {i}. `{aid}` — {persona}"
            )
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(
                    agent,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                )
            )
            lines.append("```")
            lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()