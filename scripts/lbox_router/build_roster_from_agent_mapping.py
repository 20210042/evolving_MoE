#!/usr/bin/env python3
"""Convert an agent-id mapping package into a runnable roster JSON list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-experts", type=int)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Expected a non-empty agent mapping object: {args.mapping}")

    roster = [{"id": agent_id, **metadata} for agent_id, metadata in mapping.items()]
    if args.expected_experts is not None and len(roster) != args.expected_experts:
        raise ValueError(
            f"Expected {args.expected_experts} experts, found {len(roster)}"
        )
    for expert in roster:
        if not expert.get("system_prompt"):
            raise ValueError(f"Missing system_prompt for expert {expert['id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(roster, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(roster)} experts to {args.output}")


if __name__ == "__main__":
    main()
