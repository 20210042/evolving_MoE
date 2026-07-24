#!/usr/bin/env python3
"""Tag every QASC train question with one science subject using the gemma backbone.

QASC ships no human-prior metadata, so this produces an LLM prior for the
embed_expert_viz panel (labeled as model-tagged there, not human).
Output: results/embed_viz/qasc_llm_tags.json  {id: subject}
Resume: existing output ids are skipped, chunks are flushed as they finish.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIG = REPO / "configs" / "qasc_eval_a4b.yaml"
SOURCE = REPO / "export" / "qasc" / "qasc_train.jsonl"
OUT = REPO / "results" / "embed_viz" / "qasc_llm_tags.json"
CHUNK = 512

TAXONOMY = ["biology", "ecology", "earth science", "chemistry", "physics",
            "astronomy", "health", "other"]

SYSTEM = "You are a precise science curriculum classifier."
USER_TMPL = """Classify the following science question into exactly one subject.
Subjects: biology, ecology, earth science, chemistry, physics, astronomy, health, other.
Answer with the subject name only.

Question:
{q}
"""


def parse_subject(text: str) -> str:
    t = (text or "").strip().lower()
    for s in TAXONOMY:
        if s in t:
            return s
    return "other"


def main() -> None:
    from utils.llm import llm_service_from_yaml_config

    rows = [json.loads(l) for l in open(SOURCE, encoding="utf-8") if l.strip()]
    tags: dict[str, str] = {}
    if OUT.is_file():
        tags = json.load(open(OUT, encoding="utf-8"))
        print(f"resume: {len(tags)}개 태그 로드")
    todo = [r for r in rows if str(r["id"]) not in tags]
    print(f"태깅 대상 {len(todo)}/{len(rows)}")
    if not todo:
        return

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    llm = llm_service_from_yaml_config(cfg["model"], cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        msgs = [[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": USER_TMPL.format(q=r["instruction"])}]
                for r in chunk]
        outs = llm.chat_batch(msgs, max_tokens=16, temperature=0.0,
                              enable_thinking=False)
        for r, o in zip(chunk, outs):
            tags[str(r["id"])] = parse_subject(o)
        json.dump(tags, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"진행 {min(start + CHUNK, len(todo))}/{len(todo)} → {OUT}")

    from collections import Counter
    print("분포:", dict(Counter(tags.values()).most_common()))


if __name__ == "__main__":
    main()
