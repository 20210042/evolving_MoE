#!/usr/bin/env python3
"""Tag LBox examples into a fixed legal taxonomy with a local LLM."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.helpers import extract_json_object
from utils.llm import LLMService


CATEGORIES = {
    "civil_property_obligation": "민법 - 물권법/채권법/구상권같은 소송 일체",
    "civil_family_inheritance": "민법 - 친족상속법",
    "criminal_property": "형법 - 재산죄: 사기/횡령/배임",
    "criminal_non_property": "형법 - 비재산죄: 살인, 폭행, 상해",
    "admin_traffic": "행정법 - 도로교통법",
    "admin_labor": "행정법 - 근로기준법",
    "admin_other": "행정법 - 행정법 기타",
    "family_case": "가사법",
    "patent_ip": "특허법",
}


SYSTEM_PROMPT = """You are a Korean legal dataset tagger.
Assign each LBox example to exactly one category from the fixed taxonomy.
Use only the instruction and facts. Do not solve the legal classification task.
Return only one JSON object."""


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars * 3 // 4
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n[TRUNCATED_FOR_CONTEXT: middle omitted for legal category tagging]\n\n"
        + text[-tail:].lstrip(),
        True,
    )


def build_user_prompt(item: dict[str, Any], *, max_facts_chars: int = 0, max_instruction_chars: int = 0) -> str:
    categories = "\n".join(f"- {key}: {desc}" for key, desc in CATEGORIES.items())
    instruction = str(item.get("instruction") or "").strip()
    facts = str(item.get("facts") or "").strip()
    if not facts and instruction:
        facts = instruction
    instruction, instruction_truncated = truncate_middle(instruction, max_instruction_chars)
    facts, facts_truncated = truncate_middle(facts, max_facts_chars)
    truncation_note = (
        f"Truncation applied: instruction={instruction_truncated}, facts={facts_truncated}. "
        "If truncated, classify from the visible beginning/end context."
    )
    return f"""Fixed taxonomy:
{categories}

Tie-breaking rules:
- If the example is about divorce, custody, child support, family court procedure, domestic relations, or family registration, choose family_case.
- If the example is about inheritance, succession shares, wills, or heir property disputes without a stronger family-court/procedure signal, choose civil_family_inheritance.
- If the example is about traffic crime or Road Traffic Act, choose admin_traffic even when criminal penalties are involved.
- If the example is about wages, severance pay, dismissal, industrial accident, or labor standards, choose admin_labor.
- If no category is perfect, choose the closest category and lower confidence.

Return JSON with this schema:
{{
  "primary_category": "<one of: {', '.join(CATEGORIES)}>",
  "confidence": <number from 0.0 to 1.0>,
  "rationale": "<short Korean reason, one sentence>"
}}

LBox example id: {item.get("id")}
Task type: {item.get("task_type")}
Case type: {item.get("casetype")}
{truncation_note}

Instruction:
{instruction}

Facts:
{facts}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                done.add(str(row["id"]))
    return done


def normalize_tag(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if raw in CATEGORIES:
        return raw
    lowered = raw.lower()
    for key in CATEGORIES:
        if key.lower() == lowered:
            return key
    for key, desc in CATEGORIES.items():
        if raw in desc or desc in raw:
            return key
    return None


def parse_response(text: str) -> tuple[str | None, float | None, str, str]:
    obj = extract_json_object(text)
    if not isinstance(obj, dict):
        return None, None, "", "parse_failed"
    tag = normalize_tag(str(obj.get("primary_category") or ""))
    confidence = obj.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    rationale = str(obj.get("rationale") or "").strip()
    status = "ok" if tag else "invalid_category"
    return tag, confidence, rationale, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="export/lbox")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--output-dir", default="results/lbox_legal_category_tags/gemma4_a4b")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-facts-chars", type=int, default=12000)
    parser.add_argument("--max-instruction-chars", type=int, default=4000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_dir) / f"lbox_{args.split}.jsonl"
    rows = load_jsonl(data_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lbox_{args.split}_legal_categories.jsonl"
    summary_path = out_dir / f"lbox_{args.split}_legal_categories_summary.json"

    done_ids = set() if args.no_resume else load_done_ids(out_path)
    todo = [row for row in rows if str(row.get("id")) not in done_ids]
    logging.info("split=%s total=%d done=%d todo=%d output=%s", args.split, len(rows), len(done_ids), len(todo), out_path)
    if not todo:
        return

    llm = LLMService(
        args.model,
        mode="vllm",
        vllm_kwargs={
            "tensor_parallel_size": args.tp_size,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
            "max_num_seqs": max(args.batch_size, 1),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": False,
            "trust_remote_code": True,
        },
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.0,
    )

    counts: dict[str, int] = {}
    with out_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            row,
                            max_facts_chars=args.max_facts_chars,
                            max_instruction_chars=args.max_instruction_chars,
                        ),
                    },
                ]
                for row in batch
            ]
            outputs = llm.chat_batch(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.0,
                enable_thinking=False,
            )
            for row, raw in zip(batch, outputs):
                tag, confidence, rationale, status = parse_response(raw)
                record = {
                    "id": row.get("id"),
                    "split": args.split,
                    "task_type": row.get("task_type"),
                    "task_config": row.get("task_config"),
                    "casetype": row.get("casetype"),
                    "primary_category": tag,
                    "primary_category_name": CATEGORIES.get(tag or ""),
                    "confidence": confidence,
                    "parse_status": status,
                    "rationale": rationale,
                    "raw_response": raw,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[str(tag or status)] = counts.get(str(tag or status), 0) + 1
            out.flush()
            logging.info("tagged %d/%d", min(start + len(batch), len(todo)), len(todo))

    all_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            all_counts[str(rec.get("primary_category"))] = all_counts.get(str(rec.get("primary_category")), 0) + 1
            status_counts[str(rec.get("parse_status"))] = status_counts.get(str(rec.get("parse_status")), 0) + 1
    summary = {
        "split": args.split,
        "input": str(data_path),
        "output": str(out_path),
        "model": args.model,
        "taxonomy": CATEGORIES,
        "counts": all_counts,
        "parse_status_counts": status_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("summary=%s", summary_path)


if __name__ == "__main__":
    main()
