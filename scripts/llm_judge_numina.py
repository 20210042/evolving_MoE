#!/usr/bin/env python3
"""Judge NuminaMath-CoT JSONL outputs with an LLM.

The judge compares a model prediction against the dataset's full ``solution``
field, not the shorter ``ground_truth`` / answer field saved in eval outputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SYSTEM_PROMPT = """You are a strict but fair mathematical answer judge.
Your task is to decide whether the candidate response solves the problem correctly.

Use the reference solution as the authoritative solution. The candidate does not
need to match the wording or every derivation step, but its final answer and
reasoning must be mathematically valid for the problem. Penalize responses with
wrong final answers, unsupported guesses, contradictions, or reasoning that only
accidentally reaches the answer.

Return only valid JSON with this schema:
{
  "score": 0 or 1,
  "verdict": "correct" or "incorrect",
  "reason": "brief explanation"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-judge NuminaMath-CoT JSON/JSONL outputs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL file or directory containing *.jsonl/*.json files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for judged JSONL files. Defaults to the input file/dir.",
    )
    parser.add_argument("--split", default="test", help="NuminaMath-CoT_filtered split to load.")
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_JUDGE_MODEL", "google/gemma-4-26B-A4B-it"),
        help="OpenAI-compatible judge model name.",
    )
    parser.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("SKIML_BASE_URL"),
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api_key_env",
        default=None,
        help=(
            "Env var containing API key. Defaults to OPENAI_API_KEY, then SKIML_API_KEY. "
            "For localhost OpenAI-compatible servers, a dummy key is used if none is set."
        ),
    )
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_problem_chars", type=int, default=6000)
    parser.add_argument("--max_solution_chars", type=int, default=10000)
    parser.add_argument("--max_prediction_chars", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per file.")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(list(path.glob("*.jsonl")) + list(path.glob("*.json")))
        return [
            p
            for p in files
            if not p.name.endswith(".llm_judge.jsonl")
            and not p.name.endswith(".summary.json")
            and p.name != "llm_judge_aggregate.summary.json"
        ]
    raise FileNotFoundError(f"Input path not found: {path}")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = [item for item in data if isinstance(item, dict)]
        else:
            raise ValueError(f"JSON input must be an object or a list of objects: {path}")
        return records[:limit] if limit is not None else records

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSON in %s:%d: %s", path, line_no, exc)
            if limit is not None and len(records) >= limit:
                break
    return records


def load_numina_solutions(split: str) -> dict[str, dict[str, Any]]:
    from data.loader import get_dataset

    items = get_dataset("numina_cot", split=split, seed=None)
    return {str(item["id"]): item for item in items}


def extract_problem(record: dict[str, Any], ref_item: dict[str, Any] | None) -> str:
    if ref_item and ref_item.get("instruction"):
        return str(ref_item["instruction"])
    if record.get("instruction"):
        return str(record["instruction"])
    messages = record.get("input")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return ""


def get_prediction(record: dict[str, Any]) -> str:
    for key in ("prediction", "response", "final_output", "final_answer", "output"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n\n[TRUNCATED: {omitted} chars omitted]"


def build_messages(
    record: dict[str, Any],
    ref_item: dict[str, Any],
    *,
    max_problem_chars: int,
    max_solution_chars: int,
    max_prediction_chars: int,
) -> list[dict[str, str]]:
    problem = clip(extract_problem(record, ref_item), max_problem_chars)
    solution = clip(str(ref_item.get("solution", "")), max_solution_chars)
    prediction = clip(get_prediction(record), max_prediction_chars)
    user_prompt = f"""Problem:
{problem}

Reference solution from dataset:
{solution}

Candidate response:
{prediction}

Is the candidate response mathematically correct for the problem?"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def make_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The 'openai' package is required. Install it with: pip install openai") from exc

    api_key_env = args.api_key_env
    api_key = os.environ.get(api_key_env) if api_key_env else None
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("SKIML_API_KEY")
    if not api_key and args.base_url and "localhost" in args.base_url:
        api_key = "local"
    if not api_key and args.base_url and "127.0.0.1" in args.base_url:
        api_key = "local"
    if not api_key:
        raise SystemExit(
            "Set OPENAI_API_KEY/SKIML_API_KEY, or pass --base_url for a localhost OpenAI-compatible server."
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return OpenAI(**kwargs)


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    score = int(data.get("score", 0))
    score = 1 if score == 1 else 0
    verdict = str(data.get("verdict") or ("correct" if score else "incorrect")).lower()
    if verdict not in {"correct", "incorrect"}:
        verdict = "correct" if score else "incorrect"
    return {
        "score": score,
        "verdict": verdict,
        "reason": str(data.get("reason", ""))[:2000],
    }


def judge_one(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or "{}"
            judged = parse_judge_json(text)
            judged["raw_judge_response"] = text
            return judged
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    return {
        "score": 0,
        "verdict": "incorrect",
        "reason": f"judge_error: {last_error}",
        "raw_judge_response": "",
        "error": str(last_error),
    }


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path):
        if record.get("id") is not None:
            ids.add(str(record["id"]))
    return ids


def write_records(path: Path, records: list[dict[str, Any]], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_file(
    path: Path,
    output_dir: Path,
    client: Any,
    args: argparse.Namespace,
    refs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = read_jsonl(path, limit=args.limit)
    output_path = output_dir / f"{path.stem}.llm_judge.jsonl"
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    done_ids = existing_ids(output_path) if args.resume else set()
    pending: list[tuple[int, dict[str, Any], list[dict[str, str]]]] = []
    missing_solution_ids: list[str] = []

    for idx, record in enumerate(records):
        item_id = str(record.get("id", ""))
        if item_id in done_ids:
            continue
        ref_item = refs.get(item_id)
        if not ref_item or not ref_item.get("solution"):
            missing_solution_ids.append(item_id)
            continue
        messages = build_messages(
            record,
            ref_item,
            max_problem_chars=args.max_problem_chars,
            max_solution_chars=args.max_solution_chars,
            max_prediction_chars=args.max_prediction_chars,
        )
        pending.append((idx, record, messages))

    judged_records: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(judge_one, client, args.model, messages): (idx, record)
            for idx, record, messages in pending
        }
        for future in as_completed(futures):
            _, record = futures[future]
            judge = future.result()
            out = dict(record)
            out["llm_judge_model"] = args.model
            out["llm_judge_score"] = judge["score"]
            out["llm_judge_verdict"] = judge["verdict"]
            out["llm_judge_reason"] = judge["reason"]
            out["llm_judge_raw"] = judge.get("raw_judge_response", "")
            previous_combined = float(out.get("combined_score", 0.0) or 0.0)
            out["combined_score_before_llm_judge"] = previous_combined
            out["combined_score"] = 1.0 if max(previous_combined, float(judge["score"])) >= 1.0 else 0.0
            if judge.get("error"):
                out["llm_judge_error"] = judge["error"]
            judged_records.append((idx, out))

    judged_records.sort(key=lambda item: item[0])
    write_records(
        output_path,
        [record for _, record in judged_records],
        append=args.resume and output_path.exists(),
    )

    all_judged = read_jsonl(output_path)
    total = len(all_judged)
    correct = sum(int(r.get("llm_judge_score", 0)) for r in all_judged)
    combined_correct = sum(1 for r in all_judged if float(r.get("combined_score", 0.0) or 0.0) >= 1.0)
    previous_combined_correct = sum(
        1 for r in all_judged if float(r.get("combined_score_before_llm_judge", 0.0) or 0.0) >= 1.0
    )
    score = correct / total if total else 0.0
    summary = {
        "input": str(path),
        "output": str(output_path),
        "judge_model": args.model,
        "total_records_in_input": len(records),
        "judged": total,
        "correct": correct,
        "llm_judge_score": score,
        "combined_correct_before_llm_judge": previous_combined_correct,
        "combined_score_before_llm_judge": previous_combined_correct / total if total else 0.0,
        "combined_correct_after_llm_judge_or": combined_correct,
        "combined_score_after_llm_judge_or": combined_correct / total if total else 0.0,
        "missing_solution_count": len(missing_solution_ids),
        "missing_solution_ids_first_20": missing_solution_ids[:20],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    input_path = Path(args.input)
    files = iter_input_files(input_path)
    if not files:
        raise SystemExit(f"No JSONL files found in {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_path if input_path.is_dir() else input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading NuminaMath-CoT_filtered solutions from split=%s", args.split)
    refs = load_numina_solutions(args.split)
    logging.info("Loaded %d reference solutions.", len(refs))

    client = make_client(args)
    summaries = []
    for path in files:
        logging.info("Judging %s", path)
        summary = judge_file(path, output_dir, client, args, refs)
        summaries.append(summary)
        logging.info(
            "Done %s: %.4f (%d/%d)",
            path.name,
            summary["llm_judge_score"],
            summary["correct"],
            summary["judged"],
        )

    if len(summaries) > 1:
        judged = sum(s["judged"] for s in summaries)
        correct = sum(s["correct"] for s in summaries)
        aggregate = {
            "input": str(input_path),
            "judge_model": args.model,
            "files": len(summaries),
            "judged": judged,
            "correct": correct,
            "llm_judge_score": correct / judged if judged else 0.0,
            "combined_correct_before_llm_judge": sum(s["combined_correct_before_llm_judge"] for s in summaries),
            "combined_score_before_llm_judge": (
                sum(s["combined_correct_before_llm_judge"] for s in summaries) / judged if judged else 0.0
            ),
            "combined_correct_after_llm_judge_or": sum(s["combined_correct_after_llm_judge_or"] for s in summaries),
            "combined_score_after_llm_judge_or": (
                sum(s["combined_correct_after_llm_judge_or"] for s in summaries) / judged if judged else 0.0
            ),
            "per_file": summaries,
        }
        aggregate_path = output_dir / "llm_judge_aggregate.summary.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Aggregate LLM-judge score: {aggregate['llm_judge_score']:.4f} ({correct}/{judged})")


if __name__ == "__main__":
    main()
