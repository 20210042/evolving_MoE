"""LiveCodeBench official-style scoring (``lcb_runner``)."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

from paths import livecodebench_paths
from utils.helpers import extract_code_block

_LCB_BENCHMARK_CACHE: dict = {}


def _ensure_lcb_on_path() -> None:
    for p in livecodebench_paths():
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def score_lcb_data(
    result_lines: list[str],
    result_file: str,
    dataset: str = "livecodebench",
    timeout: int = 10,
    release_version: str = "release_v5",
) -> tuple[float, int]:
    from lcb_runner.benchmarks import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    _ensure_lcb_on_path()

    global _LCB_BENCHMARK_CACHE
    version = release_version
    if version not in _LCB_BENCHMARK_CACHE:
        data = load_code_generation_dataset(release_version=version)
        _LCB_BENCHMARK_CACHE[version] = data
        _LCB_BENCHMARK_CACHE[f"{version}_dict"] = {str(p.question_id): p for p in data}

    benchmark_dict = _LCB_BENCHMARK_CACHE[f"{version}_dict"]

    generations_list = []
    samples_list = []
    matched_indices = []

    for idx, line in enumerate(result_lines):
        try:
            item = json.loads(line)
            qid = str(item.get("id"))
            if qid not in benchmark_dict:
                continue

            multi_outputs = item.get("all_persona_outputs")
            if isinstance(multi_outputs, list):
                codes = []
                for out in multi_outputs:
                    prediction = out.get("code") or out.get("final_output") or ""
                    extracted = extract_code_block(prediction)
                    codes.append(extracted if extracted else prediction)
                generations_list.append(codes)
            else:
                prediction = item.get("final_output") or item.get("code") or ""
                extracted = extract_code_block(prediction)
                generations_list.append([extracted if extracted else prediction])

            samples_list.append(benchmark_dict[qid].get_evaluation_sample())
            matched_indices.append(idx)

        except Exception:
            continue

    if not generations_list:
        return 0.0, 0

    max_k = max(len(g) for g in generations_list) if generations_list else 0
    for g in generations_list:
        while len(g) < max_k:
            g.append("")

    with contextlib.redirect_stdout(StringIO()):
        metrics_result = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[max_k if max_k > 0 else 1],
            num_process_evaluate=2,
            timeout=timeout,
        )

    raw_eval_details = metrics_result[1]
    solved_count = 0
    save_path = result_file.replace(".jsonl", "_scored.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for i, original_line_idx in enumerate(matched_indices):
            item = json.loads(result_lines[original_line_idx])

            item_eval = None
            if isinstance(raw_eval_details, dict):
                item_eval = raw_eval_details.get(i)
            elif isinstance(raw_eval_details, list) and i < len(raw_eval_details):
                item_eval = raw_eval_details[i]

            persona_results = {}
            if item_eval is not None:
                if isinstance(item_eval, dict) and "results" in item_eval:
                    raw_results = item_eval["results"]
                else:
                    raw_results = item_eval

                multi_outputs = item.get("all_persona_outputs", [])

                if isinstance(multi_outputs, list) and len(multi_outputs) > 0:
                    for p_idx, p_out in enumerate(multi_outputs):
                        p_id = p_out.get("persona_id") or p_out.get("persona") or f"p_{p_idx}"
                        if p_idx < len(raw_results):
                            res_list = raw_results[p_idx]
                            is_pass = (
                                all(x > 0 for x in res_list)
                                if isinstance(res_list, list) and len(res_list) > 0
                                else False
                            )
                            persona_results[p_id] = is_pass
                        else:
                            persona_results[p_id] = False

                    if any(persona_results.values()):
                        solved_count += 1
                else:
                    res_list = raw_results[0] if (isinstance(raw_results, list) and len(raw_results) > 0) else []
                    is_pass = (
                        all(x > 0 for x in res_list)
                        if isinstance(res_list, list) and len(res_list) > 0
                        else False
                    )
                    persona_results["default"] = is_pass
                    if is_pass:
                        solved_count += 1

            item["pass_fail"] = persona_results
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg_score = (solved_count / len(matched_indices)) * 100.0 if matched_indices else 0.0
    return avg_score, len(matched_indices)


def score_lcb_item(
    question_id: str,
    prediction_code: str,
    *,
    timeout: int = 10,
    release_version: str = "release_v5",
) -> float:
    """
    Pass@1 style score for a single LCB problem (0.0 or 100.0).
    """
    line = json.dumps(
        {"id": question_id, "final_output": prediction_code, "persona": "scouting"}
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        avg, n = score_lcb_data(
            [line],
            tmp_path,
            timeout=timeout,
            release_version=release_version,
        )
        if n == 0:
            return 0.0
        return 100.0 if avg >= 99.99 else 0.0
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_path.replace(".jsonl", "_scored.jsonl")).unlink(missing_ok=True)
        except OSError:
            pass
