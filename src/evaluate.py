#!/usr/bin/env python3
"""
SFT 모델 평가 스크립트.

Usage:
    python src/evaluate.py \\
        --model checkpoints/my_sft_model \\
        --dataset bigmath \\
        --split test \\
        --wandb_project sft_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import wandb

from data.loader import get_dataset
from evaluation.metrics import (
    exact_match_score,
    extract_answer,
    f1_score_tokens,
    math_verify_accuracy,
    numerical_match,
)
from llm import LLMService
from prompts.baseline_prompts import (
    CODING_GEN_SYSTEM,
    CODING_GEN_USER,
    MATH_GEN_SYSTEM,
    MATH_GEN_USER,
)

METRIC_KEYS = ["math_verify", "numerical_match", "exact_match", "f1"]


def _build_messages(item: dict, dataset: str) -> list:
    instruction = item["instruction"]
    ds = dataset.lower()
    if ds in ("bigmath", "math"):
        return [
            {"role": "system", "content": MATH_GEN_SYSTEM},
            {"role": "user", "content": MATH_GEN_USER.format(instruction=instruction)},
        ]
    if ds in ("mbpp", "humaneval", "livecodebench", "ds1000"):
        return [
            {"role": "system", "content": CODING_GEN_SYSTEM},
            {"role": "user", "content": CODING_GEN_USER.format(instruction=instruction)},
        ]
    return [{"role": "user", "content": instruction}]


def _get_categories(item: dict) -> list[str]:
    cat = item.get("category", "unknown")
    return cat if isinstance(cat, list) else [str(cat)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT model on math/QA datasets")
    parser.add_argument("--model", required=True, help="모델 경로 또는 HuggingFace ID")
    parser.add_argument("--dataset", default="bigmath")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="평가할 카테고리 목록 (공백 구분). 미지정 시 전체 + 카테고리별 자동 분리.",
    )
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_to_wandb", action="store_true", help="wandb 로깅 활성화")
    parser.add_argument("--wandb_project", default="sft_eval")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--output_dir", default="results/eval_sft")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    if args.log_to_wandb:
        run_name = args.wandb_run_name or f"eval_{args.dataset}_{Path(args.model).name}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    output_dir = Path(args.output_dir) / Path(args.model).name / args.dataset / args.split
    output_dir.mkdir(parents=True, exist_ok=True)


    ## ---------- 데이터셋 로딩 및 카테고리를 지정한 경우, 해당 카테고리만 필터링 ---------
    logger.info("데이터셋 로딩: %s (split=%s)", args.dataset, args.split)
    all_items = get_dataset(
        args.dataset,
        split=args.split,
        data_ratio=args.data_ratio,
        seed=args.seed,
    )
    logger.info("총 %d개 로드", len(all_items))

    ## 카테고리를 지정한 경우, 지정한 카테고리의 데이터셋에 대해서만 평가 
    if args.categories:
        specified = set(args.categories)
        items = [item for item in all_items if any(cat in specified for cat in _get_categories(item))]
        if not items:
            logger.error("지정한 categories에 해당하는 아이템이 없습니다: %s", args.categories)
            sys.exit(1)
        logger.info("카테고리 필터링 후: %d개", len(items))
        
    ## 카테고리를 지정하지 않은 경우, 모든 카테고리에 대해서 평가 
    else:
        items = all_items

    # ----------- Step 2: LLM generate (필터링된 데이터셋 전체에 대해서 한번만 generate) -----------
    logger.info("generate 시작: %d개 아이템", len(items))
    logger.info("모델 로딩: %s", args.model)
    
    llm = LLMService(model_name=args.model, mode="vllm", tp_size=args.tp_size)
    messages_list = [_build_messages(it, args.dataset) for it in items]
    prompts = [
        llm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    predictions = llm.generate(prompts, max_tokens=args.max_tokens, temperature=args.temperature)

    # ----------- Step 3: 채점 및 overall 집계 -----------
    records: List[dict] = []
    metric_sums: Dict[str, float] = defaultdict(float)

    for item, pred in zip(items, predictions):
        ref = str(item.get("ground_truth", ""))
        extracted = extract_answer(pred)
        scores = {
            "math_verify": math_verify_accuracy(extracted, ref),
            "numerical_match": numerical_match(extracted, ref),
            "exact_match": exact_match_score(extracted, ref),
            "f1": f1_score_tokens(pred, ref),
        }
        for k, v in scores.items():
            metric_sums[k] += v
        records.append({
            **item,
            "prediction_raw": pred,
            "prediction_extracted": extracted,
            **{f"score_{k}": v for k, v in scores.items()},
        })

    overall_avg = {k: metric_sums[k] / len(items) for k in METRIC_KEYS}
    metrics_by_category: Dict[str, Dict[str, float]] = {"overall": overall_avg}

    logger.info("=== [overall] %d items ===", len(items))
    for k, v in overall_avg.items():
        logger.info("  overall / %s : %.4f", k, v)

    (output_dir / "overall.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    # ----------- Step 4: 카테고리별 집계 -----------
    has_categories = any("category" in it for it in items)
    if has_categories:
        records_by_category: Dict[str, List[dict]] = defaultdict(list)
        for rec in records:
            for cat in _get_categories(rec):
                records_by_category[cat].append(rec)

        for cat, cat_records in sorted(records_by_category.items()):
            logger.info("=== [%s] %d items ===", cat, len(cat_records))
            cat_sums: Dict[str, float] = defaultdict(float)
            for rec in cat_records:
                for k in METRIC_KEYS:
                    cat_sums[k] += rec[f"score_{k}"]
            avg = {k: cat_sums[k] / len(cat_records) for k in METRIC_KEYS}
            metrics_by_category[cat] = avg

            for k, v in avg.items():
                logger.info("  %s / %s : %.4f", cat, k, v)

            cat_safe = cat.replace("/", "_").replace(" ", "_")
            (output_dir / f"{cat_safe}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in cat_records),
                encoding="utf-8",
            )

    # ----------- wandb 로그 -----------
    if args.log_to_wandb:
        log_dict: Dict[str, object] = {}

        for cat, m in metrics_by_category.items():
            for k, v in m.items():
                log_dict[f"eval/{cat}/{k}"] = v

        for k in METRIC_KEYS:
            rows = []
            for cat, m in sorted(metrics_by_category.items()):
                n = len(records_by_category[cat]) if has_categories and cat != "overall" else len(items)
                rows.append([cat, round(m[k], 4), n])
            log_dict[f"eval/table/{k}"] = wandb.Table(
                columns=["category", k, "n_items"],
                data=rows,
            )

        wandb.log(log_dict)
        wandb.finish()

    logger.info("평가 완료. 결과: %s", output_dir)
    logger.info("최종 메트릭: %s", json.dumps(overall_avg, indent=2))


if __name__ == "__main__":
    main()
