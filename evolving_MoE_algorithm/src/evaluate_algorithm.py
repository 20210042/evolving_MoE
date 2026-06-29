"""ACC algorithm evaluation entry point.

This script evaluates vanilla or LoRA-adapted models on the ACC algorithm test split. It generates
answers with `LLMService`, resumes safely from partial JSONL outputs, scores coding tasks through
stdin/stdout execution, keeps the old math metrics for compatibility, and writes both per-example
results and aggregate summaries."""

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import wandb
from transformers import HfArgumentParser

from data.loader_algorithm import get_dataset
from evaluation.metrics import (
    exact_match_score,
    math_verify_score,
    numerical_match_score,
    token_f1_score,
)
from evaluation.scorer_algorithm import score_one
from utils.helpers import extract_math_answer, set_all_seeds
from utils.llm import LLMService

logger = logging.getLogger(__name__)
MATH_DATASETS = {"bigmath", "math"}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="google/gemma-4-31B-it")
    dtype: str = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="eager")
    trust_remote_code: bool = field(default=True)
    finetuned_lora_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    test_dataset: str = field(default="acc_algorithm")
    data_dir: Optional[str] = field(
        default="/home/minjikim/minji_link/evolving_MoE/data/acc_algorithm"
    )
    categories: Optional[List[str]] = field(default=None)
    data_ratio: float = field(default=1.0)


@dataclass
class ExtraArguments:
    inference_mode: str = field(default="vllm")
    tensor_parallel_size: int = field(default=1)
    max_model_len: int = field(default=16384)
    max_new_tokens: int = field(default=8192)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=64)
    repetition_penalty: float = field(default=1.05)
    gpu_memory_utilization: float = field(default=0.90)
    enable_thinking: bool = field(default=True)

    eval_batch_size: int = field(default=8)
    resume: bool = field(default=True)

    output_dir: str = field(default="./results")
    wandb_run_name: str = field(default="eval_algorithm")
    wandb_project: str = field(default="evolving-moe")
    seed: int = field(default=42)


def evaluate_item(
    item: dict,
    prediction: str,
    is_math_dataset: bool = True,
) -> dict:
    if is_math_dataset:
        extracted = extract_math_answer(prediction)

        return {
            "extracted_answer": extracted,
            "exact_match_score": exact_match_score(
                extracted,
                item["ground_truth"],
            ),
            "token_f1_score": token_f1_score(
                extracted,
                item["ground_truth"],
            ),
            "numerical_match_score": numerical_match_score(
                extracted,
                item["ground_truth"],
            ),
            "math_verify_score": math_verify_score(
                extracted,
                item["ground_truth"],
            ),
        }

    return {
        "pass_score": score_one(
            item,
            prediction,
        )
    }


def atomic_write_jsonl(
    path: str,
    rows: list[dict],
) -> None:
    """Write JSONL through a temporary file and atomically replace the target path.

    Used when repairing partially written result files or rewriting final sorted rows.
    """
    temp_path = f"{path}.tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temp_path,
        path,
    )


def load_existing_results(
    path: str,
) -> tuple[list[dict], set[str]]:
    """Load an existing result JSONL file and return rows plus completed item IDs.

    If a previous job stopped mid-write, or if duplicate IDs were appended, this function drops the
    bad rows and rewrites the file so resume mode starts from a clean checkpoint.
    """
    if not os.path.exists(path):
        return [], set()

    rows: list[dict] = []
    completed_ids: set[str] = set()
    repair_required = False

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSONL line %d in %s.",
                    line_number,
                    path,
                )
                repair_required = True
                continue

            if row.get("id") is None:
                logger.warning(
                    "Skipping result without id at line %d in %s.",
                    line_number,
                    path,
                )
                repair_required = True
                continue

            row_id = str(row["id"])

            if row_id in completed_ids:
                logger.warning(
                    "Skipping duplicate result id=%s.",
                    row_id,
                )
                repair_required = True
                continue

            rows.append(row)
            completed_ids.add(row_id)

    if repair_required:
        logger.warning(
            "Repairing existing result file: %s",
            path,
        )

        atomic_write_jsonl(
            path,
            rows,
        )

    return rows, completed_ids


def append_batch(
    path: str,
    rows: list[dict],
) -> None:
    """Append one completed evaluation batch to the result JSONL file.

    The explicit flush and fsync make completed batches durable even if a long SLURM job is stopped
    before the full evaluation finishes.
    """
    if not rows:
        return

    with open(
        path,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        handle.flush()
        os.fsync(handle.fileno())


def build_summary(
    results: list[dict],
) -> dict:
    def avg(
        rows: list[dict],
        key: str,
    ) -> float:
        values = [
            row[key]
            for row in rows
            if key in row
        ]

        if not values:
            return 0.0

        return sum(values) / len(values)

    score_keys = sorted(
        {
            key
            for row in results
            for key in row
            if key.endswith("_score")
        }
    )

    overall = {
        key: avg(
            results,
            key,
        )
        for key in score_keys
    }

    log_dict = {
        "num_examples": len(results),
        **{
            f"overall/{key}": value
            for key, value in overall.items()
        },
    }

    logger.info(
        "[overall] %d examples  %s",
        len(results),
        "  ".join(
            f"{key}={value:.4f}"
            for key, value in overall.items()
        ),
    )

    category_groups: dict[str, list[dict]] = defaultdict(list)

    for row in results:
        categories = row.get(
            "category",
            [],
        )

        if categories is None:
            categories = []
        elif isinstance(
            categories,
            str,
        ):
            categories = [categories]

        for category in categories:
            category_groups[
                str(category)
            ].append(row)

    for category in sorted(category_groups):
        rows = category_groups[category]

        category_average = {
            key: avg(
                rows,
                key,
            )
            for key in score_keys
        }

        log_dict.update(
            {
                f"{category}/{key}": value
                for key, value in category_average.items()
            }
        )

        logger.info(
            "[%s] %d examples  %s",
            category,
            len(rows),
            "  ".join(
                f"{key}={value:.4f}"
                for key, value in category_average.items()
            ),
        )

    return log_dict


def main() -> None:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            ExtraArguments,
        ),
        description="ACC Algorithm Evaluate Script",
    )

    model_args, data_args, extra_args = (
        parser.parse_args_into_dataclasses()
    )

    logging.basicConfig(
        format=(
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.INFO,
    )

    if extra_args.eval_batch_size <= 0:
        raise ValueError(
            "eval_batch_size must be positive, "
            f"got {extra_args.eval_batch_size}."
        )

    set_all_seeds(
        seed=extra_args.seed,
    )

    wandb_run = None

    wandb_disabled = os.environ.get(
        "WANDB_DISABLED",
        "",
    ).lower() in {
        "1",
        "true",
        "yes",
    }

    if (
        not wandb_disabled
        and os.environ.get("WANDB_API_KEY")
    ):
        wandb_run = wandb.init(
            project=extra_args.wandb_project,
            name=extra_args.wandb_run_name,
        )
    else:
        os.environ.setdefault(
            "WANDB_DISABLED",
            "true",
        )
        os.environ.setdefault(
            "WANDB_MODE",
            "disabled",
        )

        logger.info(
            "Weights & Biases disabled for eval run."
        )

    logger.info(
        "Loading dataset: %s",
        data_args.test_dataset,
    )

    items_list = get_dataset(
        name=data_args.test_dataset,
        split="test",
        local_dir=data_args.data_dir,
        categories=data_args.categories,
        data_ratio=data_args.data_ratio,
        seed=extra_args.seed,
    )

    logger.info(
        "Loaded %d examples.",
        len(items_list),
    )

    dataset_ids = [
        str(item["id"])
        for item in items_list
    ]

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(
            "Duplicate IDs were found in the test dataset. "
            "Resume requires unique item IDs."
        )

    dataset_id_set = set(dataset_ids)

    os.makedirs(
        extra_args.output_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        extra_args.output_dir,
        f"{data_args.test_dataset}_results.jsonl",
    )

    summary_path = os.path.join(
        extra_args.output_dir,
        f"{data_args.test_dataset}_summary.json",
    )

    if extra_args.resume:
        _, completed_ids = load_existing_results(
            out_path
        )
    else:
        completed_ids = set()

        if os.path.exists(out_path):
            logger.warning(
                "resume=false: removing existing result file %s",
                out_path,
            )
            os.remove(out_path)

        if os.path.exists(summary_path):
            os.remove(summary_path)

    unknown_ids = completed_ids - dataset_id_set

    if unknown_ids:
        raise ValueError(
            "The existing result file contains IDs "
            "that do not exist in the current dataset. "
            f"Unknown IDs include: {sorted(unknown_ids)[:5]}. "
            "Use a different output_dir or run with "
            "--resume false."
        )

    pending_items = [
        item
        for item in items_list
        if str(item["id"]) not in completed_ids
    ]

    logger.info(
        "Resume status: completed=%d pending=%d "
        "total=%d batch_size=%d",
        len(completed_ids),
        len(pending_items),
        len(items_list),
        extra_args.eval_batch_size,
    )

    is_math_dataset = (
        data_args.test_dataset.lower()
        in MATH_DATASETS
    )

    logger.info(
        "Scoring as %s dataset.",
        "math"
        if is_math_dataset
        else "coding",
    )

    if pending_items:
        logger.info(
            "Loading model: %s (mode=%s)",
            model_args.model_name_or_path,
            extra_args.inference_mode,
        )

        try:
            llm = LLMService(
                model_name=(
                    model_args.model_name_or_path
                ),
                mode=extra_args.inference_mode,
                max_model_len=(
                    extra_args.max_model_len
                ),
                lora_path=(
                    model_args.finetuned_lora_path
                ),
                tp_size=(
                    extra_args.tensor_parallel_size
                ),
                gpu_memory_utilization=(
                    extra_args.gpu_memory_utilization
                ),
            )
        except Exception:
            logger.exception(
                "Model initialization failed for %s "
                "in %s mode.",
                model_args.model_name_or_path,
                extra_args.inference_mode,
            )
            raise

        total_pending = len(pending_items)

        for batch_start in range(
            0,
            total_pending,
            extra_args.eval_batch_size,
        ):
            batch_end = (
                batch_start
                + extra_args.eval_batch_size
            )

            batch_items = pending_items[
                batch_start:batch_end
            ]

            batch_prompts = [
                llm.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": item[
                                "instruction"
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=(
                        extra_args.enable_thinking
                    ),
                )
                for item in batch_items
            ]

            batch_predictions = llm.generate(
                batch_prompts,
                max_tokens=(
                    extra_args.max_new_tokens
                ),
                temperature=(
                    extra_args.temperature
                ),
                top_p=extra_args.top_p,
                top_k=extra_args.top_k,
                repetition_penalty=(
                    extra_args.repetition_penalty
                ),
            )

            if (
                len(batch_predictions)
                != len(batch_items)
            ):
                raise RuntimeError(
                    "Generated prediction count does not "
                    "match batch size: "
                    f"{len(batch_predictions)} "
                    f"!= {len(batch_items)}"
                )

            batch_results: list[dict] = []

            for item, prediction in zip(
                batch_items,
                batch_predictions,
            ):
                scores = evaluate_item(
                    item,
                    prediction,
                    is_math_dataset=(
                        is_math_dataset
                    ),
                )

                batch_results.append(
                    {
                        "id": item["id"],
                        "problem_id": item.get(
                            "problem_id"
                        ),
                        "prediction": prediction,
                        "ground_truth": item[
                            "ground_truth"
                        ],
                        "category": item.get(
                            "categories",
                            [],
                        ),
                        "main_critic_category": (
                            item.get(
                                "main_critic_category"
                            )
                        ),
                        **scores,
                    }
                )

            append_batch(
                out_path,
                batch_results,
            )

            completed_ids.update(
                str(row["id"])
                for row in batch_results
            )

            completed_current_run = min(
                batch_start
                + len(batch_items),
                total_pending,
            )

            logger.info(
                "Saved batch: current_run=%d/%d "
                "overall=%d/%d output=%s",
                completed_current_run,
                total_pending,
                len(completed_ids),
                len(items_list),
                out_path,
            )

    else:
        logger.info(
            "All examples are already complete. "
            "Skipping model loading."
        )

    saved_results, _ = load_existing_results(
        out_path
    )

    result_by_id = {
        str(row["id"]): row
        for row in saved_results
    }

    results = [
        result_by_id[item_id]
        for item_id in dataset_ids
        if item_id in result_by_id
    ]

    missing_ids = [
        item_id
        for item_id in dataset_ids
        if item_id not in result_by_id
    ]

    if not missing_ids:
        atomic_write_jsonl(
            out_path,
            results,
        )

    log_dict = build_summary(results)

    log_dict["num_examples_total"] = len(
        items_list
    )
    log_dict["num_examples_missing"] = len(
        missing_ids
    )
    log_dict["is_complete"] = not missing_ids

    if wandb_run is not None:
        wandb.log(log_dict)

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            log_dict,
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.flush()
        os.fsync(handle.fileno())

    logger.info(
        "Saved results: %s",
        out_path,
    )
    logger.info(
        "Saved summary: %s",
        summary_path,
    )

    if missing_ids:
        logger.warning(
            "%d examples are still missing. "
            "Run the same command again with "
            "--resume true.",
            len(missing_ids),
        )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()