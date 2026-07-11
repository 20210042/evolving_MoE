import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import wandb
from transformers import HfArgumentParser

from data import get_dataset
from evaluation.metrics import (
    exact_match_score,
    mc_score,
    math_verify_score,
    numerical_match_score,
    token_f1_score,
)
from evaluation.scorer import score_one
from prompts.coding import build_baseline_prompt
from prompts.math import build_generation_prompt as build_math_prompt
from utils.domains import task_family
from utils.helpers import extract_math_answer, set_all_seeds

logger = logging.getLogger(__name__)

MATH_DATASETS = {"bigmath", "math", "numina_cot"}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "HuggingFace 모델 ID 또는 로컬 경로."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "모델 data type: bfloat16 (추천), float16, float32."},
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={"help": "Attention 구현체: flash_attention_2, sdpa 등. None이면 기본값."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "원격 코드 신뢰 여부. EXAONE 등 커스텀 모델에 필요."},
    )
    finetuned_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "SFT LoRA 체크포인트 경로. 제공 시 LoRA 어댑터를 붙여서 추론."},
    )


@dataclass
class DataArguments:
    test_dataset: str = field(
        default="bigmath",
        metadata={"help": "평가 데이터셋 이름."},
    )
    split: str = field(
        default="test",
        metadata={"help": "평가 split. QASC/LBox 로컬 jsonl은 <dataset>_<split>.jsonl을 찾음."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "로컬 데이터 디렉토리. QASC/LBox는 보통 export/qasc, export/lbox."},
    )
    categories: Optional[List[str]] = field(
        default=None,
        metadata={"help": "카테고리 필터 (e.g. --categories calculus algebra). None이면 전체."},
    )
    data_ratio: float = field(
        default=1.0,
        metadata={"help": "데이터 사용 비율 (0.0~1.0). 0.1이면 10%만 사용."},
    )


@dataclass
class ExtraArguments:
    inference_mode: str = field(
        default="vllm",
        metadata={"help": "추론 백엔드: vllm (추천), hf, 또는 api."},
    )
    max_model_len: int = field(
        default=32768,
        metadata={"help": "vLLM max_model_len."},
    )
    max_new_tokens: int = field(
        default=4096,
        metadata={"help": "생성할 최대 토큰 수."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "생성 temperature. 0이면 greedy."},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "결과 JSONL 파일 저장 디렉토리."},
    )
    wandb_run_name: str = field(
        default="eval",
        metadata={"help": "WandB run name."},
    )
    wandb_project: str = field(
        default="evolving-moe",
        metadata={"help": "WandB 프로젝트 이름."},
    )
    seed: int = field(default=42, metadata={"help": "모든 시드 고정"})
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "reasoning 활성화. vllm/hf: chat template enable_thinking=True, api: reasoning_effort=medium."},
    )
    use_category_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "numina_cot 평가 시 전체 run에 고정해서 사용할 category persona 이름."},
    )


def evaluate_item(item: dict, prediction: str, is_math_dataset: bool = True) -> dict:
    if is_math_dataset:
        extracted = extract_math_answer(prediction)
        problem_text = item.get("instruction", "")
        ground_truth = item["ground_truth"]

        em = exact_match_score(extracted, ground_truth)
        numerical = numerical_match_score(extracted, ground_truth)
        math_verify = math_verify_score(extracted, ground_truth)
        multiple_choice = mc_score(extracted, ground_truth, problem_text)
        combined = 1.0 if max(em, numerical, math_verify, multiple_choice) >= 1.0 else 0.0
        token_f1 = token_f1_score(extracted, ground_truth)

        return {
            "extracted_answer": extracted,
            "em_score": em,
            "token_f1_score": token_f1,
            "numerical_match_score": numerical,
            "math_verify_score": math_verify,
            "multiple_choice_score": multiple_choice,
            "combined_score": combined,
        }

    return {"pass_score": score_one(item, prediction)}


def _safe_filename_component(value: str) -> str:
    value = str(value or "").strip().rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "unknown"


def _job_id_for_output() -> str:
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or f"local{os.getpid()}"
    )


def build_output_path(output_dir: str, dataset_name: str, model_name: str, lora_path: str | None = None) -> str:
    model_component = _safe_filename_component(lora_path or model_name)
    dataset_component = _safe_filename_component(dataset_name)
    job_id = _safe_filename_component(_job_id_for_output())
    filename = f"{dataset_component}_{model_component}_{job_id}.jsonl"
    return os.path.join(output_dir, filename)


def build_eval_prompt(item: dict, dataset_name: str, model_name: str, fixed_math_category: str | None = None):
    dataset_key = (item.get("dataset") or dataset_name).lower()
    family = task_family(dataset=dataset_key, domain=item.get("domain"))
    instruction = item["instruction"]

    if family == "math":
        metadata = {"category": fixed_math_category} if fixed_math_category else item
        return build_math_prompt(instruction, metadata=metadata)

    return build_baseline_prompt(
        instruction,
        dataset=dataset_key,
        model_name=model_name,
        starter_code=item.get("starter_code"),
        domain=item.get("domain"),
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments),
        description="Evaluate Script",
    )
    model_args, data_args, extra_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    set_all_seeds(seed=extra_args.seed)
    wandb_run = wandb.init(project=extra_args.wandb_project, name=extra_args.wandb_run_name)

    logger.info("데이터셋 로딩: %s/%s", data_args.test_dataset, data_args.split)
    items_list = get_dataset(
        name=data_args.test_dataset,
        split=data_args.split,
        local_dir=data_args.data_dir,
        categories=data_args.categories,
        data_ratio=data_args.data_ratio,
        seed=extra_args.seed,
    )
    logger.info("총 %d개 예제 로드 완료.", len(items_list))

    logger.info("=" * 60)
    logger.info("평가 설정")
    logger.info("  모델       : %s", model_args.model_name_or_path)
    logger.info("  LoRA       : %s", model_args.finetuned_lora_path or "없음")
    logger.info("  inference mode  : %s", extra_args.inference_mode)
    logger.info("  enable thinking : %s", extra_args.enable_thinking)
    logger.info("  데이터셋   : %s/%s (ratio=%s)", data_args.test_dataset, data_args.split, data_args.data_ratio)
    logger.info("  data_dir   : %s", data_args.data_dir or "HF/default")
    logger.info("  max_tokens : %s", extra_args.max_new_tokens)
    logger.info("  출력 디렉토리: %s", extra_args.output_dir)
    logger.info("  job id     : %s", _job_id_for_output())
    logger.info("=" * 60)

    from utils.llm import LLMService

    llm = LLMService(
        model_name=model_args.model_name_or_path,
        mode=extra_args.inference_mode,
        max_model_len=extra_args.max_model_len,
        lora_path=model_args.finetuned_lora_path,
    )

    os.makedirs(extra_args.output_dir, exist_ok=True)
    out_path = build_output_path(
        extra_args.output_dir,
        data_args.test_dataset,
        model_args.model_name_or_path,
        model_args.finetuned_lora_path,
    )
    logger.info("결과 파일: %s", out_path)

    done_ids: set = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        logger.info("기존 결과 %d개 발견 - 건너뜀.", len(done_ids))

    items_todo = [item for item in items_list if item["id"] not in done_ids]
    logger.info("처리할 항목: %d개 / 전체: %d개", len(items_todo), len(items_list))

    dataset_key = data_args.test_dataset.lower()
    family = task_family(dataset=dataset_key, domain=(items_list[0].get("domain") if items_list else None))
    is_math_dataset = family == "math"
    use_numina_category_prompt = bool(extra_args.use_category_prompt) and dataset_key == "numina_cot"
    logger.info("task family: %s", family)
    logger.info("카테고리별 system prompt 고정값: %s", extra_args.use_category_prompt or "없음")

    chunk_size = 1000
    with open(out_path, "a", encoding="utf-8") as out_f:
        for chunk_start in range(0, len(items_todo), chunk_size):
            chunk = items_todo[chunk_start : chunk_start + chunk_size]
            messages_chunk = [
                build_eval_prompt(
                    item,
                    dataset_name=data_args.test_dataset,
                    model_name=model_args.model_name_or_path,
                    fixed_math_category=(extra_args.use_category_prompt if use_numina_category_prompt else None),
                )
                for item in chunk
            ]

            logger.info("예측 중: %d~%d / %d", chunk_start + 1, chunk_start + len(chunk), len(items_todo))
            predictions = llm.chat_batch(
                messages_chunk,
                max_tokens=extra_args.max_new_tokens,
                temperature=extra_args.temperature if extra_args.inference_mode != "api" else None,
                enable_thinking=extra_args.enable_thinking,
            )

            for item, messages, prediction in zip(chunk, messages_chunk, predictions):
                scores = evaluate_item(item, prediction, is_math_dataset=is_math_dataset)
                out_f.write(json.dumps({
                    "id": item["id"],
                    "input": messages,
                    "prediction": prediction,
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category") or item.get("categories", []),
                    "dataset": item.get("dataset") or data_args.test_dataset,
                    "domain": item.get("domain"),
                    **scores,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            logger.info("청크 저장 완료: %d/%d", chunk_start + len(chunk), len(items_todo))

    results = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    if not results:
        logger.warning("결과가 없습니다: %s", out_path)
        return

    def _avg(rows: list, key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    def _wandb_key(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")

    score_keys = [k for k in results[0] if k.endswith("_score")]
    overall = {k: _avg(results, k) for k in score_keys}
    summary_dict = {
        "overall/num_examples": len(results),
        **{f"overall/{k}": v for k, v in overall.items()},
    }
    logger.info("[전체] %d개  %s", len(results), "  ".join(f"{k}={v:.4f}" for k, v in overall.items()))

    cat_groups: dict = defaultdict(list)
    for r in results:
        cats = r.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            cat_groups[cat].append(r)

    for cat in sorted(cat_groups):
        rows = cat_groups[cat]
        cat_avg = {k: _avg(rows, k) for k in score_keys}
        cat_key = _wandb_key(cat)
        summary_dict[f"{cat_key}/num_examples"] = len(rows)
        summary_dict.update({f"{cat_key}/{k}": v for k, v in cat_avg.items()})
        logger.info("  [%s] %d개  %s", cat, len(rows), "  ".join(f"{k}={v:.4f}" for k, v in cat_avg.items()))

    wandb_run.summary.update(summary_dict)
    wandb.log(summary_dict)
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
