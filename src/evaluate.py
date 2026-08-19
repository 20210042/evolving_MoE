import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
LUCA_SYSTEM_PROMPT = "You are a helpful assistant."
PROMPT_SYSTEM_CHOICES = {"baseline", "luca", "category"}
SYSTEM_PROMPT_PRESETS = PROMPT_SYSTEM_CHOICES


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
        metadata={"help": "데이터 사용 비율 (0.0~1.0). 0.1이면 10 percent만 사용."},
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
    wandb_entity: str = field(
        default="jongbin-kr-skiml_moe",
        metadata={"help": "WandB entity 이름."},
    )
    seed: int = field(default=42, metadata={"help": "모든 시드 고정"})
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "reasoning 활성화. vllm/hf: chat template enable_thinking=True, api: reasoning_effort=medium."},
    )
    use_category_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "category prompt 모드에서 전체 run에 고정할 math category 이름. 생략 시 item metadata 사용."},
    )
    prompt_system: str = field(
        default="baseline",
        metadata={"help": "Deprecated alias for --system_prompt. Preset name: baseline | luca | category."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "System prompt preset name or literal prompt text. Presets: baseline | luca | category. "
                "Any other string is used directly as the system prompt."
            )
        },
    )
    luca_roster_path: str = field(
        default="configs/roster_init.json",
        metadata={"help": "system_prompt=luca일 때 LUCA system_prompt를 읽을 roster json 경로."},
    )
    fixed_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated alias for passing literal text to --system_prompt."},
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


def build_output_path(
    output_dir: str,
    dataset_name: str,
    model_name: str,
    lora_path: str | None = None,
    system_prompt_label: str | None = None,
) -> str:
    model_component = _safe_filename_component(lora_path or model_name)
    dataset_component = _safe_filename_component(dataset_name)
    prompt_component = _safe_filename_component(system_prompt_label or "baseline")
    job_id = _safe_filename_component(_job_id_for_output())
    filename = f"{dataset_component}_{model_component}_{prompt_component}_{job_id}.jsonl"
    return os.path.join(output_dir, filename)


def load_luca_system_prompt(roster_path: str | None = None) -> str:
    """Load LUCA's system prompt from the original roster file, with a stable fallback."""
    if not roster_path:
        return LUCA_SYSTEM_PROMPT

    path = Path(roster_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        roster = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        logger.warning("LUCA roster를 읽지 못해 기본값 사용: %s (%s)", path, exc)
        return LUCA_SYSTEM_PROMPT

    if not isinstance(roster, list):
        logger.warning("LUCA roster 형식이 list가 아니어서 기본값 사용: %s", path)
        return LUCA_SYSTEM_PROMPT

    for agent in roster:
        if not isinstance(agent, dict):
            continue
        if str(agent.get("id", "")).lower() == "luca" or str(agent.get("name", "")).lower() == "luca":
            prompt = str(agent.get("system_prompt") or "").strip()
            return prompt or LUCA_SYSTEM_PROMPT

    logger.warning("LUCA agent를 roster에서 찾지 못해 기본값 사용: %s", path)
    return LUCA_SYSTEM_PROMPT


def _override_system_prompt(messages, system_prompt: str):
    if isinstance(messages, list) and messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": system_prompt}, *messages[1:]]
    if isinstance(messages, list):
        return [{"role": "system", "content": system_prompt}, *messages]
    return messages


def build_eval_prompt(
    item: dict,
    dataset_name: str,
    model_name: str,
    fixed_math_category: str | None = None,
    system_prompt: str = "baseline",
    luca_system_prompt: str = LUCA_SYSTEM_PROMPT,
):
    dataset_key = (item.get("dataset") or dataset_name).lower()
    family = task_family(dataset=dataset_key, domain=item.get("domain"))
    instruction = item["instruction"]
    system_prompt_raw = system_prompt or "baseline"
    system_prompt_key = system_prompt_raw.lower()
    is_preset = system_prompt_key in SYSTEM_PROMPT_PRESETS
    literal_system_prompt = None if is_preset else system_prompt_raw

    if family == "math":
        if system_prompt_key == "luca":
            messages = _override_system_prompt(build_math_prompt(instruction, metadata=item), luca_system_prompt)
            return _override_system_prompt(messages, literal_system_prompt) if literal_system_prompt else messages
        metadata = {"category": fixed_math_category} if fixed_math_category else item
        if system_prompt_key == "baseline":
            metadata = None
        messages = build_math_prompt(instruction, metadata=metadata)
        return _override_system_prompt(messages, literal_system_prompt) if literal_system_prompt else messages

    messages = build_baseline_prompt(
        instruction,
        dataset=dataset_key,
        model_name=model_name,
        starter_code=item.get("starter_code"),
        domain=item.get("domain"),
    )
    if system_prompt_key == "luca":
        messages = _override_system_prompt(messages, luca_system_prompt)
    if literal_system_prompt:
        return _override_system_prompt(messages, literal_system_prompt)
    return messages


def resolve_system_prompt_spec(extra_args: ExtraArguments) -> tuple[str, str, bool]:
    """Return (system prompt spec, compact filename/log label, is_preset)."""
    spec = extra_args.system_prompt
    if spec is None:
        spec = extra_args.fixed_system_prompt or extra_args.prompt_system
    spec = spec or "baseline"
    key = spec.lower()
    if key in SYSTEM_PROMPT_PRESETS:
        return key, key, True
    return spec, "custom_system", False


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments),
        description="Evaluate Script",
    )
    model_args, data_args, extra_args = parser.parse_args_into_dataclasses()
    system_prompt_spec, system_prompt_label, system_prompt_is_preset = resolve_system_prompt_spec(extra_args)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    set_all_seeds(seed=extra_args.seed)
    os.environ["WANDB_ENTITY"] = extra_args.wandb_entity
    os.environ["WANDB_PROJECT"] = extra_args.wandb_project
    wandb_run = wandb.init(
        entity=extra_args.wandb_entity,
        project=extra_args.wandb_project,
        name=extra_args.wandb_run_name,
    )

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
    logger.info("  system prompt: %s", system_prompt_label if system_prompt_is_preset else system_prompt_spec)
    if system_prompt_label == "luca":
        logger.info("  LUCA roster: %s", extra_args.luca_roster_path)
    elif system_prompt_label == "category":
        logger.info("  category prompt fixed value: %s", extra_args.use_category_prompt or "item metadata")
    logger.info("  출력 디렉토리: %s", extra_args.output_dir)
    logger.info("  job id     : %s", _job_id_for_output())
    logger.info("=" * 60)

    luca_system_prompt = load_luca_system_prompt(extra_args.luca_roster_path)
    if system_prompt_label == "luca":
        logger.info("LUCA system prompt: %s", luca_system_prompt)

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
        system_prompt_label,
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
    use_fixed_category_prompt = bool(extra_args.use_category_prompt) and system_prompt_label == "category"
    logger.info("task family: %s", family)
    logger.info("system prompt 선택: %s", system_prompt_label if system_prompt_is_preset else "literal")

    chunk_size = 1000
    with open(out_path, "a", encoding="utf-8") as out_f:
        for chunk_start in range(0, len(items_todo), chunk_size):
            chunk = items_todo[chunk_start : chunk_start + chunk_size]
            messages_chunk = [
                build_eval_prompt(
                    item,
                    dataset_name=data_args.test_dataset,
                    model_name=model_args.model_name_or_path,
                    fixed_math_category=(extra_args.use_category_prompt if use_fixed_category_prompt else None),
                    system_prompt=system_prompt_spec,
                    luca_system_prompt=luca_system_prompt,
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
                    "prompt_system": system_prompt_label,
                    "system_prompt_spec": system_prompt_spec,
                    "system_prompt": (
                        messages[0].get("content")
                        if isinstance(messages, list) and messages and messages[0].get("role") == "system"
                        else None
                    ),
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
