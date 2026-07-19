import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from data import get_dataset
from prompts.coding import build_baseline_prompt
from prompts.math import build_generation_prompt as build_math_prompt
from utils.domains import task_family
from utils.helpers import set_all_seeds

logger = logging.getLogger(__name__)


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
        metadata={"help": "SFT LoRA 체크포인트 경로. 제공 시 베이스 모델에 merge 후 SFT LoRA 초기화."},
    )


@dataclass
class DataArguments:
    train_dataset: str = field(
        default="bigmath",
        metadata={"help": "학습 데이터셋 이름."},
    )
    eval_dataset: str = field(
        default="bigmath",
        metadata={"help": "평가 데이터셋 이름."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "학습 split. 로컬 jsonl은 <dataset>_<split>.jsonl을 찾음."},
    )
    eval_split: str = field(
        default="validation",
        metadata={"help": "평가 split. 로컬 jsonl은 <dataset>_<split>.jsonl을 찾음."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "로컬 데이터 디렉토리. QASC/LBox는 보통 export/qasc, export/lbox."},
    )
    eval_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "평가용 로컬 데이터 디렉토리. 생략 시 data_dir 사용."},
    )
    categories: Optional[List[str]] = field(
        default=None,
        metadata={"help": "카테고리 필터 (e.g. --categories calculus algebra). None이면 전체."},
    )
    data_ratio: float = field(
        default=1.0,
        metadata={"help": "학습 데이터 사용 비율 (0.0~1.0). 0.1이면 10%만 사용."},
    )
    label_package: Optional[str] = field(
        default=None,
        metadata={"help": "전문가별 SFT용 export/<domain>_binning_seed*/ 패키지 경로."},
    )
    expert_id: Optional[str] = field(
        default=None,
        metadata={"help": "label_package에서 사용할 expert id. 'best'면 train_pass_at_1 최고 전문가."},
    )
    source_jsonl: Optional[str] = field(
        default=None,
        metadata={"help": "label_package 라벨과 id-join할 원본 train jsonl. 생략 시 summary.json의 source_train_jsonl."},
    )
    max_n_solved: Optional[int] = field(
        default=None,
        metadata={"help": "label_package 필터: n_solved <= 이 값인 문제만 사용(전원이 푸는 쉬운 공통문제 제외). None이면 전부."},
    )
    min_n_solved: Optional[int] = field(
        default=None,
        metadata={"help": "label_package 필터: n_solved >= 이 값인 문제만 사용. shared 어댑터(공통 core) 학습용."},
    )


@dataclass
class ExtraArguments:
    train_sft_with_lora: bool = field(
        default=False,
        metadata={"help": "LoRA 사용 여부 (Parameter-Efficient Fine-Tuning)."},
    )
    sft_lora_rank: int = field(
        default=64,
        metadata={"help": "LoRA 랭크 (r). 클수록 표현력↑, 메모리↑. 추천: 16~128."},
    )
    sft_lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA 스케일링 팩터 (alpha). 추천: rank의 1~2배."},
    )
    sft_lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA 드롭아웃 확률. 과적합 방지."},
    )
    sft_lora_target_modules: str = field(
        default="all-linear",
        metadata={"help": "LoRA 적용 대상 모듈. 'all-linear' 또는 'q_proj,v_proj' 형식."},
    )
    wandb_project: str = field(
        default="evolving-moe-sft",
        metadata={"help": "WandB 프로젝트 이름."},
    )


def stringify_completion(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_prompt_messages(item: dict, dataset_name: str, model_name: str) -> list[dict] | str:
    dataset_key = (item.get("dataset") or dataset_name).lower()
    family = task_family(dataset=dataset_key, domain=item.get("domain"))
    instruction = item["instruction"]

    if family == "math":
        metadata = item if dataset_key == "numina_cot" else None
        return build_math_prompt(instruction, metadata=metadata)

    return build_baseline_prompt(
        instruction,
        dataset=dataset_key,
        model_name=model_name,
        starter_code=item.get("starter_code"),
        domain=item.get("domain"),
    )


def build_regular_hf_dataset(
    name: str,
    split: str,
    *,
    model_name: str,
    data_dir: str | None = None,
    data_ratio: float = 1.0,
    seed: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Dataset:
    items = get_dataset(
        name,
        split=split,
        local_dir=data_dir,
        categories=categories,
        data_ratio=data_ratio,
        seed=seed,
    )

    rows = []
    for item in items:
        rows.append({
            "prompt": build_prompt_messages(item, name, model_name),
            "completion": [{"role": "assistant", "content": stringify_completion(item.get("solution", item["ground_truth"]))}],
        })
    return Dataset.from_list(rows)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_expert_id(mapping: dict[str, dict], requested: str | None) -> str:
    if not mapping:
        raise ValueError("agent_mapping.json is empty.")
    if requested in (None, "", "best"):
        return max(mapping, key=lambda aid: float(mapping[aid].get("train_pass_at_1", 0.0)))
    if requested not in mapping:
        known = ", ".join(sorted(mapping)[:20])
        raise ValueError(f"Unknown expert_id={requested!r}. Known examples: {known}")
    return requested


def source_path_from_package(package_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    summary_path = package_dir / "summary.json"
    if summary_path.is_file():
        summary = json.load(open(summary_path, encoding="utf-8"))
        src = summary.get("source_train_jsonl")
        if src:
            return Path(src)
    raise ValueError("--source_jsonl is required when summary.json lacks source_train_jsonl.")


def build_expert_label_dataset(
    *,
    package_dir: str,
    expert_id: str | None,
    source_jsonl: str | None,
    data_ratio: float,
    seed: Optional[int],
    model_name: str,
    max_n_solved: Optional[int] = None,
    min_n_solved: Optional[int] = None,
) -> tuple[Dataset, str, int]:
    package = Path(package_dir)
    labels_path = package / "binning_labels.jsonl"
    mapping_path = package / "agent_mapping.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing {labels_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing {mapping_path}")

    mapping = json.load(open(mapping_path, encoding="utf-8"))
    # shared 어댑터: 특정 expert가 아니라 "공통 core"(n_solved>=min) 전부 학습.
    # per_expert 필터 없이 n_solved 밴드만으로 선택.
    is_shared = str(expert_id).lower() in ("shared", "common")
    chosen = "shared" if is_shared else resolve_expert_id(mapping, expert_id)

    src_path = source_path_from_package(package, source_jsonl)
    source_rows = {str(r["id"]): r for r in _load_jsonl(src_path)}
    labels = _load_jsonl(labels_path)
    dataset_name = str(labels[0].get("dataset") or "") if labels else ""

    def _band_ok(ns: int) -> bool:
        return (max_n_solved is None or ns <= max_n_solved) and \
               (min_n_solved is None or ns >= min_n_solved)

    selected = [
        source_rows[str(r["id"])]
        for r in labels
        if (is_shared or int((r.get("per_expert") or {}).get(chosen, 0)) == 1)
        and _band_ok(int(r.get("n_solved", 0)))
        and str(r["id"]) in source_rows
    ]
    if seed is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(selected)
    if data_ratio < 1.0:
        selected = selected[: max(1, int(len(selected) * data_ratio))]

    # 합의(2026-07-13): 프롬프트는 baseline GEN으로 통일(페르소나 미사용, eval과 동일 경로),
    # expert 간 차이는 학습 데이터 분할뿐.
    rows = []
    for item in selected:
        rows.append({
            "prompt": build_prompt_messages(item, dataset_name, model_name),
            "completion": [{"role": "assistant", "content": stringify_completion(item.get("solution", item["ground_truth"]))}],
        })
    return Dataset.from_list(rows), chosen, len(selected)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments, SFTConfig),
        description="SFT Training Script",
    )
    model_args, data_args, extra_args, sft_config = parser.parse_args_into_dataclasses()

    sft_config.report_to = ["wandb"]
    os.environ.setdefault("WANDB_PROJECT", extra_args.wandb_project)
    os.environ.setdefault("WANDB_ENTITY", "jongbin-kr-skiml_moe")
    if sft_config.run_name:
        os.environ.setdefault("WANDB_RUN_NAME", sft_config.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if sft_config.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info("모델 인자: %s", model_args)
    logger.info("데이터 인자: %s", data_args)
    logger.info("학습 인자: %s", sft_config)

    set_all_seeds(sft_config.seed)

    if data_args.label_package:
        logger.info("전문가별 라벨 패키지 로딩: %s", data_args.label_package)
        train_dataset, chosen_expert, n_rows = build_expert_label_dataset(
            package_dir=data_args.label_package,
            expert_id=data_args.expert_id,
            source_jsonl=data_args.source_jsonl,
            data_ratio=data_args.data_ratio,
            seed=sft_config.seed,
            model_name=model_args.model_name_or_path,
            max_n_solved=data_args.max_n_solved,
            min_n_solved=data_args.min_n_solved,
        )
        logger.info("전문가 SFT 학습셋: expert=%s, examples=%d, max_n_solved=%s, min_n_solved=%s",
                    chosen_expert, n_rows, data_args.max_n_solved, data_args.min_n_solved)
    else:
        logger.info("학습 데이터셋 로딩: %s/%s", data_args.train_dataset, data_args.train_split)
        train_dataset = build_regular_hf_dataset(
            data_args.train_dataset,
            split=data_args.train_split,
            model_name=model_args.model_name_or_path,
            data_dir=data_args.data_dir,
            data_ratio=data_args.data_ratio,
            seed=sft_config.seed,
            categories=data_args.categories,
        )
        chosen_expert = None
    logger.info("학습 데이터셋: %d개 예제", len(train_dataset))

    logger.info("평가 데이터셋 로딩: %s/%s", data_args.eval_dataset, data_args.eval_split)
    eval_dataset = build_regular_hf_dataset(
        data_args.eval_dataset,
        split=data_args.eval_split,
        model_name=model_args.model_name_or_path,
        data_dir=data_args.eval_data_dir or data_args.data_dir,
        seed=sft_config.seed,
        categories=data_args.categories,
    )
    logger.info("평가 데이터셋: %d개 예제", len(eval_dataset))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.finetuned_lora_path:
        if extra_args.train_sft_with_lora:
            logger.info("기존 LoRA 이어 학습: %s", model_args.finetuned_lora_path)
            model = PeftModel.from_pretrained(
                model,
                model_args.finetuned_lora_path,
                is_trainable=True,
            )
            logger.info("기존 LoRA를 trainable adapter로 로드 완료.")
        else:
            logger.info("파인튜닝된 LoRA 병합: %s", model_args.finetuned_lora_path)
            model = PeftModel.from_pretrained(model, model_args.finetuned_lora_path)
            model = model.merge_and_unload()
            logger.info("LoRA 병합 완료.")

    sft_peft_config = None
    if extra_args.train_sft_with_lora and not model_args.finetuned_lora_path:
        sft_peft_config = LoraConfig(
            r=extra_args.sft_lora_rank,
            lora_alpha=extra_args.sft_lora_alpha,
            lora_dropout=extra_args.sft_lora_dropout,
            target_modules=extra_args.sft_lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )

    # ZeRO-3의 coalesced all-gather는 혼합 dtype 파라미터를 거부한다(TypeError).
    # TRL 기본 경로는 어댑터를 fp32로 캐스팅하므로, deepspeed일 때는 어댑터를
    # 베이스와 같은 dtype으로 직접 만들어 넘긴다.
    if sft_peft_config is not None and sft_config.deepspeed:
        model = get_peft_model(model, sft_peft_config, autocast_adapter_dtype=False)
        sft_peft_config = None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=sft_peft_config,
        args=sft_config,
    )

    logger.info("=" * 60)
    logger.info("SFT 학습 시작")
    logger.info("  모델: %s", model_args.model_name_or_path)
    logger.info(
        "  LoRA: %s, finetuned_path=%s, rank=%s, alpha=%s",
        extra_args.train_sft_with_lora,
        model_args.finetuned_lora_path or "없음",
        extra_args.sft_lora_rank,
        extra_args.sft_lora_alpha,
    )
    logger.info("  학습 데이터셋: %s, 예제 수: %d", data_args.label_package or data_args.train_dataset, len(train_dataset))
    if chosen_expert:
        logger.info("  expert_id: %s", chosen_expert)
    logger.info("  에폭: %s, LR: %s", sft_config.num_train_epochs, sft_config.learning_rate)
    logger.info("  배치: %s, GA: %s", sft_config.per_device_train_batch_size, sft_config.gradient_accumulation_steps)
    logger.info("  출력 디렉토리: %s", sft_config.output_dir)
    logger.info("=" * 60)

    train_result = trainer.train()

    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("학습 완료! 모델 저장 위치: %s", sft_config.output_dir)


if __name__ == "__main__":
    main()
