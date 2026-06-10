import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from data import get_dataset
from prompts.math import build_generation_prompt as build_math_prompt
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
    categories: Optional[List[str]] = field(
        default=None,
        metadata={"help": "카테고리 필터 (e.g. --categories calculus algebra). None이면 전체."},
    )
    data_ratio: float = field(
        default=1.0,
        metadata={"help": "학습 데이터 사용 비율 (0.0~1.0). 0.1이면 10%만 사용."},
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
        metadata={"help": "LoRA 스케일링 팩터 (α). α/r이 실제 스케일링. 추천: rank의 1~2배."},
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


def build_hf_dataset(
    name: str,
    split: str,
    tokenizer: AutoTokenizer,
    data_ratio: float = 1.0,
    seed: Optional[int] = None,
    categories: Optional[str] = None,
) -> Dataset:
    items = get_dataset(name, split=split, categories=categories, data_ratio=data_ratio, seed=seed)
    is_math = name.lower() in {"bigmath", "math", "numina_cot"}

    rows = []
    for item in items:
        prompt_messages = (
            build_math_prompt(item["instruction"])
            if is_math
            else [{"role": "user", "content": item["instruction"]}]
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        completion_text = str(item.get("solution", item["ground_truth"]))
        if tokenizer.eos_token and not completion_text.endswith(tokenizer.eos_token):
            completion_text += tokenizer.eos_token

        rows.append(
            {
                "prompt": prompt_text,
                # 문자열 prompt+completion으로 두면 TRL의 completion mask prefix 가정이 깨지지 않는다.
                "completion": completion_text,
            }
        )

    return Dataset.from_list(rows)


def main():
    
    # 인자 처리
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments, SFTConfig),
        description="SFT Training Script",
    )
    model_args, data_args, extra_args, sft_config = parser.parse_args_into_dataclasses()
    
    
    
    # WandB
    sft_config.report_to = ["wandb"]
    os.environ.setdefault("WANDB_PROJECT", extra_args.wandb_project)
    if sft_config.run_name:
        os.environ.setdefault("WANDB_RUN_NAME", sft_config.run_name)
    
    
    
    # 로깅 및 시드 설정
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if sft_config.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info(f"모델 인자: {model_args}")
    logger.info(f"데이터 인자: {data_args}")
    logger.info(f"학습 인자: {sft_config}")
    
    set_all_seeds(sft_config.seed)
    
    
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터 로드
    logger.info(f"학습 데이터셋 로딩: {data_args.train_dataset}")
    train_dataset = build_hf_dataset(
        data_args.train_dataset,
        split="train",
        tokenizer=tokenizer,
        data_ratio=data_args.data_ratio,
        seed=sft_config.seed,
        categories=data_args.categories,
    )
    logger.info(f"학습 데이터셋: {len(train_dataset)}개 예제")


    logger.info(f"평가 데이터셋 로딩: {data_args.eval_dataset}")
    eval_dataset = build_hf_dataset(
        data_args.eval_dataset,
        split="validation",
        tokenizer=tokenizer,
        seed=sft_config.seed,
        categories=data_args.categories,
    )
    logger.info(f"평가 데이터셋: {len(eval_dataset)}개 예제")



    # 모델 로드
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.finetuned_lora_path:
        logger.info(f"파인튜닝된 LoRA 병합: {model_args.finetuned_lora_path}")
        model = PeftModel.from_pretrained(model, model_args.finetuned_lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA 병합 완료.")
    
    # SFT LoRA 설정
    sft_peft_config = None
    if extra_args.train_sft_with_lora:
        sft_peft_config = LoraConfig(
            r=extra_args.sft_lora_rank,
            lora_alpha=extra_args.sft_lora_alpha,
            lora_dropout=extra_args.sft_lora_dropout,
            target_modules=extra_args.sft_lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
    
        
    
    # Trainer 정의
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=sft_peft_config,
        args=sft_config,
    )
    
    
    ## 학습 시작
    logger.info("=" * 60)
    logger.info("SFT 학습 시작")
    logger.info(f"  모델: {model_args.model_name_or_path}")
    logger.info(f"  LoRA: {extra_args.train_sft_with_lora}, rank={extra_args.sft_lora_rank}, alpha={extra_args.sft_lora_alpha}")
    logger.info(f"  학습 데이터셋: {data_args.train_dataset}, 비율: {data_args.data_ratio}, 예제 수: {len(train_dataset)}")
    logger.info(f"  에폭: {sft_config.num_train_epochs}, LR: {sft_config.learning_rate}")
    logger.info(f"  배치: {sft_config.per_device_train_batch_size}, GA: {sft_config.gradient_accumulation_steps}")
    logger.info(f"  출력 디렉토리: {sft_config.output_dir}")
    logger.info("=" * 60)

    train_result = trainer.train()

    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"학습 완료! 모델 저장 위치: {sft_config.output_dir}")


if __name__ == "__main__":
    main()
