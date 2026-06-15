import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import wandb
from transformers import HfArgumentParser

from data import get_dataset
from evaluation.metrics import exact_match_score, token_f1_score, numerical_match_score, math_verify_score, mc_aware_math_score
from evaluation.scorer import score_one
from utils.helpers import extract_math_answer, set_all_seeds
from utils.llm import LLMService

logger = logging.getLogger(__name__)

MATH_DATASETS = {"bigmath", "math"}     ## 수학 데이터셋이냐 코딩 데이터셋이냐에 따라 메트릭이 달라짐.

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
        metadata={"help": "추론 백엔드: vllm (추천) 또는 hf."},
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
    
    seed: int = field(
        default=42,
        metadata={"help": "모든 시드 고정"}
    )




def evaluate_item(item: dict, prediction: str, is_math_dataset: bool=True) -> dict:
    if is_math_dataset:
        extracted = extract_math_answer(prediction)
        scores = {
            "extracted_answer": extracted,
            "exact_match_score": exact_match_score(extracted, item["ground_truth"]),
            "token_f1_score": token_f1_score(extracted, item["ground_truth"]),
            "numerical_match_score": numerical_match_score(extracted, item["ground_truth"]),
            "math_verify_score": math_verify_score(extracted, item["ground_truth"]),
            # MC-aware: 객관식 보기↔값 복수정답(??)
            "math_verify_mc_score": mc_aware_math_score(extracted, item["ground_truth"], item.get("instruction", "")),
        }

    else:
        scores = {"pass_score": score_one(item, prediction)}
    return scores






def main():
    # 인자 처리
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments),
        description="Evaluate Script",
    )
    model_args, data_args, extra_args = parser.parse_args_into_dataclasses()

    # 로그 및 시드 설정
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    set_all_seeds(seed=extra_args.seed)
    
    # wandb
    wandb.init(project=extra_args.wandb_project, name=extra_args.wandb_run_name)
    
    
    
    # 데이터셋 로드
    logger.info(f"데이터셋 로딩: {data_args.test_dataset}")
    items_list = get_dataset(
        name=data_args.test_dataset,
        split="test",
        local_dir=None,
        categories=data_args.categories,
        data_ratio=data_args.data_ratio,
    )
    logger.info(f"총 {len(items_list)}개 예제 로드 완료.")
    
    
    
    # 모델 + LoRA 로드
    logger.info(f"모델 로딩: {model_args.model_name_or_path} (mode={extra_args.inference_mode})")
    llm = LLMService(
        model_name=model_args.model_name_or_path,
        mode=extra_args.inference_mode,
        max_model_len=extra_args.max_model_len,
        lora_path=model_args.finetuned_lora_path,
    )
    
    
    # 배치 예측 생성
    logger.info("프롬프트 구성 중...")
    prompts = [
        llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in items_list
    ]

    logger.info(f"{len(prompts)}개 프롬프트 배치 생성 시작...")
    predictions = llm.generate(
        prompts,
        max_tokens=extra_args.max_new_tokens,
        temperature=extra_args.temperature,
    )
    
    
    
    # 채점
    if data_args.test_dataset.lower() in {"math", "bigmath"}:
        is_math_dataset = True
        logger.info("수학 데이터셋 채점 중...")
    else: 
        logger.info("코딩 데이터셋 채점 중...")
        is_math_dataset = False
    
    
    results = []
    for item, prediction in zip(items_list, predictions):
        scores = evaluate_item(item, prediction, is_math_dataset=is_math_dataset)
        results.append({
            "id": item["id"],
            "prediction": prediction,
            "ground_truth": item["ground_truth"],
            "category": item.get("categories", []),
            **scores,
        })

    # 결과 저장
    os.makedirs(extra_args.output_dir, exist_ok=True)
    out_path = os.path.join(extra_args.output_dir, f"{data_args.test_dataset}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    
    # 집계 + 로깅
    def _avg(rows: list, key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    score_keys = [k for k in results[0] if k.endswith("_score")]
        

    # 전체 평균
    overall = {k: _avg(results, k) for k in score_keys}
    log_dict = {"num_examples": len(results), **{f"overall/{k}": v for k, v in overall.items()}}
    logger.info(f"[전체] {len(results)}개  " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items()))

    # 카테고리별 평균
    cat_groups: dict = defaultdict(list)
    for r in results:
        cats = r.get("category", [])
    
        for cat in cats:
            cat_groups[cat].append(r)

    if cat_groups:
        logger.info("── 카테고리별 ──")
        for cat in sorted(cat_groups):
            rows = cat_groups[cat]
            cat_avg = {k: _avg(rows, k) for k in score_keys}
            log_dict.update({f"{cat}/{k}": v for k, v in cat_avg.items()})
            logger.info(f"  [{cat}] {len(rows)}개  " + "  ".join(f"{k}={v:.4f}" for k, v in cat_avg.items()))

    wandb.log(log_dict)
    logger.info(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
