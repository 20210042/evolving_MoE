"""ACC algorithm SFT training entry point.

This is the algorithm-specific fork of the original SFT trainer. It loads the balanced ACC algorithm
JSONL splits, formats each problem/reference solution as chat SFT data, supports Gemma/Llama causal
LMs with optional LoRA, infers a large-model device map, resumes from the latest checkpoint, and
writes HuggingFace/TRL trainer outputs under the requested checkpoint directory."""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from huggingface_hub import hf_hub_download
from accelerate import infer_auto_device_map, init_empty_weights
import transformers.modeling_utils as modeling_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, HfArgumentParser, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from data.loader_algorithm import get_dataset
from utils.helpers import set_all_seeds

logger = logging.getLogger(__name__)


def latest_checkpoint(output_dir: str) -> Optional[str]:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = []
    for child in path.iterdir():
        if child.is_dir():
            match = re.fullmatch(r"checkpoint-(\d+)", child.name)
            if match:
                checkpoints.append((int(match.group(1)), child))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def build_algorithm_device_map(model_name_or_path: str, trust_remote_code: bool, device_map: Optional[str]):
    if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device_map

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    max_memory = {"cpu": os.environ.get("ACC_CPU_MAX_MEMORY", "128GiB")}
    for device_index in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(device_index).total_memory // 1024**3
        reserve_gib = int(os.environ.get("ACC_GPU_MEMORY_RESERVE_GIB", "4"))
        max_memory[device_index] = f"{max(1, total_gib - reserve_gib)}GiB"
    inferred = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        dtype=torch.bfloat16,
        clean_result=False,
    )
    first_device = 0 if torch.cuda.device_count() else "cpu"
    for name in ["model.vision_tower.std_bias", "model.vision_tower.std_scale"]:
        if name not in inferred:
            inferred[name] = first_device
    return inferred


def load_algorithm_processor(model_name_or_path: str, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    except (ValueError, OSError) as error:
        logger.warning(
            "AutoProcessor failed for %s (%s). Falling back to tokenizer.json.",
            model_name_or_path,
            error,
        )

    tokenizer_json = hf_hub_download(model_name_or_path, "tokenizer.json")
    tokenizer_config_path = hf_hub_download(model_name_or_path, "tokenizer_config.json")
    try:
        chat_template_path = hf_hub_download(model_name_or_path, "chat_template.jinja")
    except Exception:
        chat_template_path = None
    with open(tokenizer_config_path, "r", encoding="utf-8") as config_file:
        tokenizer_config = json.load(config_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token=tokenizer_config.get("bos_token", "<bos>"),
        eos_token=tokenizer_config.get("eos_token", "<eos>"),
        unk_token=tokenizer_config.get("unk_token", "<unk>"),
        pad_token=tokenizer_config.get("pad_token", "<pad>"),
    )
    if chat_template_path:
        with open(chat_template_path, "r", encoding="utf-8") as template_file:
            tokenizer.chat_template = template_file.read()
    elif tokenizer_config.get("chat_template"):
        tokenizer.chat_template = tokenizer_config["chat_template"]
    extra_special_tokens = tokenizer_config.get("extra_special_tokens") or []
    if extra_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": extra_special_tokens})
    return tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-4-31B-it",
        metadata={"help": "HuggingFace model ID or local path."},
    )
    dtype: str = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="eager")
    trust_remote_code: bool = field(default=True)
    finetuned_lora_path: Optional[str] = field(default=None)
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Use auto to shard Gemma 31B across visible GPUs."},
    )


@dataclass
class DataArguments:
    train_dataset: str = field(default="acc_algorithm")
    eval_dataset: str = field(default="acc_algorithm")
    data_dir: Optional[str] = field(default="/home/minjikim/minji_link/evolving_MoE/data/acc_algorithm")
    categories: Optional[List[str]] = field(default=None)
    data_ratio: float = field(default=1.0)


@dataclass
class ExtraArguments:
    train_sft_with_lora: bool = field(default=False)
    sft_lora_rank: int = field(default=64)
    sft_lora_alpha: int = field(default=128)
    sft_lora_dropout: float = field(default=0.05)
    sft_lora_target_modules: str = field(default="all-linear")
    wandb_project: str = field(default="evolving-moe-sft")


def build_hf_dataset(
    name: str,
    split: str,
    processor,
    data_ratio: float = 1.0,
    seed: Optional[int] = None,
    categories: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
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
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": item["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["ground_truth"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        completion = full_text[len(prompt) :] if full_text.startswith(prompt) else item["ground_truth"]
        rows.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(rows)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExtraArguments, SFTConfig),
        description="ACC Algorithm SFT Training Script",
    )
    model_args, data_args, extra_args, sft_config = parser.parse_args_into_dataclasses()

    if os.environ.get("WANDB_API_KEY"):
        sft_config.report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", extra_args.wandb_project)
        if sft_config.run_name:
            os.environ.setdefault("WANDB_RUN_NAME", sft_config.run_name)
    else:
        sft_config.report_to = []
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("WANDB_MODE", "disabled")

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if sft_config.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info(f"model_args={model_args}")
    logger.info(f"data_args={data_args}")
    logger.info(f"sft_config={sft_config}")

    set_all_seeds(sft_config.seed)

    processor = load_algorithm_processor(
        model_args.model_name_or_path,
        model_args.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading train dataset: {data_args.train_dataset}")
    train_dataset = build_hf_dataset(
        data_args.train_dataset,
        split="train",
        processor=processor,
        data_ratio=data_args.data_ratio,
        seed=sft_config.seed,
        categories=data_args.categories,
        data_dir=data_args.data_dir,
    )
    logger.info(f"Train examples: {len(train_dataset)}")

    logger.info(f"Loading eval dataset: {data_args.eval_dataset}")
    eval_dataset = build_hf_dataset(
        data_args.eval_dataset,
        split="validation",
        processor=processor,
        seed=sft_config.seed,
        categories=data_args.categories,
        data_dir=data_args.data_dir,
    )
    logger.info(f"Eval examples: {len(eval_dataset)}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.dtype, torch.bfloat16)

    device_map = build_algorithm_device_map(
        model_args.model_name_or_path,
        model_args.trust_remote_code,
        model_args.device_map,
    )
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    if model_args.finetuned_lora_path:
        logger.info(f"Merging existing LoRA: {model_args.finetuned_lora_path}")
        model = PeftModel.from_pretrained(model, model_args.finetuned_lora_path)
        model = model.merge_and_unload()

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

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=sft_peft_config,
        args=sft_config,
    )

    logger.info("=" * 60)
    logger.info("ACC Algorithm SFT start")
    logger.info(f"  model: {model_args.model_name_or_path}")
    logger.info(f"  categories: {data_args.categories or 'ALL'}")
    logger.info(f"  LoRA: {extra_args.train_sft_with_lora}, rank={extra_args.sft_lora_rank}, alpha={extra_args.sft_lora_alpha}")
    logger.info(f"  epochs: {sft_config.num_train_epochs}, lr: {sft_config.learning_rate}")
    logger.info(f"  output_dir: {sft_config.output_dir}")
    logger.info("=" * 60)

    resume_checkpoint = latest_checkpoint(sft_config.output_dir)
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training complete: {sft_config.output_dir}")


if __name__ == "__main__":
    main()
