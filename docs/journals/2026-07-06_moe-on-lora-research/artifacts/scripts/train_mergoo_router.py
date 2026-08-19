import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments

from data import get_dataset
from mergoo_compat import patch_mergoo_for_llama31_lora_moe
from prompts.math import build_generation_prompt as build_math_prompt
from utils.helpers import set_all_seeds

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="checkpoints/mergoo_lora_moe_algebra_geometry_top1",
        metadata={"help": "Composed mergoo MoE-on-LoRA checkpoint path."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: bfloat16, float16, or float32."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote/custom model code."},
    )
    patch_llama3_rope: bool = field(
        default=True,
        metadata={"help": "Patch mergoo's Llama runtime to use transformers' Llama 3.1 RoPE."},
    )


@dataclass
class DataArguments:
    train_dataset: str = field(default="numina_cot")
    eval_dataset: str = field(default="numina_cot")
    categories: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Category filter. None means all categories."},
    )
    data_ratio: float = field(default=1.0)
    max_seq_length: int = field(default=2048)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class ExtraArguments:
    wandb_project: str = field(default="evolving-moe-router")
    save_full_model: bool = field(
        default=False,
        metadata={"help": "Save the full 16GB+ MoE checkpoint at the end. Router weights are always saved."},
    )
    router_num_experts_per_tok_for_training: Optional[int] = field(
        default=None,
        metadata={"help": "Override LoRAMoe top-k during router training. Use >1 for differentiable routing."},
    )
    router_state_name: str = field(default="router_model.safetensors")
    router_summary_name: str = field(default="router_trainable_summary.json")


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.bfloat16)


def parameter_kind(name: str) -> str:
    if ".gate.weight" in name:
        return "router_gate"
    if ".lora_A." in name or ".lora_B." in name:
        return "lora_expert"
    if ".base_layer." in name:
        return "base_layer"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_or_head"
    return "other"


def set_router_only_trainable(model: torch.nn.Module) -> Dict[str, Any]:
    total = 0
    trainable = 0
    by_kind: Dict[str, int] = {}
    trainable_by_kind: Dict[str, int] = {}
    examples: List[str] = []

    for name, param in model.named_parameters():
        kind = parameter_kind(name)
        count = param.numel()
        total += count
        by_kind[kind] = by_kind.get(kind, 0) + count
        param.requires_grad = kind == "router_gate"
        if param.requires_grad:
            trainable += count
            trainable_by_kind[kind] = trainable_by_kind.get(kind, 0) + count
            if len(examples) < 20:
                examples.append(name)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "params_by_kind": by_kind,
        "trainable_params_by_kind": trainable_by_kind,
        "trainable_examples": examples,
    }


def repair_lora_moe_gate_dimensions(model: torch.nn.Module) -> Dict[str, Any]:
    repaired: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        expected_in_features = getattr(module, "in_features", None)
        num_experts = getattr(module, "num_experts", None)
        if not isinstance(gate, torch.nn.Linear) or expected_in_features is None or num_experts is None:
            continue
        if gate.in_features == expected_in_features:
            continue

        new_gate = torch.nn.Linear(
            expected_in_features,
            num_experts,
            bias=gate.bias is not None,
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        module.gate = new_gate
        repaired.append(
            {
                "name": name,
                "old_in_features": gate.in_features,
                "new_in_features": expected_in_features,
                "num_experts": num_experts,
            }
        )

    return {"repaired_gate_count": len(repaired), "repaired_examples": repaired[:20]}


def override_router_topk_for_training(model: torch.nn.Module, topk: Optional[int]) -> Dict[str, Any]:
    if topk is None:
        return {"overridden_layer_count": 0, "topk": None}

    changed: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        num_experts = getattr(module, "num_experts", None)
        current_topk = getattr(module, "num_experts_per_tok", None)
        if num_experts is None or current_topk is None:
            continue
        if topk < 1 or topk > num_experts:
            raise ValueError(f"router_num_experts_per_tok_for_training={topk} is invalid for {name} with {num_experts} experts")
        if current_topk == topk:
            continue
        module.num_experts_per_tok = topk
        changed.append({"name": name, "old_topk": current_topk, "new_topk": topk})

    if hasattr(model.config, "num_experts_per_tok"):
        model.config.num_experts_per_tok = topk

    return {"overridden_layer_count": len(changed), "topk": topk, "overridden_examples": changed[:20]}


def build_message_rows(
    name: str,
    split: str,
    data_ratio: float,
    seed: Optional[int],
    categories: Optional[List[str]],
    max_samples: Optional[int],
) -> list[dict]:
    items = get_dataset(name, split=split, categories=categories, data_ratio=data_ratio, seed=seed)
    if max_samples is not None:
        items = items[:max_samples]

    dataset_name = name.lower()
    is_math = dataset_name in {"bigmath", "math", "numina_cot"}
    use_numina_category_prompt = dataset_name == "numina_cot"

    rows = []
    for item in items:
        prompt_messages = (
            build_math_prompt(
                item["instruction"],
                metadata=item if use_numina_category_prompt else None,
            )
            if is_math
            else [{"role": "user", "content": item["instruction"]}]
        )
        completion_text = str(item.get("solution", item["ground_truth"]))
        completion_messages = [{"role": "assistant", "content": completion_text}]
        rows.append({"prompt": prompt_messages, "completion": completion_messages})
    return rows


def tokenize_rows(rows: list[dict], tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    tokenized = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if labels and all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }
        )
    return Dataset.from_list(tokenized)


class CausalLMCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def save_router_state(model: torch.nn.Module, output_dir: str, filename: str) -> str:
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if parameter_kind(name) == "router_gate"
    }
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return str(path)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, ExtraArguments, TrainingArguments))
    model_args, data_args, extra_args, training_args = parser.parse_args_into_dataclasses()

    os.environ.setdefault("WANDB_PROJECT", extra_args.wandb_project)
    os.environ.setdefault("WANDB_ENTITY", "jongbin-kr-skiml_moe")
    if training_args.run_name:
        os.environ.setdefault("WANDB_RUN_NAME", training_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info("Router training model args: %s", model_args)
    logger.info("Router training data args: %s", data_args)
    logger.info("Router training args: %s", training_args)
    set_all_seeds(training_args.seed)

    if model_args.patch_llama3_rope:
        patch_mergoo_for_llama31_lora_moe()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = build_message_rows(
        data_args.train_dataset,
        split="train",
        data_ratio=data_args.data_ratio,
        seed=training_args.seed,
        categories=data_args.categories,
        max_samples=data_args.max_train_samples,
    )
    eval_rows = build_message_rows(
        data_args.eval_dataset,
        split="validation",
        data_ratio=1.0,
        seed=training_args.seed,
        categories=data_args.categories,
        max_samples=data_args.max_eval_samples,
    )
    train_dataset = tokenize_rows(train_rows, tokenizer, data_args.max_seq_length)
    eval_dataset = tokenize_rows(eval_rows, tokenizer, data_args.max_seq_length)
    logger.info("Tokenized train/eval sizes: %d / %d", len(train_dataset), len(eval_dataset))

    from mergoo.models.modeling_llama import LlamaForCausalLM

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False
    original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if (
        original_max_position_embeddings is not None
        and data_args.max_seq_length < original_max_position_embeddings
    ):
        config.max_position_embeddings = data_args.max_seq_length
        logger.info(
            "Reduced runtime max_position_embeddings from %d to %d to avoid mergoo causal-mask OOM.",
            original_max_position_embeddings,
            config.max_position_embeddings,
        )
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype(model_args.dtype),
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    gate_repair_summary = repair_lora_moe_gate_dimensions(model)
    if gate_repair_summary["repaired_gate_count"]:
        logger.info("Repaired LoRAMoe gate dimensions: %s", json.dumps(gate_repair_summary, indent=2))
    topk_override_summary = override_router_topk_for_training(
        model, extra_args.router_num_experts_per_tok_for_training
    )
    if topk_override_summary["overridden_layer_count"]:
        logger.info("Overrode router top-k for training: %s", json.dumps(topk_override_summary, indent=2))
    trainable_summary = set_router_only_trainable(model)
    trainable_summary["gate_repair"] = gate_repair_summary
    trainable_summary["topk_override"] = topk_override_summary
    trainable_summary["runtime_max_position_embeddings"] = getattr(model.config, "max_position_embeddings", None)
    trainable_summary["original_max_position_embeddings"] = original_max_position_embeddings
    logger.info("Trainable summary: %s", json.dumps(trainable_summary, indent=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
        tokenizer=tokenizer,
    )

    train_result = trainer.train(
        resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None)
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / extra_args.router_summary_name
    with summary_path.open("w") as f:
        json.dump(trainable_summary, f, indent=2)
        f.write("\n")
    router_path = save_router_state(model, training_args.output_dir, extra_args.router_state_name)
    logger.info("Saved router state: %s", router_path)
    logger.info("Saved router summary: %s", summary_path)

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if extra_args.save_full_model:
        if original_max_position_embeddings is not None:
            model.config.max_position_embeddings = original_max_position_embeddings
        trainer.save_model(training_args.output_dir)
        logger.info("Saved full MoE checkpoint: %s", training_args.output_dir)


if __name__ == "__main__":
    main()
