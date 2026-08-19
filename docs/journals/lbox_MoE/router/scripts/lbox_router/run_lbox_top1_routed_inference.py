#!/usr/bin/env python3
"""Route LBox examples with one saved MLP and generate with the selected LoRA."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from evaluate import build_eval_prompt, evaluate_item
from train_lbox_router_baseline import Router


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GENERALIST_SYSTEM_PROMPT = (
    "You are a Korean legal classification generalist. Given Korean legal facts, return the exact "
    "requested case name, charge, or statutory provision in the required one-line format without explanation."
)
LEGAL_CATEGORY_NAMES = {
    "admin_labor": "Administrative Labor Law",
    "admin_other": "Administrative Law General",
    "admin_traffic": "Road Traffic Law",
    "civil_family_inheritance": "Civil Family and Inheritance Law",
    "civil_property_obligation": "Civil Property and Obligation Law",
    "criminal_non_property": "Criminal Non-property Offenses",
    "criminal_property": "Criminal Property Offenses",
    "family_patent_special": "Family and Patent Special Cases",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_low5_high6_persona_bank(base_bank: dict[str, Any], agent_mapping_path: Path) -> dict[str, Any]:
    """Reuse the low5/high6 router order, swapping in persona-trained LoRAs and prompts."""
    mapping = json.loads(agent_mapping_path.read_text(encoding="utf-8"))
    models = []
    for model in base_bank["models"]:
        persona_model = dict(model)
        if model["id"] == "generalist_high6":
            persona_model["lora_path"] = "checkpoints/sft_lbox_generalist_high6_persona_eval_ep5"
            persona_model["system_prompt"] = GENERALIST_SYSTEM_PROMPT
            persona_model["prompt_system"] = "persona"
        else:
            source_id = str(model.get("source_expert_id") or "").strip()
            if not source_id or source_id not in mapping:
                raise KeyError(f"Missing source_expert_id/system prompt for {model}")
            persona_model["lora_path"] = f"checkpoints/sft_lbox_roster_{source_id}_low5_persona_eval_ep5"
            persona_model["system_prompt"] = str(mapping[source_id]["system_prompt"])
            persona_model["prompt_system"] = "persona"
        models.append(persona_model)
    bank = dict(base_bank)
    bank["label"] = "low5 specialists + high6 generalist, persona-trained LoRA and persona inference prompt"
    bank["models"] = models
    return bank


def load_legal_category_bank(router_dir: Path) -> dict[str, Any]:
    """Build an 8-way LBox legal-category expert bank in the router output order."""
    metrics = json.loads((router_dir / "metrics.json").read_text(encoding="utf-8"))
    categories = list(metrics["categories"])
    models = []
    for category in categories:
        name = LEGAL_CATEGORY_NAMES.get(category, category)
        models.append(
            {
                "id": category,
                "name": name,
                "slug": f"lbox_category_{category}",
                "lora_path": f"Jongbin-kr/llama3_lbox_category_{category}_step5000",
                "prompt_system": "baseline",
                "system_prompt": "baseline",
                "legal_category": category,
            }
        )
    return {
        "label": "Gemma4-tagged legal-category router with 8 category LoRA experts",
        "models": models,
    }


def resolve_lora_path(lora_path: str, cache_dir: Path) -> str:
    """Return a local LoRA adapter path, downloading Hub repos when needed."""
    path = Path(lora_path)
    if (path / "adapter_config.json").is_file():
        return str(path.resolve())
    if "/" not in lora_path:
        raise FileNotFoundError(f"Missing LoRA adapter: {lora_path}")

    from huggingface_hub import snapshot_download

    local_dir = cache_dir / lora_path.replace("/", "__")
    snapshot = snapshot_download(
        repo_id=lora_path,
        local_dir=str(local_dir),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    if not (Path(snapshot) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Downloaded LoRA is missing adapter_config.json: {lora_path}")
    return snapshot


def task_name(row: dict[str, Any]) -> str:
    if row.get("task_type") == "casename":
        return f"casename_{row.get('casetype')}"
    return str(row.get("task_type"))


def route_examples(
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    feature_dir: Path,
    split: str,
    router_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_path = feature_dir / f"lbox_{split}_hs_mean.npy"
    ids_path = feature_dir / f"lbox_{split}_hs_ids.json"
    if not feature_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(f"Missing router features: {feature_path} / {ids_path}")

    features = np.load(feature_path, mmap_mode="r")
    feature_ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8"))]
    if len(features) != len(feature_ids):
        raise RuntimeError(f"Feature/ID length mismatch: {len(features)} != {len(feature_ids)}")
    feature_index = {item_id: index for index, item_id in enumerate(feature_ids)}
    row_ids = [str(row["id"]) for row in rows]
    missing = [item_id for item_id in row_ids if item_id not in feature_index]
    if missing:
        raise RuntimeError(f"{len(missing)} dataset IDs are absent from router features")
    aligned = np.asarray(features[[feature_index[item_id] for item_id in row_ids]], dtype=np.float32)

    normalizer = np.load(router_dir / "normalizer.npz")
    mean = normalizer["mean"].astype(np.float32)
    std = normalizer["std"].astype(np.float32)
    if aligned.shape[1:] != mean.shape[1:] or mean.shape != std.shape:
        raise RuntimeError(
            f"Normalizer shape mismatch: features={aligned.shape}, mean={mean.shape}, std={std.shape}"
        )
    aligned = (aligned - mean) / std

    architecture = json.loads((router_dir / "metrics.json").read_text(encoding="utf-8"))["architecture"]
    router = Router(
        input_dim=aligned.shape[1],
        hidden_dim=int(architecture["hidden"]),
        experts=len(models),
        dropout=float(architecture["dropout"]),
    )
    router.load_state_dict(load_file(router_dir / f"router_seed{seed}.safetensors"))
    router.eval()

    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(aligned), 2048):
            batch = torch.from_numpy(aligned[offset : offset + 2048])
            logits.append(router(batch).float().numpy())
    all_logits = np.concatenate(logits)
    selected = all_logits.argmax(axis=1)
    return selected, all_logits


def write_routing(
    path: Path,
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    selected: np.ndarray,
    logits: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, expert_index, row_logits in zip(rows, selected.tolist(), logits):
            ordered = np.sort(row_logits)
            model = models[expert_index]
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "task": task_name(row),
                        "expert_index": expert_index,
                        "expert_id": model["id"],
                        "expert_name": model["name"],
                        "expert_slug": model["slug"],
                        "top1_logit": float(ordered[-1]),
                        "top1_margin": float(ordered[-1] - ordered[-2]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_baseline(path: Path, expected_ids: set[str]) -> dict[str, float | int | str]:
    rows = load_jsonl(path)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise RuntimeError(f"Baseline does not match routed dataset: {path}")
    correct = sum(float(row.get("pass_score", 0.0)) > 0 for row in rows)
    return {
        "path": str(path),
        "examples": len(rows),
        "correct": correct,
        "accuracy": 100.0 * correct / len(rows),
    }


def summarize(
    output_dir: Path,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    models: list[dict[str, Any]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> None:
    by_id = {str(row["id"]): row for row in predictions}
    if len(by_id) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} unique predictions, found {len(by_id)}")

    ordered = [by_id[str(row["id"])] for row in rows]
    scores = np.asarray([float(row.get("pass_score", 0.0)) > 0 for row in ordered])
    tasks: dict[str, list[int]] = defaultdict(list)
    experts: dict[str, list[int]] = defaultdict(list)
    for index, (source, prediction) in enumerate(zip(rows, ordered)):
        tasks[task_name(source)].append(index)
        experts[str(prediction["routed_expert_name"])].append(index)

    expected_ids = {str(row["id"]) for row in rows}
    metrics: dict[str, Any] = {
        "protocol": {
            "bank": args.bank,
            "feature": "hs_mean",
            "router_seed": args.seed,
            "routing": "top1_argmax",
            "base_model": args.base_model,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "prompt_mode": args.bank,
        },
        "examples": len(rows),
        "correct": int(scores.sum()),
        "accuracy": float(100.0 * scores.mean()),
        "accuracy_by_task": {
            task: {
                "examples": len(indices),
                "accuracy": float(100.0 * scores[indices].mean()),
            }
            for task, indices in sorted(tasks.items())
        },
        "selection_by_expert": {
            name: {
                "examples": len(indices),
                "accuracy": float(100.0 * scores[indices].mean()),
            }
            for name, indices in sorted(experts.items())
        },
        "selection_counts": dict(
            Counter(models[index]["name"] for index in selected.tolist()).most_common()
        ),
    }
    if args.vanilla_baseline:
        metrics["vanilla_baseline"] = load_baseline(args.vanilla_baseline, expected_ids)
    if args.dense_baseline:
        metrics["dense_sft_baseline"] = load_baseline(args.dense_baseline, expected_ids)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# LBox {args.bank} seed-{args.seed} top-1 routed inference",
        "",
        f"- Examples: {metrics['examples']}",
        f"- Routed accuracy: **{metrics['accuracy']:.2f}%**",
        "",
        "| Method | Accuracy (%) |",
        "|---|---:|",
    ]
    if "vanilla_baseline" in metrics:
        lines.append(f"| Vanilla Llama-3.1-8B | {metrics['vanilla_baseline']['accuracy']:.2f} |")
    if "dense_sft_baseline" in metrics:
        lines.append(f"| Dense SFT | {metrics['dense_sft_baseline']['accuracy']:.2f} |")
    lines.append(f"| {args.bank} routed SFT | **{metrics['accuracy']:.2f}** |")
    lines.extend(["", "| Task | Examples | Accuracy (%) |", "|---|---:|---:|"])
    for task, values in metrics["accuracy_by_task"].items():
        lines.append(f"| {task} | {values['examples']} | {values['accuracy']:.2f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank-config",
        type=Path,
        default=Path("configs/lbox_router/lbox_router_banks.json"),
    )
    parser.add_argument(
        "--bank",
        default="low5_high6",
        choices=[
            "low5_high6",
            "task_prior",
            "low5_high6_persona_eval_ep5",
            "legal_category_gemma4_merged",
        ],
    )
    parser.add_argument(
        "--agent-mapping",
        type=Path,
        default=Path("results/lbox_binning_seed20210311/agent_mapping.json"),
    )
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("results/embed_viz_test"))
    parser.add_argument("--data-file", type=Path, default=Path("export/lbox/lbox_test.jsonl"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--lora-cache-dir", type=Path, default=Path("results/hf_lora_cache"))
    parser.add_argument("--vanilla-baseline", type=Path)
    parser.add_argument("--dense-baseline", type=Path)
    args = parser.parse_args()

    bank_config = json.loads(args.bank_config.read_text(encoding="utf-8"))
    if args.bank == "low5_high6_persona_eval_ep5":
        bank = load_low5_high6_persona_bank(bank_config["low5_high6"], args.agent_mapping)
    elif args.bank == "legal_category_gemma4_merged":
        bank = load_legal_category_bank(args.router_dir)
    else:
        bank = bank_config[args.bank]
    models = bank["models"]

    rows = load_jsonl(args.data_file)
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.route_only:
        selected, logits = route_examples(
            rows, models, args.feature_dir, args.split, args.router_dir, args.seed
        )
        write_routing(args.output_dir / "routing.jsonl", rows, models, selected, logits)
        selection_counts = Counter(models[index]["name"] for index in selected.tolist())
        print(f"Routing complete: {len(rows)} examples", flush=True)
        print(json.dumps(selection_counts, ensure_ascii=False, indent=2), flush=True)
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Initialize vLLM before running the CPU PyTorch router. Forking the engine
    # after PyTorch/OpenMP work can deadlock the vLLM worker during CUDA setup.
    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enable_lora=True,
    )
    tokenizer = llm.get_tokenizer()

    selected, logits = route_examples(
        rows, models, args.feature_dir, args.split, args.router_dir, args.seed
    )
    write_routing(args.output_dir / "routing.jsonl", rows, models, selected, logits)
    selection_counts = Counter(models[index]["name"] for index in selected.tolist())
    print(f"Routing complete: {len(rows)} examples", flush=True)
    print(json.dumps(selection_counts, ensure_ascii=False, indent=2), flush=True)

    args.lora_cache_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        model["resolved_lora_path"] = resolve_lora_path(
            str(model["lora_path"]),
            args.lora_cache_dir,
        )

    lora_requests = [
        LoRARequest(model["slug"], index + 1, str(model["resolved_lora_path"]))
        for index, model in enumerate(models)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    predictions_path = args.output_dir / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.is_file():
        done = {str(row["id"]): row for row in load_jsonl(predictions_path)}
    remaining_indices = [
        index for index, row in enumerate(rows) if str(row["id"]) not in done
    ]
    print(f"Generation remaining: {len(remaining_indices)}/{len(rows)}", flush=True)

    generated = 0
    with predictions_path.open("a", encoding="utf-8") as handle:
        for expert_index, lora_request in enumerate(lora_requests):
            expert = models[expert_index]
            expert_indices = [
                index
                for index in remaining_indices
                if int(selected[index]) == expert_index
            ]
            print(
                f"Generating {len(expert_indices)} examples with {expert['name']}",
                flush=True,
            )
            for offset in range(0, len(expert_indices), args.chunk_size):
                indices = expert_indices[offset : offset + args.chunk_size]
                chunk_rows = [rows[index] for index in indices]
                messages = [
                    build_eval_prompt(
                        row,
                        dataset_name="lbox",
                        model_name=args.base_model,
                        system_prompt=str(expert.get("system_prompt") or "baseline"),
                    )
                    for row in chunk_rows
                ]
                prompts = [
                    tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for message in messages
                ]
                outputs = llm.generate(
                    prompts,
                    sampling,
                    lora_request=[lora_request] * len(indices),
                    use_tqdm=True,
                )
                for index, row, message, output in zip(
                    indices, chunk_rows, messages, outputs
                ):
                    prediction = output.outputs[0].text
                    result = {
                        "id": row["id"],
                        "input": message,
                        "prediction": prediction,
                        "ground_truth": row["ground_truth"],
                        "category": row.get("category") or row.get("categories", []),
                        "dataset": row.get("dataset") or "lbox",
                        "domain": row.get("domain"),
                        "prompt_system": str(expert.get("prompt_system") or "baseline"),
                        "system_prompt_spec": str(expert.get("system_prompt") or "baseline"),
                        "system_prompt": (
                            message[0].get("content")
                            if isinstance(message, list)
                            and message
                            and message[0].get("role") == "system"
                            else None
                        ),
                        "router_seed": args.seed,
                        "routed_expert_id": expert["id"],
                        "routed_expert_name": expert["name"],
                        "routed_expert_slug": expert["slug"],
                        **evaluate_item(row, prediction, is_math_dataset=False),
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                generated += len(indices)
                print(
                    f"Generated {generated}/{len(remaining_indices)} remaining examples",
                    flush=True,
                )

    summarize(
        args.output_dir,
        rows,
        load_jsonl(predictions_path),
        models,
        selected,
        args,
    )


if __name__ == "__main__":
    main()
