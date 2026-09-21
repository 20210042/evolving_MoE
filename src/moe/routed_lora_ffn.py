"""Token-routed LoRA experts for a dense Llama FFN.

Each expert represents a complete LoRA-modified SwiGLU FFN.  The dense
projection weights are shared in memory, while every expert owns independent
LoRA A/B matrices for gate_proj, up_proj, and down_proj.  For top-k > 1 we
evaluate the selected expert FFNs independently and mix their final outputs;
this avoids the cross terms produced by mixing projection deltas before the
SwiGLU non-linearity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class RoutedLoraConfig:
    base_model_name_or_path: str
    expert_names: tuple[str, ...]
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    top_k: int = 2

    @property
    def num_experts(self) -> int:
        return len(self.expert_names)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        result = asdict(self)
        result["expert_names"] = list(self.expert_names)
        result["num_experts"] = self.num_experts
        return result


class RoutedLoraProjection(nn.Module):
    """One frozen dense projection plus per-expert LoRA matrices."""

    def __init__(self, base_layer: nn.Linear, config: RoutedLoraConfig):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = config.num_experts
        self.rank = config.rank
        self.scaling = config.scaling
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.num_experts,
                base_layer.out_features,
                self.rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        for expert_index in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[expert_index], a=5**0.5)
        self.base_layer.requires_grad_(False)

    def forward_expert(self, inputs: torch.Tensor, expert_index: int) -> torch.Tensor:
        base = self.base_layer(inputs)
        dropped = self.dropout(inputs)
        delta = F.linear(F.linear(dropped, self.lora_A[expert_index]), self.lora_B[expert_index])
        return base + delta.to(base.dtype) * self.scaling


class RoutedLoraMLP(nn.Module):
    """A Llama SwiGLU MLP with token-level top-k LoRA expert routing."""

    def __init__(self, dense_mlp: nn.Module, config: RoutedLoraConfig):
        super().__init__()
        self.hidden_size = dense_mlp.gate_proj.in_features
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate_proj = RoutedLoraProjection(dense_mlp.gate_proj, config)
        self.up_proj = RoutedLoraProjection(dense_mlp.up_proj, config)
        self.down_proj = RoutedLoraProjection(dense_mlp.down_proj, config)
        self.act_fn = dense_mlp.act_fn
        self.router = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            device=dense_mlp.gate_proj.weight.device,
            dtype=torch.float32,
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        self.last_router_logits: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_logits = self.router(flat_states.float())
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        self.last_router_logits = router_logits

        output = torch.zeros_like(flat_states)
        for expert_index in range(self.num_experts):
            assignment_slot, token_index = torch.where(
                top_k_indices.transpose(0, 1) == expert_index
            )
            if token_index.numel() == 0:
                continue
            expert_inputs = flat_states[token_index]
            gate = self.gate_proj.forward_expert(expert_inputs, expert_index)
            up = self.up_proj.forward_expert(expert_inputs, expert_index)
            intermediate = self.act_fn(gate) * up
            expert_output = self.down_proj.forward_expert(intermediate, expert_index)
            weight = top_k_weights[token_index, assignment_slot, None].to(expert_output.dtype)
            output.index_add_(0, token_index, expert_output * weight)
        return output.reshape(batch_size, sequence_length, hidden_size)


def iter_routed_layers(model) -> Iterable[RoutedLoraMLP]:
    core = model.module if hasattr(model, "module") else model
    for layer in core.model.layers:
        if not isinstance(layer.mlp, RoutedLoraMLP):
            raise TypeError(f"Expected RoutedLoraMLP, found {type(layer.mlp).__name__}")
        yield layer.mlp


def attach_routed_lora_ffns(model, config: RoutedLoraConfig) -> None:
    if not 1 <= config.top_k <= config.num_experts:
        raise ValueError("top_k must be between 1 and num_experts")
    for layer in model.model.layers:
        layer.mlp = RoutedLoraMLP(layer.mlp, config)
    model.config.use_cache = False
    model.config.routed_lora_num_experts = config.num_experts
    model.config.routed_lora_top_k = config.top_k
    model.config.routed_lora_expert_names = list(config.expert_names)


def _source_key(layer_index: int, projection: str, matrix: str) -> str:
    return (
        f"base_model.model.model.layers.{layer_index}.mlp.{projection}."
        f"lora_{matrix}.weight"
    )


@torch.no_grad()
def load_expert_adapters(model, adapter_dirs: list[str | Path], config: RoutedLoraConfig) -> dict:
    if len(adapter_dirs) != config.num_experts:
        raise ValueError(f"Expected {config.num_experts} adapters, got {len(adapter_dirs)}")
    layers = list(iter_routed_layers(model))
    copied = 0
    for expert_index, raw_dir in enumerate(adapter_dirs):
        adapter_dir = Path(raw_dir)
        source_config = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        targets = set(source_config.get("target_modules") or [])
        if targets != set(PROJECTIONS):
            raise ValueError(f"{adapter_dir}: expected FFN targets, got {sorted(targets)}")
        if int(source_config.get("r", -1)) != config.rank:
            raise ValueError(f"{adapter_dir}: rank mismatch")
        if int(source_config.get("lora_alpha", -1)) != config.alpha:
            raise ValueError(f"{adapter_dir}: alpha mismatch")
        if source_config.get("base_model_name_or_path") != config.base_model_name_or_path:
            raise ValueError(f"{adapter_dir}: base model mismatch")
        tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
        for layer_index, layer in enumerate(layers):
            for projection_name in PROJECTIONS:
                projection = getattr(layer, projection_name)
                for matrix_name in ("A", "B"):
                    source = tensors[_source_key(layer_index, projection_name, matrix_name)]
                    target = getattr(projection, f"lora_{matrix_name}")[expert_index]
                    if source.shape != target.shape:
                        raise ValueError(
                            f"{adapter_dir}: layer={layer_index} {projection_name} {matrix_name} "
                            f"shape {tuple(source.shape)} != {tuple(target.shape)}"
                        )
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                    copied += 1
        del tensors
    return {"experts": config.num_experts, "layers": len(layers), "copied_tensors": copied}


def router_logits(model) -> tuple[torch.Tensor, ...]:
    result = tuple(layer.last_router_logits for layer in iter_routed_layers(model))
    if any(item is None for item in result):
        raise RuntimeError("Router logits requested before every routed layer ran")
    return result  # type: ignore[return-value]


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    core = model.module if hasattr(model, "module") else model
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    }


def save_routed_lora(model, output_dir: str | Path, config: RoutedLoraConfig) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        trainable_state_dict(model),
        str(output / "adapter_model.safetensors"),
        metadata={"format": "pt", "architecture": "routed_lora_ffn"},
    )
    (output / "routed_lora_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_routed_lora_state(model, checkpoint_dir: str | Path) -> None:
    model = model.module if hasattr(model, "module") else model
    state = load_file(str(Path(checkpoint_dir) / "adapter_model.safetensors"), device="cpu")
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and name in result.missing_keys
    ]
    if unexpected or missing_trainable:
        raise RuntimeError(
            f"Routed LoRA checkpoint mismatch: unexpected={unexpected[:10]} "
            f"missing_trainable={missing_trainable[:10]}"
        )


def trainable_parameter_groups(model) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups = {"router": [], "expert_lora": [], "unexpected": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".router." in name:
            groups["router"].append((name, parameter))
        elif ".lora_A" in name or ".lora_B" in name:
            groups["expert_lora"].append((name, parameter))
        else:
            groups["unexpected"].append((name, parameter))
    return groups
