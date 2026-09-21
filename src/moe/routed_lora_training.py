"""Shared objectives for token-routed LoRA-MoE training.

Dataset-specific row construction and training recipes stay in their entrypoint
scripts; these objectives are shared by SNI and LBox routed-LoRA runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_router_loss(
    layer_logits: tuple[torch.Tensor, ...],
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize router probability mass outside each row's target experts."""
    batch_size, sequence_length = attention_mask.shape
    token_mask = attention_mask.reshape(-1).bool()
    losses = []
    for logits in layer_logits:
        log_probs = F.log_softmax(logits.float(), dim=-1).reshape(
            batch_size, sequence_length, -1
        )
        expanded_targets = target_mask[:, None, :].expand(-1, sequence_length, -1)
        target_log_mass = torch.logsumexp(
            log_probs.masked_fill(~expanded_targets, -torch.inf), dim=-1
        )
        losses.append((-target_log_mass.reshape(-1)[token_mask]).mean())
    return torch.stack(losses).mean()


def load_balancing_loss(
    layer_logits: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Encourage balanced top-k assignments and router importance."""
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    batch_size, sequence_length = attention_mask.shape
    per_layer = []
    for logits in layer_logits:
        probabilities = F.softmax(logits.float(), dim=-1).reshape(
            batch_size, sequence_length, -1
        )
        if top_k > probabilities.shape[-1]:
            raise ValueError(
                f"top_k={top_k} exceeds router experts={probabilities.shape[-1]}"
            )
        selected = torch.topk(probabilities, top_k, dim=-1).indices
        assignments = (
            F.one_hot(selected, probabilities.shape[-1]).float().sum(dim=-2) / top_k
        )
        mask = attention_mask[..., None].to(probabilities.dtype)
        denominator = mask.sum().clamp_min(1.0)
        load = (assignments * mask).sum(dim=(0, 1)) / denominator
        importance = (probabilities * mask).sum(dim=(0, 1)) / denominator
        per_layer.append(probabilities.shape[-1] * torch.sum(load * importance))
    return torch.stack(per_layer).mean()


def router_z_loss(
    layer_logits: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
) -> torch.Tensor:
    """Regularize the router log-partition values for numerical stability."""
    token_mask = attention_mask.reshape(-1).bool()
    return torch.stack(
        [
            torch.logsumexp(logits.float(), dim=-1).square()[token_mask].mean()
            for logits in layer_logits
        ]
    ).mean()


__all__ = ["load_balancing_loss", "router_z_loss", "supervised_router_loss"]
