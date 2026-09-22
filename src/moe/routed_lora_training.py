"""Shared objectives for token-routed LoRA-MoE training.

Dataset-specific row construction and training recipes stay in their entrypoint
scripts; these objectives are shared by SNI and LBox routed-LoRA runs.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from .routed_lora_ffn import (
    RoutedLoraConfig,
    load_routed_lora_state,
    router_logits,
    save_routed_lora,
    trainable_parameter_groups,
)


LOGGER = logging.getLogger(__name__)


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


class CompletionRouteCollator:
    """Tokenize chat rows while supervising only the assistant completion."""

    def __init__(self, tokenizer, max_length: int, num_experts: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_experts = num_experts

    def _tokenize(self, row: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt = [
            {"role": "system", "content": str(row["system"])},
            {"role": "user", "content": str(row["user"])},
        ]
        full = prompt + [{"role": "assistant", "content": str(row["target"])}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        full_ids = self.tokenizer.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"Chat template prefix mismatch for row {row['id']}")
        completion_ids = full_ids[len(prompt_ids) :]
        if not completion_ids:
            raise RuntimeError(f"Empty completion for row {row['id']}")
        available_prompt = self.max_length - len(completion_ids)
        if available_prompt < 2:
            raise RuntimeError(f"Completion exceeds max_length for row {row['id']}")
        if len(prompt_ids) > available_prompt:
            head = available_prompt // 2
            prompt_ids = prompt_ids[:head] + prompt_ids[-(available_prompt - head) :]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return torch.tensor(input_ids), torch.tensor(labels), len(prompt_ids)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        tokenized = [self._tokenize(row) for row in features]
        input_ids = pad_sequence(
            [item[0] for item in tokenized],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            [item[1] for item in tokenized], batch_first=True, padding_value=-100
        )
        attention_mask = pad_sequence(
            [torch.ones_like(item[0]) for item in tokenized],
            batch_first=True,
            padding_value=0,
        )
        route_target_mask = torch.zeros(
            len(features), self.num_experts, dtype=torch.bool
        )
        for row_index, row in enumerate(features):
            route_target_mask[row_index, row["route_targets"]] = True
        if not all(
            torch.all(labels[i, :prompt_length] == -100)
            for i, (*_, prompt_length) in enumerate(tokenized)
        ):
            raise RuntimeError("Completion-only invariant failed: prompt token has a label")
        if not all(torch.any(labels[i] != -100) for i in range(len(features))):
            raise RuntimeError("Completion-only invariant failed: empty supervised completion")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "route_target_mask": route_target_mask,
        }


class RoutedLoraTrainer(Trainer):
    """Trainer for dense-base, routed-LoRA experts and router objectives."""

    def __init__(
        self,
        *args,
        routed_config: RoutedLoraConfig,
        route_loss_coef: float,
        load_balance_coef: float,
        router_z_loss_coef: float,
        router_lr: float,
        lora_lr: float,
        router_warmup_steps: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.routed_config = routed_config
        self.route_loss_coef = route_loss_coef
        self.load_balance_coef = load_balance_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router_lr = router_lr
        self.lora_lr = lora_lr
        self.router_warmup_steps = router_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_mask = inputs.pop("route_target_mask")
        outputs = model(**inputs, use_cache=False, num_items_in_batch=num_items_in_batch)
        logits = router_logits(model)
        route = supervised_router_loss(logits, target_mask, inputs["attention_mask"])
        balance = load_balancing_loss(
            logits, inputs["attention_mask"], top_k=self.routed_config.top_k
        )
        z_loss = router_z_loss(logits, inputs["attention_mask"])
        loss = (
            outputs.loss
            + self.route_loss_coef * route
            + self.load_balance_coef * balance
            + self.router_z_loss_coef * z_loss
        )
        self._latest_losses = {
            "router_supervised_loss": route.detach(),
            "router_load_balance_loss": balance.detach(),
            "router_z_loss": z_loss.detach(),
        }
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            groups = trainable_parameter_groups(self.model)
            if groups["unexpected"]:
                raise RuntimeError(
                    f"Unexpected trainable parameters: {[n for n, _ in groups['unexpected'][:10]]}"
                )
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [p for _, p in groups["router"]],
                        "lr": self.router_lr,
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for _, p in groups["expert_lora"]],
                        "lr": self.lora_lr,
                        "weight_decay": 0.0,
                    },
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=torch.cuda.is_available(),
            )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step < self.router_warmup_steps:
            for _, parameter in trainable_parameter_groups(model)["expert_lora"]:
                parameter.grad = None
        return loss

    def log(self, logs, *args, **kwargs):
        for name, value in getattr(self, "_latest_losses", {}).items():
            logs[name] = float(value.cpu())
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        save_routed_lora(self.model, output_dir, self.routed_config)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_routed_lora_state(model or self.model, resume_from_checkpoint)

    def _load_best_model(self):
        """Restore the custom routed-LoRA state selected by eval_loss."""
        if not self.state.best_model_checkpoint:
            return
        LOGGER.info(
            "Loading best routed-LoRA checkpoint from %s",
            self.state.best_model_checkpoint,
        )
        load_routed_lora_state(self.model, self.state.best_model_checkpoint)


def verify_gradients(model, batch, args):
    """Verify router and expert gradients while the dense base stays frozen."""
    model.train()
    batch = {name: value.to(model.device) for name, value in batch.items()}
    target_mask = batch.pop("route_target_mask")
    outputs = model(**batch, use_cache=False)
    logits = router_logits(model)
    route = supervised_router_loss(logits, target_mask, batch["attention_mask"])
    balance = load_balancing_loss(logits, batch["attention_mask"], top_k=args.top_k)
    z_loss = router_z_loss(logits, batch["attention_mask"])
    loss = (
        outputs.loss
        + args.route_loss_coef * route
        + args.load_balance_coef * balance
        + args.router_z_loss_coef * z_loss
    )
    loss.backward()
    groups = trainable_parameter_groups(model)
    nonzero = {
        group: sum(
            p.grad is not None and bool(torch.count_nonzero(p.grad).item())
            for _, p in values
        )
        for group, values in groups.items()
    }
    if groups["unexpected"] or nonzero["router"] == 0 or nonzero["expert_lora"] == 0:
        raise RuntimeError(f"Gradient verification failed: nonzero={nonzero}")
    if any(
        parameter.grad is not None
        for _, parameter in model.named_parameters()
        if not parameter.requires_grad
    ):
        raise RuntimeError("Frozen base parameter received a gradient")
    LOGGER.info(
        "Gradient verification passed: loss=%.6f lm=%.6f route=%.6f balance=%.6f z=%.6f nonzero=%s",
        float(loss.detach()),
        float(outputs.loss.detach()),
        float(route.detach()),
        float(balance.detach()),
        float(z_loss.detach()),
        nonzero,
    )
    model.zero_grad(set_to_none=True)


__all__ = [
    "CompletionRouteCollator",
    "RoutedLoraTrainer",
    "load_balancing_loss",
    "router_z_loss",
    "supervised_router_loss",
    "verify_gradients",
]
