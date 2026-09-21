from types import SimpleNamespace

import torch
from torch import nn

from moe.routed_lora_ffn import (
    RoutedLoraConfig,
    RoutedLoraMLP,
    load_routed_lora_state,
    save_routed_lora,
    trainable_parameter_groups,
)
from moe.routed_lora_training import (
    load_balancing_loss,
    router_z_loss,
    supervised_router_loss,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = nn.SiLU()


def make_config(top_k=2):
    return RoutedLoraConfig(
        base_model_name_or_path="tiny",
        expert_names=("a", "b", "generalist"),
        rank=2,
        alpha=4,
        dropout=0.0,
        top_k=top_k,
    )


def expert_output(module, inputs, expert):
    gate = module.gate_proj.forward_expert(inputs, expert)
    up = module.up_proj.forward_expert(inputs, expert)
    return module.down_proj.forward_expert(module.act_fn(gate) * up, expert)


def test_top2_is_weighted_sum_of_complete_expert_ffns():
    torch.manual_seed(4)
    module = RoutedLoraMLP(TinyMLP(), make_config(top_k=2))
    with torch.no_grad():
        for projection in (module.gate_proj, module.up_proj, module.down_proj):
            projection.lora_B.normal_(std=0.1)
        module.router.weight.normal_(std=0.2)
    inputs = torch.randn(2, 3, 4)
    actual = module(inputs)
    flat = inputs.reshape(-1, 4)
    probabilities = torch.softmax(module.router(flat.float()), dim=-1)
    weights, indices = torch.topk(probabilities, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(2):
            expert = int(indices[token, slot])
            expected[token] += weights[token, slot] * expert_output(module, flat[token : token + 1], expert)[0]
    torch.testing.assert_close(actual, expected.reshape_as(actual), rtol=1e-5, atol=1e-6)


def test_router_objectives_are_finite_and_differentiable():
    logits = tuple(torch.randn(6, 3, requires_grad=True) for _ in range(2))
    targets = torch.tensor([[True, False, True], [False, False, True]])
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    loss = (
        supervised_router_loss(logits, targets, attention)
        + 0.01 * load_balancing_loss(logits, attention, top_k=2)
        + 0.001 * router_z_loss(logits, attention)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(item.grad is not None and torch.count_nonzero(item.grad) for item in logits)


def test_adapter_only_save_and_load(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].mlp = RoutedLoraMLP(TinyMLP(), make_config())

    # SimpleNamespace is not registered as an nn.Module; expose the layer through a registered holder.
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = RoutedLoraMLP(TinyMLP(), make_config())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name or ".router." in name)
    groups = trainable_parameter_groups(model)
    assert groups["router"] and groups["expert_lora"] and not groups["unexpected"]
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    save_routed_lora(model, tmp_path, make_config())
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_routed_lora_state(model, tmp_path)
    after = {name: value.detach() for name, value in model.named_parameters() if value.requires_grad}
    for name in before:
        torch.testing.assert_close(after[name], before[name])
