"""Compatibility helpers for using mergoo with newer Llama checkpoints."""

from __future__ import annotations

from typing import Any


def patch_mergoo_llama3_rope() -> None:
    """Patch mergoo's vendored Llama attention to support Llama 3.1 RoPE.

    mergoo 0.0.10 vendors an older Llama implementation whose ``_init_rope``
    expects the pre-4.43 ``rope_scaling`` schema with ``{"type", "factor"}``.
    Llama 3.1 checkpoints use ``{"rope_type": "llama3", ...}``.

    Transformers 4.43 already implements the correct Llama 3.1 rotary embedding
    math. This patch swaps mergoo's attention-layer rotary initialization to use
    that implementation while leaving mergoo's MoE/LoRA layers untouched.
    """

    import mergoo.models.modeling_llama as mergoo_llama
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    def _init_rope(self: Any) -> None:
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    mergoo_llama.LlamaAttention._init_rope = _init_rope
    if hasattr(mergoo_llama, "LlamaFlashAttention2"):
        mergoo_llama.LlamaFlashAttention2._init_rope = _init_rope
    if hasattr(mergoo_llama, "LlamaSdpaAttention"):
        mergoo_llama.LlamaSdpaAttention._init_rope = _init_rope


def patch_mergoo_lora_moe_gate_dimensions() -> None:
    """Patch mergoo LoRA MoE routers to use each projection's real input size.

    mergoo 0.0.10 initializes every ``LoRAMoeLayer.gate`` with
    ``config.hidden_size`` input features. That is correct for attention
    projections and MLP up/gate projections, but wrong for ``mlp.down_proj``,
    whose input is ``config.intermediate_size``. Patching the constructor keeps
    saved checkpoints loadable after down-proj router gates have been trained
    and saved with the correct shape.
    """

    import torch
    from mergoo.compose_layers import LoRAMoeLayer

    if getattr(LoRAMoeLayer, "_evolving_moe_gate_dim_patch", False):
        return

    original_init = LoRAMoeLayer.__init__

    def _init_with_projection_gate(self: Any, config: Any, in_features: int, out_features: int, bias: bool) -> None:
        original_init(self, config, in_features, out_features, bias)
        if self.gate.in_features != in_features:
            self.gate = torch.nn.Linear(
                in_features,
                config.num_experts,
                bias=False,
                device=self.gate.weight.device,
                dtype=self.gate.weight.dtype,
            )

    LoRAMoeLayer.__init__ = _init_with_projection_gate
    LoRAMoeLayer._evolving_moe_gate_dim_patch = True


def patch_mergoo_for_llama31_lora_moe() -> None:
    """Apply all mergoo runtime patches needed by this MoE-on-LoRA workflow."""

    patch_mergoo_llama3_rope()
    patch_mergoo_lora_moe_gate_dimensions()
