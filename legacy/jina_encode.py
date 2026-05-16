"""Jina retrieval: last-token pooling + L2-normalized embeddings (train/eval 공통)."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def last_token_pooling(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = outputs.last_hidden_state.shape[0]
    return outputs.last_hidden_state[
        torch.arange(batch_size, device=outputs.last_hidden_state.device),
        sequence_lengths,
    ]


def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        outputs = model(**inputs)
    emb = last_token_pooling(outputs, inputs["attention_mask"])
    return F.normalize(emb.float(), p=2, dim=1)


def load_jina_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    *,
    train: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
    load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    try:
        model = AutoModel.from_pretrained(model_id, attn_implementation="flash_attention_2", **load_kw)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kw)
    if train:
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad = True
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    model = model.to(device)
    return model, tokenizer
