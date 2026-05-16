"""Frozen/fine-tuned Jina bi-encoder routing: pick roster persona by max cosine similarity."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _last_token_pooling(outputs, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = outputs.last_hidden_state.shape[0]
    return outputs.last_hidden_state[
        torch.arange(batch_size, device=outputs.last_hidden_state.device),
        sequence_lengths,
    ]


class JinaPersonaRouter:
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        max_length_query: int = 2048,
        max_length_doc: int = 512,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, padding_side="left", trust_remote_code=True)
        try:
            self.model = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def _encode(self, texts: List[str], max_len: int) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        emb = _last_token_pooling(outputs, inputs["attention_mask"])
        return F.normalize(emb.float(), p=2, dim=1)

    def route(self, problem_instruction: str, roster: List[Dict[str, Any]]) -> str:
        """Returns persona id with highest cosine similarity."""
        if not roster:
            return "default"
        q = f"Query: {problem_instruction}"
        docs = [
            f"Document: {p.get('system_prompt') or p.get('persona_description') or p.get('description', '')}"
            for p in roster
        ]
        if not any(len(d.strip()) > len("Document:") for d in docs):
            logging.warning("JinaPersonaRouter: empty roster prompts; fallback to first id")
            return roster[0].get("id", "default")

        q_emb = self._encode([q], self.max_length_query)
        # 문서는 배치가 아니라 건별 인코딩(짧은 문서 배치 시 임베딩 붕괴 방지, EDA와 동일)
        doc_embs = [self._encode([d], self.max_length_doc) for d in docs]
        d_emb = torch.cat(doc_embs, dim=0)
        sims = (q_emb @ d_emb.T).squeeze(0)
        best = int(sims.argmax().item())
        return roster[best].get("id", "default")
