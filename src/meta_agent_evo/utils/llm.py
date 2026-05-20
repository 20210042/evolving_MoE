"""LLM backends (vLLM default)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

import torch
from transformers import AutoTokenizer, pipeline

Message = Union[str, List[MutableMapping[str, str]]]


def llm_service_from_yaml_config(model_name: str, cfg: Mapping[str, Any]) -> "LLMService":
    llm = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    vllm = cfg.get("vllm") if isinstance(cfg.get("vllm"), dict) else {}
    sampling = llm.get("sampling") if isinstance(llm.get("sampling"), dict) else {}

    kwargs = dict(vllm)
    kwargs.pop("tp_size", None)
    kwargs["tensor_parallel_size"] = int(vllm.get("tp_size", 1))
    if llm.get("max_model_len") is not None:
        kwargs["max_model_len"] = int(llm["max_model_len"])

    return LLMService(
        model_name,
        vllm_kwargs=kwargs,
        max_tokens=int(llm.get("max_tokens", 8192)),
        temperature=float(sampling.get("temperature", 0.7)),
        top_p=float(sampling.get("top_p", 0.8)),
        top_k=int(sampling.get("top_k", 20)),
        repetition_penalty=float(sampling.get("repetition_penalty", 1.05)),
    )


class LLMService:
    def __init__(
        self,
        model_name: str,
        mode: str = "vllm",
        *,
        vllm_kwargs: Mapping[str, Any] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        self.model_name = model_name
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        if mode == "vllm":
            if LLM is None or SamplingParams is None:
                raise ImportError("vllm is not installed.")
            kw = dict(vllm_kwargs or {})
            kw.setdefault("trust_remote_code", True)
            self.model = LLM(model_name, **kw)
            self.tokenizer = self.model.get_tokenizer()
        elif mode == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _to_prompt(self, messages: Sequence[MutableMapping[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        msgs = [dict(m) for m in messages]
        if msgs and msgs[-1].get("role") == "assistant":
            return str(
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            )
        return str(
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def chat(
        self,
        messages: Message,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        return self.chat_batch(
            [messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )[0]

    def chat_batch(
        self,
        messages_batch: List[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        prompts = [
            m if isinstance(m, str) else self._to_prompt(m)
            for m in messages_batch
        ]
        return self.generate(
            prompts,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            top_k=self.top_k if top_k is None else top_k,
            repetition_penalty=self.repetition_penalty if repetition_penalty is None else repetition_penalty,
            stop=stop,
        )

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        if self.mode == "vllm":
            assert SamplingParams is not None and self.model is not None
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            return [o.outputs[0].text for o in self.model.generate(prompts, sp)]

        assert self.tokenizer is not None and self.model is not None
        out = []
        for prompt in prompts:
            gen = self.model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1e-5,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                stop_sequence=stop,
            )
            out.append(gen[0]["generated_text"][len(prompt) :])
        return out
