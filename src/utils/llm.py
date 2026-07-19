"""LLM backends (vLLM default)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union

# pyarrow를 vllm보다 먼저 로드해야 한다. sklearn/scipy 설치(2026-07-15) 이후
# vllm 엔진 import 체인이 이들의 번들 네이티브 라이브러리를 먼저 올리면
# 마지막에 로드되는 pyarrow가 segfault한다(로드 순서 충돌, n03/n04/로그인 노드 재현).
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None

try:
    import httpx
    from openai import OpenAI as _OpenAI
except ImportError:
    httpx = None
    _OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
        max_model_len: int | None = None,
        lora_path: str | None = None,
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
        self.lora_request = None
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
            if max_model_len is not None:
                kw["max_model_len"] = max_model_len
            if lora_path is not None:
                kw["enable_lora"] = True
            self.model = LLM(model_name, **kw)
            self.tokenizer = self.model.get_tokenizer()
            if lora_path is not None:
                self.lora_request = LoRARequest("adapter", 1, lora_path)
                
                
        elif mode == "hf":
            from transformers import AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
            if lora_path is not None:
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, lora_path)
                base_model = base_model.merge_and_unload()
                
            self.model = pipeline(
                "text-generation",
                model=base_model,
                tokenizer=self.tokenizer,
            )
            
            
        elif mode == "api":
            if _OpenAI is None or httpx is None:
                raise ImportError("openai and httpx are required for api mode.")
            self.model = _OpenAI(
                api_key=os.environ["SKIML_API_KEY"],
                base_url=os.environ["SKIML_BASE_URL"],
                http_client=httpx.Client(verify=False),
            )
            self.tokenizer = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _to_prompt(
        self,
        messages: Sequence[MutableMapping[str, str]],
        enable_thinking: bool | None = None,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        msgs = [dict(m) for m in messages]
        template_kwargs: Dict[str, Any] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if msgs and msgs[-1].get("role") == "assistant":
            return str(
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    **template_kwargs,
                )
            )
        return str(
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
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
        enable_thinking: bool | None = None,
    ) -> str:
        return self.chat_batch(
            [messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            enable_thinking=enable_thinking,
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
        enable_thinking: bool | None = None,
    ) -> List[str]:
        if self.mode == "api":
            return self._chat_batch_api(
                messages_batch,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p if top_p is None else top_p,
                stop=stop,
                reasoning_effort="medium" if enable_thinking else None,
            )
        prompts = [
            m if isinstance(m, str) else self._to_prompt(m, enable_thinking=enable_thinking)
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

    def _chat_batch_api(
        self,
        messages_batch: List[Message],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        stop: Optional[List[str]] = None,
        reasoning_effort: str | None = None,
        max_workers: int = 5,
    ) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        assert self.model is not None

        def _call(idx: int, messages: Message) -> tuple[int, str]:
            msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
            kwargs: Dict[str, Any] = dict(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                stop=stop,
            )
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            response = self.model.chat.completions.create(**kwargs)
            return idx, response.choices[0].message.content

        results = [None] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call, i, m): i for i, m in enumerate(messages_batch)}
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
        return results

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
        if self.mode == "api":
            return self._chat_batch_api(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
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
            return [o.outputs[0].text for o in self.model.generate(prompts, sp, lora_request=self.lora_request)]

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
