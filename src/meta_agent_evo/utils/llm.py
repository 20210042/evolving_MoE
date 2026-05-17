from typing import List, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
from transformers import AutoTokenizer, pipeline
import torch


class LLMService:
    def __init__(
        self,
        model_name: str,
        mode: str = "vllm",
        tp_size: int = 1,
        max_model_len: int = 32768,
    ):
        self.model_name = model_name
        self.mode = mode
        self.model = None
        self.tokenizer = None

        if self.mode == "vllm":
            if LLM is None:
                raise ImportError("vllm is not installed.")
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
            )
            self.tokenizer = self.model.get_tokenizer()
        elif self.mode == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ) -> str:
        if isinstance(messages, str):
            return self.generate(
                [messages],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )[0]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )[0]

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
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            outputs = self.model.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]

        if self.mode == "hf":
            results = []
            for prompt in prompts:
                output = self.model(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1e-5,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    stop_sequence=stop,
                )
                results.append(output[0]["generated_text"][len(prompt) :])
            return results

        raise ValueError(f"Unknown mode: {self.mode}")
