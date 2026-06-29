"""Model generation service used by evaluation scripts.

Wraps vLLM and HuggingFace text-generation behind one `LLMService` interface, with optional LoRA
loading for ACC algorithm checkpoints. The algorithm evaluator uses this class for batched chat
prompt generation while preserving the older HF fallback path."""

from typing import List, Optional

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    LoRARequest = None
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class LLMService:
    def __init__(
        self,
        model_name: str,
        mode: str = "vllm",
        tp_size: int = 1,
        max_model_len: int = 16384,
        lora_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.lora_request = None
        
        ## vllm mode
        if self.mode == "vllm":
            if LLM is None: raise ImportError("vllm is not installed.")
            
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                enable_lora=lora_path is not None,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
            )
            
            self.tokenizer = self.model.get_tokenizer()
            
            if lora_path:
                self.lora_request = LoRARequest("adapter", 1, lora_path)
                
        ## hf mode
        elif self.mode == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            if lora_path:
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, lora_path)
                base_model = base_model.merge_and_unload()
            
            
            self.model = pipeline(
                "text-generation",
                model=base_model,
                tokenizer=self.tokenizer,
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
            outputs = self.model.generate(prompts, sampling_params, lora_request=self.lora_request)
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
