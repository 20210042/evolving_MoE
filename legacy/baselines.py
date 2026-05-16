from src.agents.base import Agent
from src.prompts import baseline_prompts
from src.utils.helpers import check_stop_condition
import re

class BasePipeline:
    def __init__(self, agent: Agent, domain: str = "coding"):
        self.agent = agent
        self.domain = domain.lower()

    def run(self, input_item: dict):
        raise NotImplementedError
    def check_stop_condition(self, feedback: str) -> bool:
        return check_stop_condition(feedback)


class RawPipeline(BasePipeline):
    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

        
        messages = []
        is_qwen3 = "qwen3" in self.agent.llm.model_name.lower()
        if self.domain == "coding":
            # Detect if it's LiveCodeBench to use appropriate I/O format
            # In baselines, we stay 'Stock' (no special persona/ChatML)
            if "### Problem Description" in instruction or "stdin" in instruction.lower() or "livecodebench" in str(input_item.get("id", "")):
                gen_sys = baseline_prompts.LCB_GEN_SYSTEM
                gen_user = baseline_prompts.LCB_GEN_USER.format(instruction=instruction)
            else:
                gen_sys = baseline_prompts.CODING_GEN_SYSTEM
                gen_user = baseline_prompts.CODING_GEN_USER.format(instruction=instruction)
            
            if is_qwen3:
                # Use neutral ChatML for Qwen3 but keep Stock identity
                messages = (
                    f"<|im_start|>system\n{gen_sys}\n<|im_end|>\n"
                    f"<|im_start|>user\n{gen_user}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n```python\n"
                )
            else:
                messages = [
                    {"role": "system", "content": gen_sys},
                    {"role": "user", "content": gen_user}
                ]
        elif self.domain == "math":
            messages = [
                {"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM},
                {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)}
            ]
        elif self.domain == "ds":
            messages = [
                {"role": "system", "content": baseline_prompts.DS1000_GEN_SYSTEM},
                {"role": "user", "content": baseline_prompts.DS1000_GEN_USER.format(instruction=instruction)}
            ]
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
            
        output = self.agent.chat(messages, temperature=0.0)
        return {
            "id": input_item.get("id"),
            "initial_output": output,
            "final_output": output
        }

class SelfRefinePipeline(BasePipeline):
    def __init__(self, agent: Agent, domain: str = "coding", max_iterations: int = 4):
        super().__init__(agent, domain)
        self.max_iterations = max_iterations

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

        
        # 1. Initial Generation
        if self.domain == "coding":
            if "### Problem Description" in instruction or "stdin" in instruction.lower() or "livecodebench" in str(input_item.get("id", "")):
                gen_system = baseline_prompts.LCB_GEN_SYSTEM
                gen_user = baseline_prompts.LCB_GEN_USER.format(instruction=instruction)
            else:
                gen_system = baseline_prompts.CODING_GEN_SYSTEM
                gen_user = baseline_prompts.CODING_GEN_USER.format(instruction=instruction)
        elif self.domain == "math":
            gen_system = baseline_prompts.MATH_GEN_SYSTEM
            gen_user = baseline_prompts.MATH_GEN_USER.format(instruction=instruction)
        elif self.domain == "ds":
            gen_system = baseline_prompts.DS1000_GEN_SYSTEM
            gen_user = baseline_prompts.DS1000_GEN_USER.format(instruction=instruction)
            
        is_qwen3 = "qwen3" in self.agent.llm.model_name.lower()
        if is_qwen3:
            # Use neutral ChatML for Qwen3 but keep Stock identity
            gen_msgs = (
                f"<|im_start|>system\n{gen_system}\n<|im_end|>\n"
                f"<|im_start|>user\n{gen_user}\n<|im_end|>\n"
                f"<|im_start|>assistant\n```python\n"
            )
        else:
            gen_msgs = [{"role": "system", "content": gen_system}, {"role": "user", "content": gen_user}]
            
        current_output = self.agent.chat(gen_msgs, temperature=0.0)
        history = [{"step": "initial", "output": current_output}]

        for i in range(self.max_iterations):
            # 2. Critic (No Persona/Neutral)
            if self.domain == "coding" or self.domain == "ds":
                if is_qwen3:
                    critic_msgs = (
                        f"<|im_start|>system\n{baseline_prompts.CODING_CRITIC_SYSTEM}\n<|im_end|>\n"
                        f"<|im_start|>user\n{baseline_prompts.CODING_CRITIC_USER.format(instruction=instruction, code=current_output)}\n<|im_end|>\n"
                        f"<|im_start|>assistant\nFeedback: "
                    )
                else:
                    critic_msgs = [
                        {"role": "system", "content": baseline_prompts.CODING_CRITIC_SYSTEM},
                        {"role": "user", "content": baseline_prompts.CODING_CRITIC_USER.format(instruction=instruction, code=current_output)}
                    ]
            else:
                critic_msgs = [
                    {"role": "system", "content": baseline_prompts.MATH_CRITIC_SYSTEM},
                    {"role": "user", "content": baseline_prompts.MATH_CRITIC_USER.format(instruction=instruction, solution=current_output)}
                ]
            
            feedback = self.agent.chat(critic_msgs)
            
            # Use Textual Setop
            stop = self.check_stop_condition(feedback)
            history.append({"step": f"critic_{i}", "feedback": feedback, "stop_signal": stop})

            if stop:
                print(f"Stopping early at iteration {i} due to positive feedback.")
                break

            # 3. Refine
            if self.domain == "coding" or self.domain == "ds":
                if is_qwen3:
                    ref_msgs = (
                        f"<|im_start|>system\n{baseline_prompts.CODING_REVISION_SYSTEM}\n<|im_end|>\n"
                        f"<|im_start|>user\n{baseline_prompts.CODING_REVISION_USER.format(instruction=instruction, code=current_output, feedback=feedback)}\n<|im_end|>\n"
                        f"<|im_start|>assistant\n```python\n"
                    )
                else:
                    ref_msgs = [
                        {"role": "system", "content": baseline_prompts.CODING_REVISION_SYSTEM},
                        {"role": "user", "content": baseline_prompts.CODING_REVISION_USER.format(instruction=instruction, code=current_output, feedback=feedback)}
                    ]
            else:
                ref_msgs = [
                    {"role": "system", "content": baseline_prompts.MATH_REVISION_SYSTEM},
                    {"role": "user", "content": baseline_prompts.MATH_REVISION_USER.format(instruction=instruction, solution=current_output, feedback=feedback)}
                ]
            
            refined_output = self.agent.chat(ref_msgs, temperature=0.0)
            current_output = refined_output
            history.append({"step": f"refine_{i}", "output": current_output})

        return {
            "id": input_item.get("id"),
            "initial_output": history[0]["output"],
            "final_output": current_output,
            "history": history
        }
