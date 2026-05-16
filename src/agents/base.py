from src.utils.llm import LLMService

class Agent:
    def __init__(self, llm_service: LLMService, role: str = "Assistant"):
        self.llm = llm_service
        self.llm_service = llm_service
        self.role = role
        

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.llm.generate([prompt], **kwargs)
        return outputs[0]

    def chat(self, messages: list, **kwargs) -> str:
        return self.llm.chat(messages, **kwargs)
