from llm import LLMService


class BasePipeline:
    def __init__(self, llm: LLMService, domain: str = "coding"):
        self.llm = llm
        self.domain = domain.lower()

    def run(self, input_item: dict):
        raise NotImplementedError
