from agents.base import Agent


class BasePipeline:
    def __init__(self, agent: Agent, domain: str = "coding"):
        self.agent = agent
        self.domain = domain.lower()

    def run(self, input_item: dict):
        raise NotImplementedError

    def run_batch(self, items: list) -> list:
        """Default: sequential single-item runs. Override for batched vLLM."""
        return [self.run(item) for item in items]
