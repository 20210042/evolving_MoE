from meta_agent_evo.agents.base import Agent


class BasePipeline:
    def __init__(self, agent: Agent, domain: str = "coding"):
        self.agent = agent
        self.domain = domain.lower()

    def run(self, input_item: dict):
        raise NotImplementedError
