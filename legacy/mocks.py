class MockLLMService:
    def __init__(self):
        pass
    def generate(self, prompts, **kwargs):
        return ["Mock output" for _ in prompts]
    def chat(self, messages, **kwargs):
        return "Mock output from chat."
