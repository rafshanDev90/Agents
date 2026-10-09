class AgentSession:
    def __init__(self, agent):
        self.agent = agent
        self.history = []

    def run(self, query: str, deps) -> str:
        result = self.agent.run_sync(query, deps=deps, message_history=self.history)
        self.history.extend(result.new_messages())
        return result.output