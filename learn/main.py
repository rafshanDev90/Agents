"""Local project-notes assistant.

This agent helps a small team turn natural-language requests into project notes,
checklists, and reports, and retrieve those files later. File access is confined
to the sandbox, while saved sessions let conversations resume after restarting.
"""

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from pathlib import Path
from sandbox import SandboxedFileTools
from session import agentSession
from storage_engine import PersistentSessionManager
from search_engine import SearchEngine
import json


class SeniorAgent:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.search_engine = SearchEngine()
        self.console = Console()
        self.file_sandbox = SandboxedFileTools()

        # Registry mapping function names to execution functions
        self.available_tools = {
            "add_numbers": self.add_numbers,
            "write_file": self.file_sandbox.write_file,
            "read_file": self.file_sandbox.read_file,
            "search_web": self.search_web
        }

        schema_path = Path(__file__).with_name("tool_schemas.json")
        with schema_path.open(encoding="utf-8") as schema_file:
            self.tool_schemas = json.load(schema_file)

    def add_numbers(self, a: float, b: float) -> str:
        return str(float(a) + float(b))

    def search_web(self, query: str) -> str:
        """Wrapper method executing safe interface operations with the SearchEngine."""
        try:
            result = self.search_engine.search(query)
            return str(result)
        except Exception as e:
            return f"Search execution failed: {str(e)}"

    def run(self, session: agentSession, user_input: str) -> str:
        session.add_message("user", user_input)

        with self.console.status("[bold green]Agent is thinking...") as status:
            while True:
                response = self.client.chat.completions.create(
                    model="qwen3.5:4b",
                    messages=session.history,
                    tools=self.tool_schemas,
                )

                message = response.choices[0].message

                if message.tool_calls:
                    session.add_llm_response(message)

                    for tool_call in message.tool_calls:
                        name = tool_call.function.name
                        status.update(f"[bold yellow]Executing: {name}...")

                        try:
                            if name not in self.available_tools:
                                raise ValueError(f"Unknown tool: {name}")
                            args = json.loads(tool_call.function.arguments or "{}")
                            
                            # Execute dynamically
                            result = self.available_tools[name](**args)
                            content = str(result)
                        except Exception as error:
                            content = f"Tool failure ({name}): {error}"
                            self.console.print(f"[bold red]{content}")

                        session.add_tool_call(tool_call.id, name, content)

                    status.update("[bold green]Processing changes...")
                    continue

                session.add_message("assistant", message.content or "")
                return message.content or ""

if __name__ == "__main__":
    from rich.markdown import Markdown  # Ensure Markdown is imported
    
    agent = SeniorAgent()
    manager = PersistentSessionManager()
    
    # Load or initialize session
    session_id = "user_dev_session_01"
    session = manager.load(session_id)
    
    agent.console.print(f"[bold blue]Active Session: {session.session_id}[/bold blue]\n")
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                manager.save(session)
                agent.console.print("[bold red]Session persisted. Goodbye![/bold red]")
                break

            # 1. Fetch the raw response string from the agent loop
            response = agent.run(session, user_input)
            
            # 2. Print a clean, professional separator prefix
            agent.console.print("\n[bold green]Agent:[/bold green]")
            
            # 3. Parse the Markdown and render it beautifully through the console panel
            beautiful_markdown = Markdown(response)
            agent.console.print(beautiful_markdown)
            
            # Add an extra newline for clean visual breathing room in the CLI
            agent.console.print("")
            
            # Auto-save checkpoint after every turn
            manager.save(session) 
            
    except KeyboardInterrupt:
        manager.save(session)
        agent.console.print("\n[bold red]Interrupted. Session saved safely.[/bold red]")
