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
from session_store import SessionStore
from commands import AppContext, CommandDispatcher, CommandCompleter, install_readline
from search_engine import SearchEngine
from timing import TurnTrace
import json
import time
import uuid

MAX_TOOL_ROUNDS = 5


class SeniorAgent:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.search_engine = SearchEngine()
        self.console = Console()
        self.file_sandbox = SandboxedFileTools()
        self.last_trace = None

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

    @staticmethod
    def _iter_json_objects(text: str):
        """Yield whole JSON objects found in text, ignoring surrounding prose."""
        decoder = json.JSONDecoder()
        index = 0
        while True:
            start = text.find("{", index)
            if start == -1:
                return
            try:
                obj, end = decoder.raw_decode(text[start:])
            except ValueError:
                index = start + 1
                continue
            if isinstance(obj, dict):
                yield obj
            index = start + end

    def _recover_text_tool_calls(self, content: str) -> list:
        """Rescue tool calls some models print as text instead of structured calls.

        qwen2.5-coder via Ollama emits `<tool_call>{...}</tool_call>` (or bare JSON)
        that Ollama's parser drops, leaving the call in `message.content`. We accept
        only objects naming a known tool, so ordinary JSON answers stay answers.
        """
        calls = []
        for obj in self._iter_json_objects(content or ""):
            function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
            name = obj.get("name") or function.get("name")
            if name not in self.available_tools:
                continue
            args = obj.get("arguments", function.get("arguments", obj.get("parameters")))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
        return calls

    def run(self, session: agentSession, user_input: str) -> str:
        session.add_message("user", user_input)
        trace = TurnTrace()
        self.last_trace = trace

        rounds = 0
        with self.console.status("[bold green]Agent is thinking...") as status:
            while True:
                if rounds > MAX_TOOL_ROUNDS:
                    notice = (
                        f"Stopped after {MAX_TOOL_ROUNDS} tool rounds to avoid "
                        "running indefinitely. Please narrow the request and try again."
                    )
                    session.add_message("assistant", notice)
                    return notice

                call_started = time.perf_counter()
                response = self.client.chat.completions.create(
                    model="qwen3:8b",
                    messages=session.history,
                    tools=self.tool_schemas,
                )
                trace.record_round(time.perf_counter() - call_started, response.usage)

                message = response.choices[0].message
                content = message.content or ""

                if message.tool_calls:
                    session.add_llm_response(message)
                    calls = [
                        {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                        for tool_call in message.tool_calls
                    ]
                else:
                    calls = self._recover_text_tool_calls(content)
                    if calls:
                        session.add_llm_response(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": call["name"],
                                            "arguments": call["arguments"],
                                        },
                                    }
                                    for call in calls
                                ],
                            }
                        )

                if calls:
                    for call in calls:
                        name = call["name"]
                        status.update(f"[bold yellow]Executing: {name}...")

                        try:
                            if name not in self.available_tools:
                                raise ValueError(f"Unknown tool: {name}")
                            args = json.loads(call["arguments"] or "{}")

                            # Execute dynamically, timed even if the tool raises
                            tool_started = time.perf_counter()
                            try:
                                result = self.available_tools[name](**args)
                            finally:
                                trace.record_tool(name, time.perf_counter() - tool_started)
                            tool_result = str(result)
                        except Exception as error:
                            tool_result = f"Tool failure ({name}): {error}"
                            self.console.print(f"[bold red]{tool_result}")

                        session.add_tool_call(call["id"], name, tool_result)

                    rounds += 1
                    status.update("[bold green]Processing changes...")
                    continue

                session.add_message("assistant", content)
                return content

if __name__ == "__main__":
    from rich.markdown import Markdown  # Ensure Markdown is imported
    
    agent = SeniorAgent()
    store = SessionStore()
    migrated = store.import_legacy_json()
    if migrated:
        agent.console.print(
            f"[dim]Imported {migrated} legacy JSON session(s) into SQLite[/dim]"
        )

    # Every launch starts fresh; old conversations are one /session open away
    ctx = AppContext(store=store, session=store.fresh(), console=agent.console, agent=agent)
    dispatcher = CommandDispatcher(ctx)
    install_readline(CommandCompleter(ctx, dispatcher.command_names))

    agent.console.print(
        f"[bold blue]Fresh session:[/bold blue] {ctx.session.session_id}"
        " [dim](nothing loaded — /session lists saved ones, Tab completes commands)[/dim]"
    )
    agent.console.print("[dim]Type /help for commands.[/dim]\n")
    
    try:
        while True:
            user_input = input("You: ")

            # Slash commands run locally — they never reach the model
            if user_input.strip().startswith("/"):
                if dispatcher.dispatch(user_input):
                    saved = ctx.save_session()
                    farewell = "Session persisted." if saved else "Nothing to save."
                    agent.console.print(f"[bold red]{farewell} Goodbye![/bold red]")
                    break
                continue

            if user_input.lower() in ["exit", "quit"]:
                saved = ctx.save_session()
                farewell = "Session persisted." if saved else "Nothing to save."
                agent.console.print(f"[bold red]{farewell} Goodbye![/bold red]")
                break
            if not user_input.strip():
                continue

            # 1. Fetch the raw response string from the agent loop
            response = agent.run(ctx.session, user_input)
            
            # 2. Print a clean, professional separator prefix
            agent.console.print("\n[bold green]Agent:[/bold green]")
            
            # 3. Parse the Markdown and render it beautifully through the console panel
            beautiful_markdown = Markdown(response)
            agent.console.print(beautiful_markdown)
            
            # Add an extra newline for clean visual breathing room in the CLI
            agent.console.print("")
            
            # Auto-save checkpoint after every turn
            save_started = time.perf_counter()
            ctx.save_session()
            if agent.last_trace is not None:
                agent.last_trace.record_save(time.perf_counter() - save_started)
                agent.console.print(agent.last_trace.render(), style="dim")
            
    except (KeyboardInterrupt, EOFError):
        ctx.save_session()
        agent.console.print("\n[bold red]Interrupted. Session saved safely.[/bold red]")
