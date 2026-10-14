import json
import uuid
from typing import List, Dict, Any
from openai import OpenAI
from rich.console import Console


class agentSession:
    def __init__(self,session_id: str = None, system_prompt: str = "You are a helpful assistant."):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def add_llm_response(self, message_obj:Any):
        self.history.append(message_obj)
        
    def add_tool_call(self, tool_call_id: str, name: str, content: str):
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "tool_name": name,
                "content": content,
            }
        )


class Agent:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.console = Console()
        self.history = [{"role": "system", "content": system_prompt}]
        self.available_tools = {"add_numbers": self.add_numbers}
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Use this tool to add two numerical numbers together.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "The first number"},
                            "b": {"type": "number", "description": "The second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

    def add_numbers(self, a: float, b: float) -> float:
        """Adds two numbers together."""
        return float(a) + float(b)

    def chat_with_model(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        with self.console.status("[bold green]Agent is thinking...") as status:
            while True:
                response = self.client.chat.completions.create(
                    model="qwen3.5:4b",
                    messages=self.history,
                    tools=self.tool_schemas,
                )

                message = response.choices[0].message

                if message.tool_calls:
                    self.history.append(message)

                    for tool_call in message.tool_calls:
                        name = tool_call.function.name
                        status.update(f"[bold yellow]Agent is running tool: {name}...")

                        try:
                            if name not in self.available_tools:
                                raise ValueError(f"Unknown tool: {name}")
                            args = json.loads(tool_call.function.arguments or "{}")
                            result = self.available_tools[name](**args)
                            content = str(result)
                        except (ValueError, TypeError, json.JSONDecodeError) as error:
                            content = f"Tool {name} failed: {error}"
                            self.console.print(f"[bold red]{content}")
                        except Exception as error:
                            content = f"Tool {name} failed: {error}"
                            self.console.print(f"[bold red]{content}")

                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": name,
                                "tool_name": name,
                                "content": content,
                            }
                        )

                    status.update("[bold green]Agent is processing tool results...")
                    continue

                self.history.append({"role": "assistant", "content": message.content})
                return message.content or ""

if __name__ == "__main__":
    agent = Agent()
    console = agent.console
    console.print("[bold blue]Starting the agent session...[/bold blue]")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting the agent session.[/bold red]")
                break

            response = agent.chat_with_model(user_input)
            console.print(f"[bold green]Agent: {response}[/bold green]")
    except KeyboardInterrupt:
        console.print("[bold red]Session interrupted by user.[/bold red]")
    except Exception as error:
        console.print(f"[bold red]An error occurred: {error}[/bold red]")