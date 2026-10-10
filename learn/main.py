import json  # Added to parse the tool arguments string
from openai import OpenAI
from rich.console import Console  # Imported Rich Console

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
console = Console()  # Initialized Rich Console

history = [{
    "role": "system",
    "content": "You are a helpful assistant."
}]

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Use this tool to add two numerical numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

AVAILABLE_TOOLS = {
    "add_numbers": add_numbers,
}

def chat_with_model(user_input: str) -> str:
    history.append({"role": "user", "content": user_input})

    # Wrap the entire LLM + Tool Execution loop in the console status animation
    with console.status("[bold green]Agent is thinking...") as status:
        while True:
            response = client.chat.completions.create(
                model="qwen3.5:4b",
                messages=history,
                tools=tool_schemas
            )
            
            message = response.choices[0].message
        
            # Check if the model decided to call a tool instead of replying with text
            if message.tool_calls:
                # 1. Save the model's tool call intent to history
                history.append(message) 

                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    
                    # Dynamically update the spinner text to show which tool is running
                    status.update(f"[bold yellow]Agent is running tool: {name}...")

                    try:
                        if name not in AVAILABLE_TOOLS:
                            raise ValueError(f"Unknown tool: {name}")
                        args = json.loads(tool_call.function.arguments or "{}")
                        result = AVAILABLE_TOOLS[name](**args)
                        content = str(result)
                    except (ValueError, TypeError, json.JSONDecodeError) as error:
                        content = f"Tool {name} failed: {error}"
                        console.print(f"[bold red]{content}") # Styled print
                    except Exception as error:
                        content = f"Tool {name} failed: {error}"
                        console.print(f"[bold red]{content}") # Styled print

                    # Return one tool result for every tool call so the model can continue.
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": content
                    })
                
                # Reset the status text for the next LLM iteration
                status.update("[bold green]Agent is processing tool results...")
                continue  # Go back to the start of the loop to get the model's next response

            else:
                history.append({"role": "assistant", "content": message.content})
                return message.content


if __name__ == "__main__":
    console.print("[bold cyan]Chat started. Type 'exit' or 'quit' to stop.[/bold cyan]")
    try:
        while True:
            user_input = input("\nyou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                console.print("[bold red]Goodbye![/bold red]")
                break
            if not user_input:
                continue
            
            # Get the answer (the spinner runs inside this function call)
            response_content = chat_with_model(user_input)
            
            # Print the final output cleanly using Rich
            console.print(f"[bold blue]assistant:[/bold blue] {response_content}")
            
    except (EOFError, KeyboardInterrupt):
        console.print("\n[bold red]Goodbye![/bold red]")
