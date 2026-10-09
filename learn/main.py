from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

history = [{
    "role":"system",
    "content":"You are a helpful assistant."
}]

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

tool = [
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

def chat_with_model(user_input:str) ->str:
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="qwen3.5:4b",
        messages=history,
        tools=tool
    )
    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message

print(chat_with_model("Hello,I am Tim"))
print (chat_with_model("What is the sum of 5 and 10?"))
print(history.count({"role": "user", "content": "Hello,I am Tim"}))