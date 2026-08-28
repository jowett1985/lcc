import os

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MODEL", "Qwen3.8-27B-4bit")
TIMEOUT = float(os.environ.get("TIMEOUT", "1024"))
SYSTEM = "You are a helpful assistant. Use tools when necessary."

# OpenAI-style tool definition: a single bash tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
