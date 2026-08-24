import os

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("MODEL", "/run/media/limeng/DATA/Qwen3.8-27B-UD-IQ2_XXS.gguf")
TIMEOUT = float(os.environ.get("TIMEOUT", "1024"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "-1"))
SYSTEM = "You are a helpful assistant. Use tools when necessary."

# OpenAI-style tool definition: a single bash tool.
TOOLS = [{
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
}]