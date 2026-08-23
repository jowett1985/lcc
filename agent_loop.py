#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import requests
from rich.table import Table
from rich.console import Console
from rich import box

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MODEL", "/run/media/limeng/DATA/Qwen3.8-27B-UD-IQ2_XXS.gguf")
CONNECT_TIMEOUT = float(os.environ.get("CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.environ.get("READ_TIMEOUT", "90"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "-1"))
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}."
    " Use the bash tool to solve tasks. Act, don't explain."
)

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


def run_bash(command: str) -> str:
    """Execute a shell command and return its combined output."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: dangerous command blocked"
    try:
        r = subprocess.run(
            ["bash","-c", command], shell=False, cwd=os.getcwd(),
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


class ChatError(RuntimeError):
    """The local OpenAI-compatible server could not complete a request."""


def chat(messages: list) -> dict:
    """One chat-completions call to the local server."""
    url = BASE_URL.rstrip("/") + "/chat/completions"
    request_messages = messages
    if not messages or messages[0].get("role") != "system":
        request_messages = [{"role": "system", "content": SYSTEM}] + messages
    try:
        resp = requests.post(
            url,
            json={
                "model": MODEL,
                "messages": request_messages,
                "tools": TOOLS,
                "max_tokens": MAX_TOKENS,
                "stream": True
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            stream=True
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            payload = line.removeprefix("data: ")
            json_data = json.loads(payload)
            if not json_data["choices"][0]["finish_reason"]:
                reasoning_content = json_data["choices"][0]["delta"].get("reasoning_content")
                tool_calls = json_data["choices"][0]["delta"].get("tool_calls")
                if reasoning_content:
                    print(reasoning_content, end="", flush=True)
                if tool_calls:
                    print(tool_calls[0][""], end="", flush=True)
            else:
                break
        
        return ""
    except requests.exceptions.ConnectTimeout as e:
        raise ChatError(f"Timed out connecting to {url} after {CONNECT_TIMEOUT:g}s") from e
    except requests.exceptions.ReadTimeout as e:
        raise ChatError(
            f"No response from {url} after {READ_TIMEOUT:g}s; "
            "check that the model server is healthy or raise READ_TIMEOUT."
        ) from e
    except requests.exceptions.RequestException as e:
        raise ChatError(f"Request to {url} failed: {e}") from e
    except ValueError as e:
        print(e)
        raise ChatError(f"Server at {url} returned invalid JSON") from e


def _parse_args(raw) -> dict:
    """llama.cpp may return arguments as a JSON string or a dict."""
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def agent_loop(messages: list) -> dict:
    """Run the loop until the model stops calling tools.

    Mutates `messages` (the shared transcript) in place and returns the
    final assistant message (dict).
    """
    body = chat(messages)
    #msg = body["choices"][0]["message"]
    #tool_calls = msg.get("tool_calls")

    # Append the assistant turn to the transcript.
    '''messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls or [],
    })'''

        # No tool calls -> the model is done.
        #if not tool_calls:
        #    return msg

        # Execute each tool call and feed the results back.
    '''results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = _parse_args(tc["function"].get("arguments"))
            if name == "bash":
                cmd = args.get("command", "")
                print(f"\033[33m$ {cmd}\033[0m")
                out = run_bash(cmd)
            else:
                out = f"Error: unknown tool {name}"
            print(out[:200])
            results.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": out,
            })
    messages.extend(results)'''

    # Safety: ran out of steps.
    return {"role": "assistant", "content": "(stopped: max steps reached)"}


def _final_text(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return str(msg)


def main():
    console = Console()
    table = Table(">_ LCC", box=box.SQUARE, show_lines=False)
    table.add_row("model", "qwen3.8 27B")
    table.add_row("directory", os.getcwd())
    console.print(table)

    # REPL.
    print("Enter a task, press Enter to run. Type q to quit.\n")
    history = []
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)


if __name__ == "__main__":
    main()
