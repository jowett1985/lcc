import subprocess
import os
import json
import shlex
import glob as g
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(os.getcwd()).resolve()


def safe_path(path: str | Path) -> Path:
    """
    Resolves a path and ensures it is within the BASE_DIR.
    """
    target_path = Path(path).resolve()
    if not target_path.is_relative_to(BASE_DIR):
        raise ValueError(f"Access denied: path {path} is outside the sandbox.")
    return target_path


class ToolRegistry:
    """
    Maps tool names to Python functions.
    """

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        function: Callable[..., Any],
    ):
        self._tools[name] = function

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)


def execute_tool_call(
    registry: ToolRegistry,
    tool_call: dict[str, Any],
) -> str:
    function = tool_call["function"]
    name = function["name"]
    raw_arguments = function.get(
        "arguments",
        "{}",
    )

    # --------------------------------------------------------
    # Parse JSON arguments
    # --------------------------------------------------------
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        # This generally means that the model generated
        # malformed JSON arguments.
        raise ValueError(f"Invalid arguments for tool '{name}': {raw_arguments}") from e

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------
    result = registry.execute(
        name,
        arguments,
    )

    # --------------------------------------------------------
    # Convert result to string.
    #
    # The `content` field of a tool message is normally
    # serialized text.
    # --------------------------------------------------------
    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        ensure_ascii=False,
    )


# Cap tool output so a single large result (e.g. `cat bigfile`)
# cannot blow up the model's context window.
MAX_TOOL_OUTPUT = 8000


def _truncate(text: str) -> str:
    """Keep the head and tail of oversized tool output."""
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    omitted = len(text) - MAX_TOOL_OUTPUT
    return f"{text[:half]}\n... [truncated {omitted} chars] ...\n{text[-half:]}"


def execute_bash(
    command: str,
    timeout: int = 30 * 60,
) -> dict[str, Any]:
    """
    Execute a bash command and return stdout/stderr/exit code.

    stdout/stderr are truncated to MAX_TOOL_OUTPUT characters.

    WARNING:
    This executes arbitrary commands with the privileges of
    the Python process. Only expose this to a trusted model/user.
    """

    try:
        result = subprocess.run(
            shlex.split(command),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": result.returncode,
            "stdout": _truncate(result.stdout),
            "stderr": _truncate(result.stderr),
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": (f"Command timed out after {timeout} seconds"),
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
        }

def run_read(path, limit=None):
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)

def run_write(path, content):
    safe_path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"

def run_edit(path, old_text, new_text):
    text = safe_path(path).read_text(encoding="utf-8")
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"

def run_glob(pattern):
    all_matches = g.glob(pattern, recursive=True)
    safe_matches = []
    for m in all_matches:
        try:
            p = safe_path(m)
            safe_matches.append(str(p))
        except ValueError:
            continue
            
    matches = sorted(set(safe_matches))
    shown = matches[:200]
    if len(matches) > 200:
        shown.append("... (more matches omitted; narrow the pattern)")
    return "\n".join(shown)
