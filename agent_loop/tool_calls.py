import subprocess
import json
import glob as g
from pathlib import Path
from typing import Any, Callable
from .constants import WORKDIR, TIMEOUT, MAX_TOOL_OUTPUT, DENY_LIST, DESTRUCTIVE_COMMAND_WORD


PERMISSION_RULES = [
    {
        "tools": ["read_file", "write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / json.loads(args)["path"]).resolve().is_relative_to(WORKDIR),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: contains_destructive_command(json.loads(args)["command"]) or any(
            kw in args for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list."
        return None


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def check_rules(tool_name: str, args: dict) -> str | None:
    print(f"check_rules. tool_name: {tool_name}, args: {args}")
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


def check_permission(tool_call) -> bool:
    if tool_call["name"] == "bash":
        reason = check_deny_list(tool_call["arguments"])
        if reason:
            print(f"\n⛔ {reason}")
            return False

    reason = check_rules(tool_call["name"], tool_call["arguments"])
    if reason:
        decision = ask_user(tool_call["name"], tool_call["arguments"], reason)
        if decision == "deny":
            return False

    return True


def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


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


def _truncate(text: str) -> str:
    """Keep the head and tail of oversized tool output."""
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    omitted = len(text) - MAX_TOOL_OUTPUT
    return f"{text[:half]}\n... [truncated {omitted} chars] ...\n{text[-half:]}"


def execute_bash(
    command: str,
    timeout: int = TIMEOUT,
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
            command,
            shell=True,
            executable="/bin/bash",
            cwd=WORKDIR,
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

def run_edit(path, old_content, new_content):
    text = safe_path(path).read_text(encoding="utf-8")
    if old_content not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_content, new_content, 1), encoding="utf-8")
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
