import subprocess
import json
import shlex
from typing import Any, Callable


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
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Execute a command and return stdout/stderr/exit code.

    Note: This implementation uses shlex to split the command, which
    means shell features like pipes (|), redirections (>), and
    environment variables ($VAR) are NOT supported. This is a security
    measure to prevent command injection.

    stdout/stderr are truncated to MAX_TOOL_OUTPUT characters.

    WARNING:
    This executes commands with the privileges of the Python process.
    Only expose this to a trusted model/user.
    """

    try:
        args = shlex.split(command)
        result = subprocess.run(
            args,
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
