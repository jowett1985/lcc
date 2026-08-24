import subprocess
import os
import json
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
            raise ValueError(
                f"Unknown tool: {name}"
            )
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
        arguments = json.loads(
            raw_arguments
        )
    except json.JSONDecodeError as e:
        # This generally means that the model generated
        # malformed JSON arguments.
        raise ValueError(
            f"Invalid arguments for "
            f"tool '{name}': "
            f"{raw_arguments}"
        ) from e

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


def execute_bash(
    command: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Execute a bash command and return stdout/stderr/exit code.

    WARNING:
    This executes arbitrary commands with the privileges of
    the Python process. Only expose this to a trusted model/user.
    """

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                f"Command timed out after "
                f"{timeout} seconds"
            ),
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
        }