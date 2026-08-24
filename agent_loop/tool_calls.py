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