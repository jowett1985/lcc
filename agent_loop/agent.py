import json
import subprocess
import httpx
from rich.table import Table
from rich.console import Console
from rich import box
from typing import Any, Optional, Iterator
from .constants import BASE_URL, SYSTEM, MODEL, TOOLS, MAX_TOKENS, TIMEOUT
from .accumulators import StreamAccumulator
from .tool_calls import ToolRegistry, execute_tool_call


class ChatCompletionsStreamClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def stream_chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Send a streaming request to:

            POST /v1/chat/completions

        and yield parsed SSE JSON chunks.
        """

        url = (
            f"{self.base_url}"
            "/v1/chat/completions"
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature

        with httpx.Client(
            timeout=self.timeout
        ) as client:
            with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    # SSE uses blank lines to separate events.
                    if not line:
                        continue
                    # We only care about "data:" events.
                    if not line.startswith("data:"):
                        continue
                    data = line[
                        len("data:"):
                    ].strip()
                    # OpenAI-compatible termination.
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as e:
                        print(
                            "Invalid SSE JSON:",
                            data,
                            e,
                        )
                        continue
                    yield chunk


def chat_with_tools(
    client: ChatCompletionsStreamClient,
    registry: ToolRegistry,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tool_rounds: int = 10,
) -> dict[str, Any]:
    """
    Complete agent loop.

    The function modifies `messages` in-place.

    It keeps calling the model until the model returns
    a normal assistant response instead of tool_calls.
    """

    for _ in range(max_tool_rounds):
        accumulator = StreamAccumulator()

        # ====================================================
        # 1. Call model
        # ====================================================
        itr = client.stream_chat_completion(
            model=model,
            messages=messages,
            tools=tools,
        )
        for chunk in itr:
            events = accumulator.process_chunk(chunk)
            # -----------------------------------------------
            # Stream normal answer to stdout
            # -----------------------------------------------
            for event in events:
                if event["type"] == "content":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "reasoning":
                    print(event["content"], end="", flush=True)
        print()

        # ====================================================
        # 2. Inspect complete response
        # ====================================================
        result = (
            accumulator.get_result()
        )
        finish_reason = (
            result["finish_reason"]
        )

        # ====================================================
        # 3. Normal assistant response
        # ====================================================
        if finish_reason != "tool_calls":
            # Add assistant response to conversation.
            messages.append(
                accumulator.get_assistant_message()
            )

            return result

        # ====================================================
        # 4. Model requested tools
        # ====================================================
        tool_calls = (
            result["tool_calls"]
        )
        if not tool_calls:
            raise RuntimeError(
                "finish_reason='tool_calls' "
                "but no tool_calls were received"
            )

        # ----------------------------------------------------
        # Add assistant's tool-call message
        # ----------------------------------------------------
        assistant_message = (
            accumulator.get_assistant_message()
        )

        messages.append(
            assistant_message
        )

        # ====================================================
        # 5. Execute every tool call
        # ====================================================
        for tool_call in tool_calls:
            tool_call_id = (
                tool_call["id"]
            )
            function_name = (
                tool_call["function"]["name"]
            )
            print(
                f"[tool] "
                f"{function_name}"
            )
            try:
                tool_result = (
                    execute_tool_call(
                        registry,
                        tool_call,
                    )
                )
            except Exception as e:
                # Send tool errors back to the model
                # instead of crashing the whole conversation.
                tool_result = json.dumps({
                    "error": str(e),
                })

            print(
                f"[tool result] "
                f"{tool_result}"
            )

            # ------------------------------------------------
            # Add tool result to conversation
            # ------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": (
                    tool_call_id
                ),
                "content": tool_result,
            })

        # ====================================================
        # 6. Loop back to model
        # ====================================================

    raise RuntimeError(
        "Maximum tool-call rounds exceeded"
    )
