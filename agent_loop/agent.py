import json
import httpx
from typing import Any, Optional, Iterator

from .accumulators import StreamAccumulator
from .tool_calls import ToolRegistry, execute_tool_call
from utils.printer import Printer


class ChatCompletionsStreamClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.client = httpx.Client(timeout=self.timeout)

    def close(self):
        self.client.close()

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

        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = self.api_key

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

        with self.client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
        ) as response:
            if response.is_error:
                detail = response.read().decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"{response.status_code} response from {url}: {detail}",
                    request=response.request,
                    response=response,
                )
            for line in response.iter_lines():
                # SSE uses blank lines to separate events.
                if not line:
                    continue
                # We only care about "data:" events.
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
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
    printer: Printer,
    max_tool_rounds: int = 1024,
) -> dict[str, Any]:
    """
    Complete agent loop.

    The function modifies `messages` in-place.

    It keeps calling the model until the model returns
    a normal assistant response instead of tool_calls.

    Raises RuntimeError if no final answer is produced within
    ``max_tool_rounds`` model calls.
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
                    printer.response(event["content"])
                elif event["type"] == "reasoning":
                    printer.thinking(event["content"])
        printer.newline()

        # ====================================================
        # 2. Inspect complete response
        # ====================================================
        result = accumulator.get_result()
        finish_reason = result["finish_reason"]

        # ====================================================
        # 3. Normal assistant response
        # ====================================================
        if not result["tool_calls"] and finish_reason != "tool_calls":
            # Add assistant response to conversation.
            messages.append(accumulator.get_assistant_message())
            return result

        # ====================================================
        # 4. Model requested tools
        # ====================================================
        tool_calls = result["tool_calls"]
        if not tool_calls:
            raise RuntimeError(
                "finish_reason='tool_calls' but no tool_calls were received"
            )

        # ----------------------------------------------------
        # Add assistant's tool-call message
        # ----------------------------------------------------
        assistant_message = accumulator.get_assistant_message()
        messages.append(assistant_message)

        # ====================================================
        # 5. Execute every tool call
        # ====================================================
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            params = tool_call["function"]["arguments"]
            printer.tool_name(f"{function_name} {params}")
            try:
                tool_result = execute_tool_call(registry, tool_call)
            except Exception as e:
                # Send tool errors back to the model
                # instead of crashing the whole conversation.
                tool_result = json.dumps(
                    {
                        "error": str(e),
                    }
                )

            printer.tool_result(tool_result)
            printer.newline()

            # ------------------------------------------------
            # Add tool result to conversation
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (tool_call_id),
                    "content": tool_result,
                }
            )

        # ====================================================
        # 6. Loop back to model
        # ====================================================

    raise RuntimeError(
        f"exceeded {max_tool_rounds} model rounds without a final answer"
    )
