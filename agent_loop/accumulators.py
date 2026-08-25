from typing import Any, Optional


class ToolCallAccumulator:
    def __init__(self):
        # index -> complete tool call
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add(
        self,
        delta: dict[str, Any],
    ):
        """
        Accumulate one streamed tool_call delta.

        A tool call can arrive as:

            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"ci"
                }
            }

        followed by:

            {
                "index": 0,
                "function": {
                    "arguments": "ty\":\"San"
                }
            }

        etc.
        """

        index = delta.get("index")

        if index is None:
            raise ValueError("Tool call delta has no index")

        if index not in self.tool_calls:
            self.tool_calls[index] = {
                "id": None,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        current = self.tool_calls[index]

        # ----------------------------------------------------
        # Tool call ID
        # ----------------------------------------------------
        if delta.get("id"):
            current["id"] = delta["id"]

        # ----------------------------------------------------
        # Tool call type
        # ----------------------------------------------------
        if delta.get("type"):
            current["type"] = delta["type"]

        # ----------------------------------------------------
        # Function
        # ----------------------------------------------------
        function_delta = delta.get(
            "function",
            {},
        )

        # Function name
        if function_delta.get("name"):
            current["function"]["name"] += function_delta["name"]

        # Function arguments
        if function_delta.get("arguments"):
            current["function"]["arguments"] += function_delta["arguments"]

    def get_tool_calls(
        self,
    ) -> list[dict[str, Any]]:
        return [self.tool_calls[index] for index in sorted(self.tool_calls.keys())]


class StreamAccumulator:
    def __init__(self):
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.content = ""
        self.reasoning_content = ""
        self.tool_calls = ToolCallAccumulator()
        self.finish_reason: Optional[str] = None
        self.usage: Optional[dict[str, Any]] = None

    def process_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events = []

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------
        if chunk.get("id"):
            self.id = chunk["id"]
        if chunk.get("model"):
            self.model = chunk["model"]

        # ----------------------------------------------------
        # Usage
        # ----------------------------------------------------
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        # ----------------------------------------------------
        # Choices
        # ----------------------------------------------------
        choices = chunk.get(
            "choices",
            [],
        )

        # Some OpenAI-compatible servers can send
        # a final usage-only chunk with choices=[].
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get(
            "delta",
            {},
        )

        # ----------------------------------------------------
        # Normal text
        # ----------------------------------------------------
        content = delta.get("content")
        if content:
            self.content += content
            events.append(
                {
                    "type": "content",
                    "content": content,
                }
            )

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------
        reasoning = delta.get("reasoning_content")
        if reasoning:
            self.reasoning_content += reasoning
            events.append(
                {
                    "type": "reasoning",
                    "content": reasoning,
                }
            )

        # ----------------------------------------------------
        # Tool calls
        # ----------------------------------------------------

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                self.tool_calls.add(tool_call)
                events.append(
                    {
                        "type": "tool_call_delta",
                        "index": tool_call.get("index"),
                        "id": tool_call.get("id"),
                        "function": tool_call.get(
                            "function",
                            {},
                        ),
                    }
                )

        # ----------------------------------------------------
        # Finish reason
        # ----------------------------------------------------
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason
            events.append(
                {
                    "type": "finish",
                    "reason": finish_reason,
                }
            )

        return events

    def get_assistant_message(
        self,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": (self.content if self.content else None),
        }

        tool_calls = self.tool_calls.get_tool_calls()

        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    def get_result(
        self,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "content": self.content,
            "reasoning_content": (self.reasoning_content),
            "tool_calls": (self.tool_calls.get_tool_calls()),
            "finish_reason": (self.finish_reason),
            "usage": self.usage,
        }
