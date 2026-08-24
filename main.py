#!/usr/bin/env python3

import os
from rich.table import Table
from rich.console import Console
from rich import box
from agent_loop.agent import chat_with_tools, ChatCompletionsStreamClient
from agent_loop.tool_calls import ToolRegistry, execute_bash
from agent_loop.constants import MODEL, TOOLS, BASE_URL, TIMEOUT


def main():
    console = Console()
    table = Table(">_ LCC", box=box.SQUARE, show_lines=False)
    table.add_row("model", "qwen3.8 27B")
    table.add_row("directory", os.getcwd())
    console.print(table)

    client = ChatCompletionsStreamClient(BASE_URL, TIMEOUT)
    registry = ToolRegistry()
    registry.register("bash", execute_bash)

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
        chat_with_tools(
            client,
            registry,
            MODEL,
            history,
            TOOLS,
        )


if __name__ == "__main__":
    main()
