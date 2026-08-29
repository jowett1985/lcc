# LCC

LCC is a small CLI agent loop. It talks to a local [OpenAI-compatible](https://platform.openai.com/docs/api-reference) model server and lets the model drive shell commands and file tools in a REPL.

Built for running local models: it works with any server that exposes a streaming `POST /v1/chat/completions` endpoint — [llama.cpp](https://github.com/ggml-org/llama.cpp) (`server`), [oMLX](https://github.com/mlx-ai/mlx_lm), or similar.

## How it works

1. You enter a task at the `agent >>` prompt.
2. The model's streaming response is sent to the server and printed as it arrives (answer text, reasoning, and tool calls).
3. When the model requests tool calls, LCC executes them, feeds the results back, and keeps looping until the model produces a final answer.
4. The conversation history is kept across turns for the whole session.

## Requirements

- Python ≥ 3.14 (or just [uv](https://docs.astral.sh/uv/) — see below)
- A local OpenAI-compatible model server (e.g. `llama.cpp` serving on `127.0.0.1:8080`)

## Quick start

```sh
# create the venv and install locked dependencies
make sync        # uv venv .venv + uv sync (uses pyproject.toml / uv.lock)

# point at your model server (defaults shown)
export BASE_URL=http://127.0.0.1:8080
export MODEL=Qwen3.8-27B-4bit

# run the REPL
python .venv/bin/python main.py
```

Or run through `uv` directly:

```sh
uv run python main.py
```

Inside the REPL:

```text
agent >> list the python files in this repo and summarize them
```

Type `q` (or `exit`, or an empty input) to quit.

## Configuration

| Environment variable | Default                 | Description                        |
| --------------------- | ------------------------ | ---------------------------------- |
| `BASE_URL`            | `http://127.0.0.1:8080`  | Base URL of the model server       |
| `MODEL`               | `Qwen3.8-27B-4bit`       | Model name passed to the server    |
| `TIMEOUT`             | `1024`                   | HTTP timeout in seconds            |

## Tools

The model can use the following tools, registered in `main.py`:

- `bash` — run a shell command (arbitrary command execution: only expose this to a trusted model/user)
- `read_file` — read file content
- `write_file` — write file content
- `edit_file` — replace `old_content` with `new_content` in a file
- `glob` — find files by pattern

File tools are sandboxed to the current directory (`safe_path`), and tool output is truncated to keep the model's context window manageable.

## Build a standalone binary

```sh
make build   # PyInstaller, one-file executable written to dist/lcc
```

## Project structure

```text
main.py               REPL entry point
agent_loop/
  agent.py            streaming client + agent loop
  accumulators.py     SSE chunk accumulation (content, reasoning, tool calls)
  constants.py        env-configured BASE_URL / MODEL / TIMEOUT, tool schemas
  tool_calls.py       tool registry and tool implementations
utils/
  printer.py          rich-based console output
```

## Make targets

```sh
make help     list targets
make venv     create the project virtual environment
make sync     sync dependencies into .venv
make format   format Python source with Ruff
make build    build a standalone executable in dist/
make clean    remove generated build artifacts
```

## License

[Apache License 2.0](./LICENSE)
