# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a personal learning repository for exploring AI agent concepts through hands-on experiments. Each sub-project is a self-contained study of a different protocol or pattern, paired with a step-by-step Jupyter notebook that teaches the concept incrementally.

## Environment Setup

All projects share one virtualenv and one `requirements.txt`:

```bash
python3 -m venv <your_env_name>
source <your_env_name>/bin/activate
pip install -r requirements.txt
```

The active virtualenv for this repo is at `/home/klismam/pyenvs/agent_study/`. Always activate it before running any Python code.

Ollama must be running as a background server for the RAG app, MCP app, and A2A orchestrator:

```bash
ollama serve          # start the server
ollama pull llama3.2  # download the model (first time only)
```

Install Ollama itself via `./install_ollama.sh`. Node.js (required for `mcp dev` inspector) via `mcp_app/install_node.sh`.

## Projects

### `rag_app/` — Retrieval-Augmented Generation

A Streamlit web app. Upload PDFs → chunks are embedded with `sentence-transformers` → stored in FAISS → questions answered by `llama3.2` via Ollama.

```bash
streamlit run rag_app/rag_app.py
```

FAISS indexes are persisted to `rag_app/faiss_indexes/` keyed by an MD5 of the uploaded file content hashes, so re-uploading the same PDFs reuses the existing index.

### `mcp_app/` — Model Context Protocol

An MCP server (`calculator_server.py`) exposing arithmetic tools and a math cheat-sheet resource. The notebook (`step_by_step_mcp.ipynb`) walks through server anatomy, client discovery, error handling, resources, stateful tools, and wiring Ollama as an MCP client.

```bash
# Inspect any MCP server interactively (requires Node.js)
mcp dev mcp_app/calculator_server.py
```

### `a2a_app/` — Agent-to-Agent Protocol

Multiple A2A agents communicating over HTTP. The notebook (`step_by_step_a2a.ipynb`) builds up from a single echo agent to a multi-agent system with an LLM-powered orchestrator.

Agent ports (defined in the notebook):
- `10001` — Echo Agent
- `10002` — Math Agent
- `10003` — Writer Agent
- `10004` — Orchestrator Agent
- `10005` — Stats Agent (Task-path demo)

Agents are written as `.py` files by notebook cells and then launched as subprocesses. If a port is stuck after a kernel interrupt, close the leaked file descriptors from within the kernel:

```python
import os
for fd in [9, 11, 13, 28]:   # FD numbers from: ss -tlnp | grep 1000
    try: os.close(fd)
    except OSError: pass
```

Or from the terminal using gdb:
```bash
gdb -p <PID> --batch -ex "call close(<fd>)"
```

## A2A SDK Patterns (a2a-sdk 0.3.x)

### Server side

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

class MyExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        await event_queue.enqueue_event(new_agent_text_message("reply"))
    async def cancel(self, context, event_queue): pass
```

### Client side

`A2AClient` is **deprecated**. Use `ClientFactory`:

```python
from a2a.client import ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import Task, Message

client = await ClientFactory.connect("http://localhost:9001")
async for event in client.send_message(create_text_message_object(content="hello")):
    if isinstance(event, tuple):          # (Task, None) — normal path
        task, _ = event
        for msg in reversed(task.history or []):
            if msg.role.value == "agent":
                print(msg.parts[0].root.text)
    elif isinstance(event, Message):      # direct message reply
        print(event.parts[0].root.text)
```

`send_message` is an **async generator**. One `async for` iteration yields the complete response (non-streaming agents).

## Notebook Conventions

Each notebook follows the same structure:
1. Concept overview with ASCII diagrams
2. Install/import cell
3. Incremental build-up: write a `.py` file → run it as a subprocess → interact with it → extend it
4. Wire in Ollama as the intelligence layer
5. Summary table

Notebooks use `nest_asyncio` to allow `asyncio.get_event_loop().run_until_complete()` inside Jupyter.
