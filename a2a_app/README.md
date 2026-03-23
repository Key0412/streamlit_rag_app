# A2A App — Agent-to-Agent Protocol

A hands-on study of the **Agent-to-Agent (A2A) protocol**, built incrementally through a step-by-step Jupyter notebook. Agents communicate over HTTP using JSON-RPC 2.0 and discover each other via a standard Agent Card endpoint.

For a step-by-step walkthrough, see `step_by_step_a2a.ipynb`.

---

## Architecture

```
User / Notebook (A2A Client)
        │
        │  HTTP  POST /
        ▼
┌──────────────────────────┐
│   Orchestrator Agent     │  port 10004
│   (routes with llama3.2) │
└─────────┬────────────────┘
          │  A2A (HTTP)
    ┌─────┴──────┐
    ▼            ▼
Math Agent   Writer Agent
port 10002   port 10003
(llama3.2)   (llama3.2)
```

Agents communicate using the A2A standard: any caller can discover an agent's capabilities by fetching its **Agent Card** at `/.well-known/agent.json`, then send messages via `POST /`.

---

## Agents

| Agent | Port | Description | Response type |
|---|---|---|---|
| `echo_agent.py` | 10001 | Echoes input back — minimal A2A example | `Message` |
| `math_agent.py` | 10002 | Solves math problems via `llama3.2` | `Message` |
| `writer_agent.py` | 10003 | Writing, editing, and summarization via `llama3.2` | `Message` |
| `orchestrator_agent.py` | 10004 | Routes requests to Math or Writer using LLM classification | `Message` |
| `stats_agent.py` | 10005 | Returns word/char counts as a structured artifact — Task-path demo | `(Task, None)` |

---

## Key Concepts

### Message vs Task response

An A2A agent can reply in two ways depending on what its executor enqueues:

| Executor enqueues | Client receives | Use when |
|---|---|---|
| `new_agent_text_message(...)` | `Message` | Simple text reply |
| `TaskStatusUpdateEvent` + `TaskArtifactUpdateEvent` | `(Task, None)` | Structured output, multi-step work |

### Agent Card

Every agent advertises itself at `GET /.well-known/agent.json`:

```json
{
  "name": "Math Agent",
  "url": "http://localhost:10002/",
  "capabilities": { "streaming": false },
  "skills": [{ "id": "math", "name": "Math Solver", ... }]
}
```

Clients use this to discover the agent's transport, capabilities, and skills before sending any messages.

---

## Running

Requires Ollama running locally (`ollama serve`) with `llama3.2` pulled.

```bash
# Activate the shared virtualenv
source /home/klismam/pyenvs/agent_study/bin/activate

# Run agents individually
python3 echo_agent.py          # port 10001
python3 math_agent.py          # port 10002
python3 writer_agent.py        # port 10003
python3 orchestrator_agent.py  # port 10004 (requires math + writer running)
python3 stats_agent.py         # port 10005
```

Or follow the notebook (`step_by_step_a2a.ipynb`), which writes and launches each agent as a subprocess.

---

## Client pattern (a2a-sdk 0.3.x)

```python
import httpx
from a2a.client import ClientFactory, ClientConfig
from a2a.client.helpers import create_text_message_object
from a2a.types import Task, Message

# Always set a long timeout — Ollama calls can take 4–30s
config = ClientConfig(httpx_client=httpx.AsyncClient(timeout=60.0))
client = await ClientFactory.connect("http://localhost:10002", client_config=config)

async for event in client.send_message(create_text_message_object(content="What is 12 * 34?")):
    if isinstance(event, tuple):      # (Task, None)
        task, _ = event
        for msg in reversed(task.history or []):
            if msg.role.value == "agent":
                print(msg.parts[0].root.text)
    elif isinstance(event, Message):  # direct Message
        print(event.parts[0].root.text)
```

> `A2AClient` is deprecated in 0.3.x. Use `ClientFactory.connect()` instead.

---

## Stuck ports

If a kernel interrupt leaves a port occupied:

```python
# From inside the Jupyter kernel
import os
for fd in [9, 11, 13, 28]:  # FD numbers from: ss -tlnp | grep 1000
    try: os.close(fd)
    except OSError: pass
```

Or from the terminal:

```bash
gdb -p <PID> --batch -ex "call close(<fd>)"
```
