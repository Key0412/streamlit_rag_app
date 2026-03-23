# orchestrator_agent.py
# An A2A agent that routes requests to specialist agents via A2A protocol.

import httpx
import uvicorn
import ollama
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Task, Message
from a2a.client import ClientFactory, ClientConfig
from a2a.client.helpers import create_text_message_object
from a2a.utils import new_agent_text_message

MATH_AGENT_URL   = "http://localhost:10002"
WRITER_AGENT_URL = "http://localhost:10003"

# 60s timeout — Ollama calls on the specialist agents can take 4–30s
_client_config = ClientConfig(httpx_client=httpx.AsyncClient(timeout=120.0))

ROUTER_PROMPT = """You are a routing assistant. Classify the request as exactly one word:
- "math"    if it involves numbers, calculations, equations, or formulas
- "writing" if it involves text, drafting, editing, or summarization
Reply with ONLY the single word."""


def extract_text(event) -> str:
    if isinstance(event, tuple):
        task, _ = event
        for msg in reversed(task.history or []):
            if msg.role.value == "agent":
                for part in msg.parts:
                    if hasattr(part.root, "text"):
                        return part.root.text
    elif isinstance(event, Message):
        for part in event.parts:
            if hasattr(part.root, "text"):
                return part.root.text
    return str(event)


class OrchestratorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        # ── Step 1: Classify the request with LLM ─────────────────────────────
        classification = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": user_input},
            ],
        )["message"]["content"].strip().lower()

        # ── Step 2: Route to the right specialist ─────────────────────────────
        if "math" in classification:
            target_url  = MATH_AGENT_URL
            target_name = "Math Agent"
        else:
            target_url  = WRITER_AGENT_URL
            target_name = "Writer Agent"

        # ── Step 3: Call the specialist agent via A2A ─────────────────────────
        specialist_client = await ClientFactory.connect(target_url, client_config=_client_config)
        specialist_text = ""
        async for event in specialist_client.send_message(
            create_text_message_object(content=user_input)
        ):
            specialist_text = extract_text(event)
            break

        # ── Step 4: Return the specialist's answer to the original caller ────
        reply = f"[Routed to {target_name}]\n\n{specialist_text}"
        await event_queue.enqueue_event(new_agent_text_message(reply))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


agent_card = AgentCard(
    name="Orchestrator Agent",
    description="Routes user requests to Math or Writer specialist agents.",
    url="http://localhost:10004/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="route",
            name="Smart Router",
            description="Automatically delegates tasks to the right specialist agent.",
            tags=["orchestration", "routing"],
        )
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
)

request_handler = DefaultRequestHandler(
    agent_executor=OrchestratorExecutor(),
    task_store=InMemoryTaskStore(),
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10004, log_level="warning")