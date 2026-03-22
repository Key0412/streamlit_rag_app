# echo_agent.py — the simplest possible A2A agent

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message


class EchoAgentExecutor(AgentExecutor):
    """Receives any text and echoes it back with a prefix."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # context.get_user_input() extracts the text from the latest user message
        user_input = context.get_user_input()

        # enqueue_event is async in a2a-sdk >= 0.3.x — must be awaited
        await event_queue.enqueue_event(
            new_agent_text_message(f"Echo: {user_input}")
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Called if a client cancels the task mid-flight
        pass


# --- Agent Card -----------------------------------------------------------
agent_card = AgentCard(
    name="Echo Agent",
    description="Echoes back whatever you send it.",
    url="http://localhost:10001/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="echo",
            name="Echo",
            description="Echo the input message back.",
            tags=["echo", "test"]
        )
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
)

# --- Wire everything together -------------------------------------------
request_handler = DefaultRequestHandler(
    agent_executor=EchoAgentExecutor(),
    task_store=InMemoryTaskStore(),  # stores task state in memory
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10001, log_level="warning")