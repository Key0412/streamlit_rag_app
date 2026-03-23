# math_agent.py — an A2A agent that solves math using llama3.2

import uvicorn
import ollama
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

SYSTEM_PROMPT = """You are a math expert. Solve math problems step by step.
Show your reasoning clearly. Be concise."""


class MathAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        # Call the local LLM via Ollama
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_input},
            ],
        )
        answer = response["message"]["content"]
        # enqueue_event is async in a2a-sdk >= 0.3.x — must be awaited
        await event_queue.enqueue_event(new_agent_text_message(answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


agent_card = AgentCard(
    name="Math Agent",
    description="Solves math problems using llama3.2 running via Ollama.",
    url="http://localhost:10002/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="math-solver",
            name="Math Solver",
            description="Solve arithmetic, algebra, and basic calculus problems.",
            tags=["math", "calculation", "algebra"],
        )
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
)

request_handler = DefaultRequestHandler(
    agent_executor=MathAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10002, log_level="warning")