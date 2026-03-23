# writer_agent.py — A2A agent for writing and summarization tasks

import uvicorn
import ollama
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

SYSTEM_PROMPT = """You are a skilled writer and editor.
Help with drafting emails, stories, and essays. Improve tone and clarity.
Summarize long texts into clear, concise bullet points.
Keep responses focused and polished."""


class WriterAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
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
    name="Writer Agent",
    description="Drafts, edits, and summarizes text using llama3.2.",
    url="http://localhost:10003/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="write",
            name="Writing Assistant",
            description="Draft, edit, and improve written content.",
            tags=["writing", "editing", "email"],
        ),
        AgentSkill(
            id="summarize",
            name="Summarizer",
            description="Summarize long text into concise bullet points.",
            tags=["summarization", "tldr"],
        ),
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
)

request_handler = DefaultRequestHandler(
    agent_executor=WriterAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10003, log_level="warning")