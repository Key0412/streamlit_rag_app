# stats_agent.py — demonstrates the Task response path
#
# Unlike the Echo Agent (which enqueues a Message via new_agent_text_message),
# this agent enqueues Task-typed events, which forces the SDK to return a
# (Task, None) tuple to the client instead of a plain Message.

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard, AgentCapabilities, AgentSkill,
    TaskState, TaskStatus,
    TaskStatusUpdateEvent,   # signals state changes (working, completed, …)
    TaskArtifactUpdateEvent, # attaches structured output to the Task
    Artifact, Part, DataPart,
)


class StatsAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text    = context.get_user_input()
        task_id = context.task_id
        ctx_id  = context.context_id

        # Step 1 — Signal that work has started.
        # This creates the Task object in the server's TaskStore.
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=ctx_id,
            status=TaskStatus(state=TaskState.working),
            final=False,   # more events will follow
        ))

        # Step 2 — Compute the result and attach it as a structured artifact.
        # DataPart holds arbitrary JSON-serialisable data (dict, list, etc.).
        stats = {
            "words":           len(text.split()),
            "chars":           len(text),
            "chars_no_spaces": len(text.replace(" ", "")),
        }
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id=ctx_id,
            artifact=Artifact(
                artifact_id="stats",
                name="Text statistics",
                parts=[Part(DataPart(data=stats))],
            ),
        ))

        # Step 3 — Signal completion. final=True tells the aggregator to
        # stop consuming and return the Task to the client.
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=ctx_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


agent_card = AgentCard(
    name="Stats Agent",
    description="Returns word and character counts for any input text as a structured artifact.",
    url="http://localhost:10005/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="stats",
            name="Text Statistics",
            description="Count words and characters in a string.",
            tags=["stats", "text"],
        )
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["application/json"],
)

request_handler = DefaultRequestHandler(
    agent_executor=StatsAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10005, log_level="warning")