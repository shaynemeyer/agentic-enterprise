import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi_cache.decorator import cache
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from app.api.v1.auth import CurrentUser, get_current_user, user_key
from app.core.config import settings
from app.core.context import REQUEST_ID_HEADER, get_request_id
from app.core.exceptions import AgenticException, MaxRecursionError
from app.core.llm import get_sovereign_llm
from app.core.security import limiter
from app.database import get_db
from app.graph.engine import workflow
from app.graph.gc import _conn, sweep
from app.graph.history import (
    checkpoint_config,
    is_interrupted,
    thread_timeline,
)
from app.graph.tools import tools
from app.models import AgentExecution
from app.schemas.agent_schema import (
    AgentRequest,
    AgentResponse,
    AskResponse,
    ConversationHistory,
    HistoryTurn,
)
from app.schemas.stream import StreamEvent

logger = logging.getLogger(__name__)

router = APIRouter()


async def log_agent_activity(data: str):
    # Simulate a slow IO task like writing to a secure audit log
    import asyncio

    await asyncio.sleep(1)
    print(f"AUDIT LOG: {data}")


_STATS_QUERY = """
SELECT
  (SELECT count(*) FROM checkpoints)                   AS checkpoints,
  (SELECT count(*) FROM checkpoint_writes)             AS checkpoint_writes,
  (SELECT count(*) FROM checkpoint_blobs)              AS checkpoint_blobs,
  (SELECT count(DISTINCT thread_id) FROM checkpoints)  AS threads,
  (SELECT min((checkpoint ->> 'ts')::timestamptz) FROM checkpoints)
                                                       AS oldest_checkpoint
"""


@router.post("/admin/gc")
async def run_gc(user: CurrentUser = Depends(get_current_user)):
    """Trigger a checkpoint retention sweep now. Same logic as the hourly task."""
    return await sweep(workflow.checkpointer, settings.checkpoint_retention_days)


@router.get("/admin/gc/stats")
async def gc_stats(user: CurrentUser = Depends(get_current_user)):
    """Row counts per checkpoint table and the age of the oldest live thread."""
    async with _conn(workflow.checkpointer) as conn:
        cur = await conn.execute(_STATS_QUERY)
        row = await cur.fetchone()
    # dict_row (from_conn_string) vs tuple (pool) - key by name, fall back to index.
    get = (lambda k, i: row[k]) if isinstance(row, dict) else (lambda k, i: row[i])
    oldest = get("oldest_checkpoint", 4)
    return {
        "checkpoints": get("checkpoints", 0),
        "checkpoint_writes": get("checkpoint_writes", 1),
        "checkpoint_blobs": get("checkpoint_blobs", 2),
        "threads": get("threads", 3),
        "oldest_checkpoint": oldest.isoformat() if oldest else None,
        "retention_days": settings.checkpoint_retention_days,
    }


@router.post("/run/stream")
@limiter.limit("5/minute", key_func=user_key)
async def run_agent_stream(
    request: Request,
    payload: AgentRequest,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    request_id = get_request_id()

    execution = AgentExecution(
        request_id=payload.request_id,
        agent_id=payload.agent_id,
        status="running",
    )
    db.add(execution)

    thread_id = payload.conversation_id or str(payload.request_id)

    async def events():
        initial_state = {"messages": [HumanMessage(payload.task_description)]}
        try:
            request_llm = get_sovereign_llm().bind_tools(tools)

            async for mode, chunk in workflow.astream(
                initial_state,
                stream_mode=["updates", "custom"],
                context={"llm": request_llm},
                config={"configurable": {"thread_id": thread_id}},
            ):
                if mode == "custom":
                    ev = StreamEvent(phase="status", content=chunk["status"])
                else:  # "updates": {node_name: delta}
                    node = next(iter(chunk))
                    ev = StreamEvent(phase="node", node=node, content="step complete")
                yield {
                    "event": "message",
                    "id": request_id,
                    "data": ev.model_dump_json(),
                }

            execution.status = "completed"
            done = StreamEvent(phase="done", content="run finished")
            yield {"event": "done", "id": request_id, "data": done.model_dump_json()}

        except GraphRecursionError:
            execution.status = "failed"
            err = StreamEvent(phase="error", content="recursion limit exceeded")
            yield {"event": "error", "id": request_id, "data": err.model_dump_json()}

    return EventSourceResponse(events(), headers={REQUEST_ID_HEADER: request_id})


@router.post("/run", response_model=AgentResponse)
@limiter.limit("5/minute", key_func=user_key)
async def run_agent(
    request: Request,
    payload: AgentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    execution = AgentExecution(
        request_id=payload.request_id,
        agent_id=payload.agent_id,
        status="running",
    )
    db.add(execution)

    thread_id = payload.conversation_id or str(payload.request_id)

    try:
        # Crucial: Use ainvoke for non-blocking execution
        initial_state = {"messages": [HumanMessage(payload.task_description)]}

        request_llm = get_sovereign_llm().bind_tools(tools)

        # The engine works while the CPU handles other requests
        result = await workflow.ainvoke(
            initial_state,
            context={"llm": request_llm},
            config={"configurable": {"thread_id": thread_id}},
        )

        background_tasks.add_task(log_agent_activity, payload.task_description)
        execution.status = "completed"

        return AgentResponse(
            request_id=payload.request_id,
            output=result["messages"][-1].content,
            status="success",
        )
    except GraphRecursionError:
        execution.status = "failed"
        raise MaxRecursionError(details={"agent_id": payload.agent_id})
    except AgenticException:
        execution.status = "failed"
        raise
    # No bare `except Exception`. An unexpected error becomes a 500 via the
    # framework default; the `get_db` dependency still rolls back the session.


@router.get("/ask", response_model=AskResponse)
@limiter.limit("5/minute", key_func=user_key)
@cache(expire=300)
async def ask(
    request: Request,
    q: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Read-shaped agent query. Identical `q` within 5 minutes is served from Redis.

    No AgentExecution row is written here - a cache hit would skip it anyway, so
    the audit trail stays on POST /run. Use /run for anything that must be logged.
    """
    initial_state = {"messages": [HumanMessage(q)]}

    request_llm = get_sovereign_llm().bind_tools(tools)

    # No request_id on this route - key the thread per user so a user's /ask
    # calls share one conversation. (Cache hits skip the graph entirely.)
    result = await workflow.ainvoke(
        initial_state,
        context={"llm": request_llm},
        config={"configurable": {"thread_id": f"ask:{user.username}"}},
    )
    return AskResponse(query=q, output=result["messages"][-1].content)


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Replay a thread's messages from the persistent checkpoint store.

    Reads the latest checkpoint for this thread_id - no graph run. Returns an
    empty list if the thread was never written or has been pruned.
    """
    config = {"configurable": {"thread_id": conversation_id}}
    snapshot = await workflow.aget_state(config)

    messages: list[BaseMessage] = snapshot.values.get("messages", [])
    turns = [HistoryTurn(role=m.type, content=str(m.content)) for m in messages]
    return ConversationHistory(conversation_id=conversation_id, turns=turns)


@router.get("/admin/threads/{thread_id}/history")
async def thread_history(
    thread_id: str,
    limit: int = 50,
    user: CurrentUser = Depends(get_current_user),
):
    """Every checkpoint for a thread, newest first. Empty list if unknown."""
    timeline = await thread_timeline(workflow, thread_id, limit=limit)
    return {
        "thread_id": thread_id,
        "interrupted": await is_interrupted(workflow, thread_id),
        "checkpoint_count": len(timeline),
        "checkpoints": timeline,
    }


@router.post("/admin/threads/{thread_id}/resume", response_model=AgentResponse)
async def resume_thread(
    thread_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Finish an interrupted run: execute the nodes that were still pending.

    Invokes with None as input, so LangGraph continues from the last checkpoint
    instead of starting the graph over.
    """
    if not await is_interrupted(workflow, thread_id):
        raise HTTPException(
            status_code=404,
            detail="No interrupted run for this thread_id (nothing pending).",
        )

    request_llm = get_sovereign_llm().bind_tools(tools)
    result = await workflow.ainvoke(
        None,  # resume, do not append
        context={"llm": request_llm},
        config=checkpoint_config(thread_id),
    )
    return AgentResponse(
        request_id=user.username,  # no request_id on a resume; use the caller
        output=result["messages"][-1].content,
        status="success",
    )


@router.post("/admin/threads/{thread_id}/fork")
async def fork_thread(
    thread_id: str,
    checkpoint_id: str,
    message: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Branch a thread: re-run from `checkpoint_id` with a new message.

    Does not modify the original chain - the fork is a sibling branch sharing
    history up to checkpoint_id.
    """
    request_llm = get_sovereign_llm().bind_tools(tools)
    result = await workflow.ainvoke(
        {"messages": [HumanMessage(message)]},
        context={"llm": request_llm},
        config=checkpoint_config(thread_id, checkpoint_id),
    )
    forked = await workflow.aget_state(checkpoint_config(thread_id))
    return {
        "thread_id": thread_id,
        "forked_from": checkpoint_id,
        "new_checkpoint_id": forked.config["configurable"]["checkpoint_id"],
        "output": result["messages"][-1].content,
    }
