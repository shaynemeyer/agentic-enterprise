import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi_cache.decorator import cache
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from app.api.v1.auth import (
    CurrentUser,
    get_current_user,
    require_thread_owner,
    user_key,
)
from app.core.config import settings
from app.core.context import REQUEST_ID_HEADER, get_request_id
from app.core.exceptions import AgenticException, MaxRecursionError
from app.core.llm import get_sovereign_llm
from app.core.security import limiter
from app.database import get_db
from app.graph.engine import workflow
from app.graph.gc import _conn, sweep
from app.graph.history import (
    branch_tree,
    checkpoint_config,
    edit_checkpoint,
    is_interrupted,
    thread_timeline,
)
from app.graph.ownership import is_admin, owned_thread_ids
from app.graph.tools import tools
from app.memory.vector_store import remember, search
from app.models import AgentExecution
from app.schemas.agent_schema import (
    AgentRequest,
    AgentResponse,
    AskResponse,
    ConversationHistory,
    HistoryTurn,
)
from app.schemas.memory_schema import MemorySearchResponse, RememberResponse
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
            context={"llm": request_llm, "db": db, "username": user.username},
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
    user: CurrentUser = Depends(require_thread_owner),
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
    user: CurrentUser = Depends(require_thread_owner),
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
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_thread_owner),
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


@router.post("/admin/threads/{thread_id}/edit")
async def edit_thread_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    as_node: str,
    critique: str | None = None,
    revision_count: int | None = None,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_thread_owner),
):
    """Overwrite state at a past checkpoint - the correction, not the replay.

    Only critique/revision_count are exposed: they are the two overwrite
    (no-reducer) fields the critic/general cycle uses to decide whether to
    keep looping (Lab 24). messages and internal_logs use reducers that
    append, not overwrite, so editing them here would not mean what a caller
    expects - use fork (Lab 35) to add a message instead.
    """
    values = {
        k: v
        for k, v in {"critique": critique, "revision_count": revision_count}.items()
        if v is not None
    }
    if not values:
        raise HTTPException(
            status_code=422, detail="Provide critique and/or revision_count."
        )

    new_config = await edit_checkpoint(
        workflow, thread_id, checkpoint_id, values, as_node=as_node
    )
    return {
        "thread_id": thread_id,
        "edited_from": checkpoint_id,
        "new_checkpoint_id": new_config["configurable"]["checkpoint_id"],
        "values_written": values,
    }


@router.get("/admin/threads/{thread_id}/branches")
async def thread_branches(
    thread_id: str,
    limit: int = 200,
    user: CurrentUser = Depends(require_thread_owner),
):
    """Checkpoints grouped by parent - where a thread forked or was edited."""
    timeline = await thread_timeline(workflow, thread_id, limit=limit)
    tree = branch_tree(timeline)
    fork_points = {p: len(c) for p, c in tree.items() if p is not None and len(c) > 1}
    return {
        "thread_id": thread_id,
        "checkpoint_count": len(timeline),
        "fork_points": fork_points,
        "tree": tree,
    }


@router.post(
    "/admin/threads/{thread_id}/memory/remember", response_model=RememberResponse
)
async def remember_thread_fact(
    thread_id: str,
    text: str,
    user: CurrentUser = Depends(require_thread_owner),
):
    """Embed `text` into semantic memory, tagged with thread_id.

    Ownership-checked the same as every other /admin/threads/{id} route - require_thread_owner's thread_id param is typed Path(...), so it can only be satisfied from the URL path, not a query parameter. The route lives under /admin/threads/{thread_id}/... for that reason, matching /admin/threads/{thread_id}/edit and friends.
    """
    point_id = await remember(thread_id, text)
    return RememberResponse(point_id=point_id, thread_id=thread_id)


@router.get("/admin/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    q: str,
    limit: int = 5,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cosine-similarity search across every thread's remembered facts the caller owns (or every thread, for the admin account).

    Ownership-scoped as of this lab - previously we left this route unfiltered because search() had no thread_ids param yet and there was no reason to add one until the graph's own retrieval needed it. Same owned_thread_ids() and is_admin() used by retrieve_semantic_memories, so the route and the graph node enforce one rule, not two that could drift apart.
    """
    if is_admin(user.username):
        hits = await search(q, limit=limit)
    else:
        allowed = await owned_thread_ids(db, user.username)
        hits = await search(q, limit=limit, thread_ids=allowed)

    return MemorySearchResponse(query=q, hits=hits)
