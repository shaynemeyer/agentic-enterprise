import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi_cache.decorator import cache
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from app.api.v1.auth import CurrentUser, get_current_user, user_key
from app.core.context import REQUEST_ID_HEADER, get_request_id
from app.core.exceptions import AgenticException, MaxRecursionError
from app.core.security import limiter
from app.database import get_db
from app.graph.engine import workflow
from app.models import AgentExecution
from app.schemas.agent_schema import AgentRequest, AgentResponse, AskResponse
from app.schemas.stream import StreamEvent

logger = logging.getLogger(__name__)

router = APIRouter()


async def log_agent_activity(data: str):
    # Simulate a slow IO task like writing to a secure audit log
    import asyncio

    await asyncio.sleep(1)
    print(f"AUDIT LOG: {data}")


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

    async def events():
        initial_state = {"messages": [HumanMessage(payload.task_description)]}
        try:
            async for mode, chunk in workflow.astream(
                initial_state, stream_mode=["updates", "custom"]
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

    try:
        # Crucial: Use ainvoke for non-blocking execution
        initial_state = {"messages": [HumanMessage(payload.task_description)]}

        # The engine works while the CPU handles other requests
        result = await workflow.ainvoke(initial_state)

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
    result = await workflow.ainvoke(initial_state)
    return AskResponse(query=q, output=result["messages"][-1].content)
