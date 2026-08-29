import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AgenticException, MaxRecursionError
from app.database import get_db
from app.graph.engine import workflow
from app.models import AgentExecution
from app.schemas.agent_schema import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def log_agent_activity(data: str):
    # Simulate a slow IO task like writing to a secure audit log
    import asyncio

    await asyncio.sleep(1)
    print(f"AUDIT LOG: {data}")


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    payload: AgentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
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
