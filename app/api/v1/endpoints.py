import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

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
    except Exception:
        logger.exception("agent run failed")
        execution.status = "failed"
        raise HTTPException(status_code=500, detail="Agent execution failed")
