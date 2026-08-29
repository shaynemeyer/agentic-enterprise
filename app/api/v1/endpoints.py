import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from langchain_core.messages import HumanMessage

from app.graph.engine import workflow
from app.schemas.agent_schema import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def log_agent_activity(data: str):
    # Simulate a slow IO task like writing to a secure audit log
    import asyncio

    await asyncio.sleep(1)
    print(f"AUDIT LOG: {data}")


@router.post("/run", response_model=AgentResponse)
async def run_agent(payload: AgentRequest, background_tasks: BackgroundTasks):
    try:
        # Crucial: Use ainvoke for non-blocking execution
        initial_state = {"messages": [HumanMessage(payload.task_description)]}

        # The engine works while the CPU handles other requests
        result = await workflow.ainvoke(initial_state)

        background_tasks.add_task(log_agent_activity, payload.task_description)

        return AgentResponse(
            request_id=payload.request_id,
            output=result["messages"][-1].content,
            status="success",
        )
    except Exception:
        logger.exception("agent run failed")
        raise HTTPException(status_code=500, detail="Agent execution failed")
