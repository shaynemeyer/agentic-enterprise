from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from schemas import AgentRequest, AgentResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Startup
    async with httpx.AsyncClient(timeout=30.0) as client:
        app.state.http = client
        yield
    # Shutdown happens on exit from the async with


app = FastAPI(title="Sovereign Agentic API", lifespan=lifespan)


@app.post("/v1/agent/invoke", response_model=AgentResponse)
async def invoke_agent(payload: AgentRequest) -> AgentResponse:
    # Simulated agent logic
    return AgentResponse(
        request_id=payload.request_id,
        status="processing",
        output=f"Agent '{payload.agent_id}' has received the task: {payload.task_description}",
    )
