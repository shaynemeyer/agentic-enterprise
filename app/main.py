import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.v1 import health

from .graph.graph_core import workflow
from .schemas.schemas import SmokeTestRequest, SmokeTestResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("enterprise_agent")

SMOKE_TEST_TIMEOUT_S = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: LangGraph agents, vector DB clients, HTTP pools
    logger.info("Initializing Sovereign Agentic Core...")
    yield
    # Shutdown: close connections, flush telemetry
    logger.info("Shutting down Sovereign Agentic Core...")


app = FastAPI(
    title="Sovereign Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)


@app.get("/")
async def root():
    return {"status": "Active", "message": "Sovereign Agent Node Online"}


@app.post("/test/smoke", response_model=SmokeTestResponse)
async def run_smoke_test(request: SmokeTestRequest):
    start_time = time.perf_counter()
    try:

        # Invoking the LangGraph workflow
        initial_state = {"messages": [("user", request.payload)]}
        result = await asyncio.wait_for(
            workflow.ainvoke(initial_state), timeout=SMOKE_TEST_TIMEOUT_S
        )

        end_time = time.perf_counter()

        return SmokeTestResponse(
            test_id=request.test_id,
            graph_state="COMPLETED",
            llm_response=result["messages"][-1].content,
            latency_ms=(end_time - start_time) * 1000,
        )

    except TimeoutError:
        logger.exception("Smoke test timed out for test_id=%s", request.test_id)
        raise HTTPException(status_code=504, detail="System Ignition Timed Out")

    except Exception:
        logger.exception("Smoke test failed for test_id=%s", request.test_id)
        raise HTTPException(status_code=500, detail="System Ignition Failed")
