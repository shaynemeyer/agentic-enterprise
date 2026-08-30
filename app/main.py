import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from langgraph.errors import GraphRecursionError
from redis import asyncio as redis_asyncio
from slowapi.errors import RateLimitExceeded

from app.api.v1 import auth, endpoints, health
from app.core.context import (
    REQUEST_ID_HEADER,
    get_request_id,
    new_request_id,
    set_request_id,
)
from app.core.config import settings
from app.core.exceptions import AgenticException, MaxRecursionError
from app.core.security import limiter
from app.schemas.errors import ErrorResponse

from .graph.engine import workflow
from .schemas.agent_schema import SmokeTestRequest, SmokeTestResponse


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger("enterprise_agent")

SMOKE_TEST_TIMEOUT_S = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: LangGraph agents, vector DB clients, HTTP pools
    logger.info("Initializing Sovereign Agentic Core...")
    try:
        client = redis_asyncio.from_url(settings.redis_url)
        await client.ping()
        FastAPICache.init(RedisBackend(client), prefix="agent-cache")
        logger.info("Response cache connected: %s", settings.redis_url)
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail on cache
        logger.warning("Response cache unavailable (%s); serving uncached", exc)
    yield

    # Shutdown: close connections, flush telemetry
    logger.info("Shutting down Sovereign Agentic Core...")


app = FastAPI(
    title="Sovereign Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming or new_request_id()
    set_request_id(request_id)

    response = await call_next(request)

    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(AgenticException)
async def handle_agentic_exception(request: Request, exc: AgenticException):
    logger.warning("agentic failure: %s (%s)", exc.message, exc.error_code)
    body = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        trace_id=get_request_id() or None,
    )
    return JSONResponse(
        status_code=exc.status_code, content=body.model_dump(mode="json")
    )


@app.exception_handler(GraphRecursionError)
async def handle_graph_recursion(request: Request, exc: GraphRecursionError):
    # Translate LangGraph's own exception into our schema.
    return await handle_agentic_exception(
        request, MaxRecursionError(details={"source": "langgraph"})
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    body = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Request body failed validation.",
        details=exc.errors(),
        trace_id=get_request_id() or None,
    )
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(request: Request, exc: RateLimitExceeded):
    logger.warning("rate limit hit: %s", exc.detail)
    # slowapi's RateLimitExceeded carries the Limit; its RateLimitItem knows the
    # window length. "Wait at most one full window" is a correct Retry-After.
    retry_after = exc.limit.limit.get_expiry()
    body = ErrorResponse(
        error_code="RATE_LIMIT_EXCEEDED",
        message=f"Rate limit exceeded: {exc.detail}",
        details={"limit": str(exc.detail)},
        trace_id=get_request_id() or None,
    )
    headers = {"Retry-After": str(retry_after)}
    return JSONResponse(
        status_code=429, content=body.model_dump(mode="json"), headers=headers
    )


app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(endpoints.router, prefix="/api/v1")


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
