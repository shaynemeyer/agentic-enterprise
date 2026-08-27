from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.v1 import health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("enterprise_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: LangGraph agents, vector DB clients, HTTP pools
    logger.info("Initializing Sovereign Agentic Core...")
    yield
    # Shutdown: close connections, flush telemetry
    logger.info("Shutting down Sovereign Agentic Core...")


app = FastAPI(
    title="Sovereign Enterprise Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)


@app.get("/")
async def root():
    return {"status": "Active", "message": "Sovereign Agent Node Online"}
