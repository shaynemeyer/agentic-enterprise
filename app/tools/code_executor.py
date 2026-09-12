"""Sandboxed Python execution via a gVisor-isolated container.

gVisor (runsc) needs a real Linux kernel, which macOS cannot provide, so
this project's dev machine cannot host the sandbox itself. execute_sandboxed_code
instead reaches a Podman remote API socket inside the gvisor-lab Multipass
VM (settings.sandbox_podman_url, an ssh:// URL and launches one python:3.11-slim
container per call there, network-disabled and memory/process-capped, removed when the call finishes.

Podman, not Docker: Podman's remote API has no per-container runtime
override (Docker's containers.run(runtime=...) kwarg has no Podman
equivalent), so gvisor-lab's Podman engine is configured with runsc as
its default OCI runtime instead - every container that VM's Podman runs
uses gVisor, not just this tool's.

podman-py's Container.wait() does accept a timeout kwarg (unlike
docker-py's containers.run, which has none) - it bounds the API call
itself, no background-thread workaround needed.
"""

import asyncio
import logging

from langgraph.config import get_config
from podman import PodmanClient
from podman.errors import ContainerError, PodmanError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.database import SessionLocal
from app.models import CodeExecutionEvent

logger = logging.getLogger("enterprise_agent.code_executor")

MAX_CODE_LENGTH = 4_000
EXECUTION_TIMEOUT_S = 10
MEM_LIMIT = "128m"
PIDS_LIMIT = 32


class CodeExecutionInput(BaseModel):
    code: str = Field(
        min_length=1,
        max_length=MAX_CODE_LENGTH,
        description="Python source to execute for data analysis or math. "
        "Must print() whatever result the caller needs back.",
    )


def _run_container(client: PodmanClient, code: str) -> str:
    container = client.containers.run(
        image="python:3.11-slim",
        command=["python", "-c", code],
        network_mode="none",
        mem_limit=MEM_LIMIT,
        pids_limit=PIDS_LIMIT,
        detach=True,
        remove=False,  # removed explicitly after logs are read, below
    )
    try:
        exit_code = container.wait(timeout=EXECUTION_TIMEOUT_S)
        logs = b"".join(container.logs(stdout=True, stderr=True)).decode(
            "utf-8", "replace"
        )
        if exit_code != 0:
            raise ContainerError(
                container, exit_code, "python -c", "python:3.11-slim", logs
            )
        return logs
    finally:
        container.remove(force=True)


def execute_sandboxed_code(code: str) -> dict:
    """Run a Python snippet inside a network-disabled gVisor sandbox.

    Use this for data analysis or calculations no existing tool covers.
    The script must print() its answer - stdout is what comes back.

    Takes the validated field directly (not a CodeExecutionInput instance):
    StructuredTool.from_function calls func/coroutine with the args_schema's
    fields expanded as kwargs, matching how FastMCP's @mcp.tool() calls its
    own Annotated-per-argument tools (app/mcp/fs_server.py) rather than
    passing a model instance.
    """
    try:
        with PodmanClient(base_url=settings.sandbox_podman_url) as client:
            output = _run_container(client, code)
    except PodmanError as exc:
        return {"output": str(exc), "status": "error"}
    except Exception as exc:  # noqa: BLE001 - container/runtime failures vary widely
        return {"output": str(exc), "status": "error"}
    return {"output": output.strip(), "status": "success"}


async def _audit(
    thread_id: str | None, code_length: int, outcome: str, detail: str
) -> None:
    async with SessionLocal() as session:
        session.add(
            CodeExecutionEvent(
                thread_id=thread_id,
                code_length=code_length,
                outcome=outcome,
                detail=detail[:512],
            )
        )
        await session.commit()


async def execute_sandboxed_code_async(code: str) -> dict:
    """Async wrapper: runs execute_sandboxed_code in a thread (podman-py is
    blocking, and the SSH round-trip to gvisor-lab adds real latency), then
    records the outcome in code_execution_events regardless of success.

    This is what the graph tool actually calls (StructuredTool's coroutine=,
    app/graph/tools.py) so a sandbox call never blocks the event loop.
    thread_id comes from the run's own config, same pattern as
    app/graph/engine.py's memory_retriever node - not a model-supplied arg.
    """
    thread_id = get_config().get("configurable", {}).get("thread_id")
    result = await asyncio.to_thread(execute_sandboxed_code, code)
    await _audit(
        thread_id=thread_id,
        code_length=len(code),
        outcome="ok" if result["status"] == "success" else "error",
        detail=result["output"],
    )
    return result
