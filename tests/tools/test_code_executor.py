import pytest
from podman import PodmanClient
from podman.errors import PodmanError
from pydantic import ValidationError

from app.core.config import settings
from app.tools.code_executor import CodeExecutionInput, execute_sandboxed_code


def _sandbox_up() -> bool:
    """True if gvisor-lab's Podman socket is reachable and defaults to runsc.

    Same shape as tests/graph/test_tools.py's requires_mcp: a real
    integration dependency (here, the Multipass VM from docs/.labs/lab-47-*.md)
    that isn't running in every dev environment or in CI.
    """
    try:
        with PodmanClient(base_url=settings.sandbox_podman_url) as client:
            return client.info()["host"]["ociRuntime"]["name"] == "runsc"
    except PodmanError:
        return False


requires_sandbox = pytest.mark.skipif(
    not _sandbox_up(), reason="gvisor-lab unreachable or runsc not its default runtime"
)


@requires_sandbox
def test_execute_sandboxed_code_returns_stdout():
    result = execute_sandboxed_code("print(2 + 2)")
    assert result["status"] == "success"
    assert result["output"] == "4"


@requires_sandbox
def test_execute_sandboxed_code_reports_script_error():
    result = execute_sandboxed_code("raise ValueError('bad')")
    assert result["status"] == "error"
    assert "bad" in result["output"]


@requires_sandbox
def test_execute_sandboxed_code_blocks_network():
    result = execute_sandboxed_code(
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=2)"
    )
    assert result["status"] == "error"


def test_code_execution_input_rejects_empty_code():
    with pytest.raises(ValidationError):
        CodeExecutionInput(code="")


def test_code_execution_input_rejects_oversized_code():
    with pytest.raises(ValidationError):
        CodeExecutionInput(code="x" * 5_000)
