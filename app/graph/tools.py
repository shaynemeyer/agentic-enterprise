"""Tools the graph's agent node can call.

Each is a plain function with the @tool decorator. The decorator reads the
signature and docstring into a JSON schema that gets sent to the model on
every call, so the docstring is a prompt - write it for the model to read.
"""

from langchain_core.tools import tool


@tool
def get_deployment_status(service_name: str) -> str:
    """Return the current deployment status of one named service.

    Args:
        service_name: the service to look up, e.g. "agent-api" or "redis".
    """
    # A real implementation would query the orchestrator. Stubbed for now.
    known = {"agent-api": "healthy, v1.4.2", "redis": "healthy", "db": "healthy"}
    return known.get(service_name, f"unknown service: {service_name}")


tools = [get_deployment_status]
