from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from app.graph.engine import graph_mermaid, graph_png

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "uptime": "nominal",
        "agent_engine": "ready",
    }


@router.get("/graph", response_class=PlainTextResponse, tags=["System"])
async def graph_structure() -> str:
    """The compiled agent graph as Mermaid text. Paste into any Mermaid viewer."""
    return graph_mermaid()


@router.get("/graph.png", tags=["System"])
async def graph_image() -> Response:
    try:
        return Response(content=graph_png(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Graph PNG render failed (mermaid.ink unreachable?): {exc}",
        ) from exc
