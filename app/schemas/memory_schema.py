from pydantic import BaseModel


class MemoryHit(BaseModel):
    score: float
    text: str
    thread_id: str | None = None


class MemorySearchResponse(BaseModel):
    query: str
    hits: list[MemoryHit]


class RememberResponse(BaseModel):
    point_id: str
    thread_id: str
