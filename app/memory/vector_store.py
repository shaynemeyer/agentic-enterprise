"""Cross-thread semantic memory: embed messages into Qdrant, search by
cosine similarity.

Independent of the checkpointer (Lab 33) - that store partitions strictly
by thread_id; this one deliberately does not, so a caller can surface a
relevant fact regardless of which thread it was said in. Nothing in
app/graph/engine.py calls this module - it is exposed through its own
routes (Step 5), opted into per-request rather than run on every message.
"""

import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.core.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Embedding client for whichever OpenAI-compatible backend is configured.

    Same pattern as get_sovereign_llm() (app/core/llm.py) - Ollama locally,
    a separate embedding-serving backend in prod (Step 1).
    check_embedding_ctx_length=False per LangChain's own guidance for
    non-OpenAI providers (Ollama, vLLM, OpenRouter): it sends raw text
    straight to the server instead of pre-tokenizing locally first. Without
    it, OpenAIEmbeddings pre-tokenizes with tiktoken by default - which
    doesn't know this model - or, if tiktoken_enabled=False is set instead,
    falls back to `transformers.AutoTokenizer.from_pretrained(model_name)`,
    which requires the transformers package and would try to resolve
    "qwen3-embedding:0.6b" as a HuggingFace repo id, which it is not.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_base_url,
        check_embedding_ctx_length=False,
    )


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def ensure_collection(vector_size: int) -> None:
    """Create the collection if absent. No-op if it already exists.

    vector_size must match the configured embedding model's output width
    exactly - Qdrant rejects an upsert whose vector length disagrees with
    the collection's VectorParams.size (Troubleshooting).
    """
    client = get_client()
    existing = await client.get_collections()
    if any(c.name == settings.qdrant_collection for c in existing.collections):
        return
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


async def remember(
    thread_id: str, text: str, metadata: dict[str, Any] | None = None
) -> str:
    """Embed `text` and upsert it as a new point. Returns the point id.

    A fresh uuid4() per call, not thread_id, is the point id - Qdrant point
    ids must be an unsigned int or a UUID (source in lab notes), and reusing
    thread_id would make every remember() on the same thread overwrite the
    last one instead of accumulating memories. thread_id still goes in the
    payload, where search() can filter on it if a caller wants memories
    from one thread only.
    """
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(text)
    await ensure_collection(len(vector))

    point_id = str(uuid.uuid4())
    await get_client().upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"text": text, "thread_id": thread_id, **(metadata or {})},
            )
        ],
    )
    return point_id


async def search(
    query: str, limit: int = 5, thread_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Top `limit` memories by cosine similarity to `query`.

    thread_ids, when given, restricts results to memories whose payload
    thread_id is in that list - a Qdrant Filter, not a post-hoc Python
    filter, so a caller with 0 owned threads gets a correctly-empty result
    without first fetching (and discarding) someone else's memories.
    None (the default) searches unfiltered, matching Lab 38's original
    behavior - existing callers of search() are unaffected.
    [] if the collection does not exist yet (nothing has been remembered).
    """
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(query)

    client = get_client()
    existing = await client.get_collections()
    if not any(c.name == settings.qdrant_collection for c in existing.collections):
        return []

    query_filter = None
    if thread_ids is not None:
        if not thread_ids:
            return []
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="thread_id", match=models.MatchAny(any=thread_ids)
                )
            ]
        )

    hits = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=limit,
        query_filter=query_filter,
    )
    return [
        {
            "score": p.score,
            "text": p.payload.get("text"),
            "thread_id": p.payload.get("thread_id"),
        }
        for p in hits.points
    ]
