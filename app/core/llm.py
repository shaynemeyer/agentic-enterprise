from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_sovereign_llm():
    """
    Returns a LangChain LLM instance pointed at whichever OpenAI-compatible
    backend is configured for this environment — Ollama locally, vLLM in prod.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
