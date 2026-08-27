from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "ollama"  # "ollama" | "vllm"
    llm_base_url: str = "http://host.containers.internal:11434/v1"
    llm_model: str = "mistral-nemo:12b"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    class Config:
        env_file = ".env"


settings = Settings()
