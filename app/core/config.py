from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_provider: str = "ollama"  # "ollama" | "vllm"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "mistral-nemo:12b"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5433/agent_db"


settings = Settings()
