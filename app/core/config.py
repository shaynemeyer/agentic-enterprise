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

    checkpoint_db_url: str = "postgresql://agent:agent@localhost:5433/agent_db"
    # Session GC: delete a thread when its newest checkpoint is older than this.
    checkpoint_retention_days: int = 30
    # How often the background sweep runs. Set 0 to disable the scheduled sweep
    checkpoint_gc_interval_hours: int = 6

    jwt_secret: str = "dev-only-change-me"  # override via JWT_SECRET in .env
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30

    # Demo login. Empty password (the default) means no users are seeded.
    demo_username: str = "admin"
    demo_password: str = ""

    # Host default; compose overrides with redis://redis:6379/0 (see compose.yaml).
    redis_url: str = "redis://localhost:6379/0"

    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_api_key: str = "EMPTY"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_memories"


settings = Settings()
