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

    # Second demo login, for testing non-admin behavior (e.g. thread
    # ownership boundaries). Empty password means it is not seeded.
    demo2_username: str = "someone-else"
    demo2_password: str = ""

    # Host default; compose overrides with redis://redis:6379/0 (see compose.yaml).
    redis_url: str = "redis://localhost:6379/0"

    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_api_key: str = "EMPTY"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_memories"
    mcp_auth_token: str = "dev-only-change-me"
    agent_files_dir: str = "./agent_files"
    agent_files_max_write_bytes: int = 1_048_576  # 1 MiB
    # "dev" | "staging" | "prod". Guards that must not fire locally but
    # must fire in a real deployment check against this.
    app_env: str = "dev"

    # Lab 47: sandboxed code execution runs on gVisor (runsc), which needs a
    # real Linux kernel - macOS can't run it. This points at Podman's remote
    # API socket inside the gvisor-lab Multipass VM (see docs/.labs/lab-47-*.md),
    # reached over SSH. That VM's Podman is configured with runsc as its
    # default OCI runtime (Podman has no per-container runtime override over
    # the remote API, unlike Docker's runtime= kwarg).
    sandbox_podman_url: str = "ssh://ubuntu@192.168.252.2/run/podman/podman.sock"

    finance_api_url: str = "http://127.0.0.1:8102"
    finance_oauth_token_url: str = "http://127.0.0.1:8102/oauth/token"
    finance_oauth_client_id: str = "finance-mcp-server"
    finance_oauth_client_secret: str = "dev-only-change-me"
    finance_oauth_scope: str = "ledger.read"


settings = Settings()
