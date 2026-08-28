import os

# app.core.config.Settings reads LLM_BASE_URL at import time. The app default
# (host.containers.internal) only resolves from inside a container, so tests
# running on the bare host need a reachable override before app.main is imported.
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
