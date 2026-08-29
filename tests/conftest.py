import os

# app.core.config.Settings reads LLM_BASE_URL at import time. The app default is
# localhost, but pin it here so tests stay reachable on the bare host even if a
# container URL is ever set as the default or leaks in from the environment.
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
