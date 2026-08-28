# Agentic Enterprise Labs

An enterprise-grade platform for agents, built with LangChain/LangGraph and
FastAPI, running against an OpenAI-compatible LLM backend (Ollama locally,
vLLM in production).

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Podman (or Docker) with `compose` for containerized runs
- An OpenAI-compatible LLM backend reachable from the app — Ollama for
  local development, vLLM for production

## Running locally

```bash
uv sync
uv run uvicorn app.main:app --reload
```

The API is served at `http://localhost:8000`.

| Route         | Method | Purpose                                                                                  |
| ------------- | ------ | ----------------------------------------------------------------------------------------- |
| `/`           | GET    | Liveness check                                                                           |
| `/health`     | GET    | System health status                                                                     |
| `/test/smoke` | POST   | Runs the LangGraph workflow end-to-end against the configured LLM and returns the result |

Example:

```bash
curl -X POST http://localhost:8000/test/smoke \
  -H "Content-Type: application/json" \
  -d '{"test_id": "ST-2026-001", "payload": "System Check: Respond with '\''READY'\''"}'
```

## Running in a container

```bash
# local development, hot reload
podman compose -f compose.yaml -f compose.mac.yaml up -d

# production
podman compose -f compose.yaml -f compose.prod.yaml up -d
```

Configuration is read from `.env.mac` / `.env.production` (see
`.env.mac.example` / `.env.production.example`) and includes the LLM
backend URL, model name, and provider (`ollama` | `vllm`).

## Tests

```bash
uv run pytest                        # full suite, including live LLM calls
uv run pytest -m "not integration"   # fast, no external dependencies
```

## Project layout

```text
app/
  main.py             FastAPI app and routes
  core/                settings and LLM client
  graph/               LangGraph workflow
  schemas/             Pydantic request/response models
  api/v1/               versioned routers (health, etc.)
docs/                  architecture and pattern notes
tests/                 pytest suite
```

See [`docs/graph-core.md`](docs/graph-core.md) for how the LangGraph
workflow is wired, and
[`docs/ollama-to-vllm-pattern.md`](docs/ollama-to-vllm-pattern.md) for the
local-to-production LLM backend pattern.

## Dev container

This repo includes a [devcontainer](.devcontainer/devcontainer.json)
(Python 3.13, `curl`, `git`, `openssh-client`, with the Python, Pylance,
autoDocstring, and Even Better TOML VS Code extensions).

Open this folder in VS Code and select **Reopen in Container** (requires
the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)).
