# One Codebase, Two Backends — Develop on Ollama, Deploy to vLLM

Both backends speak the same OpenAI-compatible wire protocol, which means you don't need two versions of your application — you need one config-driven factory and environment-specific settings. This doc covers that pattern, plus the gotchas that make "identical config" not quite mean "identical behavior."

---

## The Core Idea

Never hardcode the LLM base URL or model name in application code. Drive both from environment/config, so switching backends is a **deployment concern**, not a code change. Ollama becomes your fast local dev loop on a Mac; vLLM becomes your production inference engine on the NVIDIA cluster — same LangGraph nodes, same prompts, same tool bindings, different `.env` file.

---

## 1. Config-Driven Settings

`app/core/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_provider: str = "ollama"          # "ollama" | "vllm"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "mistral-nemo:12b"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    # plus database_url, redis_url, jwt_* — see app/core/config.py

settings = Settings()
```

The compose overlays set `env_file: .env.mac` / `.env.production` on the
`agent-api` service (see `compose.mac.yaml` / `compose.prod.yaml`), so those
files supply the container's environment. The `env_file=".env"` default in
`Settings` only applies to a bare local run outside compose.

---

## 2. One Factory, Not Two

`app/core/llm.py`:

```python
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
```

**Behind the scenes:** LangGraph nodes, prompts, and tool bindings call `get_sovereign_llm()` and never know or care which backend answers. Portability lives entirely in config, not in branching code paths.

---

## 3. Environment Files, One Per Target

`.env.mac` (local dev):

```bash
LLM_PROVIDER=ollama
LLM_BASE_URL=http://host.containers.internal:11434/v1
LLM_MODEL=mistral-nemo:12b
```

`.env.production` (NVIDIA cluster):

```bash
LLM_PROVIDER=vllm
LLM_BASE_URL=http://vllm:8000/v1
LLM_MODEL=mistralai/Mistral-Nemo-Instruct-2407
LLM_MAX_TOKENS=512
```

---

## 4. Compose Layering

Keep a shared base file, layer environment-specific overrides on top — don't maintain two full stacks.

`compose.yaml` (base, shared):

```yaml
services:
  agent-api:
    build:
      context: .
      dockerfile: Containerfile
      target: runtime
    container_name: langgraph_enterprise_api
    ports:
      - '8000:8000'
    volumes:
      - .:/app:z
    userns_mode: 'keep-id:uid=1001,gid=1001'
    restart: unless-stopped
```

`compose.mac.yaml` (dev override — no `vllm` service; Ollama runs natively on the host):

```yaml
services:
  agent-api:
    env_file: .env.mac
    environment:
      - ENV=development
      - LOG_LEVEL=debug
      - WATCHFILES_FORCE_POLLING=true
    command:
      ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload']
```

`compose.prod.yaml` (NVIDIA cluster — `vllm` service included):

```yaml
services:
  agent-api:
    env_file: .env.production
    environment:
      - ENV=production
      - LOG_LEVEL=info
    command: ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000']
    depends_on:
      - vllm

  vllm:
    image: vllm/vllm-openai:latest
    command: --model /models/mistral-nemo-12b-instruct --served-model-name mistralai/Mistral-Nemo-Instruct-2407 --tensor-parallel-size 1
    volumes:
      - ./models:/models
    devices:
      - nvidia.com/gpu=all
```

Bring each environment up with:

```bash
# local dev
podman compose -f compose.yaml -f compose.mac.yaml up -d

# cluster deploy
podman compose -f compose.yaml -f compose.prod.yaml up -d
```

---

## The Catch: Portable Code Isn't Identical Behavior

This is the part worth being deliberate about — it's where "works on my Mac, weird on the cluster" bugs come from. Same architecture, same config shape, but the two servers don't guarantee identical outputs:

| Divergence Source           | Why It Matters                                                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quantization**            | Ollama pulls default to 4-bit quantized GGUF; vLLM typically serves fp16/bf16 unless explicitly quantized. Same model name, different weights under the hood — outputs can diverge even at `temperature=0`. |
| **Prompt templating**       | Each server can apply the chat template slightly differently, especially for tool-calling/function-calling formats.                                                                                         |
| **Context length defaults** | Ollama and vLLM don't always default to the same `max_model_len`.                                                                                                                                           |
| **Tool-calling maturity**   | Function-calling support and behavior can differ between server versions.                                                                                                                                   |

None of this breaks the architecture — it just means dev-on-Ollama, deploy-to-vLLM should include a **parity check** before trusting that prod behavior matches what you saw locally, rather than assuming identical config implies identical behavior.

---

## Suggested Addition: Parity Check Script

Not yet in the repo. Worth adding as `scripts/check_llm_parity.py`: run a fixed
set of prompts (`parity_prompts.json`) through both `get_sovereign_llm()`
configs — Ollama and vLLM — and report where responses, token counts, or
latency diverge. It should not assert pass/fail; quantization and prompt
templating differ legitimately between backends. The point is visibility before
trusting that "works in dev" means "works in prod", especially where the app
depends on consistent behavior: routing decisions, structured output, tool
selection.

Sketch of the prompt file it would take:

```json
[
  {"id": "basic-greeting", "messages": [{"role": "user", "content": "Confirm System Integrity."}]},
  {"id": "simple-reasoning", "messages": [{"role": "user", "content": "If a train leaves at 3pm and travels for 2 hours 45 minutes, what time does it arrive?"}]},
  {"id": "instruction-following", "messages": [{"role": "user", "content": "List exactly three colors, one per line, no punctuation."}]}
]
```


---

## Summary

|                | Ollama (Mac dev)                           | vLLM (NVIDIA cluster)                  |
| -------------- | ------------------------------------------ | -------------------------------------- |
| `LLM_MODEL`    | `mistral-nemo:12b`                         | `mistralai/Mistral-Nemo-Instruct-2407` |
| `LLM_BASE_URL` | `http://host.containers.internal:11434/v1` | `http://vllm:8000/v1`                  |
| Runs where     | Native on macOS (Metal)                    | Containerized, GPU passthrough via CDI |
| Compose file   | `compose.mac.yaml`                         | `compose.prod.yaml`                    |

Application code — LangGraph nodes, prompts, tool bindings — stays identical across both. Only `.env` and the compose override change.

---

## Visual Verification

### 1. Confirm Ollama is serving and has the model loaded

```bash
curl http://localhost:11434/v1/models
```

You already ran this and saw mistral-nemo:12b in the list — that confirms the server's up and the model is pulled. No need to re-run unless something changes.

### 2. Send an actual inference request

```bash
curl http://localhost:11434/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mistral-nemo:12b",
"messages": [{"role": "user", "content": "Confirm System Integrity."}]
}'
```

Expected result: a JSON body with `choices[0].message.content` holding the model's reply, plus a `usage` object showing `prompt_tokens`/`completion_tokens`/`total_tokens`. If you get a response with real token counts, inference is happening locally on your Mac's silicon — that's your "brain is alive" check.

### 3. If your FastAPI app is running inside Podman, verify it can reach Ollama

This is the step people usually skip and then get bitten by. The Mac-native Ollama server and a container inside the Podman VM are not on the same network namespace, so `localhost` from inside the container won't reach it.

```bash
podman compose -f compose.yaml -f compose.mac.yaml up -d
```

Get the running container's name (it's `langgraph_enterprise_api`, per `container_name` in `compose.yaml`, but confirm if unsure):

```bash
podman ps
```

```bash
podman exec -it langgraph_enterprise_api \
 curl http://host.containers.internal:11434/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{"model": "mistral-nemo:12b", "messages": [{"role": "user", "content": "Confirm System Integrity."}]}'
```

Note: `host.containers.internal` only resolves *inside* a container — running the same curl from your Mac's own terminal (outside any container) needs `localhost` instead, as in step 2 above.

If the in-container curl fails (`Connection refused` or `could not resolve host`):

1. Confirm `host.containers.internal` resolves in your Podman version:

   ```bash
   podman exec -it langgraph_enterprise_api ping host.containers.internal
   ```

2. If it doesn't resolve, bind Ollama more broadly and use the Podman VM's gateway IP instead:

   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

   Then find the gateway IP from inside the VM and use it in place of `host.containers.internal`:

   ```bash
   podman machine ssh -- ip route | grep default
   ```

### 4. Once both pass, run the actual app-level check

If `get_sovereign_llm()` in `llm.py` is pointed at the right base URL for your environment, you can just exercise the app itself rather than raw curl — e.g. hit whatever endpoint in your FastAPI app calls the LLM, and confirm the response comes back with real content and no connection errors in the logs.

```bash
podman exec -it langgraph_enterprise_api python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://host.containers.internal:11434/v1/chat/completions',
    data=json.dumps({'model': 'mistral-nemo:12b', 'messages': [{'role': 'user', 'content': 'Reply with exactly the word: OK'}]}).encode(),
    headers={'Content-Type': 'application/json'}
)
print(urllib.request.urlopen(req).read().decode())
"
```
