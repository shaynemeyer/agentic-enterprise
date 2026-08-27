# One Codebase, Two Backends — Develop on Ollama, Deploy to vLLM

Both backends speak the same OpenAI-compatible wire protocol, which means you don't need two versions of your application — you need one config-driven factory and environment-specific settings. This doc covers that pattern, plus the gotchas that make "identical config" not quite mean "identical behavior."

---

## The Core Idea

Never hardcode the LLM base URL or model name in application code. Drive both from environment/config, so switching backends is a **deployment concern**, not a code change. Ollama becomes your fast local dev loop on a Mac; vLLM becomes your production inference engine on the NVIDIA cluster — same LangGraph nodes, same prompts, same tool bindings, different `.env` file.

---

## 1. Config-Driven Settings

`app/core/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "ollama"          # "ollama" | "vllm"
    llm_base_url: str = "http://host.containers.internal:11434/v1"
    llm_model: str = "mistral-nemo:12b"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    class Config:
        env_file = ".env"

settings = Settings()
```

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
podman-compose -f compose.yaml -f compose.mac.yaml up -d

# cluster deploy
podman-compose -f compose.yaml -f compose.prod.yaml up -d
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

A `scripts/check_llm_parity.py` that runs a fixed set of test prompts through both `get_sovereign_llm()` configs (Ollama and vLLM) and flags divergence — worth having anywhere the app relies on consistent behavior: routing decisions, structured output, tool selection.

### `check_llm_parity.py`

```python
#!/usr/bin/env python3
"""
check_llm_parity.py

Runs a fixed set of test prompts against two OpenAI-compatible LLM backends
(e.g. Ollama running locally on a Mac, vLLM running on an NVIDIA cluster) and
reports how their responses, token usage, and latency compare.

This does NOT assert pass/fail — quantization, prompt templating, and context
defaults can differ legitimately between backends. The goal is visibility:
catch surprising divergence before you trust that "works in dev" means
"works in prod."

Usage:
    python check_llm_parity.py \
        --prompts parity_prompts.json \
        --backend-a ollama=http://localhost:11434/v1:mistral-nemo:12b \
        --backend-b vllm=http://localhost:8000/v1:mistralai/Mistral-Nemo-Instruct-2407

    # or rely on environment variables set per the Lab 7.5 config pattern:
    python check_llm_parity.py --prompts parity_prompts.json --use-env

Requires: openai>=1.0 (pip install openai --break-system-packages)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Install with: pip install openai --break-system-packages")
    sys.exit(1)


@dataclass
class BackendConfig:
    label: str
    base_url: str
    model: str
    api_key: str = "EMPTY"


@dataclass
class PromptResult:
    prompt_id: str
    backend_label: str
    text: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None


def parse_backend_arg(arg: str) -> BackendConfig:
    """
    Parses a backend spec of the form: label=base_url:model
    e.g. ollama=http://localhost:11434/v1:mistral-nemo:12b
    Note: base_url and model are split on the LAST colon that precedes a
    non-numeric, non-URL segment is ambiguous with ports, so we require the
    simpler explicit form: label=base_url|model (pipe-separated) instead.
    """
    if "=" not in arg or "|" not in arg:
        raise argparse.ArgumentTypeError(
            f"Backend spec must be 'label=base_url|model', got: {arg}"
        )
    label, rest = arg.split("=", 1)
    base_url, model = rest.split("|", 1)
    return BackendConfig(label=label, base_url=base_url, model=model)


def load_prompts(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def run_prompt(client: OpenAI, backend: BackendConfig, prompt_entry: dict) -> PromptResult:
    result = PromptResult(prompt_id=prompt_entry["id"], backend_label=backend.label)
    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=backend.model,
            messages=prompt_entry["messages"],
            temperature=0,
            max_tokens=512,
        )
        result.latency_seconds = time.monotonic() - start
        result.text = response.choices[0].message.content or ""
        if response.usage:
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens
            result.total_tokens = response.usage.total_tokens
    except Exception as e:
        result.latency_seconds = time.monotonic() - start
        result.error = str(e)
    return result


def similarity_flag(text_a: str, text_b: str) -> str:
    """
    Cheap heuristic, not a real semantic diff: flags identical, same-length-ish,
    or clearly divergent. Good enough to draw your eye to the interesting rows;
    read the actual text for anything flagged DIVERGENT.
    """
    if text_a == text_b:
        return "IDENTICAL"
    if not text_a or not text_b:
        return "MISSING"
    len_ratio = len(text_a) / max(len(text_b), 1)
    if 0.85 <= len_ratio <= 1.15:
        return "SIMILAR"
    return "DIVERGENT"


def print_report(results_a: list, results_b: list, label_a: str, label_b: str):
    print("\n" + "=" * 100)
    print(f"LLM PARITY REPORT — {label_a}  vs  {label_b}")
    print("=" * 100)

    by_id_a = {r.prompt_id: r for r in results_a}
    by_id_b = {r.prompt_id: r for r in results_b}

    for prompt_id in by_id_a:
        ra = by_id_a[prompt_id]
        rb = by_id_b.get(prompt_id)
        print(f"\n--- {prompt_id} ---")

        if ra.error:
            print(f"  [{label_a}] ERROR: {ra.error}")
        if rb and rb.error:
            print(f"  [{label_b}] ERROR: {rb.error}")
        if ra.error or (rb and rb.error):
            continue

        flag = similarity_flag(ra.text, rb.text) if rb else "MISSING"
        print(f"  Verdict: {flag}")
        print(f"  [{label_a}] {ra.latency_seconds:.2f}s | tokens: prompt={ra.prompt_tokens} "
              f"completion={ra.completion_tokens} total={ra.total_tokens}")
        if rb:
            print(f"  [{label_b}] {rb.latency_seconds:.2f}s | tokens: prompt={rb.prompt_tokens} "
                  f"completion={rb.completion_tokens} total={rb.total_tokens}")

        if flag in ("DIVERGENT", "SIMILAR"):
            print(f"  [{label_a}] text: {ra.text[:200]!r}")
            if rb:
                print(f"  [{label_b}] text: {rb.text[:200]!r}")

    print("\n" + "=" * 100)
    divergent = sum(
        1 for pid in by_id_a
        if by_id_b.get(pid) and similarity_flag(by_id_a[pid].text, by_id_b[pid].text) == "DIVERGENT"
    )
    print(f"Summary: {divergent} of {len(by_id_a)} prompts flagged DIVERGENT — review those before deploying.")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two OpenAI-compatible LLM backends on a fixed prompt set.")
    parser.add_argument("--prompts", required=True, help="Path to JSON file of test prompts.")
    parser.add_argument("--backend-a", type=parse_backend_arg, required=True,
                         help="label=base_url|model, e.g. ollama=http://localhost:11434/v1|mistral-nemo:12b")
    parser.add_argument("--backend-b", type=parse_backend_arg, required=True,
                         help="label=base_url|model, e.g. vllm=http://localhost:8000/v1|mistralai/Mistral-Nemo-Instruct-2407")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)

    client_a = OpenAI(base_url=args.backend_a.base_url, api_key=args.backend_a.api_key)
    client_b = OpenAI(base_url=args.backend_b.base_url, api_key=args.backend_b.api_key)

    print(f"Running {len(prompts)} prompts against '{args.backend_a.label}' and '{args.backend_b.label}'...")

    results_a = [run_prompt(client_a, args.backend_a, p) for p in prompts]
    results_b = [run_prompt(client_b, args.backend_b, p) for p in prompts]

    print_report(results_a, results_b, args.backend_a.label, args.backend_b.label)


if __name__ == "__main__":
    main()
```

#### `parity_prompts.json`

```json
[
  {
    "id": "basic-greeting",
    "messages": [{ "role": "user", "content": "Confirm System Integrity." }]
  },
  {
    "id": "simple-reasoning",
    "messages": [
      {
        "role": "user",
        "content": "If a train leaves at 3pm and travels for 2 hours 45 minutes, what time does it arrive?"
      }
    ]
  },
  {
    "id": "instruction-following",
    "messages": [
      {
        "role": "user",
        "content": "List exactly three colors, one per line, no punctuation."
      }
    ]
  },
  {
    "id": "short-summary",
    "messages": [
      {
        "role": "user",
        "content": "Summarize in one sentence: Docker and Podman both run containers, but Podman is daemonless and rootless by default, while Docker relies on a background daemon running as root."
      }
    ]
  }
]
```

---

## Summary

|                | Ollama (Mac dev)                           | vLLM (NVIDIA cluster)                  |
| -------------- | ------------------------------------------ | -------------------------------------- |
| `LLM_MODEL`    | `mistral-nemo:12b`                         | `mistralai/Mistral-Nemo-Instruct-2407` |
| `LLM_BASE_URL` | `http://host.containers.internal:11434/v1` | `http://vllm:8000/v1`                  |
| Runs where     | Native on macOS (Metal)                    | Containerized, GPU passthrough via CDI |
| Compose file   | `podman-compose.mac.yml`                   | `podman-compose.prod.yml`              |

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
podman-compose -f compose.yaml -f compose.mac.yaml up -d
```

```bash
podman exec -it <fastapi_container_name> \
 curl http://host.containers.internal:11434/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{"model": "mistral-nemo:12b", "messages": [{"role": "user", "content": "Confirm System Integrity."}]}'
```

from the terminal use `localhost`

```bash
podman exec -it <fastapi_container_name> \
 curl http://localhost:11434/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{"model": "mistral-nemo:12b", "messages": [{"role": "user", "content": "Confirm System Integrity."}]}'
```

Get your actual container name/id first if you don't have it handy:

```bash
podman ps
```

If that fails (Connection refused or could not resolve host):

Confirm `host.containers.internal` resolves in your Podman version — check with podman machine ssh -- cat /etc/hosts or just try `podman exec -it <container> ping host.containers.internal`.
If it doesn't resolve, bind Ollama more broadly and use the VM's gateway IP instead, as noted in the Ollama reference doc:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

then from inside the VM: `podman machine ssh -- ip route | grep default` to get the gateway IP to use in place of `host.containers.internal`.

### 4. Once both pass, run the actual app-level check

If get_sovereign_llm() in llm.py is pointed at the right base URL for your environment, you can just exercise the app itself rather than raw curl — e.g. hit whatever endpoint in your FastAPI app calls the LLM, and confirm the response comes back with real content and no connection errors in the logs.

Want me to also walk through wiring up and running check_llm_parity.py against just the Ollama side for now (since the vLLM/cluster side presumably isn't up yet), so you have a working baseline before the cluster deploy happens?

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
