# How `app/graph/graph_core.py` Works

## State schema

`AgentState` is the shared data structure passed through every node in the
graph. Every node reads it and returns a partial update to it.

```python
class AgentState(TypedDict):
    messages: Annotated[list[str], "The conversation history"]
    status: str
```

## The graph

One node, no branching, no loops:

```mermaid
flowchart LR
    START([START]) --> agent[agent<br/>call_model]
    agent --> END([END])
```

- `add_edge(START, "agent")` — execution begins at the `agent` node.
- `add_node("agent", call_model)` — registers `call_model` under the name `"agent"`.
- `add_edge("agent", END)` — after `agent` runs, the graph terminates.

## What `call_model` does

```python
def call_model(state: AgentState) -> dict:
    prompt = state["messages"][-1]
    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [response],
        "status": "completed",
    }
```

1. Takes the last message in state as the prompt.
2. Calls the LLM (Ollama/vLLM via `get_sovereign_llm()`) — the real network call.
3. Returns an update: `messages` with the LLM's `AIMessage` response appended,
   `status` set to `"completed"`.

LangGraph merges a node's returned dict into existing state per key. There's
no reducer declared on `messages` here, so `call_model` manually appends
(`state["messages"] + [response]`) rather than relying on automatic
accumulation.

## Compiling

```python
workflow = graph_builder.compile()
```

`compile()` turns the builder into a `CompiledStateGraph` — the runnable
object with `.invoke()` / `.ainvoke()`. This graph is compiled **without a
checkpointer**, so it has no persistence: each call starts fresh with
whatever `initial_state` it's given.

## Who calls it

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI route
    participant Graph as workflow (CompiledStateGraph)
    participant LLM as Ollama / vLLM

    Client->>FastAPI: POST request
    FastAPI->>Graph: invoke(initial_state) / await ainvoke(initial_state)
    Graph->>Graph: START -> agent
    Graph->>LLM: llm.invoke(prompt)
    LLM-->>Graph: AIMessage response
    Graph->>Graph: agent -> END
    Graph-->>FastAPI: final_state
    FastAPI-->>Client: response
```

| Caller                        | Method                                    | Notes                                                                                                                                                                |
| ----------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/main.py` (`/test/smoke`) | `await workflow.ainvoke(initial_state)`   | async; used by the smoke-test route                                                                                                                                  |
| `server.py` (`/run-agent`)    | `graph.invoke(initial_state, config=...)` | sync; passes `thread_id` in `config`, but since there's no checkpointer attached, that `thread_id` is currently inert — no memory is actually persisted across calls |

## Standalone script

The `if __name__ == "__main__":` block runs the graph directly, outside of
any FastAPI route — useful for a quick manual check:

```python
final_state = workflow.invoke(initial_input)
print(f"Response: {final_state['messages'][-1]}")
```

Note this prints the raw `AIMessage` object's `repr`, not just its text,
since `call_model` appends the full message object rather than
`response.content`.
