from pydantic import BaseModel


class AgentRequest(BaseModel):
    input_text: str
    thread_id: str = "default_user"


class AgentResponse(BaseModel):
    final_state: dict
    status: str = "success"


from fastapi import FastAPI, HTTPException

from app.graph.engine import workflow as graph

app = FastAPI(title="Enterprise Agentic Gateway V1")


@app.post("/run-agent", response_model=AgentResponse)
async def run_agent_endpoint(request: AgentRequest):
    try:
        # Initialize state for the graph
        initial_state = {"messages": [request.input_text], "status": "starting"}

        # Execute the graph (Synchronous for now, per Lab 8 architecture)
        config = {"configurable": {"thread_id": request.thread_id}}

        result = graph.invoke(initial_state, config=config)

        return AgentResponse(final_state=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
