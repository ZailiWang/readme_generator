import asyncio
import json
import os
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from readme_generator.main import (
    WorkflowInput,
    WorkflowRunner,
    prepare_enabled_stages,
    prepare_workflow_input,
)


app = FastAPI()

raw_origins = os.getenv("README_GENERATOR_CORS_ORIGINS", "*")
allow_origins = ["*"] if raw_origins.strip() == "*" else [item.strip() for item in raw_origins.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    def __init__(self) -> None:
        self.runner: Optional[WorkflowRunner] = None
        self.stages: List[str] = []
        self.step: int = 0
        self.finished: bool = False


state = State()


class WorkflowRequest(BaseModel):
    generation_mode: str = "reference"
    github_md_folder_url: str = ""
    github_js_folder_url: str = ""
    source_md_url: str = ""
    source_js_url: str = ""
    input_text: str = ""
    model_list: List[str] = Field(default_factory=list)
    github_url: List[str] = Field(default_factory=list)
    skip_stages: List[str] = Field(default_factory=list)
    remote_folder: str = ""
    ssh_config: Dict[str, Any] = Field(default_factory=dict)
    github_config: Dict[str, Any] = Field(default_factory=dict)
    ref_md: str = ""
    ref_index_js: str = ""
    family_md: str = ""
    family_index_js: str = ""
    reference_folder: Optional[str] = None
    stages: List[str] = Field(default_factory=list)


def _build_workflow_input(req: WorkflowRequest) -> WorkflowInput:
    ref_md = req.ref_md or req.family_md
    ref_index_js = req.ref_index_js or req.family_index_js
    prepared = WorkflowInput(
        generation_mode=req.generation_mode,
        github_md_folder_url=req.github_md_folder_url,
        github_js_folder_url=req.github_js_folder_url,
        source_md_url=req.source_md_url,
        source_js_url=req.source_js_url,
        input_text=req.input_text,
        model_list=req.model_list,
        github_url=req.github_url,
        skip_stages=req.skip_stages,
        remote_folder=req.remote_folder,
        ssh_config=req.ssh_config,
        github_config=req.github_config,
        ref_md=ref_md,
        ref_index_js=ref_index_js,
        reference_folder=req.reference_folder or WorkflowInput.model_fields["reference_folder"].default,
    )
    return prepare_workflow_input(prepared)


def _run_current_stage():
    if not state.runner:
        raise HTTPException(status_code=400, detail="Workflow has not started.")
    if state.step >= len(state.stages):
        state.finished = True
        return {"stage": "completed", "finished": True, "result": None}

    stage_name = state.stages[state.step]
    result = state.runner._run_stage(stage_name)
    return {"stage": stage_name, "finished": False, "result": result}


@app.post("/api/start")
def start(req: WorkflowRequest):
    workflow_input = _build_workflow_input(req)
    enabled_stages = prepare_enabled_stages(workflow_input, req.stages)
    state.runner = WorkflowRunner(
        workflow_input=workflow_input,
        enabled_stages=enabled_stages,
    )
    state.stages = state.runner.enabled_stages
    state.step = 0
    state.finished = False
    return _run_current_stage()


@app.post("/api/next")
def next_stage():
    if state.finished:
        return {"stage": "completed", "finished": True, "result": None}
    state.step += 1
    return _run_current_stage()


@app.post("/api/run")
def run_all(req: WorkflowRequest):
    workflow_input = _build_workflow_input(req)
    runner = WorkflowRunner(
        workflow_input=workflow_input,
        enabled_stages=prepare_enabled_stages(workflow_input, req.stages),
    )
    return {"finished": True, "results": runner.run()}


def _run_workflow_stream(req: WorkflowRequest, event_queue: Queue) -> None:
    def emit(event: str, data: Dict[str, Any]) -> None:
        event_queue.put({"event": event, "data": data})

    try:
        workflow_input = _build_workflow_input(req)
        runner = WorkflowRunner(
            workflow_input=workflow_input,
            enabled_stages=prepare_enabled_stages(workflow_input, req.stages),
        )
        emit("workflow_started", {"stages": runner.enabled_stages})
        for idx, stage_name in enumerate(runner.enabled_stages):
            emit("stage_started", {"index": idx, "stage": stage_name})
            result = runner._run_stage(stage_name)
            runner.state.stage_results.append(result)
            cot = str(result.get("final_output") or "")
            emit(
                "stage_completed",
                {
                    "index": idx,
                    "stage": stage_name,
                    "result": result,
                    "cot": cot,
                },
            )
        emit("completed", {"results": runner.state.stage_results})
    except Exception as e:
        emit("error", {"error": str(e)})
    finally:
        event_queue.put(None)


async def _event_generator(event_queue: Queue):
    while True:
        await asyncio.sleep(0.05)
        while not event_queue.empty():
            item = event_queue.get()
            if item is None:
                yield "event: done\ndata: {}\n\n"
                return
            event = item.get("event", "message")
            data = json.dumps(item.get("data", {}), ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"


@app.post("/api/run/stream")
async def run_all_stream(req: WorkflowRequest):
    event_queue: Queue = Queue()
    worker = Thread(target=_run_workflow_stream, args=(req, event_queue), daemon=True)
    worker.start()
    return StreamingResponse(
        _event_generator(event_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def home():
    return {
        "service": "readme_generator_backend",
        "message": "Backend-only API service. Use a separate frontend to call /api/* endpoints.",
        "stream_endpoint": "/api/run/stream",
        "cors_allow_origins": allow_origins,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        reload_dirs=["./"],
    )
