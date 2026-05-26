"""API endpoints for VeraRAG Web UI."""

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    question: str
    max_rounds: int = 5
    config_overrides: Dict[str, Any] = {}


def create_router(templates, db, config):
    """Create API router with all endpoints.

    Args:
        templates: Jinja2Templates instance
        db: Database instance
        config: VeraRAG configuration dict

    Returns:
        FastAPI APIRouter
    """
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Render the home page with query input."""
        return templates.TemplateResponse(request, "index.html", {
            "config": config or {}
        })

    @router.get("/history", response_class=HTMLResponse)
    async def history(request: Request):
        """Render the query history page."""
        queries = db.list_queries()
        return templates.TemplateResponse(request, "history.html", {
            "queries": queries
        })

    @router.get("/history/{query_id}", response_class=HTMLResponse)
    async def detail(request: Request, query_id: str):
        """Render the detail page for a specific query."""
        query = db.get_query(query_id)
        if query is None:
            return templates.TemplateResponse(request, "detail.html", {
                "query": None,
                "error": "查询不存在"
            }, status_code=404)
        return templates.TemplateResponse(request, "detail.html", {
            "query": query,
            "error": None
        })

    @router.get("/api/status")
    async def status():
        """Return system status."""
        return JSONResponse({
            "status": "running",
            "config_loaded": config is not None
        })

    @router.post("/query")
    async def stream_query(request: QueryRequest):
        """Process a query with SSE streaming."""
        import sys
        sys.path.insert(0, 'src')
        from src.pipeline.verarag import VeraRAG

        pipeline_config = config or {}
        if request.config_overrides:
            pipeline_config.update(request.config_overrides)

        event_queue = asyncio.Queue()

        def callback(event_type: str, data: dict):
            event_queue.put_nowait((event_type, data))

        async def event_generator():
            # Run pipeline in thread pool to not block the event loop
            loop = asyncio.get_event_loop()

            def run_pipeline():
                pipeline = VeraRAG(pipeline_config)
                output = pipeline.query_stream(
                    question=request.question,
                    max_rounds=request.max_rounds,
                    callback=callback
                )
                # Save to database
                query_id = db.save_query(
                    question=output.question,
                    answer=output.answer,
                    confidence=output.confidence,
                    result_json=output.to_dict()
                )
                event_queue.put_nowait(("saved", {"result_id": query_id}))
                event_queue.put_nowait(("_done", {}))

            loop.run_in_executor(None, run_pipeline)

            while True:
                try:
                    event_type, data = await asyncio.wait_for(
                        event_queue.get(), timeout=300.0
                    )
                except asyncio.TimeoutError:
                    yield {"event": "error", "data": json.dumps({"error": "timeout"})}
                    break

                if event_type == "_done":
                    break

                yield {
                    "event": event_type,
                    "data": json.dumps(data, ensure_ascii=False)
                }

        return EventSourceResponse(event_generator())

    return router
