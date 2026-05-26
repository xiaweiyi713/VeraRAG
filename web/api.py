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
        llm_cfg = db.get_llm_config()
        has_provider = bool(llm_cfg.get("provider"))
        return JSONResponse({
            "status": "running",
            "config_loaded": config is not None,
            "llm_provider": llm_cfg.get("provider", ""),
            "llm_model": llm_cfg.get("model", ""),
            "llm_configured": has_provider
        })

    @router.get("/api/config")
    async def get_config():
        """Get current LLM configuration."""
        llm_cfg = db.get_llm_config()
        # Mask API key for display
        masked = llm_cfg.copy()
        if masked.get("api_key"):
            key = masked["api_key"]
            masked["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        return JSONResponse(masked)

    class ConfigRequest(BaseModel):
        provider: str
        model: str
        api_key: str
        base_url: str = ""

    @router.put("/api/config")
    async def update_config(req: ConfigRequest):
        """Update LLM configuration."""
        db.set_llm_config(
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url
        )
        return JSONResponse({"status": "ok"})

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

    class DemoQueryRequest(BaseModel):
        question: str

    @router.post("/query/demo")
    async def demo_query(request: DemoQueryRequest):
        """Demo mode: simulate the pipeline with fake data for UI testing."""
        question = request.question

        demo_events = [
            ("stage", {"stage": "task_analysis", "status": "started"}),
            ("task_analysis", {
                "task_type": "multi-hop_qa",
                "complexity": "high",
                "keywords": ["RAG", "hallucination", "legal"],
                "requires_retrieval": True,
                "requires_conflict_check": True,
                "estimated_hops": 3
            }),
            ("stage", {"stage": "decomposition", "status": "started"}),
            ("decomposition", {
                "subquestions": [
                    {"id": "sq0", "question": f"关于「{question}」的核心事实是什么？", "status": "pending", "coverage_score": 0.0},
                    {"id": "sq1", "question": "有哪些支持或反驳的证据？", "status": "pending", "coverage_score": 0.0},
                    {"id": "sq2", "question": "现有研究结论的置信度如何？", "status": "pending", "coverage_score": 0.0}
                ]
            }),
            ("stage", {"stage": "retrieval", "round": 1, "total_rounds": 2, "status": "started"}),
            ("evidence", {
                "round": 1, "new_count": 3, "total": 3,
                "evidence": [
                    {"evidence_id": "E1", "source": "paper", "title": "RAG 在专业领域的应用研究",
                     "text_span": "实验表明 RAG 能将幻觉率从 20% 降至 8%，但在法律领域效果有限",
                     "combined_score": 0.87, "credibility_score": 0.9, "relevance_score": 0.85},
                    {"evidence_id": "E2", "source": "study", "title": "法律 AI 可靠性评估",
                     "text_span": "研究显示 RAG 系统在法律问答中仍有约 15% 的错误率",
                     "combined_score": 0.82, "credibility_score": 0.85, "relevance_score": 0.80},
                    {"evidence_id": "E3", "source": "blog", "title": "RAG 的局限性分析",
                     "text_span": "RAG 无法完全消除幻觉，只能减少约 50% 的错误",
                     "combined_score": 0.72, "credibility_score": 0.7, "relevance_score": 0.75}
                ]
            }),
            ("conflict", {"conflicts": 1, "conflict_score": 0.35,
             "edges": [{"source_id": "E1", "target_id": "E3", "conflict_type": "numeric_conflict",
                        "confidence": 0.7, "rationale": "数值差异：20%→8% vs 减少约50%"}]}),
            ("stage", {"stage": "retrieval", "round": 2, "total_rounds": 2, "status": "started"}),
            ("evidence", {
                "round": 2, "new_count": 2, "total": 5,
                "evidence": [
                    {"evidence_id": "E4", "source": "paper", "title": "法律领域 RAG 优化方案",
                     "text_span": "通过引入领域特定知识库和严格验证机制，法律 RAG 错误率可降至 5%",
                     "combined_score": 0.90, "credibility_score": 0.92, "relevance_score": 0.88},
                    {"evidence_id": "E5", "source": "report", "title": "2025 法律 AI 年度报告",
                     "text_span": "行业调查显示 73% 的法律从业者认为 RAG 辅助系统提升了工作效率",
                     "combined_score": 0.78, "credibility_score": 0.8, "relevance_score": 0.76}
                ]
            }),
            ("uncertainty", {"action": "continue_retrieval", "confidence": 0.62, "reason": "证据冲突需进一步验证"}),
            ("stage", {"stage": "reasoning", "status": "started"}),
            ("reasoning", {
                "answer": f"关于「{question}」：根据现有证据，RAG 能够显著减少但不能完全消除幻觉问题。研究表明 RAG 在法律领域仍存在约 15% 的错误率，不过通过领域优化可将错误率降至 5%。建议结合人工审核机制确保可靠性。",
                "claims": [
                    {"claim": "RAG 能将幻觉率从 20% 降至 8%", "verification_status": "supported",
                     "confidence": 0.87, "supporting_evidence": ["E1"], "conflicting_evidence": []},
                    {"claim": "法律领域 RAG 错误率仍有 15%", "verification_status": "supported",
                     "confidence": 0.82, "supporting_evidence": ["E2"], "conflicting_evidence": []},
                    {"claim": "通过优化可降至 5%", "verification_status": "not_enough_info",
                     "confidence": 0.70, "supporting_evidence": ["E4"], "conflicting_evidence": []}
                ],
                "steps": [
                    {"step": 1, "description": "检索 RAG 在专业领域的应用证据", "confidence": 0.87},
                    {"step": 2, "description": "对比不同来源的实验数据", "confidence": 0.75},
                    {"step": 3, "description": "综合分析得出结论", "confidence": 0.82}
                ]
            }),
            ("stage", {"stage": "verification", "status": "started"}),
            ("verification", {
                "claim_verifications": [
                    {"claim": "RAG 能将幻觉率从 20% 降至 8%", "status": "supported"},
                    {"claim": "法律领域 RAG 错误率仍有 15%", "status": "supported"},
                    {"claim": "通过优化可降至 5%", "status": "not_enough_info"}
                ],
                "overall_status": "supported",
                "issues": [],
                "missing_evidence_for": ["通过优化可降至 5%"],
                "has_critical_issues": False
            }),
            ("complete", {"elapsed_time": 3.45, "confidence": 0.82, "num_evidence": 5, "num_conflicts": 1}),
        ]

        async def demo_generator():
            for event_type, data in demo_events:
                await asyncio.sleep(0.4)
                yield {
                    "event": event_type,
                    "data": json.dumps(data, ensure_ascii=False)
                }

        # Save demo result
        result_id = db.save_query(
            question=question,
            answer=demo_events[-1][1].get("answer_preview", question),
            confidence=0.82,
            result_json={"demo": True, "question": question}
        )
        async def wrapper():
            async for event in demo_generator():
                yield event
            yield {"event": "saved", "data": json.dumps({"result_id": result_id})}

        return EventSourceResponse(wrapper())

    return router
