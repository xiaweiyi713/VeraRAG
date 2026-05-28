"""API endpoints for VeraRAG Web UI."""

import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure project root is on sys.path for src.* imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Thread-safe BM25 cache ---
_bm25_retriever = None
_bm25_chunks: dict = {}
_bm25_lock = threading.Lock()


def _get_bm25():
    """Lazy-load BM25 index with thread safety (double-checked locking)."""
    global _bm25_retriever, _bm25_chunks

    if _bm25_retriever is not None:
        return _bm25_retriever, _bm25_chunks

    with _bm25_lock:
        if _bm25_retriever is not None:
            return _bm25_retriever, _bm25_chunks

        corpus_path = os.path.join(_project_root, "data", "verabench", "corpus.jsonl")
        if not os.path.exists(corpus_path):
            return None, []

        import json as _json

        from src.retriever.bm25 import BM25Retriever

        documents = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(_json.loads(line))

        retriever = BM25Retriever()
        retriever.index_documents(documents)
        _bm25_retriever = retriever
        _bm25_chunks = {doc.get("id", str(i)): doc for i, doc in enumerate(documents)}
        return _bm25_retriever, _bm25_chunks


class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    question: str
    max_rounds: int = 5
    config_overrides: dict[str, Any] = {}


class ConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str
    base_url: str = ""


class DemoQueryRequest(BaseModel):
    question: str


def _import_verarag():
    """Lazy import to avoid pulling heavy deps at module load."""
    from src.pipeline.verarag import VeraRAG
    return VeraRAG


def create_router(templates, db, config):
    """Create API router with all endpoints."""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "config": config or {}
        })

    @router.get("/history", response_class=HTMLResponse)
    async def history(request: Request):
        queries = db.list_queries()
        return templates.TemplateResponse(request, "history.html", {
            "queries": queries
        })

    @router.get("/history/{query_id}", response_class=HTMLResponse)
    async def detail(request: Request, query_id: str):
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

    # --- Status & Config APIs ---

    @router.get("/api/status")
    async def status():
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
        llm_cfg = db.get_llm_config()
        masked = llm_cfg.copy()
        if masked.get("api_key"):
            key = masked["api_key"]
            masked["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        return JSONResponse(masked)

    @router.put("/api/config")
    async def update_config(req: ConfigRequest):
        db.set_llm_config(
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url
        )
        return JSONResponse({"status": "ok"})

    # --- Query endpoints ---

    @router.post("/query")
    async def stream_query(request: QueryRequest):
        """Process a query with SSE streaming using real LLM pipeline."""
        VeraRAG = _import_verarag()

        pipeline_config = json.loads(json.dumps(config or {}))  # deep copy
        if request.config_overrides:
            pipeline_config.update(request.config_overrides)

        # Merge DB-stored LLM config into pipeline config
        llm_cfg = db.get_llm_config()
        if llm_cfg.get("provider"):
            pipeline_config.setdefault("llm", {})
            pipeline_config["llm"]["provider"] = llm_cfg["provider"]
            if llm_cfg.get("model"):
                pipeline_config["llm"]["model"] = llm_cfg["model"]
            if llm_cfg.get("api_key"):
                pipeline_config["llm"]["api_key"] = llm_cfg["api_key"]
            if llm_cfg.get("base_url"):
                pipeline_config["llm"]["base_url"] = llm_cfg["base_url"]

        event_queue = asyncio.Queue()
        loop_ref = asyncio.get_event_loop()

        def callback(event_type: str, data: dict):
            loop_ref.call_soon_threadsafe(event_queue.put_nowait, (event_type, data))

        async def event_generator():
            def run_pipeline():
                try:
                    pipeline = VeraRAG(pipeline_config)
                    output = pipeline.query_stream(
                        question=request.question,
                        max_rounds=request.max_rounds,
                        callback=callback
                    )
                    query_id = db.save_query(
                        question=output.question,
                        answer=output.answer,
                        confidence=output.confidence,
                        result_json=output.to_dict()
                    )
                    event_queue.put_nowait(("saved", {"result_id": query_id}))
                except Exception as e:
                    event_queue.put_nowait(("error", {"error": str(e)}))
                finally:
                    event_queue.put_nowait(("_done", {}))

            future = loop_ref.run_in_executor(None, run_pipeline)

            try:
                while True:
                    try:
                        event_type, data = await asyncio.wait_for(
                            event_queue.get(), timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}
                        continue

                    if event_type == "_done":
                        break

                    yield {
                        "event": event_type,
                        "data": json.dumps(data, ensure_ascii=False)
                    }
            finally:
                try:
                    await asyncio.wait_for(future, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass

        return EventSourceResponse(event_generator())

    @router.post("/query/demo")
    async def demo_query(request: DemoQueryRequest):
        """Demo mode: simulate the pipeline with fake data for UI testing."""
        question = request.question
        demo_answer = (
            f"关于「{question}」：根据现有证据，RAG 能够显著减少但不能完全消除幻觉问题。"
            "研究表明 RAG 在法律领域仍存在约 15% 的错误率，"
            "不过通过领域优化可将错误率降至 5%。建议结合人工审核机制确保可靠性。"
        )

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
                "answer": demo_answer,
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

        # Save demo result with correct answer
        result_id = db.save_query(
            question=question,
            answer=demo_answer,
            confidence=0.82,
            result_json={"demo": True, "question": question, "answer": demo_answer}
        )

        async def demo_generator():
            for event_type, data in demo_events:
                await asyncio.sleep(0.4)
                yield {
                    "event": event_type,
                    "data": json.dumps(data, ensure_ascii=False)
                }
            yield {"event": "saved", "data": json.dumps({"result_id": result_id})}

        return EventSourceResponse(demo_generator())

    # --- File upload endpoint ---

    @router.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload PDF/TXT/MD, extract text, chunk and merge into BM25 index."""
        filename = file.filename or "unknown"
        suffix = Path(filename).suffix.lower()

        if suffix not in (".pdf", ".txt", ".md"):
            raise HTTPException(status_code=400, detail="仅支持 PDF、TXT、MD 文件")

        raw_bytes = await file.read()

        # Extract text
        try:
            if suffix == ".pdf":
                import fitz
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            else:
                text = raw_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"文件解析失败: {e}")

        if not text.strip():
            raise HTTPException(status_code=422, detail="文件内容为空")

        # Chunk text (~500 chars per chunk, overlap 50)
        chunk_size = 500
        overlap = 50
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += chunk_size - overlap

        # Build new documents
        new_docs = []
        for i, chunk in enumerate(chunks):
            new_docs.append({
                "id": f"upload-{uuid.uuid4().hex[:8]}-{i}",
                "text": chunk,
                "title": filename,
                "source": "upload",
            })

        # Merge into global BM25 index with thread safety
        with _bm25_lock:
            # Load existing if not yet loaded
            existing_retriever, existing_chunks = _get_bm25()

            from src.retriever.bm25 import BM25Retriever

            if existing_retriever is not None:
                # Rebuild with combined documents
                combined_docs = list(existing_chunks.values()) + new_docs
                new_retriever = BM25Retriever()
                new_retriever.index_documents(combined_docs)

                global _bm25_retriever, _bm25_chunks
                _bm25_retriever = new_retriever
                _bm25_chunks = {doc.get("id", str(i)): doc for i, doc in enumerate(combined_docs)}
            else:
                # No existing index — build from uploaded docs only
                new_retriever = BM25Retriever()
                new_retriever.index_documents(new_docs)

                _bm25_retriever = new_retriever
                _bm25_chunks = {doc.get("id", str(i)): doc for i, doc in enumerate(new_docs)}

        return JSONResponse({
            "filename": filename,
            "chunks": len(new_docs),
            "chars": len(text),
        })

    # --- Export endpoint ---

    @router.get("/history/{query_id}/export")
    async def export_result(request: Request, query_id: str, format: str = "md"):
        query = db.get_query(query_id)
        if query is None:
            return JSONResponse({"error": "查询不存在"}, status_code=404)

        if format == "pdf":
            return JSONResponse({"error": "PDF 导出暂不支持"}, status_code=501)

        if format == "md":
            result = query.get("result_json", {}) if isinstance(query.get("result_json"), dict) else {}
            claims = result.get("answer_claims", [])
            evidence = result.get("evidence", [])

            md = f"# {query['question']}\n\n"
            md += f"**置信度**: {query['confidence']:.0%}\n\n"
            md += f"## 答案\n\n{query['answer']}\n\n"

            if claims:
                md += "## 声明验证\n\n"
                for c in claims:
                    status = c.get("verification_status", "?")
                    icon = {"supported": "✅", "refuted": "❌", "not_enough_info": "⚠️"}.get(status, "○")
                    md += f"- {icon} {c.get('claim', '')} (置信度: {c.get('confidence', 0):.2f})\n"
                md += "\n"

            if evidence:
                md += "## 证据\n\n"
                for e in evidence:
                    md += f"- **{e.get('title', '')}** ({e.get('source', '')}) — 分数: {e.get('combined_score', 0):.2f}\n"
                    if e.get('text_span'):
                        md += f"  > {e['text_span'][:200]}\n"
                md += "\n"

            return Response(
                content=md,
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename=query_{query_id}.md"}
            )

        return JSONResponse({"error": "不支持的格式"}, status_code=400)

    # --- WebSocket query endpoint ---

    @router.websocket("/ws/query")
    async def ws_query(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                question = data.get("question", "").strip()
                if not question:
                    await websocket.send_json({"event": "error", "data": {"error": "问题不能为空"}})
                    continue

                VeraRAG = _import_verarag()
                pipeline_config = json.loads(json.dumps(config or {}))
                llm_cfg = db.get_llm_config()
                if llm_cfg.get("provider"):
                    pipeline_config.setdefault("llm", {})
                    pipeline_config["llm"]["provider"] = llm_cfg["provider"]
                    if llm_cfg.get("model"):
                        pipeline_config["llm"]["model"] = llm_cfg["model"]
                    if llm_cfg.get("api_key"):
                        pipeline_config["llm"]["api_key"] = llm_cfg["api_key"]

                event_queue = asyncio.Queue()
                loop_ref = asyncio.get_event_loop()

                def callback(event_type: str, d: dict):
                    loop_ref.call_soon_threadsafe(event_queue.put_nowait, (event_type, d))

                def run_pipeline():
                    try:
                        pipeline = VeraRAG(pipeline_config)
                        output = pipeline.query_stream(
                            question=question, max_rounds=5, callback=callback
                        )
                        query_id = db.save_query(
                            question=output.question, answer=output.answer,
                            confidence=output.confidence, result_json=output.to_dict()
                        )
                        event_queue.put_nowait(("saved", {"result_id": query_id}))
                    except Exception as e:
                        event_queue.put_nowait(("error", {"error": str(e)}))
                    finally:
                        event_queue.put_nowait(("_done", {}))

                future = loop_ref.run_in_executor(None, run_pipeline)

                try:
                    while True:
                        try:
                            event_type, edata = await asyncio.wait_for(
                                event_queue.get(), timeout=300.0
                            )
                        except asyncio.TimeoutError:
                            await websocket.send_json({"event": "error", "data": {"error": "timeout"}})
                            break
                        if event_type == "_done":
                            break
                        await websocket.send_json({"event": event_type, "data": edata})
                finally:
                    try:
                        await asyncio.wait_for(future, timeout=5.0)
                    except (asyncio.TimeoutError, Exception):
                        pass

        except WebSocketDisconnect:
            pass

    return router
