"""Integration test: full VeraRAG pipeline with Mock LLM."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    VeraRAGOutput, TaskType, Complexity, TaskAnalysis,
    Evidence, AnswerClaim, VerificationStatus, VerificationReport,
    UncertaintyBreakdown, ReasoningStep,
)
from src.retriever.base import RetrievalResult


class MockRetriever:
    """Mock retriever that uses BM25 internally to avoid sentence-transformers dep."""
    def __init__(self, config=None, **kwargs):
        self.docs = []
        self._bm25 = None

    def index_documents(self, documents):
        from src.retriever.bm25 import BM25Retriever
        self._bm25 = BM25Retriever()
        self._bm25.index_documents(documents)
        self.docs = documents

    def retrieve(self, query, top_k=10, **kwargs):
        return self._bm25.retrieve(query, top_k=top_k)


class MockLLMClient:
    """Mock LLM that returns structured responses based on prompt keywords."""

    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider", "mock")
        self.model = kwargs.get("model", "mock")

    def generate(self, prompt: str, **kwargs) -> str:
        # Planner: must use unique markers not present in other prompts
        if "break down the following complex question" in prompt.lower() or "分解" in prompt or "decompos" in prompt.lower():
            return json.dumps({
                "subquestions": [
                    {"question": "量子计算是什么", "required_evidence_type": "general",
                     "dependency_ids": [], "requires_counter_evidence": False},
                    {"question": "量子计算的最新进展有哪些", "required_evidence_type": "general",
                     "dependency_ids": ["sq0"], "requires_counter_evidence": False},
                ],
                "reasoning_plan": ["查找量子计算基本概念", "汇总最新进展"]
            })
        if "分析" in prompt or "任务" in prompt or "task" in prompt.lower():
            return json.dumps({
                "task_type": "multi_hop_qa",
                "complexity": "high",
                "keywords": ["测试", "关键词"],
                "requires_retrieval": True,
                "requires_conflict_check": True,
                "estimated_hops": 2,
            })
        if "答案" in prompt or "answer" in prompt.lower() or "推理" in prompt or "reason" in prompt.lower():
            return json.dumps({
                "answer": "根据证据分析，这是一个测试答案。",
                "answer_claims": [
                    {"claim": "测试声明", "supporting_evidence": ["E1"],
                     "conflicting_evidence": [], "confidence": 0.85, "verification_status": "supported",
                     "claim_type": "factual", "verifiable": True, "support_type": "direct"}
                ],
                "reasoning_chain": [
                    {"step": 1, "description": "检索证据", "evidence_ids": ["E1"], "confidence": 0.8}
                ],
            })
        if "验证" in prompt or "verif" in prompt.lower():
            return json.dumps({
                "claim_verifications": [
                    {"claim": "声明1", "status": "supported"},
                ],
                "overall_status": "supported",
                "issues": [],
            })
        if "修复" in prompt or "repair" in prompt.lower():
            return "修复后的答案"
        return "默认回答"


@pytest.fixture
def pipeline_config():
    return {
        "llm": {"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
        "pipeline": {
            "max_retrieval_rounds": 2,
            "max_subquestions": 3,
            "enable_conflict_graph": True,
            "enable_uncertainty": True,
            "enable_verification": True,
            "enable_repair": True,
        },
        "retriever": {"top_k": 5},
    }


def _create_pipeline(config):
    """Create pipeline with mock LLM and mock retriever."""
    from src.pipeline.verarag import VeraRAG
    with patch("src.pipeline.verarag.HybridRetriever", MockRetriever):
        pipeline = VeraRAG(config)
    # Replace LLM client on pipeline and all agents
    mock_llm = MockLLMClient()
    pipeline.llm_client = mock_llm
    for agent in [pipeline.task_analyzer, pipeline.planner, pipeline.retrieval_agent,
                  pipeline.reasoning_agent, pipeline.verifier_agent, pipeline.repair_agent,
                  pipeline.evidence_extractor, pipeline.conflict_graph_builder]:
        if hasattr(agent, 'llm_client'):
            agent.llm_client = mock_llm
    return pipeline


class TestPipelineIntegration:
    def test_full_pipeline_run(self, pipeline_config):
        """Test that the full pipeline runs end-to-end with mock LLM."""
        pipeline = _create_pipeline(pipeline_config)

        docs = [
            {"id": "D1", "text": "量子计算是一种利用量子力学原理进行信息处理的计算方式。"
                      "与传统计算机不同，量子计算机使用量子比特。"
                      "2024年谷歌发布了最新的量子处理器Willow。",
             "title": "量子计算技术综述"},
            {"id": "D2", "text": "人工智能在医疗影像诊断中表现出色。"
                      "研究显示AI诊断准确率达到95%。"
                      "但AI在复杂病例中仍存在局限性。",
             "title": "AI医疗诊断报告"},
            {"id": "D3", "text": "量子比特是量子计算的基本单元。"
                      "超导量子比特和离子阱是两种主要技术路线。"
                      "谷歌IBM等公司都在推进量子计算研究。",
             "title": "量子比特技术"},
            {"id": "D4", "text": "2024年量子计算领域取得多项突破。"
                      "微软发布了拓扑量子比特的研究成果。"
                      "IBM推出了超过1000量子比特的处理器。",
             "title": "2024量子计算进展"},
            {"id": "D5", "text": "量子计算的应用前景包括药物发现、密码学和优化问题。"
                      "不过目前的量子计算机还处于NISQ时代，存在噪声问题。",
             "title": "量子计算应用"},
        ]
        pipeline.index_documents(docs)

        output = pipeline.query("量子计算的最新进展是什么？")
        assert isinstance(output, VeraRAGOutput)
        assert output.question == "量子计算的最新进展是什么？"
        assert isinstance(output.answer, str)
        assert len(output.answer) > 0
        assert 0 <= output.confidence <= 1

    def test_pipeline_streaming(self, pipeline_config):
        """Test streaming callback receives events."""
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试文档内容。" * 10, "title": "测试"}]
        pipeline.index_documents(docs)

        events = []
        def callback(event_type, data):
            events.append((event_type, data))

        output = pipeline.query_stream("测试问题", max_rounds=1, callback=callback)

        assert isinstance(output, VeraRAGOutput)
        assert len(events) > 0
        event_types = [e[0] for e in events]
        assert "complete" in event_types

    def test_pipeline_with_conflicts_disabled(self, pipeline_config):
        """Test pipeline runs with conflict detection disabled."""
        pipeline_config["pipeline"]["enable_conflict_graph"] = False
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)

    def test_pipeline_with_verification_disabled(self, pipeline_config):
        """Test pipeline skips verification when disabled."""
        pipeline_config["pipeline"]["enable_verification"] = False
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)

    def test_pipeline_minimal_config(self):
        """Test pipeline with minimal configuration."""
        pipeline = _create_pipeline({})

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)
