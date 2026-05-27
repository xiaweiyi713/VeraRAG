"""End-to-end tests using real LLM API calls.

Run with: RUN_REAL_LLM_TESTS=1 pytest tests/test_e2e_real_llm.py -v
"""

import pytest

from src.pipeline.verarag import VeraRAG


def _get_test_config():
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 500,
        },
        "pipeline": {
            "max_retrieval_rounds": 1,
            "enable_conflict_graph": True,
            "enable_uncertainty": True,
            "enable_verification": True,
            "enable_repair": True,
        },
    }


@pytest.fixture
def pipeline():
    return VeraRAG(_get_test_config())


@pytest.mark.real_llm
class TestRealLLM:
    def test_simple_query_returns_answer(self, pipeline):
        output = pipeline.query("什么是人工智能？")
        assert output.answer
        assert len(output.answer) > 10
        assert 0.0 <= output.confidence <= 1.0

    def test_query_stream_events(self, pipeline):
        events = []
        output = pipeline.query_stream(
            "欧盟AI法案是什么？",
            callback=lambda t, d: events.append((t, d)),
        )
        assert output.answer
        event_types = [e[0] for e in events]
        assert "task_analysis" in event_types
        assert "reasoning" in event_types
        assert "complete" in event_types

    def test_query_with_conflict(self, pipeline):
        output = pipeline.query("谷歌量子霸权的争议是什么？")
        assert output.answer
        assert output.conflict_report is not None
