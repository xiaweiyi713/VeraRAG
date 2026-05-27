"""End-to-end tests using real LLM API calls.

Run with: RUN_REAL_LLM_TESTS=1 pytest tests/test_e2e_real_llm.py -v
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.verarag import VeraRAG


def _get_test_config():
    return {
        "llm": {
            "provider": os.getenv("TEST_LLM_PROVIDER", "openai"),
            "model": os.getenv("TEST_LLM_MODEL", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
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


class TestPipelineConfigPassthrough:
    """Verify api_key config passthrough — no real LLM needed."""

    def test_api_key_direct(self):
        config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-test-direct-key",
                "temperature": 0.0,
                "max_tokens": 200,
            },
            "pipeline": {"max_retrieval_rounds": 1},
        }
        p = VeraRAG(config)
        assert p.llm_client.api_key == "sk-test-direct-key"

    def test_api_key_env_var_fallback(self):
        config = {
            "llm": {"provider": "openai"},
        }
        p = VeraRAG(config)
        expected = os.getenv("OPENAI_API_KEY", "")
        assert p.llm_client.api_key == expected

    def test_base_url_passthrough(self):
        config = {
            "llm": {
                "provider": "openai",
                "base_url": "https://custom.api.com/v1",
            },
        }
        p = VeraRAG(config)
        assert p.llm_client.base_url == "https://custom.api.com/v1"
