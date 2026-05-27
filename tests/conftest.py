import os
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_llm: requires real LLM API (run with RUN_REAL_LLM_TESTS=1)"
    )


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_REAL_LLM_TESTS") != "1":
        skip_real = pytest.mark.skip(reason="need RUN_REAL_LLM_TESTS=1 to run")
        for item in items:
            if "real_llm" in item.keywords:
                item.add_marker(skip_real)
