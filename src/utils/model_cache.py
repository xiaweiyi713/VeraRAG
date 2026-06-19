"""Process-local cache for optional sentence-transformers models."""

import threading
from collections.abc import Callable
from typing import Any

_MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}
_MODEL_FAILURES: dict[tuple[str, str, bool], Exception] = {}
_MODEL_LOCK = threading.Lock()


def load_optional_model_once(
    kind: str,
    model_name: str,
    factory: Callable[[], Any],
    *,
    local_files_only: bool = False,
) -> tuple[Any | None, Exception | None]:
    """Load one optional model per process and remember both success and failure."""
    key = (kind, model_name, local_files_only)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key], None
        if key in _MODEL_FAILURES:
            return None, _MODEL_FAILURES[key]
        try:
            model = factory()
        except Exception as exc:
            _MODEL_FAILURES[key] = exc
            return None, exc
        _MODEL_CACHE[key] = model
        return model, None


def clear_optional_model_cache() -> None:
    """Clear caches for deterministic tests."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
        _MODEL_FAILURES.clear()
