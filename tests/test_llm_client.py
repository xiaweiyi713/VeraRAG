"""Offline tests for LLM provider adapters."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.utils.llm_client import LLMClient, create_llm_client


def _fake_openai_module(created_clients, completion_calls, content="provider answer"):
    fake_module = ModuleType("openai")

    class FakeCompletions:
        def create(self, **kwargs):
            completion_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)
            self.chat = SimpleNamespace(
                completions=FakeCompletions(),
            )

    fake_module.OpenAI = FakeOpenAI
    return fake_module


def test_openai_generate_preserves_zero_temperature_and_json_format(monkeypatch):
    created_clients = []
    completion_calls = []
    monkeypatch.setitem(
        sys.modules,
        "openai",
        _fake_openai_module(created_clients, completion_calls, content="json answer"),
    )

    client = LLMClient(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
        base_url="https://proxy.example/v1",
        temperature=0.7,
        max_tokens=900,
    )
    answer = client.generate(
        "return json",
        system_prompt="system rules",
        max_tokens=50,
        temperature=0.0,
        response_format="json",
    )

    assert answer == "json answer"
    assert created_clients == [
        {"api_key": "test-key", "base_url": "https://proxy.example/v1"}
    ]
    assert completion_calls == [
        {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "return json"},
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
    ]


@pytest.mark.parametrize(
    ("provider", "expected_base_url", "expects_json_format"),
    [
        ("deepseek", "https://api.deepseek.com", True),
        ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", False),
        ("zhipuai", "https://open.bigmodel.cn/api/paas/v4/", True),
    ],
)
def test_openai_compatible_providers_use_expected_base_urls(
    monkeypatch,
    provider,
    expected_base_url,
    expects_json_format,
):
    created_clients = []
    completion_calls = []
    monkeypatch.setitem(
        sys.modules,
        "openai",
        _fake_openai_module(created_clients, completion_calls),
    )

    answer = LLMClient(
        provider=provider,
        model="provider-model",
        api_key="provider-key",
    ).generate("question", response_format="json")

    assert answer == "provider answer"
    assert created_clients == [{"api_key": "provider-key", "base_url": expected_base_url}]
    call = completion_calls[0]
    assert call["model"] == "provider-model"
    assert call["messages"] == [{"role": "user", "content": "question"}]
    assert ("response_format" in call) is expects_json_format


def test_anthropic_generate_sends_system_prompt(monkeypatch):
    fake_module = ModuleType("anthropic")
    created_clients = []
    message_calls = []

    class FakeMessages:
        def create(self, **kwargs):
            message_calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic answer")])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)
            self.messages = FakeMessages()

    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    answer = LLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="anthropic-key",
    ).generate("hello", system_prompt="be concise", temperature=0.0)

    assert answer == "anthropic answer"
    assert created_clients == [{"api_key": "anthropic-key"}]
    assert message_calls == [
        {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2000,
            "temperature": 0.0,
            "system": "be concise",
        }
    ]


def test_ollama_generate_uses_prompt_api_and_json_format(monkeypatch):
    fake_module = ModuleType("ollama")
    created_clients = []
    generate_calls = []

    class FakeOllamaClient:
        def __init__(self, host):
            created_clients.append({"host": host})

        def generate(self, **kwargs):
            generate_calls.append(kwargs)
            return {"response": "ollama answer"}

    fake_module.Client = FakeOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    answer = LLMClient(
        provider="ollama",
        model="qwen-local",
        base_url="http://ollama.local:11434",
    ).generate("hello", system_prompt="system", response_format="json")

    assert answer == "ollama answer"
    assert created_clients == [{"host": "http://ollama.local:11434"}]
    assert generate_calls == [
        {
            "model": "qwen-local",
            "prompt": "hello",
            "options": {
                "num_predict": 2000,
                "temperature": 0.7,
            },
            "system": "system",
            "format": "json",
        }
    ]


def test_is_available_handles_api_keys_and_client_failures(monkeypatch):
    client = LLMClient(provider="openai", api_key="")
    client._client = object()
    assert client.is_available() is False

    client_with_key = LLMClient(provider="openai", api_key="test-key")
    client_with_key._client = object()
    assert client_with_key.is_available() is True

    failing = LLMClient(provider="ollama")
    failing._client = SimpleNamespace(list=lambda: (_ for _ in ()).throw(OSError("down")))
    assert failing.is_available() is False


def test_ollama_is_available_success_with_cached_client():
    client = LLMClient(provider="ollama")
    client._client = SimpleNamespace(list=lambda: ["qwen2.5:7b"])

    assert client.is_available() is True


def test_unknown_provider_errors_from_client_and_generate():
    client = LLMClient(provider="unknown")

    with pytest.raises(ValueError, match="Unknown provider"):
        client._get_client()

    client._client = object()
    with pytest.raises(ValueError, match="Unknown provider"):
        client.generate("hello")


def test_provider_name_is_normalized_and_reads_matching_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")

    client = LLMClient(provider=" DeepSeek ", model="deepseek-chat")

    assert client.provider == "deepseek"
    assert client.api_key == "env-deepseek"


def test_openai_compatible_base_url_can_be_overridden(monkeypatch):
    created_clients = []
    completion_calls = []
    monkeypatch.setitem(
        sys.modules,
        "openai",
        _fake_openai_module(created_clients, completion_calls),
    )

    answer = LLMClient(
        provider="deepseek",
        model="deepseek-chat",
        api_key="provider-key",
        base_url="https://gateway.internal/deepseek/v1",
    ).generate("question", system_prompt="system", response_format="json")

    assert answer == "provider answer"
    assert created_clients == [
        {"api_key": "provider-key", "base_url": "https://gateway.internal/deepseek/v1"}
    ]
    assert completion_calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    assert completion_calls[0]["response_format"] == {"type": "json_object"}


def test_create_llm_client_prefers_explicit_provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")

    client = create_llm_client(provider="deepseek", model="deepseek-reasoner")

    assert client.provider == "deepseek"
    assert client.model == "deepseek-reasoner"
    assert client.api_key == "env-key"


def test_create_llm_client_detects_openai_after_unavailable_ollama(monkeypatch):
    fake_ollama = ModuleType("ollama")

    class UnavailableOllamaClient:
        def __init__(self, host):
            self.host = host

        def list(self):
            raise OSError("ollama unavailable")

    fake_ollama.Client = UnavailableOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    for name in [
        "ANTHROPIC_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPUAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    client = create_llm_client()

    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client.api_key == "openai-key"


def test_create_llm_client_detects_available_ollama(monkeypatch):
    fake_ollama = ModuleType("ollama")
    created_clients = []

    class AvailableOllamaClient:
        def __init__(self, host):
            created_clients.append(host)

        def list(self):
            return ["qwen2.5:7b"]

    fake_ollama.Client = AvailableOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    for name in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPUAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    client = create_llm_client(model="local-model")

    assert created_clients == ["http://localhost:11434"]
    assert client.provider == "ollama"
    assert client.model == "local-model"


@pytest.mark.parametrize(
    ("env_name", "expected_provider", "expected_model"),
    [
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-20250514"),
        ("DASHSCOPE_API_KEY", "dashscope", "qwen-turbo"),
        ("ZHIPUAI_API_KEY", "zhipuai", "glm-4-flash"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek-chat"),
    ],
)
def test_create_llm_client_detects_non_openai_api_providers(
    monkeypatch,
    env_name,
    expected_provider,
    expected_model,
):
    fake_ollama = ModuleType("ollama")

    class UnavailableOllamaClient:
        def __init__(self, host):
            self.host = host

        def list(self):
            raise OSError("ollama unavailable")

    fake_ollama.Client = UnavailableOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    for name in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPUAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, "provider-key")

    client = create_llm_client()

    assert client.provider == expected_provider
    assert client.model == expected_model
    assert client.api_key == "provider-key"


def test_create_llm_client_defaults_to_ollama_when_no_provider_available(monkeypatch):
    fake_ollama = ModuleType("ollama")

    class UnavailableOllamaClient:
        def __init__(self, host):
            self.host = host

        def list(self):
            raise OSError("ollama unavailable")

    fake_ollama.Client = UnavailableOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    for name in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPUAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    client = create_llm_client()

    assert client.provider == "ollama"
    assert client.model == "qwen2.5:7b"
