"""LLM Client supporting multiple backends including local models."""

import os


class LLMClient:
    """
    Multi-backend LLM client supporting:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Ollama (local models)
    - 通义千问 (DashScope)
    - 智谱 AI (GLM)
    - DeepSeek
    """

    def __init__(
        self,
        provider: str = "ollama",  # 默认使用 Ollama
        model: str = "qwen2.5:7b",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy load the client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**client_kwargs)

        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == "ollama":
            import ollama
            self._client = ollama.Client(host=self.base_url or "http://localhost:11434")

        elif self.provider == "dashscope":  # 通义千问
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        elif self.provider == "zhipuai":  # 智谱 AI
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )

        elif self.provider == "deepseek":
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: str | None = None
    ) -> str:
        """Generate text from the LLM."""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        client = self._get_client()

        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(model=self.model, **kwargs)
            return response.choices[0].message.content  # type: ignore[no-any-return]

        elif self.provider == "anthropic":
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text  # type: ignore[no-any-return]

        elif self.provider == "ollama":
            # Ollama API
            kwargs = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            # Ollama returns a generator for chat
            response = client.chat(**kwargs)
            return response["message"]["content"]  # type: ignore[no-any-return]

        elif self.provider in ["dashscope", "zhipuai", "deepseek"]:
            # These use OpenAI-compatible API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}

            # 通义千问不支持 json_object format
            if response_format == "json" and self.provider != "dashscope":
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(model=self.model, **kwargs)
            return response.choices[0].message.content  # type: ignore[no-any-return]

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        try:
            client = self._get_client()

            if self.provider == "ollama":
                # Try to list models
                client.list()
                return True
            else:
                # For API providers, just check if API key is set
                return bool(self.api_key)
        except Exception as e:
            print(f"LLM check failed: {e}")
            return False


def create_llm_client(
    provider: str | None = None,
    model: str | None = None
) -> LLMClient:
    """
    Create an LLM client with automatic provider selection.

    Priority:
    1. Explicit provider
    2. Ollama (if available)
    3. OpenAI (if API key set)
    4. Anthropic (if API key set)
    """
    if provider:
        return LLMClient(provider=provider, model=model or "gpt-4o")

    # Auto-detect
    # 1. Check Ollama
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        client.list()
        print("✓ 检测到 Ollama，使用本地模型")
        return LLMClient(provider="ollama", model=model or "qwen2.5:7b")
    except Exception:
        pass

    # 2. Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("✓ 检测到 OPENAI_API_KEY，使用 OpenAI")
        return LLMClient(provider="openai", model=model or "gpt-4o")

    # 3. Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✓ 检测到 ANTHROPIC_API_KEY，使用 Claude")
        return LLMClient(provider="anthropic", model=model or "claude-sonnet-4-20250514")

    # 4. Check 通义千问
    if os.getenv("DASHSCOPE_API_KEY"):
        print("✓ 检测到 DASHSCOPE_API_KEY，使用通义千问")
        return LLMClient(provider="dashscope", model=model or "qwen-turbo")

    # 5. Check 智谱 AI
    if os.getenv("ZHIPUAI_API_KEY"):
        print("✓ 检测到 ZHIPUAI_API_KEY，使用智谱 GLM")
        return LLMClient(provider="zhipuai", model=model or "glm-4-flash")

    # 6. Check DeepSeek
    if os.getenv("DEEPSEEK_API_KEY"):
        print("✓ 检测到 DEEPSEEK_API_KEY，使用 DeepSeek")
        return LLMClient(provider="deepseek", model=model or "deepseek-chat")

    # Default to Ollama (will show error if not available)
    print("⚠ 未检测到 API Key，尝试使用 Ollama 本地模型...")
    return LLMClient(provider="ollama", model=model or "qwen2.5:7b")
