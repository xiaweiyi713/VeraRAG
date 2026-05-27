"""Base Agent class for VeraRAG."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all agents in VeraRAG."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: Any | None = None
    ):
        self.config = config or {}
        self.llm_client = llm_client

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's main task."""
        pass

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs
    ) -> str:
        """Call the LLM with a prompt."""
        if self.llm_client is None:
            raise ValueError("LLM client not configured")

        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )


class LLMClient:
    """Simple LLM client wrapper supporting multiple providers."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
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
            self._client = ollama.Client()

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
            return response.choices[0].message.content

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
            return response.content[0].text

        elif self.provider == "ollama":
            kwargs = {"model": self.model, "prompt": prompt}
            if system_prompt:
                kwargs["system"] = system_prompt
            if max_tokens:
                kwargs["num_predict"] = max_tokens

            response = client.generate(**kwargs)
            return response["response"]

        else:
            raise ValueError(f"Unknown provider: {self.provider}")
