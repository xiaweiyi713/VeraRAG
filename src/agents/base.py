"""Base Agent class for VeraRAG."""

from abc import ABC, abstractmethod
from typing import Any

# Re-export the unified LLMClient from utils (supports 6 providers)
from ..utils.llm_client import LLMClient, create_llm_client  # noqa: F401


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

        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        return response if isinstance(response, str) else str(response)
