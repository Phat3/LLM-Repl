from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from llm_repl.repl import LLMRepl


class BaseLLM(ABC):
    @property
    @abstractmethod
    def is_in_streaming_mode(self):
        """Return whether the LLM is in streaming mode."""

    @classmethod
    @abstractmethod
    def load(cls, repl: LLMRepl) -> Optional[BaseLLM]:
        """Load the LLM."""

    @abstractmethod
    def process(self, msg) -> str:
        """Process the user message and return the response."""
