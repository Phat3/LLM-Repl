from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type

from llm_repl.repls import BaseREPL


class BaseLLM(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM."""

    @property
    @abstractmethod
    def info(self) -> str:
        """Return the info of the LLM."""

    @property
    @abstractmethod
    def is_in_streaming_mode(self):
        """Return whether the LLM is in streaming mode."""

    @classmethod
    @abstractmethod
    def load(cls, repl: BaseREPL) -> BaseLLM | None:
        """Load the LLM."""

    @abstractmethod
    async def process(self, msg) -> str:
        """Process the user message and return the response."""

    # FIXME: Define a proper type for the custom command
    @property
    def custom_commands(self) -> List[Any]:
        """Return the list of the custom commands of the LLM."""
        return []


LLMS: Dict[str, Type[BaseLLM]] = {}
