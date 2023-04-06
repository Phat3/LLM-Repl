from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Type

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

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
    def process(self, msg) -> str:
        """Process the user message and return the response."""

    # FIXME: Define a proper type for the custom command
    @property
    def custom_commands(self) -> List[Any]:
        """Return the list of the custom commands of the LLM."""
        return []


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


LLMS: Dict[str, Type[BaseLLM]] = {}
