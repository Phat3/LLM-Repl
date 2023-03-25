from __future__ import annotations

import os

from typing import Optional, Dict, Any, List, Union

from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain

from rich.markdown import Markdown

from llm_repl.repl import LLMRepl
from llm_repl.llms import BaseLLM


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, repl: LLMRepl) -> None:
        super().__init__()
        self.console = repl.console
        self.server_color = repl.server_color
        self.is_code_mode = False
        self.code_block = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # FIXME: This is just an hack to make the code blocks work
        #        This should be done properly in the future
        if token == "\n\n":
            token = ""
        if token == "``" or token == "```":
            if self.code_block:
                self.console.print(Markdown(self.code_block + "```\n"))
                self.code_block = ""
                self.is_code_mode = not self.is_code_mode
                return
            self.is_code_mode = not self.is_code_mode
            self.code_block = token
        elif self.is_code_mode:
            self.code_block += token
        else:
            if token == "`\n" or token == "`\n\n":
                token = "\n"
            self.console.print(token, end="")

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


class ChatGPT(BaseLLM):
    def __init__(self, api_key: str, repl: LLMRepl):
        self.api_key = api_key
        # TODO: Make options configurable
        self.streaming_mode = True

        # TODO: Make it customizable
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """
            If AI does not know the answer to a question, it truthfully says it does not know.
            """
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        llm = ChatOpenAI(
            openai_api_key=self.api_key,
            streaming=self.streaming_mode,
            callback_manager=CallbackManager([StreamingCallbackHandler(repl)]),
            verbose=True,
            # temperature=0,
        )  # type: ignore
        memory = ConversationBufferMemory(return_messages=True)
        self.model = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    @property
    def is_in_streaming_mode(self) -> bool:
        return self.streaming_mode

    @classmethod
    def load(cls, repl: LLMRepl) -> Optional[BaseLLM]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            repl.print_error_msg(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return None

        # TODO: Add autocomplete in repl
        model = cls(api_key, repl)
        return model

    def process(self, msg: str) -> str:
        resp = self.model.predict(input=msg)
        return resp.strip()
