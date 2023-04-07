from __future__ import annotations

import os

from typing import Optional, Any, List

from langchain.callbacks.base import AsyncCallbackManager, AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain

from llm_repl.repls import BaseREPL
from llm_repl.llms import BaseLLM, LLMS


class AsyncChatGPTStreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, repl: BaseREPL) -> None:
        super().__init__()
        self.repl = repl

    async def on_llm_new_token(self, token: str, **kwargs: Any):
        """Run on new LLM token. Only available when streaming is enabled."""
        await self.repl.print(token)


class ChatGPT(BaseLLM):
    def __init__(self, api_key: str, repl: BaseREPL, model_name: str = "gpt-3.5-turbo"):
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
            callback_manager=AsyncCallbackManager(
                [AsyncChatGPTStreamingCallbackHandler(repl)]
            ),
            verbose=True,
            model_name=model_name,
        )  # type: ignore
        memory = ConversationBufferMemory(return_messages=True)
        self.model = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    @property
    def name(self) -> str:
        return "ChatGPT"

    @property
    def info(self) -> str:
        return "ChatGPT based on OpenAI's GPT-3.5-turbo model."

    @property
    def is_in_streaming_mode(self) -> bool:
        return self.streaming_mode

    def _say_hi(self) -> None:
        pass

    # FIXME: Define a proper type for the custom command
    @property
    def custom_commands(self) -> List[Any]:
        return [{"name": "say_hi", "function": self._say_hi}]

    @classmethod
    def load(cls, repl: BaseREPL) -> Optional[BaseLLM]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            repl.print_error_msg(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return None

        # TODO: Add autocomplete in repl
        model = cls(api_key, repl)
        return model

    async def process(self, msg: str) -> str:
        resp = await self.model.apredict(input=msg)
        return resp.strip()


LLMS["chatgpt"] = ChatGPT
