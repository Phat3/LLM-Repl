from __future__ import annotations

import os
import pkg_resources  # type: ignore
import yaml
import pydantic

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
from llm_repl import exceptions

DATA_FOLDER = pkg_resources.resource_filename("llm_repl", "data")
PERSONALITIES_FOLDER = os.path.join(DATA_FOLDER, "chatgpt", "personalities")
DEFAULT_PERSONALITY = os.path.join(PERSONALITIES_FOLDER, "default.yml")


class ChatGPTPersonality(pydantic.BaseModel):
    """ChatGPT personality."""

    description: str
    personality: str
    memories: List[str] | None


class AsyncChatGPTStreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, repl: BaseREPL) -> None:
        super().__init__()
        self.repl = repl

    async def on_llm_new_token(self, token: str, **kwargs: Any):
        """Run on new LLM token. Only available when streaming is enabled."""
        await self.repl.print(token)


class ChatGPT(BaseLLM):
    def __init__(
        self,
        api_key: str,
        repl: BaseREPL,
        model_name: str = "gpt-3.5-turbo",
        personality: ChatGPTPersonality | None = None,
    ):
        self.api_key = api_key
        # TODO: Make options configurable
        self.streaming_mode = True

        # TODO: Make it customizable
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    personality.personality if personality is not None else ""
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
    def load(cls, repl: BaseREPL, **llm_kwargs) -> Optional[BaseLLM]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise exceptions.MissingAPIKey("OPENAI_API_KEY")

        # Path to the yaml file containing the personality
        personality_filepath = llm_kwargs.get("personality", None)

        if personality_filepath is None or not os.path.isfile(personality_filepath):
            personality_filepath = DEFAULT_PERSONALITY

        with open(personality_filepath, "r") as f:
            personality_content = yaml.safe_load(f)
        try:
            personality = ChatGPTPersonality(**personality_content)
        except pydantic.ValidationError:
            personality = ChatGPTPersonality(
                description="Default personality", personality="", memories=None
            )

        # TODO: Add autocomplete in repl
        model = cls(api_key, repl, personality=personality)
        return model

    async def process(self, msg: str) -> str:
        resp = await self.model.apredict(input=msg)
        return resp.strip()


LLMS["chatgpt"] = ChatGPT
