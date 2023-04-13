from __future__ import annotations

import os

from llm_repl.llms import LLMS, BaseLLM
from llm_repl.repls import BaseClientHandler
from llm_repl.llms.chatgpt import ChatGPT
from llm_repl import exceptions


class ChatGPT4(ChatGPT):
    @property
    def name(self) -> str:
        return "ChatGPT-4"

    @property
    def info(self) -> str:
        return "ChatGPT based on OpenAI's GPT-4 model."

    @classmethod
    def load(cls, client_handler: BaseClientHandler, **_llm_kwargs) -> BaseLLM:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise exceptions.MissingAPIKey("OPENAI_API_KEY")

        # TODO: Add autocomplete in repl
        model = cls(api_key, client_handler, model_name="gpt-4")
        return model


LLMS["chatgpt4"] = ChatGPT4
