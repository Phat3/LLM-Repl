from __future__ import annotations

import os

from typing import Optional

from llm_repl.repls.prompt_toolkit import BaseREPL
from llm_repl.llms import BaseLLM, MODELS
from llm_repl.llms.chatgpt import ChatGPT


class ChatGPT4(ChatGPT):
    @property
    def name(self) -> str:
        return "ChatGPT-4"

    @property
    def info(self) -> str:
        return "ChatGPT based on OpenAI's GPT-4 model."

    @classmethod
    def load(cls, repl: BaseREPL) -> Optional[BaseLLM]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            repl.print_error_msg(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return None

        # TODO: Add autocomplete in repl
        model = cls(api_key, repl, model_name="gpt-4")
        return model


MODELS["chatgpt4"] = ChatGPT4
