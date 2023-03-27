from __future__ import annotations

import os

from typing import Optional

from llm_repl.repl import LLMRepl
from llm_repl.llms import BaseLLM
from llm_repl.llms.chatgpt import ChatGPT


class ChatGPT4(ChatGPT):
    @property
    def name(self) -> str:
        return "ChatGPT-4"

    @classmethod
    def load(cls, repl: LLMRepl) -> Optional[BaseLLM]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            repl.print_error_msg(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return None

        # TODO: Add autocomplete in repl
        model = cls(api_key, repl, model_name="gpt-4")
        return model
