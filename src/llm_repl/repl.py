import os

from typing import Any, Dict, List, Union, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.padding import Padding

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks.base import CallbackManager
from rich.markdown import Markdown


LLM_CMD_HANDLERS: dict[str, Callable] = {}

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, console) -> None:
        super().__init__()
        self.console = console
        self.server_color = "bold blue"
        self.is_code_mode = False
        self.code_block = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
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


class LLMRepl:
    def __init__(self, config: dict[str, Any]):
        self.console = Console()
        self.words: list[str] = [cmd for cmd in LLM_CMD_HANDLERS.keys()]
        self.completer = WordCompleter(self.words)
        self.kb = KeyBindings()
        self.session: PromptSession = PromptSession(
            completer=self.completer, key_bindings=self.kb
        )

        # FIXME: This is temporary for test. This will be passed in the configuration file
        self.client_color = config["style"]["client"]["color"]
        self.server_color = config["style"]["server"]["color"]
        self.client_padding = config["style"]["client"]["padding"]
        self.server_padding = config["style"]["server"]["padding"]

        self.streaming_mode = False

        # FIXME: Test if the model works
        self.model = ChatOpenAI(
            openai_api_key=OPENAI_KEY,
            streaming=self.streaming_mode,
            callback_manager=CallbackManager([StreamingCallbackHandler(self.console)]),
            verbose=True,
            temperature=0,
        )  # type: ignore

    def handle_enter(self, event):
        """
        This function is called when the user presses the Enter key.

        It allows the user to enter new lines in the prompt and terminates the
        prompt when the user presses Enter twice.

        :param event: The event object.
        """
        # Get the current buffer text
        text = event.app.current_buffer.text
        # Check if the last two characters are newlines
        if text.endswith("\n"):
            event.current_buffer.validate_and_handle()
        else:
            # Otherwise, insert a newline as usual
            event.app.current_buffer.insert_text("\n")

    def _get_centered_banner(self, banner: str) -> str:
        """
        Returns a centered banner surrounded by dashes that spans the entire
        width of the console.

        Example:
            >>> _get_centered_banner("LLM")
            "----------------- LLM -----------------"

        :param str banner: The banner to be centered.

        :rtype str
        :return: The centered banner.
        """
        return banner.center(self.console.size.width, "-")

    def _print_client_msg(self, msg: str):
        """
        Prints the client message in the console according to the client style.

        :param str msg: The message to be printed.
        """
        self.console.print(
            f"\n[{self.client_color}]{self._get_centered_banner('YOU')}[/{self.client_color}]"
        )
        padded_msg = Padding(msg, self.client_padding)
        self.console.print(padded_msg)
        self.console.print(
            f"[{self.client_color}]{self._get_centered_banner('-')}[/{self.client_color}]"
        )

    def _print_server_msg(self, msg: str):
        """
        Prints the server message in the console according to the server style.

        :param str msg: The message to be printed.
        """
        self.console.print(
            f"[{self.server_color}]{self._get_centered_banner('LLM')}[/{self.server_color}]"
        )
        _padded_msg = Padding(msg, self.server_padding)
        mark = Markdown(msg)
        self.console.print(mark)
        self.console.print(
            f"[{self.server_color}]{self._get_centered_banner('-')}[/{self.server_color}]\n"
        )

    def run(self):
        """
        Starts the REPL.

        The REPL will continue to run until the user presses Ctrl+C.

        The user can enter new lines in the REPL by pressing Enter once. The
        REPL will terminate when the user presses Enter twice.
        """
        while True:
            user_input = self.session.prompt("> ").rstrip()
            self._print_client_msg(user_input)

            if not self.streaming_mode:
                self.console.print("Thinking...")
            else:
                self.console.print(
                    f"[{self.server_color}]{self._get_centered_banner('LLM')}[/{self.server_color}]"
                )

            resp = self.model([HumanMessage(content=user_input)])
            if not self.streaming_mode:
                self._print_server_msg(resp.content.strip())
            else:
                self.console.print(
                    f"\n\n[{self.server_color}]{self._get_centered_banner('-')}[/{self.server_color}]\n"
                )
