from typing import Any, Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings

from rich.console import Console
from rich.markdown import Markdown


LLM_CMD_HANDLERS: dict[str, Callable] = {}


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
        self.error_color = "bold red"
        self.misc_color = "gray"
        self.model: Optional[BaseLLM] = None

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

    def print_client_msg(self, msg: str):
        """
        Prints the client message in the console according to the client style.

        :param str msg: The message to be printed.
        """
        self.console.rule(f"[{self.client_color}]YOU", style=self.client_color)
        self.console.print(Markdown(msg))
        self.console.rule(style=self.client_color)

    def print_server_msg(self, msg: str):
        """
        Prints the server message in the console according to the server style.

        :param str msg: The message to be printed.
        """
        self.console.rule(f"[{self.server_color}]LLM", style=self.server_color)
        self.console.print(Markdown(msg))
        self.console.rule(style=self.server_color)

    def print_error_msg(self, msg: str):
        """
        Prints the error message in the console.

        :param str msg: The message to be printed.
        """
        self.console.rule(f"[{self.error_color}]ERROR", style=self.error_color)
        self.console.print(f"[{self.error_color}]{msg}")
        self.console.rule(style=self.error_color)

    def print_misc_msg(self, msg: str):
        """
        Print the miscellaneous message in the console.

        :param str msg: The message to be printed.
        """
        self.console.rule(f"[{self.misc_color}]{msg}", style=self.misc_color)

    def run(self):
        """
        Starts the REPL.

        The REPL will continue to run until the user presses Ctrl+C.

        The user can enter new lines in the REPL by pressing Enter once. The
        REPL will terminate when the user presses Enter twice.
        """

        self.model = ChatGPT.load(self)
        if self.model is None:
            return

        while True:
            user_input = self.session.prompt("> ").rstrip()
            self.print_client_msg(user_input)

            if not self.model.is_in_streaming_mode:
                self.print_misc_msg("Thinking...")
            else:
                self.console.rule(f"[{self.server_color}]LLM", style=self.server_color)

            resp = self.model.process(user_input)

            if not self.model.is_in_streaming_mode:
                self.print_server_msg(resp)
            else:
                self.console.print()
                self.console.rule(style=self.server_color)


from llm_repl.llms import BaseLLM
from llm_repl.llms.chatgpt import ChatGPT
