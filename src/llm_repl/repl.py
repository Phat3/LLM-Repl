from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings

from rich.console import Console
from rich.markdown import Markdown

# from llm_repl.llms import BaseLLM

LLM_CMD_HANDLERS: dict[str, Callable] = {}


class LLMRepl:

    LOADING_MSG = "Thinking..."
    SERVER_MSG_TITLE = "LLM"
    CLIENT_MSG_TITLE = "You"
    ERROR_MSG_TITLE = "ERROR"
    EXIT_TOKEN = "exit"
    INTRO_BANNER = f"Welcome to LLM REPL! Input your message and press enter twice to send it to the LLM (type '{EXIT_TOKEN}' to quit the application)"

    def __init__(self, config: dict[str, Any]):
        self.console = Console()
        self.words: list[str] = [cmd for cmd in LLM_CMD_HANDLERS.keys()]
        self.completer = WordCompleter(self.words)
        self.kb = KeyBindings()
        self.session: PromptSession = PromptSession(
            completer=self.completer, key_bindings=self.kb
        )
        self.config = config

        # FIXME: This is temporary for test. This will be passed in the configuration file
        self.client_color = config["style"]["client"]["color"]
        self.server_color = config["style"]["server"]["color"]
        self.error_color = "bold red"
        self.misc_color = "gray"
        self.llm = None  # Optional[BaseLLM] = None

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

    def _print_msg(
        self, title: str, msg: str | Markdown, color: str, justify: str = "left"
    ):
        """
        Prints the message in the console according to the style.

        :param str msg: The message to be printed.
        :param str title: The title of the message.
        :param str color: The color of the message.
        """
        if not title:
            self.console.rule(style=color)
        else:
            self.console.rule(f"[{color}]{title}", style=color)
        self.console.print(msg, justify=justify)  # type: ignore
        self.console.rule(style=color)

    def print_client_msg(self, msg: str):
        """
        Prints the client message in the console according to the client style.

        :param str msg: The message to be printed.
        """
        self._print_msg(self.CLIENT_MSG_TITLE, Markdown(msg), self.client_color)

    def print_server_msg(self, msg: str):
        """
        Prints the server message in the console according to the server style.

        :param str msg: The message to be printed.
        """
        self._print_msg(self.SERVER_MSG_TITLE, Markdown(msg), self.server_color)

    def print_error_msg(self, msg: str):
        """
        Prints the error message in the console.

        :param str msg: The message to be printed.
        """
        self._print_msg(self.ERROR_MSG_TITLE, msg, self.error_color)

    def print_misc_msg(self, msg: str, justify: str = "left"):
        """
        Print the miscellaneous message in the console.

        :param str msg: The message to be printed.
        """
        self._print_msg("", msg, self.misc_color, justify=justify)

    def run(self, llm):
        """
        Starts the REPL.

        The REPL will continue to run until the user presses Ctrl+C.

        The user can enter new lines in the REPL by pressing Enter once. The
        REPL will terminate when the user presses Enter twice.

        :param BaseLLM llm: The LLM to use.
        """

        self.llm = llm.load(self)
        if self.llm is None:
            return

        self.print_misc_msg(self.INTRO_BANNER, justify="center")
        self.print_misc_msg(f"Loaded model: {self.llm.name}", justify="center")

        while True:
            user_input = self.session.prompt("> ").rstrip()
            if user_input == self.EXIT_TOKEN:
                self.print_misc_msg("Bye!")
                break

            self.print_client_msg(user_input)

            if not self.llm.is_in_streaming_mode:
                self.print_misc_msg(self.LOADING_MSG)
            else:
                self.console.rule(
                    f"[{self.server_color}]{self.SERVER_MSG_TITLE}",
                    style=self.server_color,
                )

            resp = self.llm.process(user_input)

            if not self.llm.is_in_streaming_mode:
                self.print_server_msg(resp)
            else:
                self.console.print()
                self.console.rule(style=self.server_color)
