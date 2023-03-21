from typing import Callable, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.padding import Padding

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
        self.client_padding = config["style"]["client"]["padding"]
        self.server_padding = config["style"]["server"]["padding"]

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
        padded_msg = Padding(msg, self.server_padding)
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
        padded_msg = Padding(msg, self.server_padding)
        self.console.print(padded_msg)
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
            response = f"I heard you say {user_input}."
            self._print_server_msg(response)
