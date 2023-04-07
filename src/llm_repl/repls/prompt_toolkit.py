import sys

from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.key_binding import KeyBindings

from rich.console import Console
from rich.markdown import Markdown

from llm_repl.llms import BaseLLM, LLMS
from llm_repl.repls import BaseREPL, REPLStyle, REPLS


class PromptToolkitREPL(BaseREPL):

    LOADING_MSG = "Thinking..."
    SERVER_MSG_TITLE = "LLM"
    CLIENT_MSG_TITLE = "You"
    ERROR_MSG_TITLE = "ERROR"
    INTRO_BANNER = "Welcome to LLM REPL! Input your message and press enter twice to send it to the LLM (type 'exit' or 'quit' to quit the application)"

    def __init__(self, **kwargs):
        self.console = Console()
        self.completer_function_table = self._basic_completer_function_table
        self.kb = KeyBindings()
        self.session: PromptSession = PromptSession(
            key_bindings=self.kb,
            vi_mode=True,
            complete_while_typing=True,
            complete_in_thread=True,
        )
        self._style: REPLStyle = kwargs["style"]
        # This will hold the reference to the model currently loaded
        self.llm: BaseLLM | None = None

    @property
    def style(self) -> REPLStyle:
        return self._style

    # ----------------------------- COMMANDS -----------------------------

    def info(self):
        """
        Print the information about the LLM currently loaded
        """
        llm_info = self.llm.info  # type: ignore
        self.print_misc_msg(llm_info)

    def exit(self):
        """
        Exit the application
        """
        self.console.print()
        self.console.rule(style=self._style.misc_msg_color)
        sys.exit(0)

    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """
        if llm_name not in LLMS:
            self.print_error_msg(f"LLM {llm_name} not found")
            return

        llm = LLMS[llm_name]
        self.llm = llm.load(self)  # type: ignore
        if self.llm is None:
            self.print_error_msg(f"Failed to load LLM {llm_name}")
            return

        # Add LLMs specific custom commands to the completer and to the function table
        custom_commands_table = {}
        for custom_command in self.llm.custom_commands:  # type: ignore
            custom_commands_table[custom_command["name"]] = custom_command["function"]

        self.completer_function_table = (
            self._basic_completer_function_table | custom_commands_table
        )
        self.session.completer = NestedCompleter.from_nested_dict(
            {cmd: None for cmd in self.completer_function_table.keys()}
        )
        self.session.app.invalidate()

    @property
    def _basic_completer_function_table(self):
        return {
            "info": self.info,
            "exit": self.exit,
            "quit": self.exit,
            "llm": self.load_llm,
        }

    # ----------------------------- END COMMANDS -----------------------------

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

    def print(self, msg: Any, **kwargs):
        """
        Simply prints the message as a normal print statement.

        :param str msg: The message to be printed.
        """
        self.console.print(msg, **kwargs)

    def print_client_msg(self, msg: str):
        """
        Prints the client message in the console according to the client style.

        :param str msg: The message to be printed.
        """
        self._print_msg(
            self.CLIENT_MSG_TITLE, Markdown(msg), self._style.client_msg_color
        )

    def print_server_msg(self, msg: str):
        """
        Prints the server message in the console according to the server style.

        :param str msg: The message to be printed.
        """
        self._print_msg(
            self.SERVER_MSG_TITLE, Markdown(msg), self._style.server_msg_color
        )

    def print_error_msg(self, msg: str):
        """
        Prints the error message in the console.

        :param str msg: The message to be printed.
        """
        self._print_msg(self.ERROR_MSG_TITLE, msg, self._style.error_msg_color)

    def print_misc_msg(self, msg: str, **kwargs):
        """
        Print the miscellaneous message in the console.

        :param str msg: The message to be printed.
        """
        justify = kwargs.pop("justify", "left")
        self._print_msg("", msg, self._style.misc_msg_color, justify=justify)

    def _setup_keybindings(self):
        """
        Setup keybindings for the prompt
        """

        @self.kb.add("enter")
        def _(event):
            self.handle_enter(event)

        # Exit gracefully with Ctrl+D
        @self.kb.add("c-d")
        def _(_):
            self.exit()

        # Exit gracefully with Ctrl+C
        @self.kb.add("c-c")
        def _(_):
            self.exit()

    async def handle_msg(self, *args, **kwargs):
        """
        Handle the user input and send it to the LLM

        :param BaseLLM llm: The LLM to send the message to
        """
        llm: BaseLLM = kwargs["llm"]
        user_input = await self.session.prompt_async("> ")
        user_input = user_input.rstrip()
        # Check if the input is a custom command
        if user_input in self.completer_function_table:
            self.completer_function_table[user_input]()
            return

        # Otherwise, process the input as a normal message that has
        # to be sent to the LLM
        self.print_client_msg(user_input)
        if not llm.is_in_streaming_mode:
            self.print_misc_msg(self.LOADING_MSG)
        # If the LLM is in streaming mode, The LLM itself will print the
        # response character by character and we just need to print a
        # the "start" and "end" rule line.
        # This can in theory be done by the StreamingCallback provided
        # by langchain but it appears to be broken in this version of
        # langchain.
        else:
            self.console.rule(
                f"[{self._style.server_msg_color}]{self.SERVER_MSG_TITLE}",
                style=self._style.server_msg_color,
            )
        resp = llm.process(user_input)
        if not llm.is_in_streaming_mode:
            self.print_server_msg(resp)
        else:
            self.console.print()
            self.console.rule(style=self._style.server_msg_color)

    async def run(self, llm_name):
        """
        Starts the REPL.

        The REPL will continue to run until the user presses Ctrl+C.

        The user can enter new lines in the REPL by pressing Enter once. The
        REPL will terminate when the user presses Enter twice.

        :param BaseLLM llm: The LLM to use.
        """
        self._setup_keybindings()
        self.load_llm(llm_name)

        if self.llm is None:
            return

        self.print_misc_msg(
            f"{self.INTRO_BANNER}\n\nLoaded model: {self.llm.name}", justify="center"
        )

        while True:
            await self.handle_msg(llm=self.llm)


REPLS["prompt_toolkit"] = PromptToolkitREPL
