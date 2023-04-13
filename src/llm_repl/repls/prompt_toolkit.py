import sys
import asyncio

from pydantic import BaseModel  # pylint: disable=no-name-in-module

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.key_binding import KeyBindings

from rich.console import Console
from rich.markdown import Markdown

from llm_repl import exceptions
from llm_repl.llms import BaseLLM, LLMS
from llm_repl.repls import BaseREPL, REPLS, BaseClientHandler

# FIXME: This is temporary for test. This will be passed in the configuration file


class REPLStyle(BaseModel):
    """
    The style of the REPL
    """

    client_msg_color: str
    server_msg_color: str
    error_msg_color: str
    misc_msg_color: str


class PromptToolkitClientHandler(BaseClientHandler):

    LOADING_MSG = "Thinking..."
    SERVER_MSG_TITLE = "LLM"
    CLIENT_MSG_TITLE = "You"
    ERROR_MSG_TITLE = "ERROR"
    INTRO_BANNER = "Welcome to LLM REPL! Input your message and press enter twice to send it to the LLM (type 'exit' or 'quit' to quit the application)"
    DEFAULT_STYLE = REPLStyle(
        client_msg_color="bold green",
        server_msg_color="bold blue",
        error_msg_color="bold red",
        misc_msg_color="gray",
    )

    def __init__(self, style: None | REPLStyle = None):
        super().__init__()
        self.console = Console()
        self.completer_function_table = self._basic_completer_function_table
        self.kb = KeyBindings()
        self.session: PromptSession = PromptSession(
            key_bindings=self.kb,
            vi_mode=True,
            complete_while_typing=True,
            complete_in_thread=True,
        )
        self._style = style if style is not None else self.DEFAULT_STYLE
        # This will hold the reference to the model currently loaded
        self.llm: BaseLLM | None = None
        self.parse_markdown = True
        self.is_code_mode = False
        self.code_block = ""
        self.queue_is_empty_condition = asyncio.Condition()

    @property
    def style(self) -> REPLStyle:
        return self._style

    @property
    def _basic_completer_function_table(self):
        return {
            "info": self.info,
            "exit": self.exit,
            "quit": self.exit,
            "llm": self.load_llm,
        }

    @property
    def start_token(self) -> str:
        """Return the marker that act as start token"""
        return "[START]"

    @property
    def end_token(self) -> str:
        """Return the marker that act as end token"""
        return "[DONE]"

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

    def load_llm(self, llm_name: str, **_llm_kwargs) -> BaseLLM:
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load

        :raises exceptions.LLMNotFound: if the LLM is not found
        :raises exceptions.LLMException: if the LLM fails to load
        """
        if llm_name not in LLMS:
            raise exceptions.LLMNotFound(llm_name)

        llm_class = LLMS[llm_name]
        llm = llm_class.load(self)

        # Add LLMs specific custom commands to the completer and to the function table
        custom_commands_table = {}
        for custom_command in llm.custom_commands:  # type: ignore
            custom_commands_table[custom_command["name"]] = custom_command["function"]

        self.completer_function_table = (
            self._basic_completer_function_table | custom_commands_table
        )
        self.session.completer = NestedCompleter.from_nested_dict(
            {cmd: None for cmd in self.completer_function_table.keys()}
        )
        self.session.app.invalidate()
        return llm

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

    async def print_loop(self):
        while True:
            msg = await self.tokens.get()
            # Print an horizontal ruler with title if the message is the start token
            if msg == self.start_token:
                self.console.rule(
                    f"[{self._style.server_msg_color}]{self.SERVER_MSG_TITLE}",
                    style=self._style.server_msg_color,
                )
                self.tokens.task_done()
                continue
            # Print an horizontal ruler if the message is the end token
            if msg == self.end_token:
                if self.llm.is_in_streaming_mode:  # type: ignore
                    self.console.print()
                self.console.rule(style=self._style.server_msg_color)
                self.tokens.task_done()
                continue
            # Skip the markdown parsing if the user has disabled it
            if not self.parse_markdown:
                self.console.print(msg, end="")
                self.tokens.task_done()
                continue
            # If the LLM is not in streaming mode, there is no need to
            # parse the markdown incrementally since we already have the
            # whole message
            if not self.llm.is_in_streaming_mode:  # type: ignore
                self.console.print(Markdown(msg), end="")
                self.tokens.task_done()
                continue
            # Otherwise, we need to parse the markdown incrementally
            # FIXME: This is just an hack to make the code blocks work
            #        This should be done properly in the future
            if msg == "\n\n":
                msg = ""
            if msg == "``" or msg == "```":
                if self.code_block:
                    self.console.print(Markdown(self.code_block + "```\n"))
                    self.code_block = ""
                    self.is_code_mode = not self.is_code_mode
                    self.tokens.task_done()
                    continue
                self.is_code_mode = not self.is_code_mode
                self.code_block = msg
            elif self.is_code_mode:
                self.code_block += msg
            else:
                if msg == "`\n" or msg == "`\n\n":
                    msg = "\n"
                self.console.print(msg, end="")
            self.tokens.task_done()

    async def start(self, llm_name: str, **llm_kwargs):
        # Setup the keybindings for the Terminal prompt
        self._setup_keybindings()
        # Load the specified LLM
        try:
            self.llm = self.load_llm(llm_name)
        except exceptions.LLMException as e:
            self.print_error_msg(e.msg)
            return
        # Start the print loop
        asyncio.create_task(self.print_loop())
        self.print_misc_msg(
            f"{self.INTRO_BANNER}\n\nLoaded model: {self.llm.name}", justify="center"  # type: ignore
        )

        while True:
            # Wait if something is getting printed
            await self.tokens.join()
            # Get the user input
            user_input = await self.session.prompt_async("> ")
            user_input = user_input.rstrip()
            # Check if the input is a custom command
            if user_input in self.completer_function_table:
                self.completer_function_table[user_input]()
                continue
            # Otherwise, process the input as a normal message that has
            # to be sent to the LLM
            self.print_client_msg(user_input)
            if not self.llm.is_in_streaming_mode:
                self.print_misc_msg(self.LOADING_MSG)
            await self.llm.process(user_input)


class PromptToolkitREPL(BaseREPL):
    def __init__(self, *_args, **kwargs):
        style = kwargs.pop("style", None)
        self.client_handler = self.get_client_handler("prompt_toolkit", style=style)

    @staticmethod
    def create_client_handler(**kwargs) -> BaseClientHandler:
        return PromptToolkitClientHandler(**kwargs)

    async def run(self, llm_name: str, **llm_kwargs):
        """
        Starts the REPL.

        The user can enter new lines in the REPL by pressing Enter once. The
        REPL will terminate when the user presses Enter twice.

        :param str llm_name: The name of the LLM to use.
        """
        await self.client_handler.start(llm_name, **llm_kwargs)


REPLS["prompt_toolkit"] = PromptToolkitREPL
