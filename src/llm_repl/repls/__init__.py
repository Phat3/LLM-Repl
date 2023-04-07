from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel  # pylint: disable=no-name-in-module

from llm_repl.repls import websocket


class REPLStyle(BaseModel):
    """
    The style of the REPL
    """

    client_msg_color: str
    server_msg_color: str
    error_msg_color: str
    misc_msg_color: str


class BaseREPL(ABC):
    """
    Base class with all the methods that a REPL should implement
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Constructor
        """

    @abstractmethod
    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """

    @abstractmethod
    async def print(self, msg: Any, **kwargs):
        """
        Simply prints the message as a normal print statement.

        :param Any msg: The message to be printed.
        """

    def print_error_msg(self, msg: str):
        """
        Prints the error message in the console.

        :param str msg: The message to be printed.
        """

    async def handle_msg(self, *args, **kwargs):
        """
        Handle client message
        """

    @abstractmethod
    async def run(self, llm_name):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """


REPLS: Dict[str, Type[BaseREPL]] = {}
