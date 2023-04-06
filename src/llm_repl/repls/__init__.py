from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel  # pylint: disable=no-name-in-module


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
    def info(self):
        """
        Print the information about the LLM currently loaded
        """

    @abstractmethod
    def exit(self):
        """
        Exit the application
        """

    @abstractmethod
    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """

    @abstractmethod
    def print(self, msg: Any, **kwargs):
        """
        Simply prints the message as a normal print statement.

        :param Any msg: The message to be printed.
        """

    @abstractmethod
    def print_client_msg(self, msg: str):
        """
        Prints the client message with the appropriate style

        :param str msg: The message to be printed.
        """

    @abstractmethod
    def print_server_msg(self, msg: str):
        """
        Prints the server message with the appropriate style

        :param str msg: The message to be printed.
        """

    @abstractmethod
    def print_error_msg(self, msg: str):
        """
        Prints the error message with the appropriate style

        :param str msg: The message to be printed.
        """

    @abstractmethod
    def print_misc_msg(self, msg: str, **kwargs):
        """
        Print the miscellaneous message with the appropriate style

        :param str msg: The message to be printed.
        """

    @abstractmethod
    async def run(self, llm_name):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """

REPLS: Dict[str, Type[BaseREPL]] = {}
