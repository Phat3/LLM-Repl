import websockets
import asyncio

from typing import Any
from websockets.server import serve

from llm_repl.repls import BaseREPL, REPLStyle


class WebsockerREPL(BaseREPL):
    """
    Base class with all the methods that a REPL should implement
    """

    @property
    def style(self) -> REPLStyle:
        """
        Return the style of the REPL
        """
        raise NotImplementedError

    def info(self) -> str:
        """
        Print the information about the LLM currently loaded
        """
        return "This is a websocket REPL"

    def exit(self):
        """
        Exit the application
        """
        pass

    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """
        pass

    def print(self, msg: Any, **kwargs):
        """
        Simply prints the message as a normal print statement.

        :param Any msg: The message to be printed.
        """

    def print_client_msg(self, msg: str):
        """
        Prints the client message with the appropriate style

        :param str msg: The message to be printed.
        """

    def print_server_msg(self, msg: str):
        """
        Prints the server message with the appropriate style

        :param str msg: The message to be printed.
        """

    def print_error_msg(self, msg: str):
        """
        Prints the error message with the appropriate style

        :param str msg: The message to be printed.
        """

    def print_misc_msg(self, msg: str, **kwargs):
        """
        Print the miscellaneous message with the appropriate style

        :param str msg: The message to be printed.
        """
    
    async def _handle_msg(self, websocket):
        async for message in websocket:
            print(f"Received message: {message}")
            await websocket.send(message)

    async def run(self, llm_name):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """
        print("Starting websocket REPL")
        async with serve(self._handle_msg, "localhost", 8765):
            await asyncio.Future()