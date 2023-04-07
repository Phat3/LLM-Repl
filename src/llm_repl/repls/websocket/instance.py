from typing import Any

from llm_repl.repls import BaseREPL
from llm_repl.llms import LLMS


class WebsocketREPLInstance(BaseREPL):
    """
    Base class with all the methods that a REPL should implement
    """

    def __init__(self, **kwargs):
        """
        Constructor

        :param REPLStyle style: The style of the REPL
        """
        self.llm = None

    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """
        llm = LLMS[llm_name]
        self.llm = llm.load(self)  # type: ignore

    def print(self, msg: Any, **kwargs):
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
        Handle the client message received from the websocket

        :param str msg: The message received from the websocket
        :param Websocket websocket: The websocket instance
        """
        message = kwargs.get("msg")
        websocket = kwargs.get("websocket")
        if message is None or websocket is None:
            return

        message = message.strip()
        print(f"Received message: {message}")
        resp = self.llm.process(message)  # type: ignore
        print(f"Sending response: {resp}")
        await websocket.send(resp)

    async def run(self, llm_name):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """
