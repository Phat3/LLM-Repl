import asyncio

from typing import Any
from websockets.server import serve

from llm_repl.repls import BaseREPL, REPLS

from llm_repl.repls.websocket.instance import WebsocketREPLInstance


class WebsocketREPL(BaseREPL):
    """
    Base class with all the methods that a REPL should implement
    """

    def __init__(self, **kwargs):
        """
        Constructor

        :param REPLStyle style: The style of the REPL
        """
        self.llm_name = None
        self.port = kwargs.get("port", 8765)

    def load_llm(self, llm_name: str):
        """
        Load the LLM specified by the name and its custom commands if any

        :param str llm_name: The name of the LLM to load
        """

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
        Handle the client messages received from the websocket.

        This method sets up a separate instance of the LLM for each client.

        :param Websocket websocket: The websocket instance
        """
        # FIXME: Add proper error handling
        if self.llm_name is None:
            return
        if len(args) == 0:
            return

        websocket = args[0]
        # TODO: Handle authentication and client identification
        # token = await websocket.recv() if websocket.open else None
        # if token is None:
        #     return

        instance = WebsocketREPLInstance(websocket=websocket)
        instance.load_llm(self.llm_name)

        async for msg in websocket:
            await instance.handle_msg(msg=msg)

    async def run(self, llm_name):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """
        self.llm_name = llm_name
        # TODO: Check if the chosen LLM can be used
        # TODO: Make the port configurable
        print("Starting websocket REPL on port", self.port, "...")
        async with serve(self.handle_msg, "localhost", self.port):
            await asyncio.Future()


REPLS["websocket"] = WebsocketREPL
