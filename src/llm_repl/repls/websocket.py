import asyncio

from websockets.server import serve

from llm_repl import exceptions
from llm_repl.llms import BaseLLM, LLMS
from llm_repl.repls import BaseREPL, BaseClientHandler, REPLS


class WebsocketClientHandler(BaseClientHandler):
    """
    Client that handles a single client SSE connection
    """

    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket
        self.llm: BaseLLM | None = None

    @property
    def start_token(self) -> str:
        """Return the marker that act as start token"""
        return ""

    @property
    def end_token(self) -> str:
        """Return the marker that act as end token"""
        return "EOF"

    def _load_llm(self, llm_name: str, **_llm_kwargs) -> BaseLLM:
        """
        Load the selected LLM
        """
        if llm_name not in LLMS:
            raise exceptions.LLMNotFound(llm_name)
        llm_class = LLMS[llm_name]
        return llm_class.load(self)

    async def start(self, llm_name, **llm_kwargs):
        """
        Load the selected LLM

        :param str llm_name: The name of the LLM to load
        """
        # TODO: Handle errors
        self.llm = self._load_llm(llm_name, **llm_kwargs)
        asyncio.create_task(self.print_loop())

    async def print_loop(self):
        """
        Process the tokens in the queue and send them to the client as websocket
        packets
        """
        while True:
            token = await self.tokens.get()
            await self.websocket.send(token)
            self.tokens.task_done()

    async def process(self, message: str):
        """
        Process the message received from the client

        :param str message: The message received from the client
        """
        await self.llm.process(message)  # type: ignore


class WebsocketREPL(BaseREPL):
    def __init__(self, port: int = 8765, **kwargs):
        """
        Constructor
        """
        self.llm_name: None | str = None
        self.port = port

    @staticmethod
    def create_client_handler(**kwargs) -> BaseClientHandler:
        return WebsocketClientHandler(**kwargs)

    async def _handle_msg(self, websocket):
        """
        Handle the client messages received from the websocket.

        This method sets up a separate instance of the LLM for each client.

        :param Websocket websocket: The websocket instance
        """
        # FIXME: Add proper error handling
        if self.llm_name is None:
            return
        # TODO: Handle authentication and client identification
        # token = await websocket.recv() if websocket.open else None
        # if token is None:
        #     return
        client_handler = WebsocketREPL.create_client_handler(websocket=websocket)
        await client_handler.start(self.llm_name)

        async for msg in websocket:
            await client_handler.process(msg)  # type: ignore

    async def run(self, llm_name: str, **_llm_kwargs):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """
        print(f"Starting Websocket REPL with LLM {llm_name} on port {self.port}")
        self.llm_name = llm_name
        # TODO: Check if the chosen LLM can be used
        # TODO: Make the port configurable
        async with serve(self._handle_msg, "localhost", self.port):
            await asyncio.Future()


REPLS["websocket"] = WebsocketREPL
