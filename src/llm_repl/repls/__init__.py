import asyncio

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Type


class BaseClientHandler(ABC):
    """BaseClass to handle client messages"""

    def __init__(self):
        # Queue to hold the tokens generated by the LLM.
        # These tokens are then consumed by the client
        self.tokens: asyncio.Queue[str] = asyncio.Queue()

    @property
    def start_token(self) -> str:
        """Return the marker that act as start token"""
        return ""

    @property
    def end_token(self) -> str:
        """Return the marker that act as end token"""
        return ""

    async def add_token(self, token: str):
        """
        Add a token to the queue to be consumed by the client

        :param str token: The token to be added to the queue
        """
        await self.tokens.put(token)

    @abstractmethod
    async def start(self, llm_name, **llm_kwargs):
        """
        Start the client handler to handle messages coming and going
        to the client and the LLM

        :param str llm_name: The name of the LLM to load
        """

    @abstractmethod
    async def print_loop(self):
        """
        Async task that prints, according to the communication protocol of the
        REPL, all the tokens in the queue
        """


class BaseREPL(ABC):
    """
    Base class with all the methods that a REPL should implement
    """

    MAX_CLIENTS = 100

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Constructor"""

    @staticmethod
    @abstractmethod
    def create_client_handler(**kwargs) -> BaseClientHandler:
        """
        Create a new client handler instance
        """

    # TODO: Add Redis or disk for secondary storage when the clients are
    #       evicted from the cache
    @classmethod
    @lru_cache(maxsize=MAX_CLIENTS)
    def get_client_handler(cls, _client_id: str, **kwargs) -> BaseClientHandler:
        return cls.create_client_handler(**kwargs)

    # @abstractmethod
    # def load_llm(self, llm_name: str, **llm_kwargs):
    #     """
    #     Load the LLM specified by the name and its custom commands if any

    #     :param str llm_name: The name of the LLM to load
    #     :param dict llm_kwargs: The kwargs to pass to the LLM
    #     """

    # @abstractmethod
    # async def print(self, msg: Any, client_id: str, **kwargs):
    #     """
    #     Simply prints the message as a normal print statement.

    #     :param Any msg: The message to be printed.
    #     """

    # @abstractmethod
    # async def handle_msg(self, *args, **kwargs):
    #     """
    #     Handle client message
    #     """

    @abstractmethod
    async def run(self, llm_name, **llm_kwargs):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        :param dict llm_kwargs: The kwargs to pass to the LLM
        """


REPLS: Dict[str, Type[BaseREPL]] = {}
