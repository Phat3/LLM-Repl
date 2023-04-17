import uvicorn
import asyncio
import json
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel, BaseSettings  # pylint: disable=no-name-in-module

from typing import List, Dict

from llm_repl import exceptions
from llm_repl.repls import BaseREPL, BaseClientHandler, REPLS
from llm_repl.llms import BaseLLM, LLMS

from sse_starlette.sse import EventSourceResponse

# FIXME: Make this configurable
SSE_PING_INTERVAL = 600  # minutes


class Settings(BaseSettings):
    llm_name: str = "chatgpt"


settings = Settings()
app = FastAPI()


class Params(BaseModel):
    model: str
    messages: List[Dict[str, str]]


@app.post("/v1/chat/completions")
async def message_stream(request: Request, params: Params):
    # FIXME: Handle memory
    message = params.messages[-1]["content"]
    # TODO: Manage reproducible client id
    client_id = uuid.uuid4().hex
    client_handler = HttpREPL.get_client_handler(client_id, request=request)
    # Setup the LLM
    await client_handler.start(settings.llm_name)  # TODO: Handle error
    # In the meantime let the LLM process the message
    asyncio.create_task(client_handler.process(message=message))  # type: ignore
    # Setup the SSE response
    event_source = EventSourceResponse(client_handler.print_loop())
    event_source.ping_interval = SSE_PING_INTERVAL
    return event_source


class HttpClientHandler(BaseClientHandler):
    """
    Client that handles a single client SSE connection
    """

    RETRY_TIMEOUT = 15000  # milisecond

    def __init__(self, request: Request):
        super().__init__()
        self.request = request
        self.llm: BaseLLM | None = None

    @property
    def start_token(self) -> str:
        """Return the marker that act as start token"""
        return ""

    @property
    def end_token(self) -> str:
        """Return the marker that act as end token"""
        return "[DONE]"

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
        self.llm = self._load_llm(llm_name, **llm_kwargs)

    async def print_loop(self):
        """
        Process the tokens in the queue and send them to the client as
        Server Sent Events (SSE)
        """
        while True:
            # If client closes connection, stop sending events
            if await self.request.is_disconnected():
                break
            # Checks for new messages and return them to client if any
            token = await self.tokens.get()
            if token:
                response = {
                    "event": "new_message",
                    "id": "message_id",
                    "retry": self.RETRY_TIMEOUT,
                }
                if token == self.end_token:
                    response["data"] = "[DONE]"
                else:
                    response["data"] = json.dumps(
                        {"choices": [{"delta": {"content": token}}]}
                    )
                self.tokens.task_done()
                yield response

    async def process(self, message: str):
        """
        Processes the message

        :param str message: The message to process
        """
        await self.llm.process(message)  # type: ignore


class HttpREPL(BaseREPL):
    def __init__(self, port: int = 8000, reload_server: bool = False, **kwargs):
        """
        Constructor
        """
        self.port = port
        self.reload = reload_server

    @staticmethod
    def create_client_handler(**kwargs) -> BaseClientHandler:
        return HttpClientHandler(**kwargs)

    async def run(self, llm_name: str, **_llm_kwargs):
        """
        Starts the REPL

        :param str llm_name: The name of the LLM to load
        """
        print(f"Starting HTTP REPL with LLM {llm_name} on port {self.port}")
        settings.llm_name = llm_name
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, reload=True)
        server = uvicorn.Server(config=config)
        await server.serve()


REPLS["http"] = HttpREPL
