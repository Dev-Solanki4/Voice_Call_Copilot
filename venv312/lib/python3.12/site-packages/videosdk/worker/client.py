import asyncio
import json
import logging
from random import random
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

from pydantic import BaseModel, Field
from websockets import WebSocketClientProtocol, connect

T = TypeVar("T")
logger = logging.getLogger(__name__)


def generate_random_number():
    return round(random() * 10000000)


class Request(BaseModel):
    request: bool = True
    id: int = Field(default_factory=generate_random_number)
    method: str
    data: Dict[str, Any]


class Response(BaseModel):
    response: bool = True
    id: int
    ok: bool = True
    data: Dict[str, Any] = {}
    errorCode: Any = None
    errorReason: Any = None


class Notification(BaseModel):
    notification: bool = True
    data: Dict[str, Any] = {}
    method: str


class Client:
    def __init__(
        self,
        handle_request: Optional[
            Callable[[Request], Optional[Awaitable[Union[True, False, Response]]]]
        ] = None,
        handle_notification: Optional[
            Callable[[Request], Optional[Awaitable[Union[True, False, Response]]]]
        ] = None,
    ):
        self._ws: WebSocketClientProtocol = None
        self.__answers: Dict[str, asyncio.Future] = {}
        self.__handle_requests: Callable[
            [Request], Optional[Awaitable[Union[True, False, Response]]]
        ] = handle_request
        self.__handle_notifications: Callable[
            [Request], Optional[Awaitable[Union[True, False, Response]]]
        ] = handle_notification

    async def connect(self, uri: str):
        try:
            self._ws = await connect(uri, subprotocols=["protoo"], ping_interval=2000)
            asyncio.create_task(self.__handle_incoming_messages())
            logger.debug("worker connected")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")

    async def __handle_incoming_messages(self):
        try:
            async for Data in self._ws:
                message: Dict[str, Any] = json.loads(Data)
                logger.debug(f"incoming ws message ${message}")
                if message.get("response"):
                    self.__handle_responses(message)
                elif message.get("request"):
                    if self.__handle_requests:
                        asyncio.create_task(self.handle_incoming_request(message))
                    else:
                        logger.debug("handle_request not implemented")
                elif message.get("notification"):
                    if self.__handle_notifications:
                        await self.__handle_notifications(Notification(**message))
        except Exception as e:
            logger.error(f"server :: error while handling incoming :: {e}")
        finally:
            logger.debug("WebSocket connection closed.")
            logger.debug(f"Close code: {self._ws.close_code}")
            logger.debug(f"Close reason: {self._ws.close_reason}")

    async def handle_incoming_request(self, message):
        response = await self.__handle_requests(Request(**message))
        if response is not None:
            await self.__respond(message, response)

    def __handle_responses(self, message: Dict[str, Any]):
        try:
            logger.debug(f"server :: response message :: {message}")
            if message.get("id") is not None:
                self.__answers[message.get("id")].set_result(Response(**message))
                del self.__answers[message.get("id")]
        except Exception as e:
            logger.error(f"server :: error while handling signal messages :: {e}")

    async def __respond(
        self, request: Request, response: Optional[Union[bool, Response]] = True
    ):
        if self._ws is None:
            raise RuntimeError("Connection not established!")
        if isinstance(response, Response):
            await self._ws.send(response.model_dump_json())
        elif response is True:
            res = Response(id=request.id)
            await self._ws.send(res.model_dump_json())
        else:
            res = Response(
                id=request.id,
                ok=False,
            )
            await self._ws.send(res.model_dump_json())

    async def request(
        self, request: Request, timeout: Optional[float] = 10, **kwargs: Any
    ):
        if self._ws is None:
            raise ("connection not established!")
        fut = self.__answers[request.id] = asyncio.get_event_loop().create_future()
        await self._ws.send(request.model_dump_json())
        res = await asyncio.wait_for(fut, timeout=timeout, **kwargs)
        return res

    async def notify(self, notification: Notification):
        if self._ws is None:
            raise ("connection not established!")
        await self._ws.send(notification.model_dump_json())
