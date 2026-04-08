import asyncio
import logging
from typing import Any, Optional

from .character_intent import MatchIntentResult, SetGuidelinesResult
from .character_memory import CharacterMemory
from .character_vision import CharacterVision
from ..schemas import (
    MatchIntentPayload,
    NotificationTypes,
    SetGuidelinesPayload,
    EndInteractionPayload,
    VisionRequestPayload,
    BroadcastPayload,
    CharacterResponseStatus,
)

from ..client import Client, Notification, Request, Response

logger = logging.getLogger(__name__)


class CharacterContext:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        interaction_id: str,
        state: Any,
        args: dict,
        response_message: str,
        state_description: Optional[str],
        next,
        client,
    ):
        self.loop = loop
        self.interaction_id = interaction_id
        self.state = state
        self.state_description = state_description
        self.args = args
        self.next = next
        self.response_status: CharacterResponseStatus = CharacterResponseStatus.SUCCESS
        self.response_message = response_message
        self.memory = CharacterMemory(interaction_id=interaction_id)
        self._c: Client = client

    def set_state(self, state: Any):
        self.state = state
        self.state_description = getattr(type(state), "_state_description", "")

    def set_response(
        self,
        response_status: CharacterResponseStatus,
        response_message: Optional[str] = None,
    ):
        self.response_status = response_status
        self.response_message = response_message or ""

    def get_args(self) -> dict:
        return self.args

    def get_memory(self) -> CharacterMemory:
        return self.memory

    def end_interaction(self, force: Optional[bool] = False) -> None:
        logger.debug("WS :: end_interaction")
        payload = EndInteractionPayload(
            interaction_id=self.interaction_id, args={"forceFully": force}
        )
        asyncio.run_coroutine_threadsafe(
            self._c.notify(
                Notification(
                    data=payload.model_dump(), method=NotificationTypes.END_INTERACTION
                )
            ),
            loop=self.loop,
        ).result()

    def match_intent(self, query: str) -> MatchIntentResult:
        payload = MatchIntentPayload(
            interaction_id=self.interaction_id, args={"query": query}
        )

        response: Response = asyncio.run_coroutine_threadsafe(
            self._c.request(
                Request(
                    data=payload.model_dump(), method=NotificationTypes.MATCH_INTENT
                )
            ),
            loop=self.loop,
        ).result(timeout=15)

        data = response.data
        return MatchIntentResult(
            success=data.get("success", False), score=data.get("score", 0.0)
        )

    def set_guidelines(self, guidelines: list[str] = []) -> SetGuidelinesResult:
        logger.debug("WS :: set_guidelines")
        payload = SetGuidelinesPayload(
            interaction_id=self.interaction_id, args={"guidelines": guidelines}
        )

        response: Response = asyncio.run_coroutine_threadsafe(
            self._c.request(
                Request(
                    data=payload.model_dump(), method=NotificationTypes.SET_GUIDELINES
                )
            ),
            loop=self.loop,
        ).result(timeout=15)

        data = response.data
        return SetGuidelinesResult(
            success=data.get("success", False), message=data.get("message", "")
        )

    async def get_vision(self) -> CharacterVision:
        logger.debug("WS :: get_vision")
        payload = VisionRequestPayload(
            interaction_id=self.interaction_id,
            args={
                "interaction_id": self.interaction_id,
                "current_state_name": type(self.state).__name__,
            },
        )

        response: Response = asyncio.run_coroutine_threadsafe(
            self._c.request(
                Request(data=payload.dict(), method=NotificationTypes.GET_VISION)
            ),
            loop=self.loop,
        ).result(timeout=15)

        data = response.data
        return CharacterVision(
            success=data.get("success", False),
            message=data.get("message", ""),
            image=data.get("image", ""),
        )

    def broadcast(self, topic: str, data: dict[str, Any]):
        logger.debug("WS :: broadcast")
        payload = BroadcastPayload(
            interaction_id=self.interaction_id,
            args={
                "current_state_name": type(self.state).__name__,
                "topic": topic,
                "data": data,
            },
        )
        asyncio.run_coroutine_threadsafe(
            self._c.notify(
                Notification(data=payload.dict(), method=NotificationTypes.BROADCAST)
            ),
            loop=self.loop,
        ).result(timeout=15)
