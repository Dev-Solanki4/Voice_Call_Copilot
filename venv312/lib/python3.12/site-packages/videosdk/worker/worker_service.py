import json
import logging
from concurrent.futures import ThreadPoolExecutor
from asyncio import AbstractEventLoop, Event
import traceback
from typing import Any, List

from .client import Client, Request, Response

from .character_state import CharacterState
from .context import CharacterContext, CharacterResponseStatus
from .schemas import StateRequest, StateResponse

logger = logging.getLogger(__name__)
SIGNALLING_BASE_URL = "worker.videosdk.live"


class CharacterWorkerService:
    def __init__(
        self,
        init_state: CharacterState,
        states: List[CharacterState],
        loop: AbstractEventLoop,
    ):
        self.init_state = init_state
        self.states = states
        self.loop = loop
        self.state_map = {type(state).__name__: idx for idx, state in enumerate(states)}
        self._c = Client(handle_request=self.handle_request)

    def get_state_by_name(self, name: str) -> CharacterState:
        return self.states[self.state_map[name]]

    async def handle_request(self, request: Request):
        response = await self.handle_state(StateRequest(**request.data))
        return Response(id=request.id, data=response)

    async def start(self, characters: list[str], token):
        characters_str = ",".join(characters)
        uri = f"wss://{SIGNALLING_BASE_URL}?characters={characters_str}&token={token}"
        await self._c.connect(uri=uri)

    async def handle_state(self, request: StateRequest) -> dict[str, Any]:
        try:
            logger.debug(
                f"Received request: {request.interaction_id}, {request.current_state_name}, {request.current_state_args}"
            )

            next_event = Event()

            def next():
                next_event.set()

            if request.current_state_name:
                current_state = self.get_state_by_name(request.current_state_name)
                context = CharacterContext(
                    self.loop,
                    interaction_id=request.interaction_id,
                    state=request.current_state_name,
                    state_description=request.current_state_description,
                    response_message="",
                    next=next,
                    args=request.current_state_args or {},
                    client=self._c,
                )

                with ThreadPoolExecutor() as executor:
                    await self.loop.run_in_executor(
                        executor, current_state.handle, context
                    )
                await next_event.wait()

                next_state = type(context.state).__name__
                next_state_description = getattr(
                    context.state, "_state_description", ""
                )
                next_state_params = getattr(context.state, "_state_params", {})
                response_status = context.response_status.value
                response_message = context.response_message
            else:
                next_state = type(self.init_state).__name__
                next_state_description = getattr(
                    self.init_state, "_state_description", ""
                )
                next_state_params = getattr(self.init_state, "_state_params", {})
                response_status = CharacterResponseStatus.SUCCESS.value
                response_message = ""

            response = StateResponse(
                interaction_id=request.interaction_id,
                next_state=next_state,
                next_state_description=next_state_description,
                next_state_params=next_state_params,
                response_status=response_status,
                response_message=response_message,
            ).model_dump()

            logger.debug(f"Send response: {response}")
            return response

        except Exception as e:
            raise Exception(f"Error in handle_state: {e}")
