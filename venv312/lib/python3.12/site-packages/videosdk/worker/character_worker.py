import asyncio
import logging
import threading
from typing import Optional
from .worker_service import CharacterWorkerService
from .character_state import CharacterState

DEFAULT_PORT = "50051"
DEFAULT_WORKERS = 10
logger = logging.getLogger(__name__)


class CharacterWorker:
    """
    CharacterWorker: state machine designed to handle multiple character states
    and manage their transitions during a conversation.

    Attributes:
        states (list[CharacterState]): A list of available character states.
        init_state (CharacterState): The initial state to begin with.
    """

    def __init__(self) -> None:
        self.states: list[CharacterState] = []
        self.init_state: CharacterState = None

    def run(self, token: str, characters: list[str]):
        """
        Start the worker thread.
        """
        loop = asyncio.get_event_loop()
        if not token:
            raise ("token is missing")
        if not characters:
            raise ("characters are missing")
        self._run(loop, characters, token)
        loop.run_forever()

    def _run(
        self,
        loop: asyncio.AbstractEventLoop,
        characters: list[str],
        token: str,
    ):
        service = CharacterWorkerService(self.init_state, self.states, loop)
        loop.run_until_complete((service.start(characters, token)))
