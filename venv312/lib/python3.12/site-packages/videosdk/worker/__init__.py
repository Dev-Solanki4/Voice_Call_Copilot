from .context import CharacterContext, CharacterMemory, CharacterVision, CharacterResponseStatus
from .character_state import CharacterState
from .character_worker import CharacterWorker

def state(description, **params):
    def decorator(cls):
        cls._state_description = description
        cls._state_params = params
        return cls

    return decorator
