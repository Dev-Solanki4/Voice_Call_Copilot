from .context import CharacterContext

class CharacterState:
    def __init__(self) -> None:
        pass
    def handle(self, context: CharacterContext):
        raise NotImplementedError
