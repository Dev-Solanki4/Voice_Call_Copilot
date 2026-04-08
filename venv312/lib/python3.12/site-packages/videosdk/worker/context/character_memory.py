import os
import requests

class Memory:
    def __init__(self, content_type, user_id, user_name, type, content) -> None:
        self.content_type = content_type
        self.user_id = user_id
        self.user_name = user_name
        self.type = type
        self.content = content

class Memories:
    def __init__(self, json_data) -> None:
        self.messages: list[Memory] = []
        for data in json_data:
            self.messages.append(
                Memory(
                    content_type=data["type"],
                    user_id=data["senderId"],
                    user_name=data["senderName"],
                    type=data["senderType"],
                    content=data["content"],
                )
            )

class CharacterMemory:
    def __init__(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        self.auth_token = os.getenv("AUTH_TOKEN")

    def get_all(self, user_id: str = None) -> list[Memory]:
        url = f"https://api.videosdk.live/ai/v1/character-interactions/{self.interaction_id}/messages"
        if user_id is not None:
            url += f"?senderId={user_id}"
        res = requests.get(
            url,
            headers={"Authorization": self.auth_token},
        )

        if res.status_code != 200:
            raise "Error while fetching memory"
        else:
            return Memories(json_data=res.json()).messages
