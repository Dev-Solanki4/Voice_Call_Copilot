import logging
from typing import Callable, Dict
from ._events import Events
from .utils import PubSubPublishConfig, PubSubSubscribeConfig
from .room_client import RoomClient

logger = logging.getLogger(__name__)


class PubSub:
    """
    PubSub class for handling publish-subscribe messaging within a room.
    """

    def __init__(self, room_client: RoomClient):
        """
        Initialize PubSub with a RoomClient instance.

        Args:
            room_client (RoomClient): The client used to interact with the room.
        """
        self.__room_client = room_client
        self.__listen()
        self.__topic_cb: Dict[str, Callable] = {}
        logger.debug("PubSub initialized")

    def __emit(self, data):
        """
        Emit data to the appropriate callback based on the topic.

        Args:
            data (dict): Data object containing topic and payload.
        """
        for topic, cb in self.__topic_cb.items():
            if topic == data["topic"]:
                cb(data)

    def __listen(self):
        """
        Start listening for PUBSUB_MESSAGE events from the room client.
        """

        @self.__room_client.on(Events.PUBSUB_MESSAGE)
        def on_pubsub_message(data):
            """
            Handle incoming PUBSUB_MESSAGE events.

            Args:
                data (dict): Data object containing topic and payload.
            """
            self.__emit(data)

    async def subscribe(self, pubsub_config: PubSubSubscribeConfig):
        """
        Subscribe to a pubsub topic with a callback function.

        Args:
            pubsub_config (PubSubSubscribeConfig): Configuration object for subscription.

        Returns:
            Any: Response from the room client after subscribing.
        """
        logger.debug(f"Subscribing to topic: {pubsub_config.get('topic')}")
        ans = await self.__room_client.async_handle_pubsub_subscribe(pubsub_config)
        if ans is not None:
            self.__topic_cb[pubsub_config.get("topic")] = pubsub_config.get("cb")
        return ans

    async def publish(self, pubsub_config: PubSubPublishConfig):
        """
        Publish a message to a pubsub topic.

        Args:
            pubsub_config (PubSubPublishConfig): Configuration object for publishing.
        """
        logger.debug(f"Publishing to topic: {pubsub_config.get('topic')}")
        await self.__room_client.async_handle_pubsub_publish(pubsub_config)

    def unsubscribe(self, topic):
        """
        Unsubscribe from a pubsub topic.

        Args:
            topic (str): The topic to unsubscribe from.
        """
        if topic in self.__topic_cb:
            del self.__topic_cb[topic]
