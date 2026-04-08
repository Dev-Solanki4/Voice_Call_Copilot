import redis
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class RedisSessionManager:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis.ping()
            logging.info("--- Connected to Redis ---")
        except Exception as e:
            logging.error(f"--- Failed to connect to Redis: {e} ---")
            self.redis = None

    def _serialize_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content}
        return {"type": "unknown", "content": str(message.content)}

    def _deserialize_message(self, msg_dict):
        if msg_dict["type"] == "human":
            return HumanMessage(content=msg_dict["content"])
        elif msg_dict["type"] == "ai":
            return AIMessage(content=msg_dict["content"])
        return HumanMessage(content=msg_dict["content"]) # Default fallback

    def get_session(self, user_id):
        import time
        start = time.time()
        if not self.redis:
            return None
        data = self.redis.get(f"session:{user_id}")
        if data:
            state = json.loads(data)
            # Deserialize messages
            if "messages" in state:
                state["messages"] = [self._deserialize_message(m) for m in state["messages"]]
            duration = time.time() - start
            print(f"\033[94m[⌚ DB] Redis get_session took {duration:.2f}s\033[0m", flush=True)
            return state
        return None

    def set_session(self, user_id, state):
        if not self.redis:
            return
        
        def _default_serializer(obj):
            import uuid
            from decimal import Decimal
            if isinstance(obj, (uuid.UUID, Decimal)):
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # Create a copy to avoid modifying the original state
        serializable_state = state.copy()
        
        # Serialize messages
        if "messages" in serializable_state:
            # Increase limit to store more context for "whole communication"
            messages = serializable_state["messages"][-50:] 
            serializable_state["messages"] = [self._serialize_message(m) for m in messages]
        
        try:
            import time
            start = time.time()
            self.redis.set(f"session:{user_id}", json.dumps(serializable_state, default=_default_serializer))
            duration = time.time() - start
            print(f"\033[94m[⌚ DB] Redis set_session took {duration:.2f}s\033[0m", flush=True)
        except Exception as e:
            logging.error(f"Redis Serialization Error: {e}")
