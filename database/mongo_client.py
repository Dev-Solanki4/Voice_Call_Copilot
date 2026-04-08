import os
import motor.motor_asyncio
from datetime import datetime, timezone
import logging
import certifi
import urllib.parse

class MongoDBClient:
    def __init__(self, uri: str = None, db_name: str = None):
        # Explicit URI configuration as requested by user
        username = urllib.parse.quote_plus("Rudra")
        password = urllib.parse.quote_plus("Rudra@1510")
        self.uri = f"mongodb+srv://{username}:{password}@cluster0.bqktq28.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        self.db_name = "restaurant_bot"
        self.collection_name = "conversations"
        
        # We add certifi for TLS validation
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            self.uri, 
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsCAFile=certifi.where()
        )
        self.db = self.client[self.db_name]
        self.conversations = self.db[self.collection_name]

    async def start_conversation(self, session_id: str, restaurant_id: str):
        """Creates a new conversation document when the call starts. ONLY stores communication data."""
        try:
            doc = {
                "session_id": session_id,
                "restaurant_id": restaurant_id,
                "start_time": datetime.now(timezone.utc),
                "end_time": None,
                "status": "active",
                "messages": []
            }
            await self.conversations.insert_one(doc)
            logging.info(f"MongoDB: Started conversation {session_id}")
        except Exception as e:
            logging.error(f"MongoDB Error starting conversation {session_id}: {e}")

    async def log_turn(self, session_id: str, user_msg: dict, agent_msg: dict, current_order: dict = None, error: dict = None):
        """Appends User and Agent messages to the conversation array."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Prepare user message
            u_msg = {
                "turn_id": user_msg.get("turn_id", 0),
                "speaker": "user",
                "text": user_msg.get("text", ""),
                "normalized_text": user_msg.get("normalized_text", ""),
                "intent": user_msg.get("intent", "UNKNOWN"),
                "confidence": user_msg.get("confidence", 0),
                "timestamp": timestamp
            }
            
            # Prepare agent message
            a_msg = {
                "turn_id": user_msg.get("turn_id", 0),
                "speaker": "agent",
                "text": agent_msg.get("text", ""),
                "timestamp": timestamp
            }
            
            updates = {
                "$push": {"messages": {"$each": [u_msg, a_msg]}}
            }
            
            if error:
                error["timestamp"] = timestamp
                updates["$set"] = {"error": error}
                
            await self.conversations.update_one(
                {"session_id": session_id},
                updates
            )
            logging.info(f"MongoDB: Logged turn {u_msg.get('turn_id', 0)} for {session_id}")
        except Exception as e:
            logging.error(f"MongoDB Error logging turn for {session_id}: {e}")

    async def end_conversation(self, session_id: str, status: str = "completed"):
        """Marks the conversation as completed or cancelled and sets the end time."""
        try:
            end_time = datetime.now(timezone.utc)
            updates = {
                "end_time": end_time,
                "status": status,
            }
            
            await self.conversations.update_one(
                {"session_id": session_id},
                {"$set": updates}
            )
            logging.info(f"MongoDB: Ended conversation {session_id} with status '{status}'")
        except Exception as e:
            logging.error(f"MongoDB Error ending conversation {session_id}: {e}")
