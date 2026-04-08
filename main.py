import asyncio
import traceback
import os
import logging
import uuid
from typing import AsyncIterator
from dotenv import load_dotenv

from videosdk.agents import (
    Agent, 
    AgentSession, 
    CascadingPipeline, 
    JobContext, 
    RoomOptions, 
    WorkerJob, 
    Options, 
    ConversationFlow, 
    InterruptConfig, 
    EOUConfig, 
    LLM, 
    LLMResponse, 
    ChatRole, 
    ChatContext
)
from videosdk.plugins.sarvamai import SarvamAISTT, SarvamAITTS

# Multi-Agent Imports
from langchain_core.messages import HumanMessage
from graph.workflow import create_restaurant_graph
from database.mongo_client import MongoDBClient
from database.postgres_service import PetPoojaDB
from database.redis_client import RedisSessionManager
from utils.ssml_processor import build_ssml

logging.basicConfig(level=logging.INFO)
# Suppress noisy Sarvam TTS plugin logger (has internal logging format bug)
logging.getLogger("videosdk.plugins.sarvamai.tts").setLevel(logging.WARNING)
load_dotenv()

# Global placeholders
RESTAURANT_GRAPH = None
MONGO_CLIENT = None
MENU_KEYTERMS = None  # Loaded from DB once
MENU_CACHE = None     # In-memory menu cache

def load_menu_cache():
    """Load entire menu database into memory at startup."""
    global MENU_CACHE
    from database.menu_cache import MenuCache
    MENU_CACHE = MenuCache.get_instance()
    MENU_CACHE.load()
    return MENU_CACHE

def load_menu_keyterms():
    """Load all menu item names from the in-memory cache for STT keyterms."""
    global MENU_KEYTERMS, MENU_CACHE
    if MENU_KEYTERMS is not None:
        return MENU_KEYTERMS
    
    try:
        if MENU_CACHE is None:
            load_menu_cache()
        
        names = MENU_CACHE.get_item_names()
        # Add common ordering phrases
        names.extend([
            "confirm order", "place order", "that's all",
            "menu", "remove", "cancel", "add", "biryani",
            "manchurian", "paratha", "tikka", "masala",
        ])
        MENU_KEYTERMS = names
        logging.info(f"--- Loaded {len(names)} menu keyterms from cache ---")
        return names
    except Exception as e:
        logging.error(f"Failed to load menu keyterms: {e}")
        return [
            "confirm order", "place order", "menu", "remove", "cancel", "add"
        ]

# Global state store for cross-turn persistence within a session
# SESSION_STATES = {}  # REPLACED BY REDIS
REDIS_MANAGER = None

# Define the custom Multi-Agent Brain
class RestaurantMultiAgentLLM(LLM):
    def __init__(self, session_id: str):
        super().__init__()
        global RESTAURANT_GRAPH, MONGO_CLIENT, REDIS_MANAGER, MENU_CACHE
        
        logging.info(f"--- Initializing MultiAgent Brain for session: {session_id} ---")
        
        if RESTAURANT_GRAPH is None:
            logging.info("--- Creating Restaurant Graph (Initial) ---")
            RESTAURANT_GRAPH = create_restaurant_graph()
            
        if MONGO_CLIENT is None:
            logging.info("--- Creating Mongo Client (Initial) ---")
            MONGO_CLIENT = MongoDBClient()

        if REDIS_MANAGER is None:
            logging.info("--- Creating Redis Manager (Initial) ---")
            REDIS_MANAGER = RedisSessionManager()
            
        if MENU_CACHE is None:
            logging.info("--- Loading Menu Cache (Initial) ---")
            load_menu_cache()

        self.graph = RESTAURANT_GRAPH
        self.session_id = session_id
        self.mongo = MONGO_CLIENT
        self.redis = REDIS_MANAGER
        
        # We will load/set state in the chat method to ensure it's always fresh from Redis
        logging.info(f"--- MultiAgent Brain initialized for {session_id} ---")

    async def chat(
        self, 
        messages: ChatContext, 
        tools: list = None, 
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        logging.info(f"!!! LLM CHAT METHOD TRIGGERED !!! Session: {self.session_id}")
        
        # Load or initialize state from Redis
        state = self.redis.get_session(self.session_id)
        if not state:
            state = {
                "messages": [],
                "restaurant_id": "73313cb0-dcd4-4f03-94e0-5ec7aaf711ad",
                "last_extracted_items": [],
                "current_order": {"items": [], "payment_status": "pending", "order_confirmed": False, "total": 0.0, "tax": 0.0},
                "intent": "Greeting",
                "retrieved_menu": [],
                "matched_menu_items": [],
                "inventory_status": [],
                "recommendation_shown": False,
                "ready_for_confirmation": False,
                "transaction_completed": False,
                "dining_type": None,
                "checkout_stage": "",
                "payment_mode": None,
                "customer_name": None,
                "menu_action": "",
                "turn_id": 0,
                "normalized_text": "",
                "confidence": 0,
                "final_response": ""
            }
        self.state = state

        # Rule 3: Restaurant ID Validation
        rid = self.state.get("restaurant_id")
        if not rid or rid == "null" or rid == "" or rid == "None":
            logging.warning(f"!!! STO-PED: Invalid restaurant_id: {rid} !!!")
            # Return empty response to stay silent as per rule
            return

        if not messages.items:
            logging.warning("!!! LLM ChatContext is empty !!!")
            return

        # Last message from user — ensure it's always a plain string
        last_user_msg = messages.items[-1].content
        if isinstance(last_user_msg, list):
            last_user_msg = " ".join([str(t) for t in last_user_msg])
        
        # ── STT HALLUCINATION FILTER ──
        # Filter out garbage/echo from TTS being picked up by STT
        cleaned = last_user_msg.strip()
        if len(cleaned) <= 2:
            logging.warning(f"--- FILTERED (too short): '{cleaned}' ---")
            return
        
        # Must have at least 2 real words
        words = cleaned.split()
        if len(words) < 2:
            logging.warning(f"--- FILTERED (single word noise): '{cleaned}' ---")
            return
        
        cleaned_lower = cleaned.lower()
        
        # Detect single word repeated (e.g. "bye bye bye" or "ha ha ha")
        unique_words = set(w.lower().strip('.,!?') for w in words)
        if len(unique_words) == 1 and len(words) >= 2:
            logging.warning(f"--- FILTERED (repeated word): '{cleaned}' ---")
            return
        
        # Common STT hallucination patterns (AI hearing itself / silence-fill)
        hallucination_patterns = [
            "thank you for watching", "thanks for watching",
            "subscribe", "like and share",
            "please subscribe", "bye bye",
            "the end", "silence",
            "music", "applause", "laughter",
            "you", "hmm", 
            "thank you", "thanks",
            "okay thank", "okay bye",
            "foreign", "subtitle",
            "uh", "um",
        ]
        # Exact match on very short phrases (these are always hallucinations)
        if cleaned_lower in hallucination_patterns:
            logging.warning(f"--- FILTERED (hallucination exact): '{cleaned}' ---")
            return
        # Substring match on longer phrases
        if any(pattern in cleaned_lower for pattern in ["thank you for watching", "thanks for watching", "please subscribe", "like and share"]):
            logging.warning(f"--- FILTERED (hallucination substring): '{cleaned}' ---")
            return
        
        # Filter if STT just echoed back what AI just said
        if self.state.get("messages") and len(self.state["messages"]) > 0:
            last_ai = self.state["messages"][-1]
            if hasattr(last_ai, 'type') and last_ai.type == 'ai':
                ai_text = last_ai.content.lower().strip()
                # If user text is a substantial substring of the last AI response, it's echo
                if len(cleaned_lower) > 5 and cleaned_lower in ai_text:
                    logging.warning(f"--- FILTERED (echo of AI): '{cleaned}' ---")
                    return
                # If >60% of words match the AI response, it's likely echo
                if len(words) >= 3:
                    ai_words = set(ai_text.split())
                    overlap = len(set(w.lower() for w in words) & ai_words)
                    if overlap / len(words) > 0.6:
                        logging.warning(f"--- FILTERED (echo overlap {overlap}/{len(words)}): '{cleaned}' ---")
                        return
        
        logging.info(f"--- USER SAYS: {last_user_msg} ---")
        
        # Prepare state for LangGraph
        self.state["messages"].append(HumanMessage(content=last_user_msg))
        self.state["recommendation_shown"] = False  # Reset for each turn
        
        # Invoke LangGraph
        try:
            logging.info("--- Starting 17-Step Workflow ---")
            import time
            start_total = time.time()
            
            self.state["turn_id"] += 1
            
            # Debug: Log state before graph
            items_before = self.state.get("current_order", {}).get("items", [])
            logging.info(f"--- Turn {self.state['turn_id']} START: Items={len(items_before)}, Ready={self.state.get('ready_for_confirmation')} ---")
            
            async for event in self.graph.astream(self.state, stream_mode="updates"):
                for node_name, updates in event.items():
                    logging.info(f"--- Workflow Node '{node_name}' finished ---")
                    if not updates:
                        continue
                    # Merge updates back into self.state
                    for key, value in updates.items():
                        if key == "messages":
                            self.state["messages"].extend(value)
                        else:
                            self.state[key] = value

            end_total = time.time()
            items_after = self.state.get("current_order", {}).get("items", [])
            logging.info(f"--- Turn {self.state['turn_id']} END: Items={len(items_after)}, Ready={self.state.get('ready_for_confirmation')}, Duration={end_total - start_total:.2f}s ---")
            logging.info(f"--- Workflow completed in {end_total - start_total:.2f}s ---")
            
            # Get the last AI message
            ai_response_text = self.state["messages"][-1].content
            logging.info(f"--- AI SAYS: {ai_response_text} ---")
            
            # Apply SSML post-processing for human-like voice
            ssml_text, detected_emotion = build_ssml(ai_response_text)
            logging.info(f"--- SSML Emotion: {detected_emotion} ---")
            
            # Formulate MongoDB structured log metrics
            u_msg = {
                "turn_id": self.state["turn_id"],
                "text": last_user_msg,
                "normalized_text": self.state.get("normalized_text", last_user_msg.lower()),
                "intent": self.state.get("intent", "UNKNOWN"),
                "confidence": self.state.get("confidence", 0)
            }
            a_msg = {
                "turn_id": self.state["turn_id"],
                "text": ai_response_text
            }
            order_data = self.state.get("current_order", {})
            
            # Write to MongoDB explicitly
            await self.mongo.log_turn(self.session_id, u_msg, a_msg, order_data)

            # Update Redis session
            self.redis.set_session(self.session_id, self.state)

            # --- STREAMING IMPLEMENTATION ---
            # Split by common sentence terminators (., !, ?, |) to stream chunks to TTS
            import re
            sentences = re.split(r'(?<=[.!?|])\s+', ssml_text)
            
            print(f"\033[96m[📡 STREAM] Starting LLM->TTS stream for {len(sentences)} chunks...\033[0m", flush=True)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                print(f"\033[96m[📡 STREAM] Yielding chunk {i+1}: '{sentence}'\033[0m", flush=True)
                
                # Yield each sentence as a separate LLMResponse to the TTS engine
                # The trailing space helps TTS separate words properly between chunks
                yield LLMResponse(content=sentence + " ", role=ChatRole.ASSISTANT)
            
            print(f"\033[96m[📡 STREAM] Finished streaming response.\033[0m", flush=True)
            
        except Exception as e:
            logging.error(f"!!! LangGraph Critical Error: {str(e)} !!!")
            import traceback
            error_details = traceback.format_exc()
            logging.error(error_details)
            
            # Log error into MongoDB gracefully without crashing STT/TTS engine completely
            err_dict = {
                "message": str(e),
                "step": "LangGraph Execution",
                "stack": error_details
            }
            u_msg = {"turn_id": self.state.get("turn_id", 0), "text": last_user_msg, "intent": "UNKNOWN", "confidence": 0}
            a_msg = {"turn_id": self.state.get("turn_id", 0), "text": "Sorry ji, samajh nahi payi. Kya aap repeat kar sakte hain?"}
            await self.mongo.log_turn(self.session_id, u_msg, a_msg, self.state.get("current_order", {}), error=err_dict)

            ssml_err, _ = build_ssml(a_msg["text"])
            yield LLMResponse(content=ssml_err, role=ChatRole.ASSISTANT)

    async def cancel_current_generation(self) -> None:
        logging.info(f"--- Cancelling generation for session {self.session_id} ---")
        pass

# Define the agent's behavior
class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful AI assistant for a restaurant POS system.",
        )

    async def on_enter(self) -> None:
        # Time-based greeting — hardcoded, runs ONCE at call start
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        hour = datetime.now(ist).hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        greeting_text = f"{greeting}! Hamare restaurant mein aapka swagat hai. Aaj kya lena chahenge aap?"
        ssml_greeting, _ = build_ssml(greeting_text)
        await self.session.say(ssml_greeting)

    async def on_exit(self) -> None:
        logging.info("Agent session ended.")

async def start_session(context: JobContext):
    print("DEBUG: start_session entry", flush=True)
    session_id = str(uuid.uuid4())
    logging.info(f"Starting session: {session_id}")

    # Load menu item names from DB for Deepgram keyterm prompting
    keyterms = load_menu_keyterms()

    # Configure Sarvam STT — Saaras v3 with Indian English support
    stt = SarvamAISTT(
        api_key=os.getenv("SARVAM_API_KEY"),
        model="saaras:v3",
        language="en-IN"         # Force English transcription (prevent auto-detect to other languages)
    )
    
    # Configure Sarvam TTS — Bulbul v2 (low-latency, Indian voices)
    tts = SarvamAITTS(
        api_key=os.getenv("SARVAM_API_KEY"),
        language="en-IN",
        model="bulbul:v3",
        speaker="priya",
        temperature=0.4,
        pace=0.85
    )


    # Use our custom Multi-Agent Brain
    llm = RestaurantMultiAgentLLM(session_id=session_id)

    # Build the cascading pipeline — TURN-BASED: No interruptions during AI speech
    interrupt_config = InterruptConfig(
        interrupt_min_words=1,                  # Accept all genuine speech from user
        interrupt_min_duration=2.0,             # User must speak for 2s to interrupt AI mid-speech
        false_interrupt_pause_duration=3.0,     # Wait 3s after false interrupt before resuming
        resume_on_false_interrupt=True          # Resume TTS if interrupt was noise/hallucination
    )
    
    eou_config = EOUConfig(
        mode="ADAPTIVE",
        min_max_speech_wait_timeout=[0.6, 1.2]  # Automatically scales wait time based on word count
    )
    
    pipeline = CascadingPipeline(
        stt=stt, 
        llm=llm, 
        tts=tts,
        interrupt_config=interrupt_config,
        eou_config=eou_config
    )
    
    agent = MyVoiceAgent()
    
    flow = ConversationFlow(
        agent=agent, 
        stt=stt, 
        llm=llm, 
        tts=tts
    )
    flow.interrupt_min_words = 1  # Let SDK pass all words; our chat() filter handles hallucinations
    
    session = AgentSession(agent=agent, pipeline=pipeline, conversation_flow=flow)

    try:
        await context.connect()
        await session.start()
        
        # Log the start of the conversation (after connected)
        await MONGO_CLIENT.start_conversation(session_id, "73313cb0-dcd4-4f03-94e0-5ec7aaf711ad")
        
        await asyncio.Event().wait()
    finally:
        await session.close()
        await context.shutdown()
        # Log backend closure
        await MONGO_CLIENT.end_conversation(session_id, "completed")
        # Cleanup memory
        SESSION_STATES.pop(session_id, None)

def make_context() -> JobContext:
    # Fetch token from environment
    token = os.getenv("VIDEOSDK_AUTH_TOKEN")
    if not token:
        logging.warning("VIDEOSDK_AUTH_TOKEN is missing in .env! Connection will likely fail.")
        
    # Set playground=True for easier local testing (generates an interact URL)
    room_options = RoomOptions(
        token=token,
        playground=True
    )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    try:
        options = Options(
            agent_id="MyTelephonyAgent",
            register=True,
            max_processes=1, # Reduced from 5 to avoid initialization timeouts
            host="localhost",
            port=8081,
        )

        job = WorkerJob(entrypoint=start_session, jobctx=make_context, options=options)
        job.start()
    except Exception:
        traceback.print_exc()