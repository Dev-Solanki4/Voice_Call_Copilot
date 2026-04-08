from graph.state import AgentState
import logging

class IntentAgent:
    """Step 1: Intent Detection — LEGACY FILE, NOT USED IN WORKFLOW.
    The ExtractionAgent now handles both intent detection and entity extraction in a single pass.
    This file is kept for reference only.
    """
    
    VALID_INTENTS = {
        "ORDER_ITEM", "MODIFY_ORDER", "ASK_MENU", "CONFIRM_ORDER",
        "DINING_SELECTION", "CANCEL", "GREETING", "UNKNOWN"
    }
    
    def __init__(self):
        pass  # No LLM needed — this agent is not used in the workflow

    async def __call__(self, state: AgentState):
        """Legacy fallback: simple keyword-based intent detection."""
        text = state["messages"][-1].content.lower().strip()
        
        # Simple keyword matching
        if any(w in text for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            intent = "GREETING"
        elif any(w in text for w in ["menu", "what do you have", "available", "show me"]):
            intent = "ASK_MENU"
        elif any(w in text for w in ["add", "want", "give me", "order", "get me"]):
            intent = "ORDER_ITEM"
        elif any(w in text for w in ["remove", "change", "modify", "cancel item"]):
            intent = "MODIFY_ORDER"
        elif any(w in text for w in ["confirm", "done", "place order", "that's all", "go ahead"]):
            intent = "CONFIRM_ORDER"
        elif any(w in text for w in ["dine in", "sitting", "online", "delivery", "take away", "parcel"]):
            intent = "DINING_SELECTION"
        elif any(w in text for w in ["cancel", "start over", "forget it"]):
            intent = "CANCEL"
        else:
            intent = "UNKNOWN"
        
        confidence = 49 if intent == "UNKNOWN" else 80
        logging.info(f"--- IntentAgent (legacy keyword) Intent: {intent}, Conf: {confidence} ---")
        return {
            "intent": intent,
            "confidence": confidence
        }
