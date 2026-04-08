from graph.state import AgentState
from langchain_core.messages import AIMessage
import logging

class PaymentAgent:
    """Payment handling — LEGACY FILE, NOT USED IN WORKFLOW.
    The TransactionAgent now handles payment selection directly in the workflow.
    This file is kept for reference only and operates deterministically.
    """
    
    def __init__(self):
        pass  # No LLM needed — logic is deterministic

    async def __call__(self, state: AgentState):
        """Legacy fallback: Basic payment prompting using state data."""
        current_order = state.get("current_order") or {"items": [], "total": 0.0}
        
        # Check for both "total" and "total_price" just in case legacy state differs
        total = float(current_order.get("total", 0.0) or current_order.get("total_price", 0.0))
        
        if total > 0:
            msg = f"Aapka grand total hai rupees {int(total)}. Would you like to proceed with payment?"
        else:
            msg = "Aapka cart khali hai. Would you like to order something first?"
            
        logging.info(f"--- PaymentAgent (legacy fallback) executed ---")
        
        return {
            "messages": [AIMessage(content=msg)],
        }
