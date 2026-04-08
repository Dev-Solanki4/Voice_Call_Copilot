from graph.state import AgentState
from langchain_core.messages import AIMessage
import logging

class SummarizationAgent:
    """Deterministic cart summary — no LLM needed. Reads all items with prices."""
    
    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        current_order = state.get("current_order") or {"items": [], "total": 0.0, "tax": 0.0}
        items = current_order.get("items", [])
        
        if not items:
            return {
                "messages": [AIMessage(content="Aapka cart abhi khali hai ji. Kya order karna chahenge?")],
                "ready_for_confirmation": False,
                "checkout_stage": ""
            }

        # Build deterministic, voice-ready summary
        item_parts = []
        for i in items:
            name = i["name"]
            qty = i["quantity"]
            price = int(i["price"])
            subtotal = int(i["subtotal"])
            if qty == 1:
                item_parts.append(f"{name} at rupees {price}")
            else:
                item_parts.append(f"{qty} {name} at rupees {price} each, subtotal {subtotal}")
        
        items_str = ", ".join(item_parts)
        total = int(current_order.get("total", 0))
        
        msg = f"Bilkul ji, aapka order hai — {items_str}. Grand total hai rupees {total}. Yeh dine-in hoga ya online?"
        
        agent_duration = time.time() - start
        logging.info(f"--- SummarizationAgent: {msg} ---")
        print(f"\033[92m[⌚ TOOL] SummarizationAgent took {agent_duration:.2f}s\033[0m", flush=True)
        
        return {
            "messages": [AIMessage(content=msg)],
            "ready_for_confirmation": True,
            "checkout_stage": "dining"
        }
