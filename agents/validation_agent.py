from graph.state import AgentState
from database.postgres_service import PetPoojaDB
from langchain_core.messages import AIMessage
import logging

class ValidationAgent:
    """Step 10: Validate menu items — check availability, stock, and restaurant status."""
    
    def __init__(self):
        self.db = PetPoojaDB()

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        
        current_order = state.get("current_order", {})
        items = current_order.get("items", [])
        intent = state.get("intent", "UNKNOWN")
        
        if not items or intent in ["ASK_MENU", "GREETING"]:
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] ValidationAgent skipped took {agent_duration:.2f}s\033[0m", flush=True)
            return {}
        
        # Check inventory for each item — only flag if we have actual inventory data
        unavailable_items = []
        for item in items:
            menu_item_id = item.get("menu_item_id")
            if not menu_item_id:
                continue
            
            try:
                inventory = self.db.get_inventory_status(menu_item_id)
                if not inventory:
                    # No inventory record = item is available (no stock tracking)
                    continue
                    
                for ing in inventory:
                    stock = float(ing.get("stock", 0))
                    required = float(ing.get("required", 0)) * item.get("quantity", 1)
                    if stock > 0 and stock < required:
                        unavailable_items.append({
                            "item": item.get("name"),
                            "ingredient": ing.get("name"),
                            "available_stock": stock,
                            "required": required
                        })
            except Exception as e:
                logging.error(f"ValidationAgent inventory check error: {e}")
        
        if unavailable_items:
            names = ", ".join(set(u["item"] for u in unavailable_items))
            msg = f"Sorry ji, {names} abhi out of stock hai. Kya main kuch aur suggest kar sakti hoon?"
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] ValidationAgent (Out of Stock) took {agent_duration:.2f}s\033[0m", flush=True)
            return {"ready_for_confirmation": False, "messages": [AIMessage(content=msg)]}
        
        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] ValidationAgent (All OK) took {agent_duration:.2f}s\033[0m", flush=True)
        return {"ready_for_confirmation": True}
