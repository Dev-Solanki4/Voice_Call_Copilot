import logging
from graph.state import AgentState
from langchain_core.messages import AIMessage

class OrderAgent:
    """Steps 6-9: Order Management — LEGACY FILE, NOT USED IN WORKFLOW.
    The OrderMathAgent now handles all order logic deterministically.
    This file is kept for reference only.
    """
    
    def __init__(self):
        pass  # No LLM needed — this agent is not used in the workflow

    async def __call__(self, state: AgentState):
        """Legacy fallback: basic order management using state data."""
        current_order = state.get("current_order") or {"items": [], "total": 0.0, "tax": 0.0}
        extracted = state.get("last_extracted_items", [])
        matched_items = state.get("matched_menu_items", [])
        intent = state.get("intent", "UNKNOWN")
        
        if intent == "ASK_MENU":
            return {"current_order": current_order}
        
        # Handle removes
        remove_items = [e for e in extracted if e.get("action") == "remove"]
        if remove_items:
            removed_names = []
            for rm in remove_items:
                rm_name = rm.get("name", "").lower()
                before_len = len(current_order["items"])
                current_order["items"] = [
                    item for item in current_order["items"]
                    if rm_name not in item.get("name", "").lower()
                ]
                if len(current_order["items"]) < before_len:
                    removed_names.append(rm.get("name", "Item"))
            
            current_order["total"] = sum(i.get("subtotal", 0) for i in current_order["items"])
            
            if removed_names:
                r_str = ", ".join(removed_names)
                if current_order["items"]:
                    items_str = ", ".join([f"{i['quantity']} {i['name']}" for i in current_order['items']])
                    total = current_order.get("total", 0)
                    msg = f"Removed {r_str}. Your current order: {items_str}. Total: ₹{total}."
                else:
                    msg = "Your order is currently empty. What would you like to order?"
                return {"current_order": current_order, "messages": [AIMessage(content=msg)]}
        
        # Handle adds from matched items
        filtered_matches = [item for item in matched_items if item.get("status") in ["matched", "multiple_matches"]]
        
        for match in filtered_matches:
            mid = match.get("menu_item_id")
            price = float(match.get("price", 0))
            name = match.get("name", "")
            qty = match.get("quantity", 1)
            
            # Check if already in order
            existing = next((i for i in current_order["items"] if i.get("menu_item_id") == mid), None)
            if existing:
                existing["quantity"] += qty
                existing["subtotal"] = float(existing["quantity"] * existing["price"])
            else:
                current_order["items"].append({
                    "menu_item_id": mid,
                    "name": name,
                    "quantity": qty,
                    "price": price,
                    "subtotal": float(qty * price)
                })
        
        # Recalculate total
        current_order["total"] = float(sum(item.get("subtotal", 0) for item in current_order["items"]))
        
        if current_order["items"]:
            items_str = ", ".join([f"{i['quantity']} {i['name']} – ₹{i['subtotal']}" for i in current_order['items']])
            msg = f"Got it. Your order: {items_str}. Total: ₹{current_order['total']}. Anything else?"
        else:
            msg = "Your order is currently empty. What would you like to order?"
        
        logging.info(f"--- OrderAgent (legacy) completed ---")
        return {"current_order": current_order, "messages": [AIMessage(content=msg)]}
