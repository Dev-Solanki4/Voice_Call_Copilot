import logging
import json
from langchain_core.messages import AIMessage
from database.menu_cache import MenuCache

logger = logging.getLogger(__name__)

class OrderMathAgent:
    """
    Step 8: Order Math Logic Node — Handles calculations and state updates numerically.
    Replaces LLM-based arithmetic for subtotal, TAX (GST), and grand totals.
    """
    
    def __init__(self):
        self.cache = MenuCache.get_instance()
        
    async def __call__(self, state: dict):
        import time
        start = time.time()
        # 1. Initialize state variables
        extracted_items = state.get("last_extracted_items", [])
        intent = state.get("intent", "UNKNOWN")
        current_order = state.get("current_order") or {"items": [], "total": 0.0, "tax": 0.0}
        
        added_list = []
        removed_list = []
        unavailable_list = []
        ambiguous_list = []

        # Handle VIEW_CART or empty items gracefully
        if not extracted_items:
            if not current_order["items"]:
                return {"messages": [AIMessage(content="Aapka cart abhi khali hai ji. Menu dekhna chahenge?")]}
            # Detailed cart summary with per-item prices
            item_parts = []
            for i in current_order["items"]:
                item_parts.append(f"{i['quantity']} {i['name']} at rupees {int(i['price'])} each, subtotal rupees {int(i['subtotal'])}")
            item_list = ". ".join(item_parts)
            summary = f"Bilkul ji, aapka current order — {item_list}. Grand total hai rupees {int(current_order['total'])} including GST."
            return {"messages": [AIMessage(content=summary)], "ready_for_confirmation": True}

        # 2. Process ALL extracted items (Supporting multi-item commands)
        last_removed_qty = None
        for item_data in extracted_items:
            query = item_data.get("name", "").lower().strip()
            qty = item_data.get("quantity", 1)
            action = item_data.get("action", "add")

            if not query: continue

            # Fuzzy search using MenuCache
            matches = self.cache.search_items(query)
            
            if not matches:
                unavailable_list.append(query)
                continue
                
            if len(matches) > 1:
                # Try exact match first
                exact = [m for m in matches if m["name"].lower() == query]
                if exact:
                    matched_item = exact[0]
                elif action == "add":
                    # Sub-item name query (e.g. "paneer") — list all matching items
                    options_str = ", ".join([f"{m['name']} at rupees {int(m['price'])}" for m in matches[:5]])
                    ambiguous_list.append({"query": query, "options": [m["name"] for m in matches[:5]], "options_str": options_str})
                    continue
                elif action == "remove":
                    # For removal, try partial match in cart
                    cart_match = [c for c in current_order["items"] if query in c["name"].lower()]
                    if len(cart_match) == 1:
                        matched_item = {"name": cart_match[0]["name"], "price": cart_match[0]["price"], "id": cart_match[0]["menu_item_id"]}
                    else:
                        ambiguous_list.append({"query": query, "options": [m["name"] for m in matches[:5]], "options_str": ", ".join([m["name"] for m in matches[:5]])})
                        continue
                else:
                    ambiguous_list.append({"query": query, "options": [m["name"] for m in matches[:5]], "options_str": ", ".join([m["name"] for m in matches[:5]])})
                    continue
            else:
                matched_item = matches[0]

            # 3. Handle Quantity Logic for the matched item
            item_name = matched_item["name"]
            price = matched_item["price"]
            item_id = matched_item["id"]

            found_in_cart = False
            if action == "remove":
                # Special Case: 'all' or 'every' (qty: -1)
                effective_qty = qty
                if qty == -1:
                    # Find total quantity in cart
                    existing = [c for c in current_order["items"] if c["name"].lower() == item_name.lower()]
                    effective_qty = sum(c["quantity"] for c in existing) if existing else 1
                    last_removed_qty = effective_qty  # Store for replacement math
                
                new_items = []
                for cart_item in current_order["items"]:
                    if cart_item["name"].lower() == item_name.lower():
                        # If effective_qty is more than or equal to current quantity, remove it entirely
                        if effective_qty == -1 or effective_qty >= cart_item["quantity"]:
                            found_in_cart = True
                            # Don't add to new_items (removes it)
                            continue
                        else:
                            cart_item["quantity"] -= effective_qty
                            cart_item["subtotal"] = cart_item["quantity"] * price
                            new_items.append(cart_item)
                            found_in_cart = True
                    else:
                        new_items.append(cart_item)
                current_order["items"] = new_items
                if found_in_cart:
                    removed_list.append(f"{effective_qty} {item_name}")
                else:
                    unavailable_list.append(f"{item_name} (not in cart)")
            else:
                # Add logic
                # Special Case: 'all' replacement (qty: -1)
                effective_qty = qty
                if qty == -1:
                    effective_qty = last_removed_qty if last_removed_qty is not None else 1
                
                for cart_item in current_order["items"]:
                    if cart_item["name"].lower() == item_name.lower():
                        cart_item["quantity"] += effective_qty
                        cart_item["subtotal"] = cart_item["quantity"] * price
                        found_in_cart = True
                        break
                if not found_in_cart:
                    current_order["items"].append({
                        "menu_item_id": item_id,
                        "name": item_name,
                        "quantity": effective_qty,
                        "price": price,
                        "subtotal": price * effective_qty
                    })
                added_list.append(f"{effective_qty} {item_name} at rupees {int(price)} each")

        # 4. PERFORM FINAL MATH (Numerical precision)
        subtotal = sum(i["subtotal"] for i in current_order["items"])
        gst_amount = round(subtotal * 0.05, 2)
        grand_total = round(subtotal + gst_amount, 2)
        
        current_order["total"] = grand_total
        current_order["tax"] = gst_amount
        
        # 5. Build Comprehensive Voice Response
        parts = []
        if removed_list:
            parts.append(f"Hata diya ji, {', '.join(removed_list)}.")
        if added_list:
            parts.append(f"Add ho gaya ji, {', '.join(added_list)}.")
        
        if unavailable_list:
            parts.append(f"Sorry ji, {', '.join(unavailable_list)} menu mein nahi mila.")
            
        if ambiguous_list:
            for amb in ambiguous_list:
                options_str = amb.get("options_str", ", or ".join(amb["options"]))
                parts.append(f"Haan ji, '{amb['query']}' ke liye multiple options hain — {options_str}. Kaunsa chahiye?")

        if current_order["items"]:
            parts.append(f"Aapka total hai {grand_total} rupees. Checkout karein?")
        elif not parts:
            parts.append("Aapka cart ab khali hai ji.")

        summary = " ".join(parts)
        logging.info(f"--- OrderMathAgent Summary: {summary} ---")
        
        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] OrderMathAgent total took {agent_duration:.2f}s\033[0m", flush=True)

        return {
            "messages": [AIMessage(content=summary)],
            "current_order": current_order,
            "ready_for_confirmation": len(current_order["items"]) > 0 and not ambiguous_list,
            "recommendation_shown": False,  # Reset so recommender can trigger on this turn
            "dining_type": None,      # Reset on cart change
            "payment_mode": None,     # Reset on cart change
        }
