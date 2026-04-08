import logging
from graph.state import AgentState
from database.vector_store import MenuVectorStore
from database.menu_cache import MenuCache
from langchain_core.messages import AIMessage

class MenuAgent:
    """Steps 2-4: Menu Matching — Search menu, check confidence, suggest alternatives. No LLM needed."""
    
    def __init__(self):
        self.vector_store = MenuVectorStore()
        self.cache = MenuCache.get_instance()

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        extracted_items = state.get("last_extracted_items", [])
        intent = state.get("intent", "UNKNOWN")
        # Use normalized_text from extractor if available (it has spelling corrections)
        text = state.get("normalized_text") or state["messages"][-1].content
        if isinstance(text, list):
            text = " ".join([str(t) for t in text])
        
        if intent not in ["ASK_MENU", "ORDER_ITEM", "MODIFY_ORDER", "PRICE_CHECK"]:
            return {} # Pass through if not a menu/order query
        
        # ── Handle PRICE_CHECK Targeted Query ──
        if intent == "PRICE_CHECK" and extracted_items:
            price_msgs = []
            for item in extracted_items:
                query = item.get("name", "")
                if not query: continue
                matches = self.cache.search_items(query)
                if matches:
                    exact = [m for m in matches if m["name"].lower() == query.lower()]
                    best_match = exact[0] if exact else matches[0]
                    price_msgs.append(f"{best_match['name']} is ₹{int(best_match['price'])}.")
                else:
                    price_msgs.append(f"Sorry, I couldn't find {query} on the menu.")
            
            msg = " ".join(price_msgs) if price_msgs else "Which item's price would you like to know?"
            print(f"--- MenuAgent PASSTHROUGH (PRICE_CHECK): {time.time() - start:.2f}s ---")
            return {
                "messages": [AIMessage(content=msg)],
                "menu_action": "list_items", # Signal to formatter to passthrough
                "confidence": 100
            }
        
        # Search for items
        all_results = []
        retrieved_objects = []
        requested_cat = None
        
        if extracted_items:
            for extract in extracted_items:
                query = extract.get("name") or extract.get("item")
                if not query or extract.get("action") == "broken_speech": continue
                
                # 1. Vector search
                results = self.vector_store.search(query, k=5)
                all_results.extend([doc.page_content for doc in results])
                for doc in results:
                    retrieved_objects.append(doc.metadata)
                
                # 2. In-memory exact search
                db_results = self.cache.search_items(query)
                for item in db_results:
                    content = f"Name: {item['name']}\nDescription: {item.get('description', '')}\nPrice: ₹{item['price']}"
                    if content not in all_results:
                        all_results.append(content)
                    meta = {"id": str(item["id"]), "name": item["name"], "price": float(item["price"])}
                    if meta not in retrieved_objects:
                        retrieved_objects.append(meta)
                        
        elif intent == "ASK_MENU":
            # Check for specific category or 'veg' request
            text_lower = text.lower()
            categories = self.cache.get_categories()
            
            # --- PRIORITY 1: Search for specific item matches (e.g. "coffee", "pizza") ---
            search_term = text.strip().lower()
            ignore_list = ["menu", "what do you have", "what is available", "show me menu", "menu card", "i want to order", "order"]
            
            if search_term not in ignore_list and len(search_term.split()) <= 4:
                clean_term = search_term[:-1] if search_term.endswith('s') else search_term
                db_items = self.cache.search_items(clean_term)
                
                if db_items:
                    items_str = ", ".join([f"{item['name']} at rupees {int(item['price'])}" for item in db_items[:8]])
                    msg = f"We have multiple options for '{search_term}': {items_str}. Which one would you like to order?"
                    print(f"--- MenuAgent Step 2 (Specific Match: {search_term}): {time.time() - start:.2f}s ---")
                    return {
                        "messages": [AIMessage(content=msg)],
                        "menu_action": "list_items",
                        "confidence": 100
                    }

            # --- PRIORITY 2: Check for Category Match ---
            requested_cat = None
            for cat in categories:
                c_low = cat.lower()
                if c_low in text_lower or text_lower in c_low:
                    requested_cat = cat
                    break
            
            if requested_cat:
                db_items = self.cache.get_items_by_category(requested_cat)
                if db_items:
                    items_str = ", ".join([f"{item['name']} at rupees {int(item['price'])}" for item in db_items])
                    msg = f"In {requested_cat}, we have {items_str}. What would you like to order?"
                    print(f"--- MenuAgent Step 2 (Category Match: {requested_cat}): {time.time() - start:.2f}s ---")
                    return {
                        "messages": [AIMessage(content=msg)],
                        "menu_action": "list_items",
                        "confidence": 100
                    }
                
            # --- PRIORITY 3: Handle "Veg" keyword ---
            if "veg" in text_lower:
                db_items = self.cache.get_veg_items()
                if db_items:
                    items_str = ", ".join([f"{item['name']} at rupees {int(item['price'])}" for item in db_items])
                    msg = f"Our vegetarian options are: {items_str}. What would you like?"
                    print(f"--- MenuAgent Step 2 (Veg Match): {time.time() - start:.2f}s ---")
                    return {
                        "messages": [AIMessage(content=msg)],
                        "menu_action": "list_items",
                        "confidence": 100
                    }

            # --- PRIORITY 4: Handle generic menu requests ---
            if any(term in text_lower for term in ignore_list):
                cats_str = ", ".join(categories)
                msg = f"We have {cats_str}. Which category would you like to explore?"
                print(f"--- MenuAgent Step 1 (All Categories): {time.time() - start:.2f}s ---")
                return {
                    "messages": [AIMessage(content=msg)],
                    "menu_action": "list_categories",
                    "confidence": 100
                }

            # --- PRIORITY 5: Complex Query Fallback (Vector Search — No LLM) ---
            if not all_results:
                results = self.vector_store.search(text, k=8)
                all_results.extend([doc.page_content for doc in results])
                for doc in results:
                    retrieved_objects.append(doc.metadata)

            # Build deterministic spoken response from vector search results
            if retrieved_objects:
                # Deduplicate by item name
                seen_names = set()
                unique_items = []
                for meta in retrieved_objects:
                    name = meta.get("name", "")
                    if name and name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        unique_items.append(meta)
                
                if unique_items:
                    items_str = ", ".join([f"{m['name']} at rupees {int(float(m.get('price', 0)))}" for m in unique_items[:10]])
                    msg = f"Haan ji, we have {items_str}. Which one would you like to order?"
                else:
                    msg = "Sorry ji, kuch nahi mila menu mein. Kya aur kuch try karein?"
            else:
                msg = "Sorry ji, yeh item hamare menu mein nahi mila. Kya aur kuch dekhna chahenge?"

            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] MenuAgent total took {agent_duration:.2f}s\033[0m", flush=True)
            return {
                "messages": [AIMessage(content=msg)],
                "retrieved_menu": retrieved_objects,
                "matched_menu_items": [],
                "confidence": 100
            }

