import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState
from langchain_core.messages import AIMessage

class RecommendationAgent:
    """Step 12: Recommend additional/complementary items after adding food."""
    
    def __init__(self, model: str = "moonshotai/kimi-k2-instruct-0905"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model=model,
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a restaurant recommendation assistant.\n"
                      "The customer just ordered these items: {order_items}\n"
                      "The full menu includes: {menu_items}\n\n"
                      "Suggest 1-2 complementary items that pair well with their order.\n"
                      "Keep it short, friendly, and conversational.\n"
                      "Example: 'Would you like to add some garlic bread or a cold drink with that?'\n\n"
                      "IMPORTANT: Do NOT repeat items already in the order.\n"
                      "Return ONLY the recommendation message, nothing else."),
            ("human", "What would you recommend with my order?")
        ])

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        
        current_order = state.get("current_order", {})
        items = current_order.get("items", [])
        intent = state.get("intent", "Greeting")
        
        # Skip if no items or not an order intent (safety check — routing already filters)
        if not items or intent not in ["ORDER_ITEM"]:
            print(f"\033[92m[⌚ TOOL] RecommendationAgent skipped\033[0m", flush=True)
            return {}
        
        # Get item names from current order
        order_item_names = [item.get("name", "") for item in items]
        
        # Get full menu from retrieved_menu metadata
        menu_metadata = state.get("retrieved_menu", [])
        menu_names = list(set([m.get("name", "") for m in menu_metadata if m.get("name", "") not in order_item_names]))
        
        # If retrieved_menu is empty (e.g. user direct-ordered without asking for menu), 
        # fallback to the MenuCache to get a list of items
        if not menu_names:
            from database.menu_cache import MenuCache
            cache = MenuCache.get_instance()
            all_items = cache.get_item_names()
            menu_names = [name for name in all_items if name not in order_item_names]
            
        if not menu_names:
            print(f"\033[93m[⌚ TOOL] RecommendationAgent skipped (no menu items found)\033[0m", flush=True)
            return {}
        
        llm_start = time.time()
        response = await (self.prompt | self.llm).ainvoke({
            "order_items": ", ".join(order_item_names),
            "menu_items": ", ".join(menu_names[:10])  # Limit to 10 for speed
        })
        llm_duration = time.time() - llm_start
        
        in_tokens = response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        out_tokens = response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        
        print(f"\033[95m[⌚ LLM] RecommendationAgent (Kimi-K2) took {llm_duration:.2f}s | In: {in_tokens} Out: {out_tokens} tokens\033[0m", flush=True)
        
        # Append recommendation to the last message
        last_messages = state.get("messages", [])
        if last_messages:
            last_msg = last_messages[-1].content
            combined = f"{last_msg} {response.content}"
        else:
            combined = response.content
        
        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] RecommendationAgent total took {agent_duration:.2f}s\033[0m", flush=True)
        return {
            "messages": [AIMessage(content=combined)],
            "recommendation_shown": True
        }
