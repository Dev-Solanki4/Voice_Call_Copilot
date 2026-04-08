from langgraph.graph import StateGraph, END
from graph.state import AgentState
import logging

def create_restaurant_graph():
    # Deferred imports to speed up worker initialization
    from agents.extraction_agent import ExtractionAgent
    from agents.menu_agent import MenuAgent
    from agents.order_math_agent import OrderMathAgent
    from agents.validation_agent import ValidationAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.transaction_agent import TransactionAgent
    from agents.summarization_agent import SummarizationAgent
    from agents.general_agent import GeneralAgent
    from agents.response_formatter import ResponseFormatter

    # Initialize agents
    extractor = ExtractionAgent()            # Step 1 & 2 Unified
    menu_agent = MenuAgent()                 # Step 4-5
    order_math = OrderMathAgent()            # Step 8 (Refined Math Node)
    validator = ValidationAgent()            # Step 10
    recommender = RecommendationAgent()      # Step 12
    summarizer = SummarizationAgent()        # Cart Summary (deterministic)
    transaction_finalizer = TransactionAgent() # Dining + Payment + Finalize
    concierge = GeneralAgent()               # Greetings, Unknown
    formatter = ResponseFormatter()          # Guideline 10

    # Define workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extractor", extractor)
    workflow.add_node("menu_specialist", menu_agent)
    workflow.add_node("order_math", order_math)
    workflow.add_node("validator", validator)
    workflow.add_node("recommender", recommender)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("transaction_finalizer", transaction_finalizer)
    workflow.add_node("concierge", concierge)
    workflow.add_node("formatter", formatter)

    # Set entry point: always extract first
    workflow.set_entry_point("extractor")

    # --- Routing Functions ---

    def route_after_extraction(state: AgentState):
        """Route based on detected intent."""
        extracted = state.get("last_extracted_items", [])
        intent = state.get("intent", "UNKNOWN")
        
        # Handle broken speech
        if any(item.get("action") == "broken_speech" for item in extracted):
            return "concierge"
            
        current_order = state.get("current_order", {})
        items = current_order.get("items", [])
        has_items = len(items) > 0
        
        logging.info(f"--- Routing: Intent={intent}, Items={len(items)}, Stage={state.get('checkout_stage', '')} ---")
        
        # ── ORDER LOCK: Post-payment ──
        if current_order.get("order_confirmed"):
            return "concierge"
        
        # Checkout flow intents
        if intent == "CONFIRM_ORDER":
            if has_items:
                return "summarizer"  # Always show summary first
            return "concierge"  # No items to confirm
        
        if intent in ["SELECT_DINING", "SELECT_PAYMENT", "PAYMENT_DONE", "PROVIDE_NAME"]:
            if has_items:
                return "transaction_finalizer"
            return "concierge"
            
        # Ordering intents (works mid-checkout too)
        if intent in ["ORDER_ITEM", "MODIFY_ORDER"]:
            return "order_math"
            
        if intent in ["ASK_MENU", "PRICE_CHECK"]:
            return "menu_specialist"
            
        if intent in ["CANCEL", "GREETING", "UNKNOWN", "GOODBYE", "NEGOTIATE"]:
            return "concierge"
        
        if intent == "DECLINE_RECOMMENDATION":
            return "concierge"
            
        return "concierge"

    def route_after_menu(state: AgentState):
        """After menu: if browsing → formatter, if ordering → order_math."""
        menu_action = state.get("menu_action", "")
        if menu_action in ["list_categories", "list_items"]:
            return "formatter"
        return "order_math"

    def route_after_order_math(state: AgentState):
        """After order math: if mid-checkout → summarizer (re-read cart), else → validator."""
        checkout_stage = state.get("checkout_stage", "")
        if checkout_stage in ["dining", "payment"]:
            # Mid-checkout modification — cycle back to summary
            return "summarizer"
        return "validator"

    def route_after_validation(state: AgentState):
        """After validation: recommend ONLY after ORDER_ITEM, and only if user hasn't opted out."""
        if not state.get("ready_for_confirmation"):
            return "formatter"
        
        intent = state.get("intent", "")
        opted_out = state.get("recommendation_opted_out", False)
        
        # Only recommend after ORDER_ITEM and if user hasn't permanently opted out
        if intent == "ORDER_ITEM" and not opted_out:
            return "recommender"
        
        return "formatter"

    # --- Wire Edges ---

    # Extractor → Next Node
    workflow.add_conditional_edges("extractor", route_after_extraction, {
        "summarizer": "summarizer",
        "transaction_finalizer": "transaction_finalizer",
        "menu_specialist": "menu_specialist",
        "order_math": "order_math",
        "concierge": "concierge"
    })

    # Menu → conditional: formatter (browsing) or order_math (ordering)
    workflow.add_conditional_edges("menu_specialist", route_after_menu, {
        "formatter": "formatter",
        "order_math": "order_math"
    })
    
    # Order Math → conditional: summarizer (mid-checkout) or validator (normal)
    workflow.add_conditional_edges("order_math", route_after_order_math, {
        "summarizer": "summarizer",
        "validator": "validator"
    })

    # Validation → recommender or formatter
    workflow.add_conditional_edges("validator", route_after_validation, {
        "recommender": "recommender",
        "formatter": "formatter"
    })

    # Simple edges to formatter
    workflow.add_edge("recommender", "formatter")
    workflow.add_edge("summarizer", "formatter")
    workflow.add_edge("transaction_finalizer", "formatter")
    workflow.add_edge("concierge", "formatter")

    # Final → END
    workflow.add_edge("formatter", END)

    return workflow.compile()
