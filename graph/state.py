from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Restaurant context
    restaurant_id: str  # UUID string
    
    # Step 2: Extraction results
    last_extracted_items: List[dict]
    # format: [{"name": "pizza", "quantity": 2, "modifiers": [], "action": "add"}]
    
    # Step 3: Intent detection
    intent: str
    
    # Step 4: Menu search results
    retrieved_menu: List[dict]       # Full metadata from vector/DB search
    matched_menu_items: List[dict]   # Structured matching results
    menu_action: str                 # "list_categories", "list_items", or "" (for ordering)
    
    # Steps 6-8: Order management
    current_order: dict
    # format: {"items": [{"menu_item_id": "uuid", "name": "pizza", "quantity": 1, "price": 350.0, "subtotal": 350.0}], "total": 0.0, "tax": 0.0}
    
    # Step 10: Validation
    inventory_status: List[dict]
    
    # Step 12: Recommendations
    recommendation_shown: bool
    recommendation_opted_out: bool  # If True, never show recommendations again this session
    
    # Steps 14-16: Confirmation flow
    ready_for_confirmation: bool
    order_confirmed: bool
    
    # Steps 17-19: Transaction
    transaction_completed: bool
    dining_type: Optional[str] # "dine-in" or "online"
    checkout_stage: str        # "", "summary", "dining", "payment", "customer_name"
    payment_mode: Optional[str] # "cash" or "online"
    customer_name: Optional[str] # Customer's name for the order
    
    # Output Tracking
    turn_id: int
    normalized_text: str
    confidence: int  # 0-100
    error: dict
    
    # Final output
    final_response: str
