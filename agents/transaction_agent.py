from database.postgres_service import PetPoojaDB
from graph.state import AgentState
from langchain_core.messages import AIMessage
import logging
import traceback

class TransactionAgent:
    """Staged checkout: dining mode → payment mode → finalize order."""
    
    def __init__(self):
        self.db = PetPoojaDB()

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        intent = state.get("intent")
        current_order = state.get("current_order")
        dining_type = state.get("dining_type")
        payment_mode = state.get("payment_mode")
        checkout_stage = state.get("checkout_stage", "")
        
        if not current_order or not current_order.get("items"):
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Empty Cart)\033[0m", flush=True)
            return {
                "messages": [AIMessage(content="Your cart is empty. What would you like to order?")],
                "checkout_stage": ""
            }

        # ── STAGE 1: Handle dining selection ──
        if intent == "SELECT_DINING":
            dining = dining_type  # Already extracted by ExtractionAgent
            if not dining:
                # Try to detect from text
                text = state["messages"][-1].content.lower() if state["messages"] else ""
                if "dine" in text or "sit" in text or "eat here" in text:
                    dining = "dine-in"
                elif "online" in text or "deliver" in text or "take" in text or "parcel" in text:
                    dining = "online"
            
            if not dining:
                return {
                    "messages": [AIMessage(content="Sorry ji, samajh nahi payi. Yeh dine-in hoga ya online?")],
                    "checkout_stage": "dining",
                    "ready_for_confirmation": True
                }
            
            msg = f"Badhiya ji, {dining} order note ho gaya. Payment kaise karenge? Cash ya online?"
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Stage: Dining->Payment)\033[0m", flush=True)
            return {
                "messages": [AIMessage(content=msg)],
                "dining_type": dining,
                "checkout_stage": "payment",
                "ready_for_confirmation": True
            }

        # ── STAGE 2: Handle payment selection ──
        if intent == "SELECT_PAYMENT":
            payment = payment_mode  # Already extracted by ExtractionAgent
            if not payment:
                text = state["messages"][-1].content.lower() if state["messages"] else ""
                if "cash" in text:
                    payment = "cash"
                elif "online" in text or "upi" in text or "gpay" in text or "card" in text:
                    payment = "online"
            
            if not payment:
                return {
                    "messages": [AIMessage(content="Sorry ji, samajh nahi payi. Cash ya online payment?")],
                    "checkout_stage": "payment",
                    "ready_for_confirmation": True
                }
            
            # Payment mode selected — now ask for customer name
            msg = f"Done ji, {payment} payment note ho gaya. Aapka shubh naam kya hai?"
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Stage: Payment->Name)\033[0m", flush=True)
            return {
                "messages": [AIMessage(content=msg)],
                "payment_mode": payment,
                "checkout_stage": "customer_name",
                "ready_for_confirmation": True
            }

        # ── STAGE 2.5: Handle customer name ──
        if intent == "PROVIDE_NAME":
            customer_name = state.get("customer_name")
            if not customer_name:
                return {
                    "messages": [AIMessage(content="Sorry ji, naam samajh nahi aaya. Please apna naam bataiye?")],
                    "checkout_stage": "customer_name",
                    "ready_for_confirmation": True
                }
            
            # Name captured — now finalize the order
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Finalizing...)\033[0m", flush=True)
            return await self._finalize_order(state, dining_type or "dine-in", payment_mode or "cash", customer_name)

        # ── STAGE 3: Payment done ──
        if intent == "PAYMENT_DONE":
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Finalizing Payment Done...)\033[0m", flush=True)
            return await self._finalize_order(state, dining_type or "dine-in", payment_mode or "cash", state.get("customer_name"))

        # ── Fallback: CONFIRM_ORDER when no dining type yet ──
        if intent == "CONFIRM_ORDER":
            if not dining_type:
                return {
                    "messages": [AIMessage(content="Yeh dine-in hoga ya online?")],
                    "checkout_stage": "dining",
                    "ready_for_confirmation": True
                }
            if not payment_mode:
                return {
                    "messages": [AIMessage(content="Payment kaise karenge? Cash ya online?")],
                    "checkout_stage": "payment",
                    "ready_for_confirmation": True
                }
            if not state.get("customer_name"):
                return {
                    "messages": [AIMessage(content="Aapka shubh naam kya hai please?")],
                    "checkout_stage": "customer_name",
                    "ready_for_confirmation": True
                }
            agent_duration = time.time() - start
            print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (Finalizing Confirm Order...)\033[0m", flush=True)
            return await self._finalize_order(state, dining_type, payment_mode, state.get("customer_name"))

        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] TransactionAgent took {agent_duration:.2f}s (No hit)\033[0m", flush=True)
        return {}

    async def _finalize_order(self, state, dining_type, payment_mode, customer_name=None):
        """Finalize the order in the database."""
        current_order = state.get("current_order")
        restaurant_id = state.get("restaurant_id")
        
        try:
            # Create order in DB
            order_result = self.db.create_order(
                restaurant_id=restaurant_id,
                total=float(current_order['total']),
                tax=float(current_order.get('tax', 0.0)),
                payment_mode=payment_mode,
                dining_type=dining_type,
                customer_name=customer_name
            )
            order_id = order_result["order_id"]
            order_number = order_result["order_number"]
            
            # Insert order items
            self.db.add_order_items(order_id, current_order['items'])
            
            # Update inventory
            self.db.update_inventory_post_order(order_id)
            
            total = int(current_order['total'])
            
            # Time-based goodbye
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            hour = datetime.now(ist).hour
            if hour < 12:
                goodbye = "Have a great morning"
            elif hour < 17:
                goodbye = "Have a wonderful afternoon"
            else:
                goodbye = "Have a lovely evening"
                
            if customer_name and str(customer_name).lower() != "none":
                goodbye = f"{goodbye}, {customer_name}"
            
            msg = (f"Wonderful! Aapka order successfully place ho gaya ji! "
                   f"Order ID hai {order_number}. "
                   f"{dining_type} order, payment by {payment_mode}. "
                   f"Total rupees {total}. Lagbhag 20 minute mein ready ho jayega. "
                   f"Bahut shukriya ji! {goodbye}!")
            
            return {
                "transaction_completed": True,
                "dining_type": dining_type,
                "payment_mode": payment_mode,
                "checkout_stage": "done",
                "current_order": {**current_order, "payment_status": "completed", "order_confirmed": True, "order_number": order_number},
                "messages": [AIMessage(content=msg)]
            }
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"Transaction Finalization Failed: {e}\n{error_details}")
            print(f"!!! TRANSACTION ERROR !!!: {e}", flush=True)
            return {
                "messages": [AIMessage(content="Oh sorry ji, order finalize karte waqt ek error aaya. Please phir se try karein.")]
            }
