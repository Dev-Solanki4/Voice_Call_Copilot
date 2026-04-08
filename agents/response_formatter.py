from graph.state import AgentState
from langchain_core.messages import AIMessage
import logging

class ResponseFormatter:
    """Final node: Passthrough formatter. All upstream agents already produce voice-ready output."""
    
    def __init__(self):
        pass  # No LLM needed — upstream agents handle formatting

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        
        base_response = state["messages"][-1].content if state["messages"] else ""
        
        if not base_response:
            base_response = "Sorry ji, kuch samajh nahi aaya. Please dobara bataiye?"
        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] ResponseFormatter PASSTHROUGH took {agent_duration:.2f}s\033[0m", flush=True)
        return {"messages": [AIMessage(content=base_response)]}
