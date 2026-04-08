import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState
from langchain_core.messages import AIMessage

class GeneralAgent:
    def __init__(self, model: str = "moonshotai/kimi-k2-instruct-0905"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model=model,
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Priya, a warm and helpful restaurant voice assistant.\n"
                       "You speak in natural Hinglish — a friendly mix of Hindi and English.\n"
                       "You are speaking on a LIVE phone call.\n"
                       "Output raw plain text only. NO markdown, NO emoji, NO asterisks.\n"
                       "Use commas generously for natural pauses.\n"
                       "Keep every sentence short, max 12 words.\n"
                       "Use warm affirmations: 'Haan ji!', 'Bilkul!', 'Sure sure!'\n"
                       "Use polite address: 'aap' not 'tum', 'ji' as suffix.\n"
                       "### CONVERSATION HISTORY ###\n"
                       "{chat_history}"),
            ("human", "{text}")
        ])

    async def __call__(self, state: AgentState):
        import time
        start = time.time()
        intent = state.get("intent", "UNKNOWN")
        text = state["messages"][-1].content
        
        # --- ORDER CONFIRMED CHECK ---
        if state.get("current_order", {}).get("order_confirmed"):
            if intent == "GOODBYE":
                return {"messages": [AIMessage(content="Bahut shukriya ji! Aapka din shubh ho.")]}
            else:
                return {"messages": [AIMessage(content="Aapka order successfully place ho chuka hai ji. Ab usme modifications nahi ho sakte. Agar aapko naya order dena hai toh please call wapas lagayein!")]}
        
        # NO NEGOTIATION: Prices and portions are fixed
        if intent == "NEGOTIATE":
            return {"messages": [AIMessage(content="Sorry ji, hamare prices fixed hain. Main discount nahi de sakti, hope you understand!")]}

        # Rule 2: NO GENERAL QUESTIONS
        if intent == "UNKNOWN":
            return {"messages": [AIMessage(content="Sorry ji, main sirf food ordering mein help kar sakti hoon. Kya order karna chahenge?")]}
        
        # DECLINE_RECOMMENDATION: User doesn't want suggestions — opt out permanently
        if intent == "DECLINE_RECOMMENDATION":
            return {
                "messages": [AIMessage(content="Bilkul ji, koi baat nahi! Aur kuch order karna hai ya checkout karein?")],
                "recommendation_opted_out": True
            }
            
        # GREETING: Only give the full welcome on the FIRST message of the call
        if intent == "GREETING":
            is_first_message = len(state.get("messages", [])) <= 1
            if is_first_message:
                from datetime import datetime, timezone, timedelta
                ist = timezone(timedelta(hours=5, minutes=30))
                hour = datetime.now(ist).hour
                if hour < 12:
                    greeting = "Good morning"
                elif hour < 17:
                    greeting = "Good afternoon"
                else:
                    greeting = "Good evening"
                return {"messages": [AIMessage(content=f"{greeting}! Hamare restaurant mein aapka swagat hai. Aaj kya lena chahenge aap?")]}
            else:
                # Mid-conversation greeting — no welcome line, just redirect
                return {"messages": [AIMessage(content="Haan ji, bataiye kya order karna chahenge?")]}

        # Override for explicit Goodbye intent
        if intent == "GOODBYE":
            return {"messages": [AIMessage(content="Bahut shukriya ji! Aapka din shubh ho. Phir milenge!")]}
            
        chat_history = ""
        if len(state["messages"]) > 1:
            chat_history = "\n".join([f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in state["messages"][-11:-1]])
            
        full_prompt = self.prompt | self.llm
        
        llm_start = time.time()
        response = await full_prompt.ainvoke({
            "chat_history": chat_history,
            "text": text
        })
        llm_duration = time.time() - llm_start
        
        in_tokens = response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        out_tokens = response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
        
        print(f"\033[95m[⌚ LLM] GeneralAgent (Kimi-K2) took {llm_duration:.2f}s | In: {in_tokens} Out: {out_tokens} tokens\033[0m", flush=True)
        
        agent_duration = time.time() - start
        print(f"\033[92m[⌚ TOOL] GeneralAgent total took {agent_duration:.2f}s\033[0m", flush=True)
        return {"messages": [AIMessage(content=response.content)]}
