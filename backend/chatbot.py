from backend.memory import MongoCustomChatHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

def get_chat_response(user_id: str, user_message: str) -> str:
    memory = MongoCustomChatHistory(connection_string=DB_URL, user_id=user_id)

    # Save human input
    memory.add_message(HumanMessage(content=user_message))

    # Optional: define system message for LLM
    system_message = SystemMessage(
        content=(
            "You are a compassionate and professional mental health assistant and therapist."
            " Your role is to listen attentively, understand emotions, and respond empathetically. "
            "Always prioritize the users emotional well-being, offer supportive guidance, and provide a safe, non-judgmental space for sharing thoughts and feelings."
            " Use a warm, understanding, and gentle tone. Validate the users emotions before suggesting coping strategies or resources. "
            "Avoid giving medical diagnoses or treatments; instead, focus on active listening, emotional support, encouragement, and practical mental wellness advice."
        )
    )
    
    # Load conversation
    messages = memory._load_messages()

    # AI response
    ai_response = chat_model.invoke(messages)

    # Save AI output
    memory.add_message(ai_response)

    return ai_response.content

# Runnable for LangSmith tracking
chat_chain = RunnableLambda(
    lambda x: get_chat_response(x["user_id"], x["message"])
)
