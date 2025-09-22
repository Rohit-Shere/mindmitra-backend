from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from backend.chatbot import chat_chain
from backend.memory import MongoCustomChatHistory   # <-- FIX
import os

app = FastAPI(title="Mental Health Chatbot API")

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str
#chat history response model
class ChatHistoryResponse(BaseModel):
    messages: List[dict]

# Response model
class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def home():
    return {"message": "Mental Health Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint with Postgres memory + LangSmith observability.
    """
    # Track per-user thread in LangSmith
    reply = chat_chain.with_config(
        {"configurable": {"thread_id": request.user_id}}
    ).invoke({"user_id": request.user_id, "message": request.message})

    # Ensure the reply is a plain string
    if isinstance(reply, dict) and "reply" in reply:
        reply = reply["reply"]
    elif not isinstance(reply, str):
        reply = str(reply)

    return ChatResponse(reply=reply)

@app.post("/history/{user_id}", response_model=ChatHistoryResponse)
def get_history(user_id: str):
    """
    Fetch full chat history for a user from Postgres.
    """
    memory = MongoCustomChatHistory(connection_string=os.getenv("MONGODB_URI"), user_id=user_id)
    messages = memory._load_messages()

    # Convert to simple dicts {role: ..., content: ...}
    formatted = []
    for m in messages:
        role = "user" if m.type == "human" else "ai"
        formatted.append({"role": role, "content": m.content})

    return ChatHistoryResponse(messages=formatted)