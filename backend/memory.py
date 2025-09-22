from typing import List
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from pymongo import MongoClient
import os


class MongoCustomChatHistory(BaseChatMessageHistory):
    def __init__(self, connection_string: str, user_id: str):
        self.connection_string = connection_string
        self.user_id = user_id

        # connect
        self.client = MongoClient(self.connection_string)
        self.db = self.client["chatbot_db"]  # database name
        self.collection = self.db["chat_history"]

        # load past history
        self.messages = self._load_messages()

    def _load_messages(self) -> List[BaseMessage]:
        """Load chat history for a user from MongoDB"""
        cursor = self.collection.find({"user_id": self.user_id}).sort("created_at", 1)

        messages = []
        for doc in cursor:
            role, content = doc["role"], doc["content"]
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def add_message(self, message: BaseMessage) -> None:
        role = "system"
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"

        doc = {
            "user_id": self.user_id,
            "role": role,
            "content": message.content,
            "created_at": __import__("datetime").datetime.utcnow(),
        }
        self.collection.insert_one(doc)

        # update in-memory copy
        self.messages.append(message)

    def clear(self) -> None:
        self.collection.delete_many({"user_id": self.user_id})
        self.messages = []
