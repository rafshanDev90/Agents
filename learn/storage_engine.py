#L6: Persistent Memory + Injection Risk
'''
The Problem: Storing entire histories in memory will eventually blow up the LLM's context window. 
Furthermore, text passing through tool outputs or user prompts risks Indirect Prompt Injection if not handled cleanly.'''

'''The Fix: We will add a JSON-based persistent storage engine and implement a sliding context window limit to keep histories lean.'''

import os
import json
from session import agentSession

class PersistentSessionManager:
    """Saves and loads sessions to/from disk to prevent data loss."""
    def __init__(self, storage_dir: str = "./sessions"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _get_path(self, session_id: str) -> str:
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def save(self, session: agentSession):
        # Note: In production, serialize OpenAI message objects carefully
        serializable_history = []
        for msg in session.history:
            if hasattr(msg, 'model_dump'):  # Handle ChatCompletionMessage objects
                serializable_history.append(msg.model_dump())
            else:
                serializable_history.append(msg)
                
        data = {"session_id": session.session_id, "history": serializable_history}
        with open(self._get_path(session.session_id), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, session_id: str) -> agentSession:
        path = self._get_path(session_id)
        if not os.path.exists(path):
            return agentSession(session_id=session_id)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        session = agentSession(session_id=data["session_id"])
        session.history = data["history"]
        return session
