"""Model and env configuration — the only place model IDs live."""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

MODEL = os.getenv("EIGHTYSIX_MODEL", "claude-opus-5")
ROUTER_MODEL = os.getenv("EIGHTYSIX_ROUTER_MODEL", "claude-haiku-4-5")

# claude-opus-5 rejects temperature/top_p/top_k with a 400, and thinking is on
# by default — max_tokens caps thinking + response together, so keep it roomy
MAX_TOKENS = 4096


def chat_model(model: str | None = None, max_tokens: int = MAX_TOKENS) -> ChatAnthropic:
    return ChatAnthropic(model=model or MODEL, max_tokens=max_tokens, max_retries=2)
