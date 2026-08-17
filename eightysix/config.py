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

# Hard bound on one turn of the agent<->tools loop, passed at invoke time by
# every interface (CLI now, Streamlit on Day 4). langgraph 1.x defaults to
# 10007 supersteps -- effectively unbounded, ~5000 model calls on a runaway
# turn. 12 supersteps = ~5 tool rounds, double what a real question needs.
RECURSION_LIMIT = 12


def chat_model(model: str | None = None, max_tokens: int = MAX_TOKENS) -> ChatAnthropic:
    return ChatAnthropic(model=model or MODEL, max_tokens=max_tokens, max_retries=2)
