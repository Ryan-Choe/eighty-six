"""Model and env configuration — the only place model IDs live."""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

MODEL = os.getenv("EIGHTYSIX_MODEL", "claude-opus-5")
ROUTER_MODEL = os.getenv("EIGHTYSIX_ROUTER_MODEL", "claude-haiku-4-5")

# claude-opus-5 rejects temperature/top_p/top_k with a 400, and thinking is on
# by default — max_tokens caps thinking + response together. 16000 leaves room
# for a long think plus a full answer; a tight cap truncates mid-sentence.
MAX_TOKENS = 16000

# Hard bound on one turn of the agent<->tools loop, passed at invoke time by
# every interface (CLI now, Streamlit on Day 4). langgraph 1.x defaults to
# 10007 supersteps -- effectively unbounded, ~5000 model calls on a runaway
# turn. 12 supersteps = ~5 tool rounds, double what a real question needs.
RECURSION_LIMIT = 12

# The demo story is calendar-dependent: the planted supplier tradeoff only
# bites on a Friday night (Roma's Saturday truck vs Valco's weekday counter).
# Pinning the clock makes the recording reproducible on any real day, and the
# README says so out loud. Unset = real wall clock.
DEMO_NOW = os.getenv("EIGHTYSIX_DEMO_NOW", "")


def scenario_now() -> str:
    from datetime import datetime
    return DEMO_NOW or datetime.now().strftime("%A %B %d, %I:%M %p")


def chat_model(model: str | None = None, max_tokens: int = MAX_TOKENS) -> ChatAnthropic:
    return ChatAnthropic(model=model or MODEL, max_tokens=max_tokens, max_retries=2)
