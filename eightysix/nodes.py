"""Graph nodes. Each returns a partial state update; LangGraph merges it in."""

import logging

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from eightysix import prompts
from eightysix.config import ROUTER_MODEL, chat_model
from eightysix.state import AgentState, Intent
from eightysix.tools import INVENTORY_TOOLS

log = logging.getLogger(__name__)


class Route(BaseModel):
    """Forced shape for the router's answer."""

    intent: Intent = Field(description="The single best intent for this message.")


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            content = message.content
            # Anthropic can return content as a list of blocks; flatten defensively
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return content
    return ""


def route(state: AgentState) -> dict:
    """Classify the turn. Haiku, one structured call, no tools.

    Cheap and exactly gradeable: the eval on Day 4 is an exact-match against
    this one field.
    """
    message = _last_user_text(state)
    llm = chat_model(ROUTER_MODEL, max_tokens=1024).with_structured_output(Route)
    try:
        decision = llm.invoke(prompts.ROUTER.format(message=message))
        return {"intent": decision.intent}
    except Exception:
        # A router that can't parse shouldn't take the whole turn down. Inventory
        # is the safe default: it's read-only, and a wrong answer there costs a
        # re-ask rather than a purchase order.
        log.exception("router failed to produce a structured intent; defaulting to inventory")
        return {"intent": "inventory"}


def inventory_agent(state: AgentState) -> dict:
    """Answer stock questions. The model picks tools; the tools do the counting."""
    llm = chat_model().bind_tools(INVENTORY_TOOLS)
    messages = [SystemMessage(content=prompts.INVENTORY_AGENT), *state["messages"]]
    return {"messages": [llm.invoke(messages)]}


def deflect(state: AgentState) -> dict:
    """Canned reply for off-topic turns and for paths that land on Day 3.

    No model call — the router already made the only decision needed.
    """
    intent = state.get("intent")
    if intent == "off_topic":
        text = prompts.DEFLECT_OFF_TOPIC
    else:
        text = prompts.DEFLECT_NOT_BUILT.format(intent=intent)
    return {"messages": [AIMessage(content=text)]}
