"""Graph nodes. Each returns a partial state update; LangGraph merges it in."""

import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from eightysix import db, prompts, purchasing, rag
from eightysix.config import ROUTER_MODEL, chat_model, scenario_now
from eightysix.state import AgentState, Intent, POChoice
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
    # a new user turn starts clean: nothing from the previous turn's draft or
    # citations should leak into this one (the interrupt-resume path never
    # re-enters route, so a pending approval is unaffected by this reset)
    fresh = {"low_stock": None, "candidates": None, "po_draft": None,
             "approved": None, "citations": None}
    llm = chat_model(ROUTER_MODEL, max_tokens=1024).with_structured_output(Route)
    try:
        decision = llm.invoke(prompts.ROUTER.format(message=message))
        return {"intent": decision.intent, **fresh}
    except Exception:
        # A router that can't parse shouldn't take the whole turn down. Inventory
        # is the safe default: it's read-only, and a wrong answer there costs a
        # re-ask rather than a purchase order.
        log.exception("router failed to produce a structured intent; defaulting to inventory")
        return {"intent": "inventory", **fresh}


def repair_tool_history(messages: list) -> list:
    """Return history with synthetic results spliced in for tool calls that
    never ran.

    A turn that aborts between the agent and the tool node (the recursion
    budget lands on an agent superstep; a crash mid-tools) checkpoints an
    AIMessage whose tool_calls have no ToolMessage answers. By the next turn
    the new HumanMessage sits AFTER that dangling message, and the Anthropic
    API requires tool results to immediately follow their tool_use -- so the
    repair must splice in place, not append at the end. Runs on every model
    input; a clean history passes through untouched.
    """
    answered = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    repaired: list = []
    for message in messages:
        repaired.append(message)
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            repaired.extend(
                ToolMessage(
                    content="(tool call was interrupted before it ran)",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                for call in message.tool_calls
                if call["id"] not in answered
            )
    return repaired


def inventory_agent(state: AgentState) -> dict:
    """Answer stock questions. The model picks tools; the tools do the counting."""
    llm = chat_model().bind_tools(INVENTORY_TOOLS)
    messages = [SystemMessage(content=prompts.INVENTORY_AGENT),
                *repair_tool_history(state["messages"])]
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


def policy_qa(state: AgentState) -> dict:
    """Answer from the policy docs, with citations from chunk metadata.

    Retrieval is a pipeline step, not a tool: grounding shouldn't depend on
    the model deciding to look.
    """
    question = _last_user_text(state)
    docs = rag.retrieve(question, "policy")
    excerpts = "\n\n".join(
        f"[{d.metadata['source']} § {d.metadata['section']}]\n{d.page_content}"
        for d in docs
    )
    answer = chat_model().invoke(
        prompts.POLICY_QA.format(excerpts=excerpts, question=question)
    )
    # citations come from what was actually retrieved, never parsed from the
    # model's text -- the citation eval can't be gamed by a hallucinated cite
    citations = [{"source": d.metadata["source"], "section": d.metadata["section"]}
                 for d in docs]
    return {"messages": [answer], "citations": citations}


def _format_candidates(candidates: list[dict]) -> str:
    return "\n".join(
        f"- {c['ingredient']}: {c['supplier']} sells {c['pack_desc']} at "
        f"${c['price_cents'] / 100:.2f}; {c['packs_to_par']} pack(s) restores par "
        f"(${c['cost_to_par_cents'] / 100:.2f}); supplier minimum "
        f"${c['supplier_min_order_cents'] / 100:.2f}"
        for c in candidates
    )


def draft_po(state: AgentState) -> dict:
    """Draft a purchase order: Python computes, the model chooses, Python
    re-prices. Two proposal attempts, then honest failure."""
    conn = db.connect()
    try:
        low = db.get_low_stock(conn)
        if not low:
            return {"messages": [AIMessage(
                "Nothing is below its reorder threshold, so there's nothing to "
                "order. Ask me again after the next rush."
            )], "po_draft": None}

        candidates = purchasing.build_candidates(conn, low)

        covered = {c["ingredient"] for c in candidates}
        uncovered = [i["name"] for i in low if i["name"] not in covered]
        if not candidates:
            return {"messages": [AIMessage(
                "Low right now: " + ", ".join(uncovered) + " -- but no tracked "
                "supplier carries them, so there's nothing I can draft. That's "
                "a gap in the supplier catalog, not in the pantry."
            )], "po_draft": None}

        # terms are retrieved per supplier with a source filter, not one
        # blended similarity search -- ranked across all docs, Roma's chunks
        # can outrank a supplier's own terms sheet and bury the pickup trap
        seen, terms_docs = set(), []
        for supplier in sorted({c["supplier"] for c in candidates}):
            for d in rag.retrieve_supplier_terms(supplier):
                key = (d.metadata["source"], d.metadata["section"])
                if key not in seen:
                    seen.add(key)
                    terms_docs.append(d)
        terms = "\n\n".join(
            f"[{d.metadata['source']} § {d.metadata['section']}]\n{d.page_content}"
            for d in terms_docs
        )
        citations = [{"source": d.metadata["source"], "section": d.metadata["section"]}
                     for d in terms_docs]

        low_text = "\n".join(
            f"- {i['name']}: {i['on_hand']}{i['unit']} on hand, threshold "
            f"{i['reorder_threshold']}{i['unit']}, par {i['par_level']}{i['unit']}"
            for i in low
        )
        llm = chat_model().with_structured_output(POChoice)
        retry_note = ""
        for _ in range(2):
            try:
                choice = llm.invoke(prompts.DRAFT_PO.format(
                    now=scenario_now(),
                    request=_last_user_text(state),
                    low_stock=low_text,
                    candidates=_format_candidates(candidates),
                    terms=terms,
                    retry_note=retry_note,
                ))
                if choice is None:
                    raise purchasing.POValidationError("model returned no structured choice")
                po = purchasing.price_po(
                    conn, choice.supplier_name,
                    [line.model_dump() for line in choice.lines],
                )
                break
            except purchasing.POValidationError as e:
                retry_note = (f"\n\nYour previous proposal was rejected: {e}. "
                              "Use only suppliers and items from the options list.")
            except Exception:
                # an API/parse failure mid-draft must not take down the turn --
                # the REPL and (Day 4) the UI keep running either way
                log.exception("draft_po model call failed")
                return {"messages": [AIMessage(
                    "I hit an error drafting the order and don't have a usable "
                    "proposal. Nothing was sent -- try asking again."
                )], "po_draft": None, "citations": citations}
        else:
            return {"messages": [AIMessage(
                "I couldn't put together a valid order from the catalog. "
                "Check the supplier options yourself before ordering."
            )], "po_draft": None, "citations": citations}

        po["expected_delivery"] = choice.expected_delivery
        po["rationale"] = choice.rationale
        fee_note = "" if po["meets_minimum"] else (
            f" (under the ${po['min_order_cents'] / 100:.0f} minimum -- expect a fee)"
        )
        lines_text = "; ".join(
            f"{l['packs']}x {l['pack_desc']} {l['ingredient']}" for l in po["lines"]
        )
        ordered = {l["ingredient"] for l in po["lines"]}
        left_out = [i["name"] for i in low if i["name"] not in ordered]
        left_note = (
            f" Not in this order: {', '.join(left_out)}." if left_out else ""
        )
        summary = AIMessage(
            f"Drafted: {lines_text} from {po['supplier_name']} -- "
            f"${po['subtotal_cents'] / 100:.2f}{fee_note}.{left_note} "
            f"{choice.expected_delivery} {choice.rationale}"
        )
        return {"messages": [summary], "po_draft": po,
                "low_stock": low, "candidates": candidates, "citations": citations}
    finally:
        conn.close()


def human_approval(state: AgentState) -> dict:
    """The graph stops here until a human decides. interrupt() raises,
    checkpoints the thread, and re-runs this node on Command(resume=...)."""
    decision = interrupt({"po_draft": state["po_draft"]})
    return {"approved": bool(decision.get("approved"))}


def send_po(state: AgentState) -> dict:
    """The only side effect in the reorder path, strictly after approval."""
    conn = db.connect()
    try:
        po = state["po_draft"]
        po_id = purchasing.create_and_send_po(conn, po)
        return {"messages": [AIMessage(
            f"Sent PO-{po_id} to {po['supplier_name']} for "
            f"${po['subtotal_cents'] / 100:.2f}. {po['expected_delivery']}"
        )], "po_draft": None}
    finally:
        conn.close()


def cancel_po(state: AgentState) -> dict:
    return {"messages": [AIMessage(
        "Cancelled -- nothing was sent. The draft is gone; ask again if you "
        "change your mind."
    )], "po_draft": None}
