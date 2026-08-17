"""Terminal REPL for eighty-six.

This is the development harness first and a fallback demo medium second. The
graph doesn't know or care which interface is driving it, which is the point:
the interrupt on Day 3 lives in the graph, not in the UI.

    python cli.py                      interactive
    python cli.py "how much mozzarella do we have?"   one question and exit
"""

import sys
import uuid

from dotenv import load_dotenv

load_dotenv()  # before anything imports a client

from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402

from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.types import Command  # noqa: E402

from eightysix.config import RECURSION_LIMIT  # noqa: E402
from eightysix.graph import build_graph  # noqa: E402
from eightysix.redaction import redact  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _text(content) -> str:
    """Anthropic streams content as blocks; pull the text out of either shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def ask(graph, question: str, thread_id: str) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
        "run_name": "cli_turn",
        "tags": ["cli"],
        "metadata": {"thread_id": thread_id, "interface": "cli"},
    }

    question = redact(question)  # before invoke: traces record graph inputs verbatim
    try:
        pending_po = _stream(graph, {"messages": [("user", question)]}, config)
        print()
        if pending_po:
            _approve(graph, pending_po, config)
    except GraphRecursionError:
        print("I hit my tool-call budget on that one without landing an answer. "
              "Try a narrower question.")


def _approve(graph, po: dict, config: dict) -> None:
    """The human half of the interrupt. The graph is checkpointed at
    human_approval; whatever we resume with becomes that node's decision."""
    print(f"\n{BOLD}-- PURCHASE ORDER pending approval --{RESET}")
    for line in po["lines"]:
        print(f"  {line['packs']}x {line['pack_desc']:<14} {line['ingredient']:<24} "
              f"${line['line_total_cents'] / 100:>7.2f}")
    print(f"  {'':44}--------")
    fee = "" if po["meets_minimum"] else (
        f"  (under ${po['min_order_cents'] / 100:.0f} minimum)")
    print(f"  {po['supplier_name']:<44}${po['subtotal_cents'] / 100:>7.2f}{fee}")
    print(f"  {DIM}{po['expected_delivery']}{RESET}")

    while True:
        try:
            answer = input(f"{BOLD}[a]pprove / [r]eject:{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "r"
        if answer in {"a", "approve", "r", "reject"}:
            break
    approved = answer.startswith("a")
    _stream(graph, Command(resume={"approved": approved}), config)
    print()


def _stream(graph, graph_input, config: dict) -> dict | None:
    """Stream one graph run. Returns the PO payload if the run paused at the
    approval interrupt, else None."""
    pending_po = None
    for mode, chunk in graph.stream(
        graph_input,
        config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            if "__interrupt__" in chunk:
                pending_po = chunk["__interrupt__"][0].value["po_draft"]
                continue
            for node, update in chunk.items():
                if node == "route":
                    print(f"{DIM}[route -> {update.get('intent')}]{RESET}")
                elif node == "inventory_tools":
                    for message in update.get("messages", []):
                        print(f"{DIM}[tool  {message.name}] {message.content[:90]}{RESET}")
        else:
            message_chunk, meta = chunk
            if meta.get("langgraph_node") in (
                "inventory_agent", "deflect", "policy_qa", "draft_po",
                "send_po", "cancel_po",
            ):
                text = _text(message_chunk.content)
                if text:
                    print(text, end="", flush=True)
    return pending_po


def main() -> None:
    graph = build_graph()
    thread_id = str(uuid.uuid4())

    if len(sys.argv) > 1:
        ask(graph, " ".join(sys.argv[1:]), thread_id)
        wait_for_all_tracers()
        return

    print(f"{BOLD}eighty-six{RESET} {DIM}- ctrl-c or 'quit' to exit{RESET}")
    try:
        while True:
            try:
                question = input(f"\n{BOLD}>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                break
            try:
                ask(graph, question, thread_id)
            except KeyboardInterrupt:
                print("\n(interrupted -- ask again when ready)")
            except Exception as e:  # an API blip must not kill the session
                print(f"\nSomething went wrong ({type(e).__name__}). "
                      "The session is still alive -- try again.")
    finally:
        wait_for_all_tracers()  # traces flush on a background thread; don't exit first


if __name__ == "__main__":
    main()
