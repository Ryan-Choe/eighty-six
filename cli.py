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

from eightysix.config import RECURSION_LIMIT  # noqa: E402
from eightysix.graph import build_graph  # noqa: E402

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

    printed_any = False
    try:
        _stream(graph, question, config)
        printed_any = True
    except GraphRecursionError:
        print("I hit my tool-call budget on that one without landing an answer. "
              "Try a narrower question.")
    if printed_any:
        print()


def _stream(graph, question: str, config: dict) -> None:
    for mode, chunk in graph.stream(
        {"messages": [("user", question)]},
        config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            for node, update in chunk.items():
                if node == "route":
                    print(f"{DIM}[route -> {update.get('intent')}]{RESET}")
                elif node == "inventory_tools":
                    for message in update.get("messages", []):
                        print(f"{DIM}[tool  {message.name}] {message.content[:90]}{RESET}")
        else:
            message_chunk, meta = chunk
            if meta.get("langgraph_node") in ("inventory_agent", "deflect"):
                text = _text(message_chunk.content)
                if text:
                    print(text, end="", flush=True)


def main() -> None:
    graph = build_graph()
    thread_id = str(uuid.uuid4())

    if len(sys.argv) > 1:
        ask(graph, " ".join(sys.argv[1:]), thread_id)
        wait_for_all_tracers()
        return

    print(f"{BOLD}eighty-six{RESET} {DIM}- ctrl-c or 'quit' to exit{RESET}")
    while True:
        try:
            question = input(f"\n{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        ask(graph, question, thread_id)

    wait_for_all_tracers()  # traces flush on a background thread; don't exit first


if __name__ == "__main__":
    main()
