"""Day-1 gate: the smallest possible graph, invoked once, trace visible in
LangSmith. If this doesn't produce a trace, fix the wiring today — not on
demo day."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, MessagesState, START, END

from eightysix.config import chat_model


def hello(state: MessagesState):
    response = chat_model().invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState).add_node("hello", hello) \
    .add_edge(START, "hello").add_edge("hello", END).compile()

if __name__ == "__main__":
    run_id = uuid.uuid4()
    result = graph.invoke(
        {"messages": [("user", "Say hello to Sal's Slice House in one sentence.")]},
        config={
            "run_id": run_id,
            "run_name": "hello_graph",
            "tags": ["day1-gate"],
            "metadata": {"purpose": "langsmith wiring check"},
        },
    )
    print(result["messages"][-1].content)
    print(f"\nrun_id: {run_id} — now find this trace in the LangSmith project.")

    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
