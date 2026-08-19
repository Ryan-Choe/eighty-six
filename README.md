# eighty-six

> The inventory copilot that keeps you from 86'ing the mozzarella.

A LangGraph agent for a small pizzeria (the fictional Sal's Slice House). It
watches stock, answers the owner's questions with real numbers, drafts
purchase orders against supplier terms, and stops for a human before any
money moves.

## What it does

- POS order batches deplete stock through a deterministic pipeline. No model
  touches the math.
- The owner asks questions in plain English: current stock, "can we make 12
  margheritas?", food-safety policy, supplier terms.
- When something runs low, the agent drafts a PO: code computes the options,
  the model picks a supplier using retrieved terms, code re-prices the pick,
  and a LangGraph interrupt holds it until a human clicks approve.

## Quickstart

```
make setup        # python 3.12 venv via uv, installs deps (torch is big)
cp .env.example .env    # add ANTHROPIC_API_KEY + LANGSMITH_API_KEY
make seed         # builds the SQLite db and the Chroma index
make run          # Streamlit UI  (or: make cli)
```

Try these, in order: click **Simulate Friday rush**, then ask *"do we have
enough fresh mozzarella for the weekend?"*, then *"draft a reorder for
whatever's low"* and approve it. After `make seed`, `make test` runs 45 tests with no API keys.

## The graph

Generated from the compiled graph with `make graph`, so it can't drift from
the code.

<!-- graph:start -->

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	route(route)
	inventory_agent(inventory_agent)
	inventory_tools(inventory_tools)
	policy_qa(policy_qa)
	draft_po(draft_po)
	human_approval(human_approval)
	send_po(send_po)
	cancel_po(cancel_po)
	deflect(deflect)
	__end__([<p>__end__</p>]):::last
	__start__ --> route;
	draft_po -.-> __end__;
	draft_po -.-> human_approval;
	human_approval -.-> cancel_po;
	human_approval -.-> send_po;
	inventory_agent -.-> __end__;
	inventory_agent -. &nbsp;tools&nbsp; .-> inventory_tools;
	inventory_tools --> inventory_agent;
	route -.-> deflect;
	route -.-> draft_po;
	route -.-> inventory_agent;
	route -.-> policy_qa;
	cancel_po --> __end__;
	deflect --> __end__;
	policy_qa --> __end__;
	send_po --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

<!-- graph:end -->

`human_approval` is where the graph stops: `interrupt()` checkpoints the
thread mid-run and nothing moves until the interface resumes it with an
approve/reject decision. The pause is invisible in the generated diagram —
it happens inside the node at runtime, not at an edge.

### State

| field | written by | read by |
|---|---|---|
| `messages` | every answering node (append reducer) | everything; the UI streams from it |
| `intent` | `route` | the branch picker |
| `low_stock`, `candidates` | `draft_po` (cleared by `route` each turn) | the drafting prompt |
| `po_draft` | `draft_po` (cleared by `route`, `send_po`, `cancel_po`) | approval card, `send_po` |
| `approved` | `human_approval`, from the resume payload | the send/cancel branch |
| `citations` | `policy_qa`, `draft_po` (reset each turn) | the UI's sources list |

## Design notes

| # | Decision | Alternative I rejected | Why |
|---|---|---|---|
| 1 | Owner-facing copilot | A CRUD inventory tracker | Delete the LLM from a tracker and nothing breaks. Here the judgment calls are the product |
| 2 | Order ingestion outside the graph | A graph node for order batches | Nobody chats a JSON file. The graph is a decision surface, not a job runner |
| 3 | The model never does arithmetic | LLM computes quantities and totals | `price_po()` re-prices every draft from the catalog; model output is a proposal |
| 4 | SQL for live state, RAG for prose | RAG over everything | If it fits in a WHERE clause it doesn't belong in a vector store. Columns win on conflict |
| 5 | Explicit agent + ToolNode loop | `create_react_agent` | ~20 extra lines and the loop shows up in the diagram above |
| 6 | `interrupt()` before `send_po` | Fully autonomous ordering | Money leaves the building. Durable pause/resume is the reason to use LangGraph at all |
| 7 | Redaction before `graph.invoke()` | A redaction node in the graph | Traces and checkpoints record graph inputs verbatim, so an in-graph redactor leaks by design |
| 8 | Mostly deterministic evaluators | LLM-as-judge for everything | Exact match where an exact answer exists; the judge only grades prose, and only pass/fail |
| 9 | `InMemorySaver` checkpointer | `SqliteSaver` | One less dependency for a demo. Durable threads are on the improve list |
| 10 | Local MiniLM embeddings | A hosted embedding API | You need two API keys to run this, not three |
| 11 | Haiku router, Opus everywhere else | Opus for routing too | The routing eval scores 8/8 on both models. Same accuracy, ~5x cheaper per route |
| 12 | Makefile, no Docker | A Dockerfile | The brief says "or". A multi-GB torch image the reviewer must mount keys into buys nothing over `make setup` |

Three things the evals changed after the fact: the usage tool divided by the
7-day window instead of days-with-data and reported 9.7 days of mozzarella
cover when the truth was 1.4; the markdown splitter strips headers out of
chunk text, so Roma's "$250 minimum" chunk didn't contain the word Roma and
three suppliers looked identical to the embeddings (fixed by re-injecting the
doc title into each chunk); and the supplier tradeoff turned out to be
calendar-dependent, so the demo runs on a pinned clock (`EIGHTYSIX_DEMO_NOW`)
— on a real Sunday the agent picked the "wrong" supplier and was right to.

## Evals

Three datasets, 18 examples, committed as JSON in `evals/datasets/` and run
with `make eval`. Routing is exact match: 8/8 on Haiku and 8/8 on Opus, which
is why the router runs on Haiku. Inventory math is exact match against
numbers verified by hand before they became the reference: 3/3. The
full-graph set (answers judged pass/fail, citations checked against retrieved
metadata, reorder decisions checked against the paused draft): 5/5, 3/3, 3/3.
The first run scored 1/5 on answers and every failure was a real bug; the
experiment history in LangSmith keeps that progression visible on purpose.

## LangSmith

Every run is traced. The `eighty-six-demo` project holds a curated set of
named runs (`make demo`), including the pair that matters: a reorder that pauses at the
approval interrupt, and its resume after approval. Share links live in
[docs/notes.md](docs/notes.md), along with setup detail and the traps we hit.

## What I'd improve

- Demand forecasting: reorders react to a threshold today; even a 7-day
  moving average would let them anticipate the weekend instead.
- A real POS webhook with idempotency keys, replacing batch JSON files.
- Retrieval-stage evals (recall@k against a golden query set) — today only
  end answers are graded, so a retrieval regression shows up indirectly.
- Multi-supplier orders: one PO per draft right now, so a mixed shortage
  bundles into one vendor or says what it left out.
- A way to ask "what did I already order?" — sent POs are written but nothing
  reads them back yet.
- Durable threads (`SqliteSaver`) so an approval survives a process restart.

## Repo map

```
eightysix/       the package: graph, nodes, state, tools, prompts, db (the
                 SQL), rag, purchasing (all money math), ingest, redaction
app.py / cli.py  two thin interfaces over the same graph
data/seed        the world: ingredients, recipes, suppliers
data/pos_orders  one Friday rush of POS orders
data/kb          five markdown docs the vector store indexes
evals/           datasets as JSON + evaluators + runners
scripts/         seed, ingest, diagram regen, demo curation, day-1 smoke test
docs/            setup notes, traps we hit, the demo script
tests/           45 tests, no API keys needed
```
