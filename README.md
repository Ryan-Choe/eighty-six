# eighty-six

> The inventory copilot that keeps you from 86'ing the mozzarella.

<!-- hero GIF goes here (Day 5): the PO citing "no weekend delivery" -->

Work in progress — built as a take-home challenge. Sections below get filled in as the
decisions land, not written at the end.

## Quickstart

```
make setup      # python 3.12 venv via uv + deps (torch is big, give it a few minutes)
cp .env.example .env   # then add your ANTHROPIC_API_KEY and LANGSMITH_API_KEY
make seed
make run        # or: make cli
```

## The graph

Generated from the compiled graph with `make graph`, so it can't drift from the code.

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

<!-- state write-map table: field / written by / read by (Day 2, hand-written) -->

## Design notes

<!-- tradeoffs table, rows added the day each decision lands -->

| # | Decision | Alternative I rejected | Why |
|---|---|---|---|

## Evals

<!-- 3 datasets, what each measures, results (Day 4) -->

## What I'd improve

<!-- Day 5 -->
