# Notes: setup detail and traps

Things that bit us during the build, kept out of the README to keep it short.

## Share links

Traces (public, read-only):

- Reorder draft, paused at the approval interrupt:
  https://smith.langchain.com/public/0250b7d8-1b67-40f6-947c-2d55336d07d7/r
- The resume after approval (same thread, second trace):
  https://smith.langchain.com/public/1bcb0f4d-2a47-415f-bd54-8e5025db0070/r
- Weekend judgment question (tools + thin-history caveat):
  https://smith.langchain.com/public/5d607c53-3723-414d-af4e-497f82bdae0d/r
- Policy answer with citations:
  https://smith.langchain.com/public/3be41de3-ad42-437b-aef4-f54c16fdf620/r

Datasets with their experiment history (the 1/5 -> 5/5 progression is in
qa-reorder on purpose):

- Routing (8/8 on Haiku and on Opus):
  https://smith.langchain.com/public/18e83381-6ad2-4495-a27e-5e0e24f5f3ac/d
- QA + reorder judgment:
  https://smith.langchain.com/public/e4627628-1376-456e-b30b-1fdda9854393/d
- Inventory math:
  https://smith.langchain.com/public/ea5276e9-aa7a-4e10-8322-7f846dc5d6d9/d

## Environment

- Python 3.12 on purpose. 3.13+ still has wheel gaps in this stack's
  transitive deps.
- `requirements.txt` shows intent; `requirements.lock` is what actually ran.
- First `make seed` downloads the MiniLM embedding model (~90MB). After that,
  `HF_HUB_OFFLINE=1` silences hub checks entirely.
- `EIGHTYSIX_DEMO_NOW` pins the scenario clock for the PO prompt. The
  supplier tradeoff is calendar-dependent: on a real Sunday, Valco's Monday
  counter genuinely beats Roma's Tuesday truck, and the agent will say so.
- Usage history windows on the real clock, not the pinned one, so
  `ingest_file` rebases the order file's timestamps to ingest time (relative
  spacing kept). The weekend question works whenever you clone this.

## Traps we hit (each cost real time)

1. `claude-opus-5` rejects `temperature` / `top_p` / `top_k` with a 400, and
   thinking is on by default — `max_tokens` caps thinking plus response, so a
   tight cap truncates mid-sentence.
2. langgraph 1.x defaults `recursion_limit` to 10007. That is not a limit.
   We pass 12 at invoke time from both interfaces.
3. langgraph 1.x `ToolNode` only converts schema errors to messages; a real
   exception in a tool kills the turn unless you pass `handle_tool_errors`.
4. A turn that dies between the agent and its tools checkpoints a dangling
   tool call. The repair has to splice the synthetic result immediately after
   the broken call — the API rejects it anywhere else — and has to scan the
   whole history, because by the next turn the new question sits after it.
5. The markdown header splitter removes headers from chunk text. Chunks that
   read fine to a human embed as anonymous lookalikes. Re-inject the doc
   title and section into the text at ingest.
6. Streamlit reruns the whole script per interaction: the compiled graph
   lives behind `st.cache_resource`, the pending approval in session state.
   And `st.markdown` renders `$...$` as LaTeX — escape dollars in anything
   a model wrote.
7. SQLite `COLLATE NOCASE` after `IN (...)` binds to the IN result, not the
   column. Compare `lower()` on both sides instead.
8. Piping python output through `tail` in a Makefile hides non-zero exit
   codes. We lost a real ImportError to that once.
9. You can't pin temperature on claude-opus-5 (it rejects the parameter), so
   an engineered scenario with two defensible answers is a coin flip at
   runtime. Our supplier trap first shipped with Valco's counter open on
   Saturdays -- the agent alternated between two correct-sounding drafts.
   Determinism has to come from the world: Valco closed weekends, one
   dominant answer, verified three-for-three.

## Running the evals

`make eval` uploads datasets (idempotent) and runs three experiments. Targets
never touch `pizzeria.db` — each example builds its own throwaway database.
The qa-reorder target runs the full graph to the interrupt and grades the
paused draft, so no eval ever "approves" anything.
