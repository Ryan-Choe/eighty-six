# Recording script — 2:45 target, 3:00 hard cap

Prep, in order:
1. `make seed` (clean world), restart the Streamlit server so the pinned
   clock loads, `LANGSMITH_PROJECT=eighty-six-demo` in .env for the take.
2. One throwaway question off-camera so nothing lazy-loads on camera.
3. Notifications off, bookmarks bar hidden, 1080p or better.

| time | shot | say |
|---|---|---|
| 0:00 | README hero + diagram | "Eighty-six is an inventory copilot for a pizzeria. One graph, four paths, and a human approval before money moves." |
| 0:15 | Click **Simulate Friday rush**; 86 board flips to 2 flagged | "Friday night's orders just landed. The math is plain Python — no model touches it. Mozzarella and pepperoni just went below threshold." |
| 0:45 | Ask: *do we have enough fresh mozzarella for the weekend?* | "It pulls live numbers through tools, judges the weekend at Friday pace, and says how thin its history is — one day of data." |
| 1:10 | Ask: *we got slammed tonight, draft a reorder for whatever's low* | "Here's the judgment call: Valco is cheaper, but it's pickup only. Roma's Saturday truck wins, and it cites the terms doc for the cutoff." |
| 1:50 | Approval card → **Approve & send** | "The graph is paused on a LangGraph interrupt. Nothing sends until I click. ...Sent." |
| 2:05 | LangSmith: the reorder trace + its resume; expand route → retrieval → interrupt | "Every run is traced. Here's the pause, here's the resume after approval — one thread, two runs." |
| 2:25 | Experiments tab: routing 8/8 both models, qa 5/5 | "Three eval datasets keep it honest. The first run scored 1 out of 5 — every failure was a real bug, and the history shows the fixes landing." |
| 2:40 | README "What I'd improve" | "With more time: forecasting and a real POS webhook. Thanks." |

Retake rule: if a beat stumbles, `make seed`, restart the server, full retake.
Partial takes read as edits.
