"""Chat-input redaction. Runs BEFORE graph.invoke, never inside the graph:
traces and checkpoints capture graph inputs verbatim, so an in-graph redactor
(or an LLM one, which must be SENT the data) leaks by construction.

POS order files never need this -- ingestion allowlists three fields and the
customer data is simply never read. This covers the other door: the owner
pasting an order confirmation or a customer text into chat.
"""

import re

PHONE = re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def redact(text: str) -> str:
    text = PHONE.sub("[PHONE]", text)
    text = EMAIL.sub("[EMAIL]", text)
    return text
