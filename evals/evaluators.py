"""Evaluators. Deterministic Python wherever the answer has one right value;
the LLM judge only grades prose quality, and only pass/fail."""

from pydantic import BaseModel

from eightysix.config import chat_model


def correct_route(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "correct_route",
            "score": outputs.get("intent") == reference_outputs["intent"]}


def stock_exact(outputs: dict, reference_outputs: dict) -> dict:
    # the reference pins the ingredients that matter for the case; the target
    # may report more. Every pinned number must match exactly.
    got = outputs.get("stock_after") or {}
    ok = (outputs.get("applied") == reference_outputs["applied"]
          and outputs.get("errors") == reference_outputs["errors"]
          and all(got.get(k) == v for k, v in reference_outputs["stock_after"].items())
          and outputs.get("low_stock") == reference_outputs["low_stock"])
    return {"key": "stock_exact", "score": ok}


def cited_expected_doc(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs.get("expected_doc")
    if not expected:
        return {"key": "cited_expected_doc", "score": None}   # honesty case: no doc to cite
    sources = {c["source"] for c in outputs.get("citations") or []}
    return {"key": "cited_expected_doc", "score": expected in sources}


def reorder_decision(outputs: dict, reference_outputs: dict) -> dict:
    if reference_outputs.get("kind") not in ("reorder",):
        return {"key": "reorder_decision", "score": None}
    drafted = outputs.get("po_draft") is not None
    ok = drafted == reference_outputs["should_draft"]
    want_supplier = reference_outputs.get("supplier")
    if ok and drafted and want_supplier:
        ok = outputs["po_draft"].get("supplier_name") == want_supplier
    return {"key": "reorder_decision", "score": ok}


class Grade(BaseModel):
    passed: bool
    reasoning: str  # one sentence; shows up next to the score in LangSmith


JUDGE_PROMPT = """You are grading an assistant that answers a pizzeria owner's
questions from the restaurant's own documents.

QUESTION: {question}
WHAT A CORRECT ANSWER MUST CONTAIN (any wording; numbers must match exactly):
{facts}
ASSISTANT ANSWER: {answer}

passed=true ONLY if every required element is present and nothing in the
answer contradicts them. If the requirement says the assistant must admit the
docs don't cover something, an invented answer fails. One sentence of
reasoning."""


def answer_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    facts = reference_outputs.get("expected_facts")
    check = reference_outputs.get("judge_check")
    if not facts and not check:
        return {"key": "answer_quality", "score": None}
    requirement = "\n".join(f"- {f}" for f in (facts or []))
    if check:
        requirement += f"\n- {check}"
    judge = chat_model().with_structured_output(Grade)
    grade = judge.invoke(JUDGE_PROMPT.format(
        question=inputs["message"], facts=requirement,
        answer=outputs.get("answer", "")))
    return {"key": "answer_quality", "score": grade.passed, "comment": grade.reasoning}
