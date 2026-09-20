"""One claim sentence. Identification.statement and result.claim() share this."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._verdict import describe_status, verdict_for


def query_phrase(query: object) -> str:
    name = type(query).__name__
    treatment = getattr(query, "treatment", None)
    outcome = getattr(query, "outcome", None)
    if isinstance(treatment, str) and isinstance(outcome, str):
        mediators = getattr(query, "mediators", None)
        if mediators:
            via = ", ".join(str(item) for item in mediators)
            return f"{name} of {outcome} from {treatment} via {via}"
        modifier = getattr(query, "modifier", None)
        if isinstance(modifier, str):
            return f"{name} of {outcome} from {treatment} given {modifier}"
        return f"{name} of {outcome} from {treatment}"
    return name


def identification_statement(
    query: object,
    status: str,
    method: str | None,
    adjustment_set: Sequence[str],
) -> str:
    """One-sentence identification state. Data, not a display hook."""
    query_text = query_phrase(query)
    verdict = verdict_for(status)
    if verdict == "not identified":
        return f"{query_text} is not identified."
    if verdict == "graph-dependent":
        phrase = f"{query_text} is graph-dependent, not a single identified effect"
    else:
        phrase = f"{query_text} is {describe_status(status)}"
    method_text = method.strip() if method else ""
    if method_text and method_text.lower() not in {"none", "unavailable"}:
        phrase = f"{phrase} by {method_text}"
    if adjustment_set:
        phrase = f"{phrase}, adjusting for {', '.join(adjustment_set)}"
    return f"{phrase}."


def result_claim(
    *,
    query: object | None,
    status: str,
    method: str | None,
    adjustment_set: Sequence[str],
    answer: Any,
    calibration: str | None,
) -> str:
    """Identification sentence plus the executed answer and calibration line."""
    query_obj = query if query is not None else "this query"
    sentence = identification_statement(query_obj, status, method, adjustment_set)
    bits = [sentence.rstrip(".")]
    kind = getattr(answer, "kind", None)
    if kind == "point" and getattr(answer, "value", None) is not None:
        bits.append(f"answer is {answer.value:g}")
    elif kind == "bounds" and getattr(answer, "bounds", None) is not None:
        lo, hi = answer.bounds
        bits.append(f"answer is the identified set [{lo:g}, {hi:g}]")
    elif kind == "response":
        bits.append("answer is a function-valued response")
    elif kind == "structured":
        bits.append("answer is structured")
    elif kind == "partial":
        detail = getattr(answer, "detail", None)
        bits.append(f"answer is withheld ({detail})" if detail else "answer is withheld")
    elif kind == "unavailable":
        detail = getattr(answer, "detail", None)
        bits.append(f"no claim ({detail})" if detail else "no claim")
    if calibration:
        bits.append(f"calibration {calibration}")
    return f"{'; '.join(bits)}."
