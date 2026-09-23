"""Assertion helper: a refusal is typed and carries a registered reason code."""

from __future__ import annotations

from antecedent import _native

REGISTERED = frozenset(_native.runtime_refusal_codes())


def assert_registered_refusal(error: BaseException) -> None:
    """``error.reason_code`` is one of the runtime's registered refusal codes."""
    code = getattr(error, "reason_code", None)
    assert code in REGISTERED, (code, str(error))
