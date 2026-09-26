"""Shared expectation for loading a licensed result from bytes alone.

Every licensed route executes through a sealed checked operation. Loading its
export either replays a verified program (``acceptance.verified``) or, when the
consumer cannot re-execute the sealed operation without the retained proof,
recognizes the artifact, verifies its contract and identities, keeps the recorded
answer, and names the checked operation in ``acceptance.unresolved``
(``acceptance.sealed``). Both keep the recorded answer; only the former is
replayable.
"""

from __future__ import annotations

from typing import Any

from antecedent._workflow import is_sealed_dependency


def assert_answer_kept(loaded: Any) -> None:
    """The load is verified or sealed, and its recorded answer is available."""
    acceptance = loaded.acceptance
    assert acceptance.recognized, acceptance.details
    assert acceptance.verified or acceptance.sealed, acceptance.details
    if acceptance.verified:
        assert acceptance.replayable
        assert acceptance.status == "verified"
    else:
        assert not acceptance.replayable
        assert acceptance.status == "sealed"
        assert acceptance.unresolved, acceptance.details
        assert all(map(is_sealed_dependency, acceptance.unresolved)), acceptance.unresolved
    assert loaded.answer.kind != "unavailable", acceptance.details
    assert repr(loaded).endswith(f"acceptance={acceptance.status}>")
