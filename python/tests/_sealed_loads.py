"""Shared expectation for loading a supported result from bytes alone.

Every supported route executes through a sealed checked operation. Loading its
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


#: Unresolved keys an independent consumer reports for any artifact body that carries
#: refutation reports: the reports have no checked replay, so the claim is not accepted.
REFUTATION_REPLAY_KEYS = (
    "dependencies.refutation_replay",
    "dependencies.scoped_refutation_replay",
)


def carries_refutation_replay(loaded: Any) -> bool:
    return any(key in REFUTATION_REPLAY_KEYS for key in loaded.acceptance.unresolved)


def assert_refutation_replay_withheld(loaded: Any) -> None:
    """An artifact whose body carries refutation reports is recognized but not accepted.

    The reports are decodable and inspectable, but the consumer has no checked replay of
    them, so it names the replay in ``acceptance.unresolved``, is not a verified program,
    and does not keep the recorded answer.
    """
    acceptance = loaded.acceptance
    assert acceptance.recognized, acceptance.details
    assert not acceptance.verified, acceptance.details
    assert not acceptance.replayable
    assert not acceptance.sealed
    assert acceptance.status == "unavailable"
    assert carries_refutation_replay(loaded), acceptance.unresolved
    assert loaded.answer.kind == "unavailable"
    assert repr(loaded).endswith("acceptance=unavailable>")


def assert_loaded_outcome(loaded: Any) -> None:
    """Refutation-bearing bodies are withheld; every other load keeps its recorded answer."""
    if carries_refutation_replay(loaded):
        assert_refutation_replay_withheld(loaded)
    else:
        assert_answer_kept(loaded)
