"""Shared expectation for loading a supported result from bytes alone.

Every Antecedent artifact is intact when it loads, and loading keeps the recorded answer
without recomputing anything. A replay is asked for with ``loaded.replay()``; it recomputes
what the artifact carries evidence for and names, in ``acceptance.not_replayable``, the parts
whose evidence it does not carry, such as a checked operation whose proof stays with the live
study.
"""

from __future__ import annotations

from typing import Any


def assert_answer_kept(loaded: Any) -> None:
    """The load is intact, nothing was replayed, and its recorded answer is available."""
    acceptance = loaded.acceptance
    assert acceptance.recognized, acceptance.details
    assert acceptance.status == "loaded", acceptance.details
    assert not acceptance.replayed
    assert not acceptance.integrity, acceptance.integrity
    assert loaded.answer.kind != "unavailable", acceptance.details
    assert repr(loaded).endswith(f"acceptance={acceptance.status}>")
    replayed = loaded.replay().acceptance
    assert not replayed.integrity, replayed.integrity
    assert not replayed.mismatches, replayed.mismatches
    assert replayed.replayed == (not replayed.not_replayable), replayed.details


#: Replay findings an independent consumer reports for any artifact body that carries
#: refutation reports: the reports have no checked replay.
REFUTATION_REPLAY_KEYS = (
    "dependencies.refutation_replay",
    "dependencies.scoped_refutation_replay",
)


def carries_refutation_replay(loaded: Any) -> bool:
    return any(key in REFUTATION_REPLAY_KEYS for key in loaded.replay().acceptance.not_replayable)


def assert_refutation_replay_withheld(loaded: Any) -> None:
    """An artifact whose body carries refutation reports keeps its answer on load, but its
    replay is withheld: the reports have no checked replay, so the replay names them as not
    replayable and the result is not reported as replayed.
    """
    acceptance = loaded.acceptance
    assert acceptance.recognized, acceptance.details
    assert acceptance.status == "loaded", acceptance.details
    assert loaded.answer.kind != "unavailable"
    replayed = loaded.replay().acceptance
    assert not replayed.replayed
    assert not replayed.integrity and not replayed.mismatches, replayed.details
    assert any(key in REFUTATION_REPLAY_KEYS for key in replayed.not_replayable), (
        replayed.not_replayable
    )


def assert_loaded_outcome(loaded: Any) -> None:
    """Every load keeps its recorded answer; a refutation-bearing body has its replay withheld."""
    if carries_refutation_replay(loaded):
        assert_refutation_replay_withheld(loaded)
    else:
        assert_answer_kept(loaded)
