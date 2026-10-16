"""Restore earlier items when a later refresh fails, including cancellation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

_T = TypeVar("_T")


def refresh_all_or_restore(
    items: Sequence[_T],
    refresh: Callable[[_T], None],
    restore: Callable[[_T], None],
) -> None:
    """Apply ``refresh`` to every item, or put the touched prefix back.

    A ``BaseException`` (cancellation, keyboard interrupt) must not leave the
    prefix on the new snapshot and the rest on the old one. ``finally`` runs
    for that too; ``except Exception`` does not. A restore failure propagates
    in place of the original error, which is the ``finally`` rule.
    """
    done = 0
    succeeded = False
    try:
        for item in items:
            refresh(item)
            done += 1
        succeeded = True
    finally:
        if not succeeded:
            for item in items[:done]:
                restore(item)
