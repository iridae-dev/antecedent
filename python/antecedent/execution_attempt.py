"""Diagnostic work observations around one original synchronous native execution.

This observer invokes its callable exactly once. It records only instrumented native
component operations reached on the calling thread and original joined workers with a
propagated observation token. It does not monitor arbitrary Python, remote services or
independently created threads, and never retries, alters a scientific result, issues
execution authority or converts failed work into successful receipt counts. A numerical
solve and its enclosing fitted model are separate operation kinds, not additive fits.
A completed component invocation can return an unidentified scientific answer; it does
not establish scientific support. Cached failure reads are distinguished from new solves.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from ._native import observe_native_attempts as _observe

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class OperationCounts:
    """Begun, successful, failed and unfinished work for one original component kind."""

    attempted: int
    completed: int
    failed: int
    unfinished: int


@dataclass(frozen=True, slots=True)
class AttemptReport:
    """Fixed-kind observation, independent of scientific authority and successful receipts."""

    operations: Mapping[str, OperationCounts]
    is_empty: bool
    is_complete: bool
    scope: str


@dataclass(frozen=True, slots=True)
class NativeAttempt(Generic[T]):
    """Original return value or original exception, plus separately observed native work."""

    value: T | None
    error: BaseException | None
    report: AttemptReport

    def unwrap(self) -> T:
        """Return the original value, or raise the original exception after inspection."""
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]


def observe_native_attempts(operation: Callable[[], T]) -> NativeAttempt[T]:
    """Invoke once and retain original value/error with actual component observations.

    Complete reuse and refusals before entering an instrumented component have zero work.
    Cancellation inside an entered component counts an unsuccessful operation; the original
    exception retains its actual cancellation/refusal code. Nested observations each see the
    same original operations once. Reports have no calibrated or resumable-state standing.
    """
    value, error, data = _observe(operation)
    wire = json.loads(data)
    return NativeAttempt(
        value,
        error,
        AttemptReport(
            MappingProxyType(
                {key: OperationCounts(**row) for key, row in wire["operations"].items()}
            ),
            wire["is_empty"],
            wire["is_complete"],
            wire["scope"],
        ),
    )


__all__ = ["OperationCounts", "AttemptReport", "NativeAttempt", "observe_native_attempts"]
