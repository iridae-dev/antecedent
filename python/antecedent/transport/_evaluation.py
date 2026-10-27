"""Declared finite transport evaluation bounds; physical policy is Execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CausalTypeError
from ..execution import _integer


@dataclass(frozen=True, slots=True)
class TransportEvaluation:
    """Bounds on formula evaluation and finite support enumeration.

    These bounds never truncate a successful scientific answer. Exhaustion is
    reported by the native evaluator. Memory and cancellation belong to Execution.
    """

    max_operations: int = 10_000_000
    max_depth: int = 256
    max_support_rows: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("max_operations", "max_depth", "max_support_rows"):
            value = getattr(self, name)
            if value is None:
                raise CausalTypeError(f"{name} must be an integer")
            _integer(value, name)


def require_evaluation(value: TransportEvaluation) -> None:
    if not isinstance(value, TransportEvaluation):
        raise CausalTypeError("evaluation must use transport.TransportEvaluation")
