"""Closed-form linear mediation summary under an execution policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _native
from ._native import MediationEffectsSummary
from .execution import Execution, native_controls


def mediation_effects_summary(
    names: list[str],
    columns: Sequence[Any],
    treatment: str,
    mediator: str,
    outcome: str,
    *,
    execution: Execution | None = None,
) -> MediationEffectsSummary:
    """Direct, indirect and total effects of ``treatment`` on ``outcome`` through ``mediator``."""
    ctl = native_controls(execution, seed="seed", threads="threads")
    runner = Execution() if execution is None else execution
    return runner._call(
        _native.mediation_effects_summary,
        names,
        columns,
        treatment,
        mediator,
        outcome,
        **ctl,
    )
