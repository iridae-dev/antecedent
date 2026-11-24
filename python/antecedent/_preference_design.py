"""Compile a decision contract with a compiled design of several outcomes."""

from __future__ import annotations

import json
from typing import Any

from .errors import CausalValueError
from .execution import Execution


def is_compiled_design(source: Any) -> bool:
    """Whether ``source`` is a compiled native joint design."""
    from ._joint_panel_designs import PreparedNativeJointDesign

    return isinstance(source, PreparedNativeJointDesign)


def compile_with_design(
    source: Any,
    contract_json: str,
    *,
    refits: int | None,
    seed: int,
    sampling: str | None,
    execution: Execution | None,
) -> tuple[Any, str | None]:
    """Compile the contract with the design's own program; returns ``(compiled, refusal)``.

    ``contract_json`` is a contract across trade-offs, or ``{"ordinary": wire}`` for an
    ordinary contract.

    The design reads its arm means, or with ``refits`` its refits under ``seed``: a
    static-regime design's aligned whole-subject refits, or, with ``sampling`` naming the
    premise that the independent units were sampled independently from a larger population,
    a randomized design's independent-unit bootstrap.
    """
    from ._panel_execution import run_panel

    if sampling is not None and refits is None:
        raise CausalValueError("sampling= declares the premise of the refits; give refits= with it")
    declared: dict[str, Any] = {"replicates": refits, "seed": seed}
    if sampling is not None:
        declared["sampling"] = sampling
    bootstrap = None if refits is None else json.dumps(declared)
    compiled, refusal = run_panel(
        execution,
        source._handle.compile_for_preference_decision,
        contract_json,
        bootstrap,
    )
    return compiled, refusal
