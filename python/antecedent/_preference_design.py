"""Compile a decision across preferences with a compiled design of several outcomes."""

from __future__ import annotations

import json
from typing import Any

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
    execution: Execution | None,
) -> tuple[Any, str | None]:
    """Compile the contract with the design's own program; returns ``(compiled, refusal)``.

    The design reads its arm means, or with ``refits`` its aligned whole-subject refits
    under ``seed``.
    """
    from ._panel_execution import run_panel

    bootstrap = None if refits is None else json.dumps({"replicates": refits, "seed": seed})
    compiled, refusal = run_panel(
        execution,
        source._handle.compile_for_preference_decision,
        contract_json,
        bootstrap,
    )
    return compiled, refusal
