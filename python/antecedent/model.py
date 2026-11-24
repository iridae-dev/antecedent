"""Interventional sampling helpers, plus one associational convenience predictor.

``sample_do`` / ``sample_interventional_distribution`` draw from a fitted graphical causal
model under ``do()``. ``predict_conditional_summary`` is different in kind: it fits a lagged
bivariate OLS with no graph and reports ``E[target_t | parent_{t-lag} = level]``. It is a
conditional prediction, not an interventional one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _native, _retained
from ._native import (
    GcmSampleResult,
    PredictSummary,
    encode_model_bundle,
    predict_conditional_summary,
)
from ._native import decode_model_bundle as _decode_model_bundle
from .counterfactual import FittedGcm, fit_gcm
from .execution import Execution, native_controls


def decode_model_bundle(
    data: bytes | list[int], *, retained_identity: str | None = None
) -> tuple[list[str], list[tuple[int, int]], int]:
    """Decode a model bundle: ``(variable_names, edges, n_mechanisms)``.

    ``retained_identity`` is :func:`antecedent.artifacts.artifact_identity` of the bundle the
    consumer kept when it was written; other bytes are refused with ``external_binding_mismatch``.
    """
    raw = bytes(data)
    _retained.admit(
        raw,
        retained_identity,
        "antecedent.model.decode_model_bundle",
        "antecedent.model.decode_model_bundle(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
    )
    return _decode_model_bundle(raw, _retained.native(retained_identity))


def sample_do(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    *,
    mechanism_wrappers: dict[str, Any] | None = None,
    shift: bool = False,
    execution: Execution | None = None,
) -> GcmSampleResult:
    """Draw ``n_draws`` rows from a model fitted here under ``do(treatment = do_value)``."""
    ctl = native_controls(execution, seed="seed", threads="threads")
    runner = Execution() if execution is None else execution
    return runner._call(
        _native.sample_do,
        names,
        columns,
        edges,
        treatment,
        do_value,
        n_draws,
        mechanism_wrappers=mechanism_wrappers,
        shift=shift,
        **ctl,
    )


def sample_interventional_distribution(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    outcome: str | None = None,
    *,
    shift: bool = False,
    execution: Execution | None = None,
) -> GcmSampleResult:
    """The interventional distribution of ``outcome`` under ``do(treatment = do_value)``."""
    ctl = native_controls(execution, seed="seed", threads="threads")
    runner = Execution() if execution is None else execution
    return runner._call(
        _native.sample_interventional_distribution,
        names,
        columns,
        edges,
        treatment,
        do_value,
        n_draws,
        outcome,
        shift=shift,
        **ctl,
    )


__all__ = [
    "FittedGcm",
    "GcmSampleResult",
    "PredictSummary",
    "decode_model_bundle",
    "encode_model_bundle",
    "fit_gcm",
    "predict_conditional_summary",
    "sample_do",
    "sample_interventional_distribution",
]
