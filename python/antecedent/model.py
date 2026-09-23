"""Interventional sampling helpers, plus one associational convenience predictor.

``sample_do`` / ``sample_interventional_distribution`` draw from a fitted graphical causal
model under ``do()``. ``predict_conditional_summary`` is different in kind: it fits a lagged
bivariate OLS with no graph and reports ``E[target_t | parent_{t-lag} = level]``. It is a
conditional prediction, not an interventional one.
"""

from __future__ import annotations

from ._native import (
    FittedGcm,
    GcmSampleResult,
    PredictSummary,
    decode_model_bundle,
    encode_model_bundle,
    fit_gcm,
    predict_conditional_summary,
    sample_do,
    sample_interventional_distribution,
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
