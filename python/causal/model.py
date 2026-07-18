"""Interventional sampling helpers."""

from __future__ import annotations

from ._native import (
    GcmSampleResult,
    sample_do,
    sample_interventional_distribution,
)

__all__ = ["GcmSampleResult", "sample_do", "sample_interventional_distribution"]
