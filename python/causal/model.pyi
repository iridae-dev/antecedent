"""Structural / GCM model helpers (DESIGN.md §25.1)."""

from ._native import (
    GcmSampleResult as GcmSampleResult,
    gcm_sample_do as gcm_sample_do,
    gcm_sample_interventional_distribution as gcm_sample_interventional_distribution,
)

__all__: list[str]
