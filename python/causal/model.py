"""Structural / GCM model helpers (DESIGN.md §25.1)."""

from __future__ import annotations

from ._native import GcmSampleResult, gcm_sample_do, gcm_sample_interventional_distribution

__all__ = ["GcmSampleResult", "gcm_sample_do", "gcm_sample_interventional_distribution"]
