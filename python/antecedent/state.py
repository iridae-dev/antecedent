"""Incremental causal state helpers."""

from __future__ import annotations

from ._native import CausalState, antecedent_state_append

__all__ = ["CausalState", "antecedent_state_append"]
