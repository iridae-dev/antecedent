"""Incremental causal-state helpers."""

from ._native import CausalState as CausalState
from ._native import causal_state_append as causal_state_append

__all__ = ["CausalState", "causal_state_append"]
