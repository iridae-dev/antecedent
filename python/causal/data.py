"""Data loading helpers (DESIGN.md §25.1)."""

from __future__ import annotations

from ._native import ArrowLoadInfo, load_float64_columns

__all__ = ["ArrowLoadInfo", "load_float64_columns"]
