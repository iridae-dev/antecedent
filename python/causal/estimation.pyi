"""Estimation / analyze wrappers (DESIGN.md §25.1)."""

from ._native import (
    AnalysisResult as AnalysisResult,
    AteAnalysisResult as AteAnalysisResult,
    analyze as analyze,
    analyze_ate as analyze_ate,
)

__all__: list[str]
