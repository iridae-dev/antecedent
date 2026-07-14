"""Estimation / analyze wrappers (DESIGN.md §25.1)."""

from __future__ import annotations

from ._native import AnalysisResult, AteAnalysisResult, analyze, analyze_ate

__all__ = ["AnalysisResult", "AteAnalysisResult", "analyze", "analyze_ate"]
