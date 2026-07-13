"""causal — Python bindings for the causal-library Rust workspace."""

from __future__ import annotations

from causal._native import (
    AnalysisResult,
    ArrowLoadInfo,
    AteAnalysisResult,
    DiscoveredLink,
    PcmciDiscoveryResult,
    analyze,
    analyze_ate,
    discover_pcmci,
    load_float64_columns,
)

__all__ = [
    "AnalysisResult",
    "ArrowLoadInfo",
    "AteAnalysisResult",
    "DiscoveredLink",
    "PcmciDiscoveryResult",
    "analyze",
    "analyze_ate",
    "discover_pcmci",
    "load_float64_columns",
    "__version__",
]

try:
    from causal._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    __version__ = "0.1.0"
