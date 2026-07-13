"""causal — Python bindings for the causal-library Rust workspace (Phase 0–5).

`analyze_ate` accepts optional `identifier`/`estimator` kwargs to select any of the
Phase 4 identification/estimation pairs (e.g. `estimator="propensity.weighting"`,
`identifier="iv", estimator="iv.2sls"`, `identifier="frontdoor",
estimator="frontdoor.two_stage"`); omitting both preserves the Phase 0–3 default
(`backdoor.adjustment` + `linear.adjustment.ate`).

`discover_pcmci` / `discover_pcmci_plus` accept `ci=` (name string) to select the
conditional-independence test; default is `parcorr`. When `ci="weighted_parcorr"`,
pass observation `weights=` (length = n rows).
"""

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
    discover_pcmci_plus,
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
    "discover_pcmci_plus",
    "load_float64_columns",
    "__version__",
]

try:
    from causal._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    __version__ = "0.1.0"
