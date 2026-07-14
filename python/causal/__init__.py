"""causal — Python bindings for the causal-library Rust workspace.

Package layout follows DESIGN.md §25.1. Prefer ``causal.analyze(...)`` and the
stage modules (``causal.discovery``, ``causal.estimation``, …); symbols remain
re-exported at the top level for compatibility.
"""

from __future__ import annotations

from . import (
    attribution,
    counterfactual,
    data,
    design,
    discovery,
    estimation,
    graph,
    identification,
    model,
    query,
    state,
    validation,
)
from ._native import (
    AnalysisResult,
    ArrowLoadInfo,
    AteAnalysisResult,
    DiscoveredLink,
    GcmIteResult,
    GcmSampleResult,
    MediationEffectsSummary,
    PcmciDiscoveryResult,
    PosteriorArtifact,
    PredictSummary,
    RpcmciDiscoverySummary,
    analyze,
    analyze_ate,
    decode_posterior_artifact,
    discover_jpcmci_plus,
    discover_lpcmci,
    discover_pcmci,
    discover_pcmci_plus,
    discover_rpcmci,
    encode_posterior_artifact,
    gcm_counterfactual_ite,
    gcm_sample_do,
    load_float64_columns,
    mediation_effects_summary,
    predict_intervened_summary,
    rank_design_eig,
    causal_state_append_demo,
)

__all__ = [
    "AnalysisResult",
    "ArrowLoadInfo",
    "AteAnalysisResult",
    "DiscoveredLink",
    "GcmIteResult",
    "GcmSampleResult",
    "MediationEffectsSummary",
    "PcmciDiscoveryResult",
    "PosteriorArtifact",
    "PredictSummary",
    "RpcmciDiscoverySummary",
    "analyze",
    "analyze_ate",
    "attribution",
    "causal_state_append_demo",
    "counterfactual",
    "data",
    "decode_posterior_artifact",
    "design",
    "discover_jpcmci_plus",
    "discover_lpcmci",
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
    "discovery",
    "encode_posterior_artifact",
    "estimation",
    "gcm_counterfactual_ite",
    "gcm_sample_do",
    "graph",
    "identification",
    "load_float64_columns",
    "mediation_effects_summary",
    "model",
    "predict_intervened_summary",
    "query",
    "rank_design_eig",
    "state",
    "validation",
    "__version__",
]

try:
    from ._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    __version__ = "0.1.0"
