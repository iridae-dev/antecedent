"""causal — Python bindings for the causal-library Rust workspace.

Phase 9 adds `discover_jpcmci_plus`, `discover_rpcmci`, `mediation_effects_summary`,
and `predict_intervened_summary` (one GIL crossing each).
Phase 6 exit: `decode_posterior_artifact` / `encode_posterior_artifact` and
`AteAnalysisResult.posterior_artifact` bytes.
"""

from __future__ import annotations

from causal._native import (
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
    "decode_posterior_artifact",
    "discover_jpcmci_plus",
    "discover_lpcmci",
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
    "encode_posterior_artifact",
    "gcm_counterfactual_ite",
    "gcm_sample_do",
    "load_float64_columns",
    "mediation_effects_summary",
    "predict_intervened_summary",
    "__version__",
]

try:
    from causal._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    __version__ = "0.1.0"
