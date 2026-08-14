"""Result view dataclasses for :mod:`antecedent.estimation`."""

from . import _html  # noqa: F401 — side effect: attaches _repr_html_ to the views below
from ._views import (
    AnalysisResult,
    ConflictSummaryView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    RefutationReport,
    ValidationView,
)
from .response import (
    CausalResponseView,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
)

__all__ = [
    "IdentificationView",
    "MediationView",
    "EstimateView",
    "ConflictSummaryView",
    "PosteriorView",
    "EffectEnvelope",
    "PredictiveCheckReport",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "PerformanceView",
    "PlanView",
    "PhysicalPlanView",
    "AnalysisResult",
    "CausalResponseView",
    "ResponseEnvelopeView",
    "ResponseUncertainty",
    "ResponseView",
    "ResponseValidationCheck",
    "ResponseValidationView",
    "SupportDiagnostic",
    "SupportReport",
]
