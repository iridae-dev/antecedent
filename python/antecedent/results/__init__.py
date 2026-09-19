"""Result view dataclasses for :mod:`antecedent.estimation`."""

from typing import TypeAlias

from . import _html  # noqa: F401 — side effect: attaches _repr_html_ to the views below
from ._execution import Answer, CalibrationInfo
from ._report import InspectionReport
from ._slots import ReasoningSlots, SlotView
from ._views import (
    AnalysisResult,
    ConflictSummaryView,
    DistributionAtomView,
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
    ProbabilityIntervalView,
    RefutationReport,
    TemporalMediationGridView,
    TemporalMediationSliceView,
    ValidationView,
)
from .response import (
    CausalResponseView,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SimultaneousBand,
    SupportDiagnostic,
    SupportReport,
)

#: Shared analyze() result. Both classes implement :class:`ResultAPI`
#: (``answer``, ``claim``, ``inspect``, ``as_point``, ``as_response``).
Analysis: TypeAlias = AnalysisResult | CausalResponseView

__all__ = [
    "Analysis",
    "Answer",
    "CalibrationInfo",
    "InspectionReport",
    "ReasoningSlots",
    "SlotView",
    "IdentificationView",
    "MediationView",
    "TemporalMediationGridView",
    "TemporalMediationSliceView",
    "EstimateView",
    "ProbabilityIntervalView",
    "DistributionAtomView",
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
    "SimultaneousBand",
    "SupportDiagnostic",
    "SupportReport",
]
