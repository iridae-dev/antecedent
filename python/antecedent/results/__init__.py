"""Result view dataclasses for :mod:`antecedent.estimation`."""

from typing import TypeAlias

from . import _html  # noqa: F401 — side effect: attaches _repr_html_ to the views below
from ._execution import Answer, CalibrationInfo
from ._plan_view import ProgramPlanView
from ._report import InspectionReport
from ._slots import ReasoningSlots, SlotView
from ._views import (
    AnalysisResult,
    ConflictSummaryView,
    DistributionAtomView,
    DropReasonView,
    EffectEnvelope,
    EstimateView,
    GroupAccountingView,
    IdentificationView,
    MediationView,
    ImputationExtraView,
    ImputationVectorView,
    ImputationView,
    ImputationWaldView,
    MissingDataView,
    MissingVariableView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    ProbabilityIntervalView,
    RefutationReport,
    RowAccountingView,
    SamplingWeightsView,
    TemporalMediationGridView,
    TemporalMediationSliceView,
    ValidationView,
)
from .coordinates import response_coordinates
from .provider import ProviderAnalysisResult
from .response import (
    CausalResponseView,
    IntervalInterpretation,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SimultaneousBand,
    SupportDiagnostic,
    SupportReport,
    SupportStatus,
    UncertaintyKind,
)
from .sources import MeanSource

#: Shared analyze() result. Both classes implement :class:`ResultAPI`
#: (``answer``, ``claim``, ``inspect``, ``as_point``, ``as_response``).
Analysis: TypeAlias = AnalysisResult | CausalResponseView | ProviderAnalysisResult

__all__ = [
    "MeanSource",
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
    "SamplingWeightsView",
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
    "ProgramPlanView",
    "MissingVariableView",
    "DropReasonView",
    "GroupAccountingView",
    "RowAccountingView",
    "ImputationWaldView",
    "ImputationExtraView",
    "ImputationVectorView",
    "ImputationView",
    "MissingDataView",
    "AnalysisResult",
    "ProviderAnalysisResult",
    "CausalResponseView",
    "ResponseEnvelopeView",
    "ResponseUncertainty",
    "ResponseView",
    "ResponseValidationCheck",
    "ResponseValidationView",
    "SimultaneousBand",
    "SupportDiagnostic",
    "SupportReport",
    "SupportStatus",
    "UncertaintyKind",
    "IntervalInterpretation",
    "response_coordinates",
]
