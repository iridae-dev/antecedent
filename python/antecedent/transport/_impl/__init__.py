"""Single-source graphical transportability specifications.

This namespace describes structural population differences. It is distinct
from statistical prior/evidence transport in :mod:`antecedent.priors`.
"""

from __future__ import annotations

from ..._transport_results import (
    TransportContrast,
    TransportGridPoint,
    TransportUncertainty,
)
from .authority import (
    _no_native_authority as _no_native_authority,
)
from .catalog import (
    DependenceGroupName,
    DistributionAvailabilityName,
    Environment,
    EvidenceCatalog,
    EvidenceCatalogDelta,
    EvidenceKindName,
    EvidenceRegime,
    RegimeBinding,
    RegimeKindName,
    SamplingDesignName,
    SelectionDiagram,
    TargetSamplingName,
    VariableCoordinate,
    VariableDomainName,
    ZTransportCandidate,
    ZTransportQuery,
    ZTransportSensitivityResult,
)
from .catalog import (
    TransportStage as TransportStage,
)
from .exact import (
    ClassicalTransportIdentification,
    ExactDiscreteLaw,
    ExactTransportData,
    ExactTransportDistribution,
    ExactTransportQuery,
    consume_exact,
    consume_identification,
    evaluate_exact,
    identify_classical,
    identify_meta,
    inspect_catalog,
    inspect_proof_graph,
)
from .exact import (
    _exact_distribution as _exact_distribution,
)
from .grid import (
    TransportResponseGrid,
    TransportResponseGridQuery,
    consume_response_grid,
)
from .grid import (
    _response_grid as _response_grid,
)
from .identify import (
    DirectFormula,
    MissingEvidenceCertificate,
    NonTransportableCertificate,
    NotCertifiedCertificate,
    OverlapDiagnostic,
    PopulationFactor,
    RecursiveFactorizationFormula,
    StandardizationFormula,
    TransportCertificate,
    TransportIdentification,
    TransportOverlapReport,
    TransportQuery,
    TrialTransportEstimate,
    consume_z_transport_artifact,
    consume_z_transport_sensitivity_artifact,
    estimate_trial_effect,
    identify,
    identify_z_transport,
    plan_z_transport_evidence,
    reload_lowered_expression,
    reload_lowered_program,
    replay_z_transport_proposal,
    restore_lowered_program,
)
from .identify import (
    TransportFormula as TransportFormula,
)
from .identify import (
    _consume_limits as _consume_limits,
)
from .identify import (
    _ReplayLimits as _ReplayLimits,
)
from .identify import (
    _response_args as _response_args,
)
from .limits import (
    _non_negative as _non_negative,
)
from .limits import (
    _optional_non_negative as _optional_non_negative,
)
from .statistical import (
    EmpiricalTable,
    LearnedCategorical,
    RegimeSample,
    StatisticalTransportData,
    StatisticalTransportDistribution,
    StatisticalTransportQuery,
    consume_statistical,
    evaluate_statistical_grid,
)
from .statistical import (
    _statistical_distribution as _statistical_distribution,
)
from .trial import (
    LearnedTrialEstimate,
    TransportInference,
    TrialAipw,
    TrialAipwData,
    TrialAipwQuery,
    TrialNuisanceDiagnostics,
    prepare_trial,
)
from .trial import (
    _learned_trial as _learned_trial,
)
from .trial import (
    _TransportLimits as _TransportLimits,
)

__all__ = [
    "DependenceGroupName",
    "DirectFormula",
    "DistributionAvailabilityName",
    "Environment",
    "EvidenceCatalog",
    "EvidenceCatalogDelta",
    "EvidenceKindName",
    "EvidenceRegime",
    "MissingEvidenceCertificate",
    "NotCertifiedCertificate",
    "NonTransportableCertificate",
    "OverlapDiagnostic",
    "PopulationFactor",
    "RecursiveFactorizationFormula",
    "reload_lowered_expression",
    "reload_lowered_program",
    "restore_lowered_program",
    "RegimeBinding",
    "RegimeKindName",
    "SamplingDesignName",
    "SelectionDiagram",
    "StandardizationFormula",
    "TargetSamplingName",
    "TransportCertificate",
    "TransportIdentification",
    "TransportOverlapReport",
    "TransportQuery",
    "ZTransportQuery",
    "ZTransportCandidate",
    "ZTransportSensitivityResult",
    "TrialTransportEstimate",
    "VariableCoordinate",
    "VariableDomainName",
    "estimate_trial_effect",
    "identify",
    "identify_z_transport",
    "consume_z_transport_artifact",
    "plan_z_transport_evidence",
    "consume_z_transport_sensitivity_artifact",
    "replay_z_transport_proposal",
    "ClassicalTransportIdentification",
    "ExactDiscreteLaw",
    "ExactTransportData",
    "ExactTransportDistribution",
    "identify_classical",
    "identify_meta",
    "evaluate_exact",
    "RegimeSample",
    "StatisticalTransportData",
    "StatisticalTransportDistribution",
    "StatisticalTransportQuery",
    "consume_statistical",
    "evaluate_statistical_grid",
    "consume_exact",
    "ExactTransportQuery",
    "inspect_catalog",
    "inspect_proof_graph",
    "TransportResponseGridQuery",
    "TransportResponseGrid",
    "consume_response_grid",
    "consume_identification",
    "EmpiricalTable",
    "LearnedCategorical",
    "EmpiricalTable",
    "LearnedCategorical",
    "TrialAipw",
    "TrialAipwData",
    "TrialAipwQuery",
    "TransportInference",
    "LearnedTrialEstimate",
    "prepare_trial",
    "TransportContrast",
    "TransportUncertainty",
    "TrialNuisanceDiagnostics",
    "TransportGridPoint",
]

_PACKAGE = "antecedent.transport._impl"


def _pin_module(value: object) -> None:
    """Keep class and function modules on the re-export package.

    Typing aliases stay as they are: a ``Literal`` form can be a shared object.
    """
    if not isinstance(value, type) and not callable(value):
        return
    module = getattr(value, "__module__", None)
    if isinstance(module, str) and module.startswith(_PACKAGE + "."):
        try:
            value.__module__ = _PACKAGE
        except (AttributeError, TypeError):
            return


for _name in (
    "DependenceGroupName",
    "DirectFormula",
    "DistributionAvailabilityName",
    "Environment",
    "EvidenceCatalog",
    "EvidenceCatalogDelta",
    "EvidenceKindName",
    "EvidenceRegime",
    "MissingEvidenceCertificate",
    "NotCertifiedCertificate",
    "NonTransportableCertificate",
    "OverlapDiagnostic",
    "PopulationFactor",
    "RecursiveFactorizationFormula",
    "reload_lowered_expression",
    "reload_lowered_program",
    "restore_lowered_program",
    "RegimeBinding",
    "RegimeKindName",
    "SamplingDesignName",
    "SelectionDiagram",
    "StandardizationFormula",
    "TargetSamplingName",
    "TransportCertificate",
    "TransportIdentification",
    "TransportOverlapReport",
    "TransportQuery",
    "ZTransportQuery",
    "ZTransportCandidate",
    "ZTransportSensitivityResult",
    "TrialTransportEstimate",
    "VariableCoordinate",
    "VariableDomainName",
    "estimate_trial_effect",
    "identify",
    "identify_z_transport",
    "consume_z_transport_artifact",
    "plan_z_transport_evidence",
    "consume_z_transport_sensitivity_artifact",
    "replay_z_transport_proposal",
    "ClassicalTransportIdentification",
    "ExactDiscreteLaw",
    "ExactTransportData",
    "ExactTransportDistribution",
    "identify_classical",
    "identify_meta",
    "evaluate_exact",
    "RegimeSample",
    "StatisticalTransportData",
    "StatisticalTransportDistribution",
    "StatisticalTransportQuery",
    "consume_statistical",
    "evaluate_statistical_grid",
    "consume_exact",
    "ExactTransportQuery",
    "inspect_catalog",
    "inspect_proof_graph",
    "TransportResponseGridQuery",
    "TransportResponseGrid",
    "consume_response_grid",
    "consume_identification",
    "EmpiricalTable",
    "LearnedCategorical",
    "TrialAipw",
    "TrialAipwData",
    "TrialAipwQuery",
    "TransportInference",
    "LearnedTrialEstimate",
    "prepare_trial",
    "TransportContrast",
    "TransportUncertainty",
    "TrialNuisanceDiagnostics",
    "TransportGridPoint",
    "_non_negative",
    "_optional_non_negative",
    "_no_native_authority",
    "TransportStage",
    "TransportFormula",
    "_response_args",
    "_ReplayLimits",
    "_consume_limits",
    "_exact_distribution",
    "_statistical_distribution",
    "_response_grid",
    "_learned_trial",
):
    _pin_module(globals()[_name])
