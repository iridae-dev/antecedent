"""Operation-level inventory of the six prepared recalculation families.

This contract selects an existing adapter, not an execution license. The adapter
must still check the request's complete identities, support and inference status.
Planner capability flags and readable artifacts never supply executable state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache

from . import _native
from .errors import CausalValueError
from .prediction import FittedEffectModel
from .recalc import RecalcSession, RecalcUnavailable, Stage
from .recalc_cell import CellSession, CrossfitSession, ScoreResumeSession


class Family(StrEnum):
    ADJUSTED = "adjusted_regression"
    DOUBLY_ROBUST = "doubly_robust_effects"
    STATIC = "static_response_transport"
    BAYESIAN = "bayesian_analysis"
    TEMPORAL = "temporal_analysis"
    DESIGN = "design_specific_effects"


class Operation(StrEnum):
    UTILITY = "utility"
    FUNCTIONAL = "contrast_functional"
    TARGET = "target_weights_law"
    ACTION_GRID = "action_grid"
    DATA = "new_rows_outcomes"
    STRUCTURE = "graph_query_regime"
    PROVIDER = "provider_prior"
    LEARNER = "learner_folds_rng"
    INFERENCE = "inference_settings"
    RESUME = "fresh_process_resume"


class RetainedKind(StrEnum):
    READABLE = "readable_result"
    LIVE_SCORES = "live_score_session"
    SCORES = "verified_portable_scores"
    PREDICTOR = "verified_portable_predictor"
    DRAWS = "aligned_posterior_draws"
    PROVIDER = "executable_provider"


@dataclass(frozen=True, slots=True)
class IdentityRequirement:
    """Complete semantic fields that must enter the shared planner's stage digest."""

    stage: Stage
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Adapter:
    retained: RetainedKind
    state_type: type[object]
    python_path: str
    rust_path: str
    operation: str
    scope: str


@dataclass(frozen=True, slots=True)
class Capability:
    family: Family
    operation: Operation
    identities: tuple[IdentityRequirement, ...]
    retained_objects: tuple[str, ...]
    adapters: tuple[Adapter, ...]
    compatible_operation: str
    refusal_code: str = "route_not_supported"
    refusal_detail: str = "recalc.capability_adapter_unavailable"
    inference: str = "Preserve the producing claim; reuse never licenses uncertainty."


_BASE = (
    IdentityRequirement(Stage.GRAPH, ("graph class", "nodes/edges", "accepted structure")),
    IdentityRequirement(
        Stage.QUERY,
        (
            "quantity",
            "treatments",
            "outcome",
            "contrast/reference levels",
            "units",
            "scale/transform",
            "horizon",
        ),
    ),
    IdentityRequirement(Stage.REGIME, ("intervention regime", "treatment coding", "history")),
    IdentityRequirement(
        Stage.EVIDENCE,
        (
            "identification proof",
            "evidence obligations",
            "consumed evidence IDs",
            "support/trust/calibration standing",
        ),
    ),
    IdentityRequirement(
        Stage.SOURCE_POPULATION,
        ("source population IDs", "sampling/dependence design", "source factor/proof leaves"),
    ),
    IdentityRequirement(
        Stage.TARGET_POPULATION,
        ("target population/law", "weights", "weight dependency variables", "target row IDs/order"),
    ),
    IdentityRequirement(
        Stage.DATA_SNAPSHOT,
        ("snapshot digest", "schema", "all used values", "outcomes", "complete-case row IDs/order"),
    ),
    IdentityRequirement(
        Stage.ROW_DESIGN,
        (
            "row/unit/cluster/time membership",
            "sampling weights",
            "overlap",
            "lag alignment",
            "missingness",
        ),
    ),
    IdentityRequirement(
        Stage.TREATMENT_GRID, ("action values/grid", "action support", "reference cell")
    ),
    IdentityRequirement(
        Stage.LEARNER_FOLDS_RNG,
        (
            "provider/version",
            "learner hyperparameters",
            "basis/features/link",
            "fold map",
            "RNG algorithm/seed/stream",
            "fit/draw/inference settings",
        ),
    ),
    IdentityRequirement(Stage.UTILITY, ("utility functional", "action mapping", "cost", "benefit")),
)

_EXTERNAL = tuple(
    IdentityRequirement(
        Stage.external_study(branch),
        (
            "external study snapshot and datum IDs",
            "source/target population",
            "graph/query/regime",
            "provider/version and request",
            "prior construction/evidence already consumed",
            "factor regime/proof leaf",
            "draw alignment/weights/axes",
            "inference standing",
        ),
    )
    for branch in range(8)
)

_FAMILY_FIELDS = {
    Family.ADJUSTED: (
        "linear/GLM link",
        "feature/basis specification",
        "treatment levels/coding",
        "joint coefficient covariance",
    ),
    Family.DOUBLY_ROBUST: (
        "AIPW/cell/DML/CATE coordinate",
        "propensity/outcome learners",
        "complete-case fold map",
        "score quantity",
        "positivity",
        "portable predictor schema and parent claim",
    ),
    Family.STATIC: (
        "identified functional and factor map",
        "source/regime proof leaves",
        "source overlap",
        "target law",
        "grid support",
        "search completeness/budget",
    ),
    Family.BAYESIAN: (
        "likelihood",
        "prior/version/construction",
        "already consumed evidence",
        "model/basis/noise",
        "draw axes/count/weights",
        "joint stream/source alignment",
        "posterior calibration",
    ),
    Family.TEMPORAL: (
        "unit/time memberships",
        "lag alignment",
        "initial-state law",
        "intervention history",
        "history/horizon support",
        "whole-unit resampling identity",
    ),
    Family.DESIGN: (
        "IV instruments/exclusion/strength",
        "RD running variable/cutoff/bandwidth/support",
        "front-door mediator models/positivity",
        "design-specific identification evidence",
    ),
}

_OBJECTS = {
    Family.ADJUSTED: (
        "PreparedAnalysis compiled plan; not a persistent fit",
        "readable adjustment result",
    ),
    Family.DOUBLY_ROBUST: (
        "RecalcSession/CrossfitSession/CellSession",
        "FrozenScores via ScoreResumeSession",
        "FittedEffectModel via verified load",
        "readable RecalcReceipt",
    ),
    Family.STATIC: (
        "prepared response/transport stage",
        "bound factor/external claim",
        "readable response/transport artifact",
    ),
    Family.BAYESIAN: (
        "Gaussian/basis fit",
        "aligned posterior draw artifact",
        "readable posterior result",
    ),
    Family.TEMPORAL: (
        "prepared temporal stage",
        "history/initial-state artifact",
        "readable temporal result",
    ),
    Family.DESIGN: ("PreparedAnalysis compiled design plan", "readable IV/RD/front-door result"),
}

_LIVE = Adapter(
    RetainedKind.LIVE_SCORES,
    CellSession,
    "antecedent.recalc_cell.CellSession.execute",
    "antecedent::analysis::recalc_cell::execute_cell_with_receipt",
    "checked selective execution",
    "Cell-AIPW at its licensed binary-treatment quantity; cross-fit AIPW uses RecalcSession.execute. Full supplied data, checked graph and native identities are required.",
)
_CROSSFIT = replace(
    _LIVE,
    state_type=CrossfitSession,
    python_path="antecedent.recalc_cell.CrossfitSession.execute",
    rust_path="antecedent::analysis::recalc_receipt::execute_with_receipt",
)
_RECALC = replace(
    _CROSSFIT,
    state_type=RecalcSession,
    python_path="antecedent.recalc.RecalcSession.execute",
)
_LIVE_ADAPTERS = (_LIVE, _CROSSFIT, _RECALC)
_SCORES = Adapter(
    RetainedKind.SCORES,
    ScoreResumeSession,
    "antecedent.recalc_cell.ScoreResumeSession.retarget",
    "antecedent::analysis::recalc_cell::execute_resumed_retarget",
    "same-row reweight/resummarize",
    "Loaded frozen AIPW/cell scores, unchanged snapshot/row IDs/folds/query/action grid; weights depend only on licensed adjustment variables; utility may change. No new-row prediction or refit.",
)
_PREDICT = Adapter(
    RetainedKind.PREDICTOR,
    FittedEffectModel,
    "antecedent.prediction.FittedEffectModel.predict",
    "antecedent_estimate::FittedEffect::predict",
    "point CATE prediction",
    "Verified DRLearner/CausalForest portable predictor with its exact parent claim and feature schema. New feature rows only; no new-outcome fitting, training residuals, covariance or shared RecalcPlan workflow.",
)
_LOAD_PREDICT = Adapter(
    RetainedKind.PREDICTOR,
    FittedEffectModel,
    "antecedent.prediction.FittedEffectModel.load",
    "antecedent_io::decode_analysis_result_artifact",
    "verified predictor load",
    _PREDICT.scope,
)
_LOAD_SCORES = Adapter(
    RetainedKind.SCORES,
    ScoreResumeSession,
    "antecedent.recalc_cell.resume_from_scores",
    "antecedent::analysis::recalc_cell::ScoreResumeSession::resume_from_score_bytes",
    "verified score resume",
    _SCORES.scope,
)


@cache
def capability_matrix() -> tuple[Capability, ...]:
    """Return all sixty immutable cells, including explicit unavailable adapters.

    Identity requirements specify what the operation must preserve or invalidate,
    not proof that the current adapter supports every variation of those fields.
    Missing families retain no executable adapter in this shared workflow yet.
    """
    rows = []
    for family in Family:
        identities = (*_BASE, *_EXTERNAL, IdentityRequirement(Stage.QUERY, _FAMILY_FIELDS[family]))
        for operation in Operation:
            adapters: tuple[Adapter, ...] = ()
            compatible = "No family adapter; readable/prepared results do not grant this operation."
            if family == Family.DOUBLY_ROBUST:
                if operation in (Operation.UTILITY, Operation.TARGET):
                    adapters = (*_LIVE_ADAPTERS, _SCORES)
                    compatible = "Reuse same-row licensed scores; recheck target weights/support and recompute law/decision as required. A new target law is unsupported."
                elif operation == Operation.DATA:
                    adapters = (*_LIVE_ADAPTERS, _PREDICT)
                    compatible = "Supplied new rows/outcomes require native refitting; a portable predictor only predicts on new feature rows."
                elif operation in (
                    Operation.FUNCTIONAL,
                    Operation.ACTION_GRID,
                    Operation.STRUCTURE,
                    Operation.LEARNER,
                ):
                    adapters = _LIVE_ADAPTERS
                    compatible = "The native adapter rechecks support/identification and refits invalidated scores for its supported cell quantity, treatment columns, graph, folds or RNG. Arbitrary learner changes and off-grid actions remain unsupported."
                elif operation == Operation.RESUME:
                    adapters = (_LOAD_SCORES, _LOAD_PREDICT)
                    compatible = "Verified scores resume same-row retargeting; verified predictors resume point prediction. Neither restores a general executable PreparedStudy."
            rows.append(
                Capability(family, operation, identities, _OBJECTS[family], adapters, compatible)
            )
    return tuple(rows)


def capability(family: Family | str, operation: Operation | str) -> Capability:
    """Look up a declared operation; reject unknown family/operation coordinates."""
    try:
        coordinate = Family(family), Operation(operation)
    except ValueError as error:
        raise CausalValueError(f"unknown recalculation capability: {family}/{operation}") from error
    return next(r for r in capability_matrix() if (r.family, r.operation) == coordinate)


def retained_kind(state: object) -> RetainedKind:
    """Inspect actual native-backed state; caller-set capability flags are ignored."""
    if isinstance(state, (RecalcSession, CrossfitSession, CellSession)):
        if (
            isinstance(
                state._handle,
                (
                    _native.RecalcSessionHandle,
                    _native.CrossfitSessionHandle,
                    _native.CellSessionHandle,
                ),
            )
            and state._handle.is_live()
        ):
            return RetainedKind.LIVE_SCORES
    elif isinstance(state, ScoreResumeSession):
        if isinstance(state._handle, _native.ScoreResumeHandle):
            return RetainedKind.SCORES
    elif isinstance(state, FittedEffectModel) and isinstance(
        state._native, _native.FittedEffectModel
    ):
        return RetainedKind.PREDICTOR
    return RetainedKind.READABLE


def require_adapter(family: Family | str, operation: Operation | str, state: object) -> Adapter:
    """Select an existing adapter for actual retained state, or refuse specifically.

    This does not run the operation or certify compatibility/reuse. Its public
    adapter must still validate the request and instrument actual execution.
    No arbitrary callback, posterior artifact or planner flag is treated as a
    provider or an executable family adapter.
    """
    row = capability(family, operation)
    kind = retained_kind(state)
    for adapter in row.adapters:
        if adapter.retained == kind and isinstance(state, adapter.state_type):
            return adapter
    raise RecalcUnavailable(
        {
            "code": row.refusal_code,
            "detail": row.refusal_detail,
            "stage": Stage.SCORE_ARTIFACT.value,
            "message": f"{row.family}/{row.operation} has no adapter for {kind}",
        }
    )


__all__ = [
    "Adapter",
    "Capability",
    "Family",
    "IdentityRequirement",
    "Operation",
    "RetainedKind",
    "capability",
    "capability_matrix",
    "require_adapter",
    "retained_kind",
]
