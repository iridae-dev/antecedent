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
from .recalc_adjusted import AdjustedSession
from .recalc_cell import CellSession, CrossfitSession, ScoreResumeSession
from .recalc_dr import DrSession


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
    LIVE_FIT = "live_adjusted_fit"
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
        "AdjustedSession checked joint fit and coefficient covariance",
        "PreparedAnalysis compiled plan; not a persistent fit",
        "readable adjustment result or receipt; no fitted-state resume",
    ),
    Family.DOUBLY_ROBUST: (
        "RecalcSession/CrossfitSession/CellSession",
        "DrSession checked DML AIPW scores and optional actual CATE model",
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
_DR = replace(
    _LIVE,
    state_type=DrSession,
    python_path="antecedent.recalc_dr.DrSession.execute",
    rust_path="antecedent::analysis::recalc_dr::execute_dr_with_receipt",
    scope="Checked DML AIPW or DR-Learner with Linear/Ridge outcome and final learners; Linear/Ridge/Logistic treatment, fixed binary contrast. PLR, trimming and arbitrary providers are unsupported.",
)
_DR_PREDICT = replace(
    _DR,
    python_path="antecedent.recalc_dr.DrSession.predict",
    rust_path="antecedent::analysis::recalc_dr::DrSession::predict",
    operation="retained CATE point prediction",
    scope="Actual retained CATE map, exact ordered feature names and finite rows; no fitting or sampling covariance.",
)
_DR_RESUME = replace(
    _DR,
    python_path="antecedent.recalc_dr.DrSession.resume",
    rust_path="antecedent::analysis::recalc_dr::DrSession::resume",
    operation="supplied-data refit at process boundary",
    scope="Receipts supply identities only; fresh processes must supply raw data and refit. Verified scores/predictors separately restore their scoped consumers.",
)
_LIVE_ADAPTERS = (_LIVE, _CROSSFIT, _RECALC, _DR)
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


_ADJUSTED = Adapter(
    RetainedKind.LIVE_FIT,
    AdjustedSession,
    "antecedent.recalc_adjusted.AdjustedSession.execute",
    "antecedent::analysis::recalc_adjusted::execute_adjusted_with_receipt",
    "checked selective adjusted execution",
    "Linear/GLM and vector/categorical mean contrasts over a checked common adjustment set. Compatible contrasts, same-row target weights and utility reuse the joint fit; raw snapshot/model/roles/coding changes refit. Full coefficient covariance retained; conditional analytic uncertainty has unmeasured calibration. Receipts and flags provide no portable fit.",
)
_ADJUSTED_PREDICT = replace(
    _ADJUSTED,
    python_path="antecedent.recalc_adjusted.AdjustedSession.predict",
    rust_path="antecedent::analysis::recalc_adjusted::AdjustedSession::predict",
    operation="measured retained-model prediction",
    scope="Point conditional means at the retained named feature schema and checked treatment support. Actual model fit count is measured as zero; no shared-stage receipt, new outcome fitting or prediction interval is supplied.",
)
_ADJUSTED_RESUME = replace(
    _ADJUSTED,
    python_path="antecedent.recalc_adjusted.AdjustedSession.resume",
    rust_path="antecedent::analysis::recalc_adjusted::AdjustedSession::resume",
    operation="supplied-data refit at process boundary",
    scope="A receipt supplies input identities only; a fresh process must supply raw data and refit. Portable-fit/score flags cannot supply executable adjusted state.",
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
            if family == Family.ADJUSTED and operation != Operation.PROVIDER:
                adapters = (_ADJUSTED_RESUME,) if operation == Operation.RESUME else (_ADJUSTED,)
                if operation == Operation.DATA:
                    adapters = (_ADJUSTED, _ADJUSTED_PREDICT)
                compatible = {
                    Operation.UTILITY: "Reuse the checked fit and law; recompute only the changed net-benefit rule.",
                    Operation.FUNCTIONAL: "Evaluate supported numeric or declared-level mean contrasts from retained joint coefficients and covariance. Other functionals have no execution license.",
                    Operation.TARGET: "Same-row finite nonnegative weights depending only on checked adjustment variables standardize the retained response model; changed row law/data refit.",
                    Operation.ACTION_GRID: "Supported finite numeric actions and declared categorical contrasts reuse the model; changed categorical level/reference/coding declarations refit and recheck support.",
                    Operation.DATA: "New rows or outcomes in a supplied snapshot invalidate the fit; scoped retained-model prediction separately validates feature schema and treatment support without fitting.",
                    Operation.STRUCTURE: "Changed causal roles, graph or adjustment set rerun checked identification and affected fits; incompatible adjustment refuses.",
                    Operation.LEARNER: "Declared OLS covariance or GLM family/fit-option and RNG changes invalidate the model. Arbitrary learners/bases are unsupported.",
                    Operation.INFERENCE: "Declared model covariance changes invalidate the joint fit. No calibrated intervals or additional resampling/inference settings are licensed.",
                    Operation.RESUME: "No adjusted fitted-state artifact loader: supplied compatible raw data refit at the fresh-process boundary; receipt-only/flag-only resume refuses.",
                }[operation]
            if family == Family.DOUBLY_ROBUST:
                if operation in (Operation.UTILITY, Operation.TARGET):
                    adapters = (*_LIVE_ADAPTERS, _SCORES)
                    compatible = "Reuse same-row licensed scores; recheck target weights/support and recompute law/decision as required. A new target law is unsupported."
                elif operation == Operation.DATA:
                    adapters = (*_LIVE_ADAPTERS, _DR_PREDICT, _PREDICT)
                    compatible = "Supplied new rows/outcomes require native refitting; a portable predictor only predicts on new feature rows."
                elif operation in (
                    Operation.FUNCTIONAL,
                    Operation.ACTION_GRID,
                    Operation.STRUCTURE,
                    Operation.LEARNER,
                ):
                    adapters = _LIVE_ADAPTERS
                    compatible = "The native adapter rechecks support/identification and refits invalidated scores for its supported cell, DML AIPW or DR-Learner quantity, treatment columns, graph, folds or RNG. Arbitrary learner changes and off-grid actions remain unsupported."
                elif operation == Operation.RESUME:
                    adapters = (_LOAD_SCORES, _LOAD_PREDICT, _DR_RESUME)
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
    if isinstance(state, AdjustedSession):
        if isinstance(state._handle, _native.AdjustedSessionHandle) and state._handle.is_live():
            return RetainedKind.LIVE_FIT
    elif isinstance(state, DrSession):
        if isinstance(state._handle, _native.DrSessionHandle) and state._handle.is_live():
            return RetainedKind.LIVE_SCORES
    elif isinstance(state, (RecalcSession, CrossfitSession, CellSession)):
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
            if adapter is _DR_PREDICT and (
                not isinstance(state, DrSession)
                or not isinstance(state._handle, _native.DrSessionHandle)
                or state._handle.prediction_columns() is None
            ):
                continue
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
