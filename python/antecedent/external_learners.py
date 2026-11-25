"""Explicit provider-backed nuisance learner fit and predict operations.

Each operation executes through :class:`ProviderRegistry`, so verification is evaluated
for that exact request. This module does not by itself provide cross-fitting or establish
that a pretrained model excludes evaluation rows.

:func:`antecedent.analyze` and :func:`antecedent.prepare` take the learners as a study
setting. ``estimator`` is the only knob on that call: ``aipw``, ``dml`` (the partially
linear score), ``dr_learner`` or ``causal_forest``. :func:`analyze_ate` keeps the older
reading, ``estimator`` and ``score`` together, as the compatibility wrapper. The direct
entries :func:`dml_ate`, :func:`aipw_ate`, :func:`dr_learner_ate`, :func:`causal_forest_ate`
and :func:`cross_fit_aipw` run the same fit without the compiler and are deprecated in
favour of :func:`antecedent.analyze`.
"""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from . import _deprecation
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution, native_controls
from .extensibility import (
    FoldPolicy,
    ProviderRegistry,
    ProviderResult,
    _digest,
    _request_digest,
    _spec_evidence,
    connect_learner,
)


@dataclass(frozen=True, slots=True)
class ExternalLearner:
    """A registered provider's declared nuisance task and fit behavior.

    Providers use ``query_family='nuisance_learner'`` and declare ``mean``. Connecting
    one to this slot goes through the interface contract (``fit``, ``predict`` and
    ``mean``). Fit returns a model in ``ProviderExecution.artifact``; prediction
    receives it as base64 in its request. A variable ``output_shape`` declaration is
    needed because folds vary in size. The fold policy is :class:`~antecedent.extensibility.FoldPolicy`.
    Weighted fitting additionally requires ``sample_weights`` in the provider's declared
    ``data_dependence``.
    """

    registry: ProviderRegistry
    name: str
    task: Literal["regression", "binary_probability"]
    supports_weights: bool = False
    training: Literal["fold_fit", "frozen", "fine_tuned"] = "fold_fit"
    configuration: Mapping[str, Any] | None = None
    _pinned_spec_digest: str | None = field(default=None, repr=False, compare=False)
    #: ``predict`` is a pure function of the artifact bytes and the rows. A provider
    #: that needs live process state is not portable. Recorded from the artifact codec.
    portable: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "portable", _artifact_is_portable(self))
        _require_honoured(self)

    def fit(self, x: Any, y: Any, weights: Any | None = None) -> FittedExternalLearner:
        """Fit or bind the model to these training rows and retain the execution receipt."""
        if self.task not in ("regression", "binary_probability"):
            raise CausalValueError("unknown nuisance prediction task")
        if self.training not in ("fold_fit", "frozen", "fine_tuned"):
            raise CausalValueError("unknown nuisance training declaration")
        spec = self.registry._entry(self.name)[0].spec
        if (
            self._pinned_spec_digest is not None
            and _digest("spec", _spec_evidence(spec)) != self._pinned_spec_digest
        ):
            raise CausalValueError("nuisance provider specification changed during the run")
        forecast_adapter = isinstance(self, ForecastNuisanceAdapter) and (
            self.information_contract == "iid_row_conditional_mean"
            and "iid_row_conditional_mean" in spec.data_dependence
        )
        connect_learner(spec, as_learner=forecast_adapter)
        if type(self.supports_weights) is not bool:
            raise CausalValueError("supports_weights must be a bool")
        if self.supports_weights and "sample_weights" not in spec.data_dependence:
            raise CausalValueError("provider does not declare sample_weights in data_dependence")
        matrix = _matrix(x)
        target = _vector(y, matrix.shape[0], "target")
        if self.task == "binary_probability" and np.any((target != 0) & (target != 1)):
            raise CausalValueError("binary probability targets must be 0 or 1")
        selected_weights = None
        if weights is not None:
            if not self.supports_weights:
                raise CausalValueError("nuisance provider does not declare sample-weight support")
            selected_weights = _vector(weights, matrix.shape[0], "weights")
            if np.any(selected_weights < 0) or not np.any(selected_weights > 0):
                raise CausalValueError("weights must be nonnegative and have positive mass")
        config = copy.deepcopy(dict(self.configuration or {}))
        digest = _digest("spec", _spec_evidence(spec))
        request = {
            "operation": "fit",
            "version": 1,
            "task": self.task,
            "training": self.training,
            "configuration": config,
            "x": matrix,
            "y": target,
            "weights": selected_weights,
        }
        request_digest = _request_digest(request)
        result = self.registry.execute(self.name, request)
        if _request_digest(request) != request_digest:
            raise CausalValueError("nuisance provider changed its fit request")
        if _digest("spec", _spec_evidence(self.registry._entry(self.name)[0].spec)) != digest:
            raise CausalValueError("nuisance provider specification changed during fit")
        if result.estimate.size != 0 or result.uncertainty is not None or result.artifact is None:
            raise CausalValueError(
                "nuisance fit must return an empty estimate and a model artifact"
            )
        return FittedExternalLearner(
            self.registry,
            self.name,
            self.task,
            config,
            _request_digest(config),
            digest,
            bytes(result.artifact),
            matrix.shape[1],
            result,
            spec.fold_policy,
        )


@dataclass(frozen=True, slots=True)
class ForecastNuisanceAdapter(ExternalLearner):
    """Opt in to IID-row conditional means from a compatible forecast provider.

    The provider must declare ``iid_row_conditional_mean`` in ``data_dependence`` and
    implement the nuisance fit/predict operations. This is not a temporal validation
    contract; time-ordered cross-fitting remains refused.
    """

    information_contract: Literal["iid_row_conditional_mean"] = "iid_row_conditional_mean"


def _snapshot_learner(learner: ExternalLearner) -> ExternalLearner:
    """Pin one run's specification and copy its configuration before any folds execute."""
    with learner.registry._execution_lock:
        spec = learner.registry._entry(learner.name)[0].spec
        return replace(
            learner,
            configuration=copy.deepcopy(dict(learner.configuration or {})),
            _pinned_spec_digest=_digest("spec", _spec_evidence(spec)),
        )


@dataclass(frozen=True, slots=True)
class FittedExternalLearner:
    """A fitted model and its fit receipt; each prediction has its own trust receipt."""

    registry: ProviderRegistry
    name: str
    task: str
    configuration: Mapping[str, Any]
    configuration_digest: str
    spec_digest: str
    model: bytes
    n_features: int
    fit_receipt: ProviderResult
    fold_policy: FoldPolicy

    def predict(self, x: Any) -> ProviderResult:
        """Predict on the supplied rows and return the registry's validated result."""
        matrix = _matrix(x)
        if matrix.shape[1] != self.n_features:
            raise CausalValueError("nuisance prediction feature count differs from fit")
        if _request_digest(self.configuration) != self.configuration_digest:
            raise CausalValueError("nuisance configuration changed since fit")
        if (
            _digest("spec", _spec_evidence(self.registry._entry(self.name)[0].spec))
            != self.spec_digest
        ):
            raise CausalValueError("nuisance provider specification changed since fit")
        request = {
            "operation": "predict",
            "version": 1,
            "task": self.task,
            "configuration": copy.deepcopy(dict(self.configuration)),
            "x": matrix,
            "model_base64": base64.b64encode(self.model).decode("ascii"),
        }
        request_digest = _request_digest(request)
        result = self.registry.execute(self.name, request)
        if _request_digest(request) != request_digest:
            raise CausalValueError("nuisance provider changed its prediction request")
        if (
            _digest("spec", _spec_evidence(self.registry._entry(self.name)[0].spec))
            != self.spec_digest
        ):
            raise CausalValueError("nuisance provider specification changed during prediction")
        if result.estimate.shape != (matrix.shape[0],) or result.uncertainty is not None:
            raise CausalValueError("nuisance prediction needs one point value per row")
        if self.task == "binary_probability" and np.any(
            (result.estimate < 0) | (result.estimate > 1)
        ):
            raise CausalValueError("nuisance probability predictions must lie in [0, 1]")
        return result


@dataclass(frozen=True, slots=True)
class CrossFittedAipwNuisances:
    """Held-out nuisance predictions, folds, and per-model provider receipts.

    Each group of three provenance entries corresponds to the control outcome,
    treated outcome, and propensity model for a fold, in fold order. The numeric
    identifier in each provenance entry links it to the fit and prediction receipts.
    """

    control: tuple[float, ...]
    treated: tuple[float, ...]
    propensity: tuple[float, ...]
    fold_ids: tuple[int, ...]
    provenance: tuple[tuple[str, ...], ...]
    receipts: tuple[tuple[int, ProviderResult, ProviderResult], ...]
    fold_policy: FoldPolicy


@dataclass(frozen=True, slots=True)
class ExternalAteEstimate:
    """A static DAG ATE estimate and the evidence for its external nuisance fits.

    Its standard error is the IID cross-fitted score standard error; its validity
    requires the declared causal graph and the usual nuisance-rate conditions.
    External provider claims and any prior training overlap remain unverified except
    to the extent of the individual exact-request receipts.
    """

    estimator: Literal["dml", "aipw", "dr_learner", "causal_forest"]
    score: Literal["aipw", "partially_linear"]
    estimate: float
    standard_error: float
    row_ids: tuple[int, ...]
    fold_ids: tuple[int, ...]
    adjustment_set: tuple[str, ...]
    provenance: tuple[tuple[str, ...], ...]
    assumptions: tuple[Mapping[str, Any], ...]
    cate: tuple[float, ...] | None
    cate_standard_errors: tuple[float, ...] | None
    receipts: tuple[tuple[int, ProviderResult, ProviderResult], ...]
    fold_policy: FoldPolicy
    #: Out-of-fold treatment propensity of each analysed row, when the fit keeps an AIPW
    #: score table (set by :func:`analyze_ate`; the direct entries leave it ``None``).
    propensity: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class Learners:
    """The outcome-regression and propensity learners a study declares.

    ``outcome`` must be a ``regression`` learner and ``treatment`` a ``binary_probability``
    learner. Each run pins both providers' specifications and configurations, and their names,
    specification digests and configuration digests become the declaration's identity, so two
    studies that differ only in their learners compile to different programs.
    """

    outcome: ExternalLearner
    treatment: ExternalLearner

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ExternalLearner) or not isinstance(
            self.treatment, ExternalLearner
        ):
            raise CausalValueError("outcome and treatment must be ExternalLearner instances")
        _refuse_shuffled_series(self.outcome)
        _refuse_shuffled_series(self.treatment)
        _refuse_unhonoured_policy(self.outcome)
        _refuse_unhonoured_policy(self.treatment)
        _require_honoured(self.outcome)
        _require_honoured(self.treatment)


def _learner_identity(learner: ExternalLearner) -> dict[str, Any]:
    return {
        "name": learner.name,
        "task": learner.task,
        "training": learner.training,
        "supports_weights": learner.supports_weights,
        "spec": learner._pinned_spec_digest,
        "configuration": _request_digest(dict(learner.configuration or {})),
        "portable": learner.portable,
    }


def _columns(data: Mapping[str, Any]) -> tuple[list[str], list[NDArray[np.float64]], int]:
    if (
        not isinstance(data, Mapping)
        or not data
        or any(not isinstance(k, str) or not k for k in data)
    ):
        raise CausalValueError("data must map nonempty column names to numeric columns")
    names = list(data)
    raw_columns = [np.asarray(data[name], dtype=np.float64) for name in names]
    n = len(raw_columns[0]) if raw_columns[0].ndim == 1 else 0
    if not n or any(
        column.shape != (n,) or not np.isfinite(column).all() for column in raw_columns
    ):
        raise CausalValueError("data columns must be nonempty, finite and equally sized")
    return names, raw_columns, n


#: Artifact codec whose bytes are a prediction map: ``predict`` reads only those
#: bytes and the rows. Any other codec needs live process state.
PORTABLE_PREDICTION_MAP = "portable_prediction_map_v1"

#: One refusal for a non-portable learner declared on smoothed dose, trial AIPW
#: or joint-cell scores. The runtime reason is ``route_not_supported``.
LEARNER_NOT_PORTABLE = "dose_response.learner_not_portable"

PORTABLE_ROUTES = ("smoothed_dose", "trial_aipw", "joint_cells")


def _artifact_is_portable(learner: ExternalLearner) -> bool:
    spec = learner.registry._entry(learner.name)[0].spec
    return spec.artifact_codec == PORTABLE_PREDICTION_MAP


def declare_portable_route(route: str, learners: Learners) -> Learners:
    """Admit ``learners`` on a route that evaluates ``predict`` from the artifact.

    ``route`` is ``smoothed_dose``, ``trial_aipw`` or ``joint_cells``. A learner
    whose predict needs live process state is refused with
    :data:`LEARNER_NOT_PORTABLE`. The same learner stays admissible on a route
    that refits in process.
    """
    if route not in PORTABLE_ROUTES:
        raise CausalValueError(f"unknown portable learner route {route!r}")
    if not isinstance(learners, Learners):
        raise CausalValueError("learners must be a Learners declaration")
    for role, learner in (("outcome", learners.outcome), ("treatment", learners.treatment)):
        if not learner.portable:
            raise CausalUnsupportedError(
                f"{LEARNER_NOT_PORTABLE}: the {role} learner's predict needs live process "
                f"state; {route} evaluates predict from the artifact bytes and the rows",
                reason_code="route_not_supported",
            )
    return learners


def _require_honoured(learner: ExternalLearner) -> None:
    """Refuse a fold policy the nuisance estimators cannot honour.

    A :class:`ForecastNuisanceAdapter` opts into an IID information contract at
    fit; its forecast policy is not this check.
    """
    if type(learner) is not ExternalLearner:
        return
    spec = learner.registry._entry(learner.name)[0].spec
    if not spec.fold_policy.estimator_honours():
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")


def _fold_policy(value: FoldPolicy | str, *, iid: FoldPolicy) -> FoldPolicy:
    """Decode a legacy word onto the fold policy. ``iid`` is this route's policy."""
    if value == "iid":
        policy = iid
    else:
        try:
            policy = FoldPolicy.decode(value)
        except (TypeError, ValueError) as error:
            raise CausalValueError(str(error)) from error
    if not policy.estimator_honours():
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")
    return policy


def _check_estimator(estimator: str, score: str, fold_policy: FoldPolicy) -> None:
    if not fold_policy.estimator_honours():
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")
    if estimator not in ("dml", "aipw", "dr_learner", "causal_forest"):
        raise CausalValueError("unknown external learner estimator")
    if score not in ("aipw", "partially_linear") or (estimator != "dml" and score != "aipw"):
        raise CausalValueError("invalid score for the requested estimator")


def analyze_ate(
    data: Mapping[str, Any],
    edges: Any,
    treatment_name: str,
    outcome_name: str,
    *,
    learners: Learners,
    folds: int = 5,
    weights: Any | None = None,
    fold_policy: FoldPolicy | str = FoldPolicy.HOST_OWNED,
    estimator: Literal["dml", "aipw", "dr_learner", "causal_forest"] = "dml",
    score: Literal["aipw", "partially_linear"] = "aipw",
    execution: Execution | None = None,
) -> ExternalAteEstimate:
    """Compatibility wrapper for a static DAG average effect with declared learners.

    Today's reading stays: the default is ``estimator="dml"`` with ``score="aipw"``, and
    any score other than ``"aipw"`` is accepted only with ``estimator="dml"``. The study
    setting that replaces this call is :func:`antecedent.analyze` / :func:`antecedent.prepare`,
    where ``estimator`` is the only knob and ``dml`` means the partially linear score.
    """
    from . import _native

    if not isinstance(learners, Learners):
        raise CausalValueError("learners must be a Learners declaration")
    _refuse_shuffled_series(learners.outcome)
    _refuse_shuffled_series(learners.treatment)
    _refuse_unhonoured_policy(learners.outcome)
    _refuse_unhonoured_policy(learners.treatment)
    policy = _fold_policy(fold_policy, iid=FoldPolicy.HOST_OWNED)
    _check_estimator(estimator, score, policy)
    outcome = _snapshot_learner(learners.outcome)
    treatment = _snapshot_learner(learners.treatment)
    identity = json.dumps(
        {"outcome": _learner_identity(outcome), "treatment": _learner_identity(treatment)},
        sort_keys=True,
        separators=(",", ":"),
    )
    names, raw_columns, n = _columns(data)
    selected_weights = None if weights is None else _vector(weights, n, "weights")
    (
        point,
        se,
        rows,
        fold_ids,
        adjustment,
        provenance,
        assumptions,
        cate,
        cate_se,
        receipts,
        propensity,
    ) = _native.declared_learner_ate(
        names,
        [col.tolist() for col in raw_columns],
        list(edges),
        treatment_name,
        outcome_name,
        outcome,
        treatment,
        identity=identity,
        folds=folds,
        weights=None if selected_weights is None else selected_weights.tolist(),
        method=estimator,
        score=score,
        **native_controls(
            execution,
            seed="seed",
            cancel="cancel",
            memory="memory_limit_bytes",
        ),
    )
    return ExternalAteEstimate(
        estimator,
        score,
        point,
        se,
        tuple(rows),
        tuple(fold_ids),
        tuple(names[index] for index in adjustment),
        tuple(tuple(item) for item in provenance),
        tuple(json.loads(assumptions)),
        None if cate is None else tuple(cate),
        None if cate_se is None else tuple(cate_se),
        tuple(receipts),
        policy,
        None if propensity is None else tuple(propensity),
    )


_ANALYZE = (
    "antecedent.analyze(data, graph=edges, query=AverageEffect(treatment, outcome), "
    "learners=Learners(outcome, treatment), estimator={estimator!r})"
)


def dml_ate(
    data: Mapping[str, Any],
    edges: Any,
    treatment_name: str,
    outcome_name: str,
    *,
    outcome: ExternalLearner,
    treatment: ExternalLearner,
    folds: int = 5,
    weights: Any | None = None,
    fold_policy: FoldPolicy | str = FoldPolicy.HOST_OWNED,
    estimator: Literal["dml", "aipw", "dr_learner", "causal_forest"] = "dml",
    score: Literal["aipw", "partially_linear"] = "aipw",
    execution: Execution | None = None,
) -> ExternalAteEstimate:
    """Identify a static DAG and fit its binary-treatment DML AIPW estimate.

    The supplied graph is the caller's causal assumption. Only IID cross-fitting
    is implemented; temporal training and validation are explicitly refused.
    Provider receipts are returned for every fold and treatment arm.

    Deprecated: :func:`antecedent.analyze` declares the learners on the study and returns the
    same numbers. This entry keeps the estimator and score arguments.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.external_learners.dml_ate", _ANALYZE.format(estimator=estimator)
    )
    return _direct_ate(
        data,
        edges,
        treatment_name,
        outcome_name,
        outcome=outcome,
        treatment=treatment,
        folds=folds,
        weights=weights,
        fold_policy=fold_policy,
        estimator=estimator,
        score=score,
        execution=execution,
    )


def _direct_ate(
    data: Mapping[str, Any],
    edges: Any,
    treatment_name: str,
    outcome_name: str,
    *,
    outcome: ExternalLearner,
    treatment: ExternalLearner,
    folds: int = 5,
    weights: Any | None = None,
    fold_policy: FoldPolicy | str = FoldPolicy.HOST_OWNED,
    estimator: Literal["dml", "aipw", "dr_learner", "causal_forest"] = "dml",
    score: Literal["aipw", "partially_linear"] = "aipw",
    execution: Execution | None = None,
) -> ExternalAteEstimate:
    from . import _native

    if not isinstance(outcome, ExternalLearner) or not isinstance(treatment, ExternalLearner):
        raise CausalValueError("outcome and treatment must be ExternalLearner instances")
    _refuse_shuffled_series(outcome)
    _refuse_shuffled_series(treatment)
    _refuse_unhonoured_policy(outcome)
    _refuse_unhonoured_policy(treatment)
    policy = _fold_policy(fold_policy, iid=FoldPolicy.HOST_OWNED)
    _check_estimator(estimator, score, policy)
    outcome, treatment = _snapshot_learner(outcome), _snapshot_learner(treatment)
    names, raw_columns, n = _columns(data)
    selected_weights = None if weights is None else _vector(weights, n, "weights")
    point, se, rows, fold_ids, adjustment, provenance, assumptions, cate, cate_se, receipts = (
        _native.external_dml_ate(
            names,
            [col.tolist() for col in raw_columns],
            list(edges),
            treatment_name,
            outcome_name,
            outcome,
            treatment,
            folds=folds,
            weights=None if selected_weights is None else selected_weights.tolist(),
            method=estimator,
            score=score,
            **native_controls(
                execution,
                seed="seed",
                cancel="cancel",
                memory="memory_limit_bytes",
            ),
        )
    )
    return ExternalAteEstimate(
        estimator,
        score,
        point,
        se,
        tuple(rows),
        tuple(fold_ids),
        tuple(names[index] for index in adjustment),
        tuple(tuple(item) for item in provenance),
        tuple(json.loads(assumptions)),
        None if cate is None else tuple(cate),
        None if cate_se is None else tuple(cate_se),
        tuple(receipts),
        policy,
    )


def aipw_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Identify a static DAG and fit the supported five-fold external AIPW route.

    Deprecated: :func:`antecedent.analyze` with ``estimator="aipw"`` returns the same numbers.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.external_learners.aipw_ate", _ANALYZE.format(estimator="aipw")
    )
    return _direct_ate(data, edges, treatment_name, outcome_name, estimator="aipw", **kwargs)


def dr_learner_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Fit external nuisances with the built-in linear DR-learner final stage.

    Deprecated: :func:`antecedent.analyze` with ``estimator="dr_learner"`` returns the same numbers.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.external_learners.dr_learner_ate",
        _ANALYZE.format(estimator="dr_learner"),
    )
    return _direct_ate(data, edges, treatment_name, outcome_name, estimator="dr_learner", **kwargs)


def causal_forest_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Fit native forest CATE trees and external nuisances for the marginal ATE.

    Deprecated: :func:`antecedent.analyze` with ``estimator="causal_forest"`` returns the same
    numbers.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.external_learners.causal_forest_ate",
        _ANALYZE.format(estimator="causal_forest"),
    )
    return _direct_ate(
        data, edges, treatment_name, outcome_name, estimator="causal_forest", **kwargs
    )


def cross_fit_aipw(
    outcome: ExternalLearner,
    treatment: ExternalLearner,
    x: Any,
    y: Any,
    t: Any,
    fold_ids: Any,
    *,
    weights: Any | None = None,
    fold_policy: FoldPolicy | str = FoldPolicy.CALLER_OWNED,
    execution: Execution | None = None,
) -> CrossFittedAipwNuisances:
    """Run native AIPW nuisance cross-fitting on an explicit fold assignment.

    This produces predictions, not a causal estimate. Every provider receives only
    the rows belonging to its fit or prediction call. The caller must choose a fold
    plan suitable for the data; no temporal validation is inferred here.

    Deprecated: :func:`antecedent.analyze` runs this cross-fit inside the compiled study.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.external_learners.cross_fit_aipw", _ANALYZE.format(estimator="dml")
    )
    from . import _native

    if not isinstance(outcome, ExternalLearner) or not isinstance(treatment, ExternalLearner):
        raise CausalValueError("outcome and treatment must be ExternalLearner instances")
    _refuse_shuffled_series(outcome)
    _refuse_shuffled_series(treatment)
    _refuse_unhonoured_policy(outcome)
    _refuse_unhonoured_policy(treatment)
    policy = _fold_policy(fold_policy, iid=FoldPolicy.CALLER_OWNED)
    outcome, treatment = _snapshot_learner(outcome), _snapshot_learner(treatment)
    matrix = _matrix(x)
    target = _vector(y, matrix.shape[0], "target")
    arms = _vector(t, matrix.shape[0], "treatment")
    folds = np.asarray(fold_ids)
    if (
        folds.shape != (matrix.shape[0],)
        or folds.dtype.kind not in "iu"
        or np.any(folds < 0)
        or np.any(folds > 65535)
    ):
        raise CausalValueError("fold ids must be nonnegative 16-bit integers aligned with rows")
    selected_weights = None if weights is None else _vector(weights, matrix.shape[0], "weights")
    m0, m1, e, provenance, receipts = _native.external_aipw_crossfit(
        outcome,
        treatment,
        matrix.tolist(),
        target.tolist(),
        arms.tolist(),
        folds.tolist(),
        weights=None if selected_weights is None else selected_weights.tolist(),
        **native_controls(
            execution,
            seed="seed",
            cancel="cancel",
            memory="memory_limit_bytes",
        ),
    )
    return CrossFittedAipwNuisances(
        tuple(m0),
        tuple(m1),
        tuple(e),
        tuple(int(v) for v in folds),
        tuple(tuple(item) for item in provenance),
        tuple(receipts),
        policy,
    )


def _matrix(value: Any) -> NDArray[np.float64]:
    matrix = np.array(value, dtype=np.float64, copy=True)
    if (
        matrix.ndim != 2
        or not matrix.shape[0]
        or not matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise CausalValueError("nuisance design must be a nonempty finite matrix")
    matrix.flags.writeable = False
    return matrix


def _vector(value: Any, length: int, name: str) -> NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise CausalValueError(f"nuisance {name} must have one finite value per row")
    vector.flags.writeable = False
    return vector


def _refuse_shuffled_series(learner: ExternalLearner) -> None:
    """Keep an origin-scoped series model off an IID cross-fit."""
    if isinstance(learner, OriginScopedLearner):
        raise CausalValueError("shuffled folds are refused for an origin-scoped series model")


def _refuse_unhonoured_policy(learner: ExternalLearner) -> None:
    """Refuse a fold policy the nuisance estimators cannot honour.

    An :class:`OriginScopedLearner` is the temporal declaration and is checked
    by :class:`OriginScopedLearners` instead.
    """
    if isinstance(learner, OriginScopedLearner):
        return
    policy = learner.registry._entry(learner.name)[0].spec.fold_policy
    if not policy.estimator_honours():
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")


@dataclass(frozen=True, slots=True)
class OriginScopedLearner(ExternalLearner):
    """A registered series model fitted forward from each origin.

    History ends at the origin and predictions are only after it. The provider's
    fold policy is ``origin_scoped``. Shuffled folds are refused. This is not
    :class:`ForecastNuisanceAdapter`: that adapter stays the IID opt-in.
    """

    def fit_at_origins(
        self,
        x: Any,
        y: Any,
        times: Sequence[int],
        origins: Sequence[int],
    ) -> tuple[Any, ...]:
        """Fit one receipt per origin. Predictions are the rows strictly after it."""
        from .forecasting import fit_at_origins

        return fit_at_origins(self, x, y, times, origins)


@dataclass(frozen=True, slots=True)
class OriginScopedLearners:
    """Outcome and treatment series models for one temporal average-effect route."""

    outcome: OriginScopedLearner
    treatment: OriginScopedLearner

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, OriginScopedLearner) or not isinstance(
            self.treatment, OriginScopedLearner
        ):
            raise CausalValueError("outcome and treatment must be OriginScopedLearner instances")
        for learner in (self.outcome, self.treatment):
            policy = learner.registry._entry(learner.name)[0].spec.fold_policy
            if policy is not FoldPolicy.ORIGIN_SCOPED:
                raise CausalValueError(
                    "shuffled folds are refused for an origin-scoped fit; the series model "
                    "must declare an origin_scoped fold policy"
                )


@dataclass(frozen=True, slots=True)
class OriginNuisance:
    """The nuisance fits of one origin, each with its own receipt."""

    origin: int
    fits: tuple[Any, ...]


def _selected_estimator(estimator: Any) -> tuple[str, str, str]:
    from .estimators import learner_estimator

    if estimator is None:
        estimator = "dml"
    if not isinstance(estimator, str):
        raise CausalValueError(
            "the learner study's only knob is estimator: aipw, dml, dr_learner or causal_forest"
        )
    try:
        method, score = learner_estimator(estimator)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return estimator, method, score


def _edge_list(graph: Any) -> list[tuple[str, str]]:
    if graph is None:
        raise CausalValueError("a learner study needs a graph")
    if isinstance(graph, Sequence) and not isinstance(graph, (str, bytes)):
        edges = list(graph)
    else:
        listed = getattr(graph, "edges", None)
        if not callable(listed):
            raise CausalTypeError("graph must be an edge list or a graph")
        edges = list(listed())
    if any(not isinstance(edge, tuple) or len(edge) != 2 for edge in edges):
        raise CausalValueError("a learner study takes a static graph of (cause, effect) edges")
    return [(str(cause), str(effect)) for cause, effect in edges]


def _decline_learner_options(declarations: Mapping[str, Any], *, temporal: bool) -> None:
    from .prepared import _AUTO_DATA_DIAGNOSTICS

    blocked: list[str] = []
    if declarations.get("discovery") is not None:
        blocked.append("discovery")
    if declarations.get("accept_discovered"):
        blocked.append("accept_discovered")
    if declarations.get("inference") is not None:
        blocked.append("inference")
    for name in (
        "identifier",
        "estimator_config",
        "class_prior",
        "max_completions",
        "participant",
        "missing",
        "population_registry",
        "validators",
        "regimes",
        "provider",
        "unit_key",
        "evidence_candidates",
        "outcome_units",
        "dose_units",
    ):
        if declarations.get(name) is not None:
            blocked.append(name)
    if declarations.get("refute") not in (None, False, "none"):
        blocked.append("refute")
    if declarations.get("retain_joint_horizons"):
        blocked.append("retain_joint_horizons")
    if declarations.get("on_refusal") not in (None, "refuse"):
        blocked.append("on_refusal")
    if declarations.get("quantity_population") not in (None, "target"):
        blocked.append("quantity_population")
    if declarations.get("quantity_transform") not in (None, "identity"):
        blocked.append("quantity_transform")
    if declarations.get("data_diagnostics") not in (None, _AUTO_DATA_DIAGNOSTICS):
        blocked.append("data_diagnostics")
    sampling = declarations.get("sampling_design")
    if temporal:
        if sampling not in (None, "dependent_time_series"):
            blocked.append("sampling_design")
    elif sampling is not None:
        blocked.append("sampling_design")
    if blocked:
        raise CausalUnsupportedError(
            "a learner study fits the declared nuisances once; "
            + ", ".join(blocked)
            + " does not apply",
            reason_code="option_not_applicable",
        )


def _require_binary_effect(query: Any) -> None:
    from .query import AverageEffect

    if not isinstance(query, AverageEffect):
        raise CausalUnsupportedError(
            "declared learners apply to a binary average effect",
            reason_code="option_not_applicable",
        )
    if query.control_level != 0 or query.active_level != 1:
        raise CausalUnsupportedError(
            "declared learners apply to the binary contrast of 0 and 1",
            reason_code="option_not_applicable",
        )
    if query.target_population is not None or query.outcome_functional is not None:
        raise CausalUnsupportedError(
            "declared learners apply to the mean effect on the observed rows",
            reason_code="option_not_applicable",
        )


def _analysis_result(
    *,
    point: float,
    se: float,
    estimator: str,
    method: str,
    adjustment: Sequence[str],
    assumptions: Sequence[Any],
    provenance: Mapping[str, Any],
    query: Any,
    cate: tuple[float, ...] | None = None,
    cate_se: tuple[float, ...] | None = None,
) -> Any:
    from .results._views import (
        AnalysisResult,
        EstimateView,
        IdentificationView,
        PerformanceView,
        ValidationView,
    )

    return AnalysisResult(
        identification=IdentificationView(
            status="NonparametricallyIdentified",
            method="adjustment",
            adjustment_set=tuple(adjustment),
            assumption_count=len(assumptions),
            derivation_step_count=1,
        ),
        estimate=EstimateView(
            ate=float(point),
            se_analytic=float(se),
            se_bootstrap=None,
            estimator_id=estimator,
            method=method,
            cate=cate,
            cate_se=cate_se,
        ),
        posterior=None,
        validation=ValidationView(passed=None, ran=False, count=0),
        performance=PerformanceView(),
        diagnostics=(),
        provenance=dict(provenance),
        query=query,
    )


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    point = float(array.mean())
    if array.size < 2:
        return point, 0.0
    return point, float(array.std(ddof=1) / np.sqrt(array.size))


def _static_result(spec: Mapping[str, Any]) -> Any:
    from . import _native

    learners = spec["learners"]
    query = spec["query"]
    _require_binary_effect(query)
    edges = _edge_list(spec["graph"])
    _selected, method, score = _selected_estimator(spec["estimator"])
    outcome = _snapshot_learner(learners.outcome)
    treatment = _snapshot_learner(learners.treatment)
    identity = json.dumps(
        {"outcome": _learner_identity(outcome), "treatment": _learner_identity(treatment)},
        sort_keys=True,
        separators=(",", ":"),
    )
    names, raw_columns, _n = _columns(spec["data"])
    (
        point,
        se,
        _rows,
        fold_ids,
        adjustment,
        learner_provenance,
        assumptions,
        cate,
        cate_se,
        receipts,
        propensity,
    ) = _native.declared_learner_ate(
        names,
        [column.tolist() for column in raw_columns],
        edges,
        query.treatment,
        query.outcome,
        outcome,
        treatment,
        identity=identity,
        folds=5,
        weights=None,
        method=method,
        score=score,
        **native_controls(
            spec["execution"],
            seed="seed",
            cancel="cancel",
            memory="memory_limit_bytes",
        ),
    )
    parsed = json.loads(assumptions)
    assumption_items = tuple(parsed) if isinstance(parsed, list) else (parsed,)
    return _analysis_result(
        point=point,
        se=se,
        estimator=method,
        method=score,
        adjustment=tuple(names[index] for index in adjustment),
        assumptions=assumption_items,
        provenance={
            "fold_receipts": tuple(receipts),
            "fold_ids": tuple(int(fold) for fold in fold_ids),
            "assumptions": assumption_items,
            "learner_provenance": tuple(tuple(item) for item in learner_provenance),
            "propensity": None if propensity is None else tuple(float(value) for value in propensity),
            "estimator": method,
            "score": score,
        },
        query=query,
        cate=None if cate is None else tuple(float(value) for value in cate),
        cate_se=None if cate_se is None else tuple(float(value) for value in cate_se),
    )


def _series_frame(data: Mapping[str, Any], query: Any) -> tuple[Any, list[int], NDArray[np.float64], tuple[str, ...]]:
    names, columns, n = _columns(data)
    if query.treatment not in names or query.outcome not in names:
        raise CausalValueError("treatment and outcome must be columns of the learner study")
    time_name = "time" if "time" in names else None
    if time_name is None:
        times = list(range(n))
    else:
        times = []
        for value in columns[names.index(time_name)]:
            whole = int(value)
            if value != whole:
                raise CausalValueError("time must be whole numbers")
            times.append(whole)
    skip = {query.treatment, query.outcome}
    if time_name is not None:
        skip.add(time_name)
    kept = tuple(name for name in names if name not in skip)
    if kept:
        matrix = np.column_stack([columns[names.index(name)] for name in kept])
    else:
        matrix = np.ones((n, 1), dtype=np.float64)
    return columns, times, matrix, kept


def _forward_origins(times: Sequence[int]) -> list[tuple[int, list[int], list[int]]]:
    ordered = sorted(set(times))
    windows = []
    for origin, nxt in zip(ordered, ordered[1:]):
        history = [index for index, time in enumerate(times) if time <= origin]
        predict = [index for index, time in enumerate(times) if time == nxt]
        if history and predict:
            windows.append((origin, history, predict))
    return windows


def _temporal_result(spec: Mapping[str, Any]) -> Any:
    from .forecasting import fit_origin

    learners = spec["learners"]
    query = spec["query"]
    _require_binary_effect(query)
    _edge_list(spec["graph"])
    _selected, method, score = _selected_estimator(spec["estimator"])
    if method not in ("aipw", "dml"):
        raise CausalUnsupportedError(
            "an origin-scoped series model supplies a temporal AIPW or DML route",
            reason_code="option_not_applicable",
        )
    columns, times, matrix, adjustment = _series_frame(spec["data"], query)
    names = list(spec["data"])
    treatment = columns[names.index(query.treatment)]
    outcome = columns[names.index(query.outcome)]
    windows = _forward_origins(times)
    groups: list[OriginNuisance] = []
    if method == "aipw":
        scores: list[float] = []
        for origin, history, predict in windows:
            hist0 = [index for index in history if treatment[index] == 0]
            hist1 = [index for index in history if treatment[index] == 1]
            if not hist0 or not hist1:
                continue
            m0 = fit_origin(
                learners.outcome, matrix, outcome, origin=origin, times=times, history=hist0, predict=predict
            )
            m1 = fit_origin(
                learners.outcome, matrix, outcome, origin=origin, times=times, history=hist1, predict=predict
            )
            propensity = fit_origin(
                learners.treatment,
                matrix,
                treatment,
                origin=origin,
                times=times,
                history=history,
                predict=predict,
            )
            for offset, row in enumerate(predict):
                e = min(max(propensity.predictions[offset], 1e-6), 1.0 - 1e-6)
                treated = float(treatment[row])
                y = float(outcome[row])
                scores.append(
                    (m1.predictions[offset] - m0.predictions[offset])
                    + (treated / e) * (y - m1.predictions[offset])
                    - ((1.0 - treated) / (1.0 - e)) * (y - m0.predictions[offset])
                )
            groups.append(OriginNuisance(origin, (m0, m1, propensity)))
        if not scores:
            raise CausalValueError("the forward plan scored no row")
        point, se = _mean_se(scores)
    else:
        residuals: list[tuple[float, float]] = []
        for origin, history, predict in windows:
            y_hat = fit_origin(
                learners.outcome, matrix, outcome, origin=origin, times=times, history=history, predict=predict
            )
            t_hat = fit_origin(
                learners.treatment,
                matrix,
                treatment,
                origin=origin,
                times=times,
                history=history,
                predict=predict,
            )
            for offset, row in enumerate(predict):
                residuals.append(
                    (float(treatment[row]) - t_hat.predictions[offset], float(outcome[row]) - y_hat.predictions[offset])
                )
            groups.append(OriginNuisance(origin, (y_hat, t_hat)))
        if not residuals:
            raise CausalValueError("the forward plan scored no row")
        numerator = sum(treated * y for treated, y in residuals)
        denominator = sum(treated * treated for treated, _y in residuals)
        if denominator == 0:
            raise CausalValueError("temporal DML treatment residual has no variation")
        point = numerator / denominator
        mean_square = denominator / len(residuals)
        influence = [
            (treated * y - point * treated * treated) / mean_square for treated, y in residuals
        ]
        se = _mean_se(influence)[1]
    assumptions = (
        "origin_scoped_fit",
        "forward_only",
        f"estimator:{method}",
        f"score:{score}",
    )
    return _analysis_result(
        point=point,
        se=se,
        estimator=method,
        method=score,
        adjustment=adjustment,
        assumptions=assumptions,
        provenance={
            "fold_receipts": tuple(groups),
            "assumptions": assumptions,
            "estimator": method,
            "score": score,
            "forward_only": True,
        },
        query=query,
    )


class _DeclaredLearnerEngine:
    """Study engine that fits the declared learners when the study is estimated."""

    _kind = "declared_learners"
    _native = None
    _transport = None
    _last_result = None
    participants = None
    validator_names = ()

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._spec = dict(spec)

    def estimate(self) -> Any:
        learners = self._spec["learners"]
        if isinstance(learners, OriginScopedLearners):
            return _temporal_result(self._spec)
        if isinstance(learners, Learners):
            return _static_result(self._spec)
        raise CausalValueError("learners must be a Learners or OriginScopedLearners declaration")

    def _bind(self, data: Any, execution: Execution) -> _DeclaredLearnerEngine:
        spec = dict(self._spec)
        spec["data"] = data
        spec["execution"] = execution
        return _DeclaredLearnerEngine(spec)

    def _branch(self, execution: Execution) -> _DeclaredLearnerEngine:
        spec = dict(self._spec)
        spec["execution"] = execution
        return _DeclaredLearnerEngine(spec)

    def plan_view(self) -> Any:
        raise CausalUnsupportedError(
            "a learner study compiles when it is estimated",
            reason_code="plan_view_unavailable",
        )

    def preflight(self) -> Any:
        return self.inspect()

    def inspect(self) -> Any:
        from .results._report import InspectionReport, SlotModel

        slot = SlotModel(
            available=False,
            reason="declared_learners",
            summary="the declared learners are fitted when the study is estimated",
        )
        return InspectionReport(
            identification=slot,
            support=slot,
            uncertainty=slot,
            assumptions=slot,
            stage="prepared",
            operations=("inspect", "estimate", "rebind", "reconfigure"),
        )


def prepare_learner_study(
    data: Any,
    *,
    execution: Execution,
    declarations: Mapping[str, Any],
) -> Any:
    """Prepare a study whose nuisance fits are the declared learners.

    Nothing is fitted. :func:`antecedent.estimate` runs the static average-effect
    route, or the origin-scoped temporal route when ``learners`` is an
    :class:`OriginScopedLearners`.
    """
    from .prepared import PreparedAnalysis

    learners = declarations["learners"]
    temporal = isinstance(learners, OriginScopedLearners)
    if not temporal and not isinstance(learners, Learners):
        raise CausalValueError("learners must be a Learners or OriginScopedLearners declaration")
    _decline_learner_options(declarations, temporal=temporal)
    _require_binary_effect(declarations["query"])
    _edge_list(declarations["graph"])
    _selected_estimator(declarations["estimator"])
    if not isinstance(data, Mapping):
        raise CausalValueError("learner data must map column names to numeric columns")
    _columns(data)
    engine = _DeclaredLearnerEngine(
        {
            "data": data,
            "execution": execution,
            "learners": learners,
            "query": declarations["query"],
            "graph": declarations["graph"],
            "estimator": declarations["estimator"],
        }
    )
    return PreparedAnalysis(
        _engine=engine,
        execution=execution,
        _declarations=declarations,
        _data=data,
    )
