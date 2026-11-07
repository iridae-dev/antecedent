"""Explicit provider-backed nuisance learner fit and predict operations.

Each operation executes through :class:`ProviderRegistry`, so verification is evaluated
for that exact request. This module does not by itself provide cross-fitting or establish
that a pretrained model excludes evaluation rows.
"""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .errors import CausalValueError
from .execution import Execution, native_controls
from .extensibility import (
    ProviderCapability,
    ProviderRegistry,
    ProviderResult,
    _digest,
    _request_digest,
    _spec_evidence,
)


@dataclass(frozen=True, slots=True)
class ExternalLearner:
    """A registered provider's declared nuisance task and fit behavior.

    Providers use ``query_family='nuisance_learner'`` and declare ``mean``. Fit returns
    a model in ``ProviderExecution.artifact``; prediction receives it as base64 in its
    request. A variable ``output_shape`` declaration is needed because folds vary in size.
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
        if (
            spec.query_family != "nuisance_learner"
            and not (spec.query_family == "forecast" and forecast_adapter)
        ) or ProviderCapability.MEAN not in spec.capabilities:
            raise CausalValueError("provider must declare nuisance_learner and the mean capability")
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
    fold_policy: Literal["iid", "temporal"] = "iid",
    estimator: Literal["dml", "aipw", "dr_learner", "causal_forest"] = "dml",
    score: Literal["aipw", "partially_linear"] = "aipw",
    execution: Execution | None = None,
) -> ExternalAteEstimate:
    """Identify a static DAG and fit its binary-treatment DML AIPW estimate.

    The supplied graph is the caller's causal assumption. Only IID cross-fitting
    is implemented; temporal training and validation are explicitly refused.
    Provider receipts are returned for every fold and treatment arm.
    """
    from . import _native

    if not isinstance(outcome, ExternalLearner) or not isinstance(treatment, ExternalLearner):
        raise CausalValueError("outcome and treatment must be ExternalLearner instances")
    if fold_policy != "iid":
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")
    if estimator not in ("dml", "aipw", "dr_learner", "causal_forest"):
        raise CausalValueError("unknown external learner estimator")
    if score not in ("aipw", "partially_linear") or (estimator != "dml" and score != "aipw"):
        raise CausalValueError("invalid score for the requested estimator")
    outcome, treatment = _snapshot_learner(outcome), _snapshot_learner(treatment)
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
    )


def aipw_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Identify a static DAG and fit the supported five-fold external AIPW route."""
    return dml_ate(data, edges, treatment_name, outcome_name, estimator="aipw", **kwargs)


def dr_learner_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Fit external nuisances with the built-in linear DR-learner final stage."""
    return dml_ate(data, edges, treatment_name, outcome_name, estimator="dr_learner", **kwargs)


def causal_forest_ate(
    data: Mapping[str, Any], edges: Any, treatment_name: str, outcome_name: str, **kwargs: Any
) -> ExternalAteEstimate:
    """Fit native forest CATE trees and external nuisances for the marginal ATE."""
    return dml_ate(data, edges, treatment_name, outcome_name, estimator="causal_forest", **kwargs)


def cross_fit_aipw(
    outcome: ExternalLearner,
    treatment: ExternalLearner,
    x: Any,
    y: Any,
    t: Any,
    fold_ids: Any,
    *,
    weights: Any | None = None,
    fold_policy: Literal["iid", "temporal"] = "iid",
    execution: Execution | None = None,
) -> CrossFittedAipwNuisances:
    """Run native AIPW nuisance cross-fitting on an explicit fold assignment.

    This produces predictions, not a causal estimate. Every provider receives only
    the rows belonging to its fit or prediction call. The caller must choose a fold
    plan suitable for the data; no temporal validation is inferred here.
    """
    from . import _native

    if not isinstance(outcome, ExternalLearner) or not isinstance(treatment, ExternalLearner):
        raise CausalValueError("outcome and treatment must be ExternalLearner instances")
    if fold_policy != "iid":
        raise CausalValueError("temporal cross-fitting is not supported by this fold contract")
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
