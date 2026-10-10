"""Selective adjusted regression using the shared native planner and receipt.

A session retains one checked joint fit. Compatible contrasts, target weights and
utility changes reuse it; data, graph, treatment coding and model changes refit.
Prediction is restricted to the fitted schema and observed treatment support.
Receipts are portable records. They contain no fitted model or raw training data;
a fresh process resumes only by supplying data and recomputing the fit.

Analytic standard errors retain their producing model assumptions and unmeasured
calibration standing. Fit reuse adds no confidence-interval or coverage license.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from numpy.typing import ArrayLike

from . import _native
from ._recalc_bounds import MAX_COLUMNS, MAX_ROWS, MAX_VALUES, _columns
from .errors import CausalSerializationError, CausalTypeError, CausalValueError
from .recalc import (
    Capabilities,
    Decision,
    Law,
    RecalcPlan,
    RecalcReceipt,
    RecalcResult,
    ResumeContext,
    Stage,
    TargetWeights,
    Utility,
    _column,
    _declared,
    _declared_json,
    _raise,
    _seed,
)

Covariance = Literal["model_based", "hc0", "hc1", "hc2", "hc3"]
GlmFamily = Literal[
    "binomial_logit", "binomial_probit", "gaussian_identity", "poisson_log", "negative_binomial"
]


@dataclass(frozen=True, slots=True)
class LinearModel:
    """One joint OLS fit, retaining its complete coefficient covariance."""

    covariance: Covariance = "model_based"


@dataclass(frozen=True, slots=True)
class GlmModel:
    """Outcome-model g-computation at the declared GLM family and fit options."""

    family: GlmFamily = "binomial_logit"
    max_iter: int = 50
    tolerance: float = 1e-8


@dataclass(frozen=True, slots=True)
class CategoricalModel:
    """Declared level labels aligned with the single treatment role column.

    The numeric role column must map one-to-one to the observed labels.
    Changed role coding or reference changes invalidate the fit contract and refit.
    """

    labels: Sequence[str]
    levels: tuple[str, ...]
    reference: str
    ordered: bool = False
    min_level_rows: int = 1
    covariance: Covariance = "model_based"


@dataclass(frozen=True, slots=True)
class NumericContrast:
    """Active and control action vectors in declared treatment order."""

    active: tuple[float, ...] = (1.0,)
    control: tuple[float, ...] = (0.0,)


@dataclass(frozen=True, slots=True)
class CategoricalContrast:
    """Mean outcome difference ``to_level - from_level``."""

    from_level: str
    to_level: str


@dataclass(frozen=True, slots=True, eq=False)
class AdjustedRequest:
    """Raw snapshot, checked graph, adjustment design and one mean contrast."""

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatments: tuple[str, ...]
    outcome: str
    adjustment: tuple[str, ...]
    utility: Utility
    model: LinearModel | GlmModel | CategoricalModel = LinearModel()
    contrast: NumericContrast | CategoricalContrast = NumericContrast()
    target: TargetWeights | None = None


def _spec(request: AdjustedRequest) -> dict[str, Any]:
    model = request.model
    model_wire: dict[str, Any]
    if isinstance(model, LinearModel):
        model_wire = {"kind": "linear", "covariance": model.covariance}
    elif isinstance(model, GlmModel):
        model_wire = {
            "kind": "glm",
            "family": model.family,
            "max_iter": model.max_iter,
            "tolerance": model.tolerance,
        }
    elif isinstance(model, CategoricalModel):
        model_wire = {
            "kind": "categorical",
            "labels": list(model.labels),
            "levels": list(model.levels),
            "reference": model.reference,
            "ordered": model.ordered,
            "min_level_rows": model.min_level_rows,
            "covariance": model.covariance,
        }
    else:
        raise CausalTypeError(
            "model must be LinearModel, GlmModel or CategoricalModel",
            reason_code="invalid_argument",
        )
    contrast = request.contrast
    if isinstance(contrast, NumericContrast):
        contrast_wire = {
            "kind": "numeric",
            "active": list(contrast.active),
            "control": list(contrast.control),
        }
    elif isinstance(contrast, CategoricalContrast):
        contrast_wire = {
            "kind": "categorical",
            "from": contrast.from_level,
            "to": contrast.to_level,
        }
    else:
        raise CausalTypeError(
            "contrast must be NumericContrast or CategoricalContrast",
            reason_code="invalid_argument",
        )
    target = request.target
    return {
        "edges": [list(edge) for edge in request.edges],
        "treatments": list(request.treatments),
        "outcome": request.outcome,
        "adjustment": list(request.adjustment),
        "model": model_wire,
        "contrast": contrast_wire,
        "benefit_per_unit": float(request.utility.benefit_per_unit),
        "cost": float(request.utility.cost),
        "target_weights": None
        if target is None
        else _column("target weights", target.weights).tolist(),
        "target_depends_on": [] if target is None else list(target.depends_on),
    }


@dataclass(frozen=True, slots=True)
class AdjustedPrediction:
    """Retained model means and measured fit count; no new inference claim."""

    values: tuple[float, ...]
    model_fits: int


class AdjustedSession:
    """Retained native adjusted fit with instrumented selective execution.

    ``receipt.totals.model_fits`` counts completed model fits, independently of
    cross-fit nuisance ``fold_fits``. Unavailable process state and unsupported
    action/model coordinates produce the shared structured recalculation refusals.
    """

    __slots__ = ("_handle",)

    def __init__(self) -> None:
        self._handle = _native.AdjustedSessionHandle()

    @classmethod
    def resume(
        cls, previous: Mapping[Stage, str] | RecalcReceipt, context: ResumeContext | None = None
    ) -> AdjustedSession:
        """Resume identities; supplied raw data permit a new fit, never reuse.

        Caller flags cannot recreate portable adjusted fits. No fitted-state
        artifact loader is provided by this family adapter.
        """
        declared = previous.requested if isinstance(previous, RecalcReceipt) else previous
        ctx = context or ResumeContext()
        session = cls.__new__(cls)
        session._handle = _native.AdjustedSessionHandle.resume(
            _declared_json(declared), json.dumps(ctx.to_wire())
        )
        return session

    @property
    def is_live(self) -> bool:
        """Whether a checked native fit is retained in this process."""
        return self._handle.is_live()

    @property
    def identities(self) -> dict[Stage, str]:
        """Declared inputs of the last successful execution or resume boundary."""
        return _declared(json.loads(self._handle.identities_json()))

    @property
    def capabilities(self) -> Capabilities:
        """Native capabilities at the actual retained-state boundary."""
        return Capabilities.from_wire(json.loads(self._handle.capabilities_json()))

    @property
    def prediction_columns(self) -> tuple[str, ...] | None:
        """Native canonical feature names/order, or ``None`` without a retained fit."""
        columns = self._handle.prediction_columns()
        return None if columns is None else tuple(columns)

    def predict(
        self, rows: Sequence[Sequence[float]], *, columns: Sequence[str]
    ) -> AdjustedPrediction:
        """Predict rows in native adjustment-then-treatment design order.

        Categorical treatment columns are dummy indicators in the canonical
        declared level order excluding the reference. Native validation rejects
        incompatible schemas, unsupported actions and absent retained state.
        Values are fitted conditional means, with no prediction intervals.
        """
        if (
            len(rows) > MAX_ROWS
            or len(columns) > MAX_COLUMNS
            or sum(len(row) for row in rows) > MAX_VALUES
        ):
            raise CausalValueError(
                "recalc.limits_exceeded: adjusted predictions exceed size limits",
                reason_code="invalid_argument",
            )
        try:
            features = [[float(value) for value in row] for row in rows]
        except (TypeError, ValueError) as error:
            raise CausalValueError("prediction rows must contain numeric features") from error
        report, refusal = self._handle.predict(features, list(columns))
        _raise(refusal)
        if report is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("adjusted prediction returned no result")
        wire = json.loads(report)
        return AdjustedPrediction(tuple(wire["values"]), wire["model_fits"])

    def plan(
        self, request: AdjustedRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        """Inspect the shared stage plan without fitting or predicting."""
        names, columns = _columns(request.data)
        wire = self._handle.plan(
            names,
            columns,
            json.dumps(_spec(request)),
            seed=_seed(seed),
            threads=threads,
        )
        return RecalcPlan.from_wire(json.loads(wire))

    def execute(
        self, request: AdjustedRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcResult:
        """Run invalidated stages and return the shared law, decision and sealed receipt."""
        names, columns = _columns(request.data)
        result, artifact, refusal = self._handle.execute(
            names,
            columns,
            json.dumps(_spec(request)),
            seed=_seed(seed),
            threads=threads,
        )
        _raise(refusal)
        if result is None or artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("adjusted recalculation returned no result")
        wire = json.loads(result)
        return RecalcResult(
            RecalcPlan.from_wire(wire["plan"]),
            RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
            Law(**wire["law"]),
            Decision(**wire["decision"]),
        )


__all__ = [
    "AdjustedPrediction",
    "AdjustedRequest",
    "AdjustedSession",
    "CategoricalContrast",
    "CategoricalModel",
    "Covariance",
    "GlmFamily",
    "GlmModel",
    "LinearModel",
    "NumericContrast",
]
