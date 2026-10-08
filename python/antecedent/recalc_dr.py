"""Selective DML AIPW and DR-Learner execution using the shared native stage planner.

Compatible utilities and same-row target weights reuse retained marginal AIPW
scores. CATE prediction separately uses the fitted final-stage map, with measured
zero fits and no pointwise interval claim. Changed snapshots, graphs, folds or
learner settings refit their dependencies. Partially linear DML and unsupported
score/population operations refuse rather than inheriting an AIPW license.

Receipts record historical work; they supply no executable model. Fresh-process
retargeting requires verified portable scores, prediction requires a verified
portable predictor, and a supplied raw snapshot permits a new fit. Both portable
objects reuse the existing artifact consumers and retain their exact identities.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import ArrayLike

from . import _native
from ._recalc_bounds import MAX_COLUMNS, MAX_ROWS, MAX_VALUES, _columns
from .errors import CausalSerializationError, CausalTypeError, CausalValueError
from .estimators import DML, DRLearner
from .prediction import FittedEffectModel
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
from .recalc_cell import FrozenScores, _frozen


@dataclass(frozen=True, slots=True, eq=False)
class DrRequest:
    """Graph-certified binary-treatment marginal mean effect and estimator config.

    The native prepared study selects and certifies the adjustment coordinates;
    callers cannot assert a different score schema or fold assignment.
    """

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatment: str
    outcome: str
    utility: Utility
    estimator: DML | DRLearner = DML()
    target: TargetWeights | None = None


@dataclass(frozen=True, slots=True)
class CatePrediction:
    """Retained CATE point values and measured fit count; uncertainty unavailable."""

    values: tuple[float, ...]
    model_fits: int

    def to_dict(self) -> dict[str, object]:
        """Prediction-only disclosure; no marginal score or interval license."""
        return {
            "values": list(self.values),
            "model_fits": self.model_fits,
            "uncertainty": {"status": "unavailable", "reason": "prediction_only"},
        }


def _spec(request: DrRequest) -> dict[str, Any]:
    estimator = request.estimator
    if not isinstance(estimator, (DML, DRLearner)):
        raise CausalTypeError("estimator must be DML or DRLearner", reason_code="invalid_argument")
    target = request.target
    return {
        "edges": [list(edge) for edge in request.edges],
        "treatment": request.treatment,
        "outcome": request.outcome,
        "kind": "dml" if isinstance(estimator, DML) else "cate",
        "config": estimator._wire(),
        "benefit_per_unit": float(request.utility.benefit_per_unit),
        "cost": float(request.utility.cost),
        "target_weights": None
        if target is None
        else _column("target weights", target.weights).tolist(),
        "target_depends_on": [] if target is None else list(target.depends_on),
    }


class DrSession:
    """Checked retained marginal scores and, for DR-Learner, a CATE point map."""

    __slots__ = ("_handle",)

    def __init__(self) -> None:
        self._handle = _native.DrSessionHandle()

    @classmethod
    def resume(
        cls, previous: Mapping[Stage, str] | RecalcReceipt, context: ResumeContext | None = None
    ) -> DrSession:
        """Resume identities; supplied raw data refit, flags do not supply fits/scores."""
        declared = previous.requested if isinstance(previous, RecalcReceipt) else previous
        ctx = context or ResumeContext()
        session = cls.__new__(cls)
        session._handle = _native.DrSessionHandle.resume(
            _declared_json(declared), json.dumps(ctx.to_wire())
        )
        return session

    @property
    def is_live(self) -> bool:
        """Whether the native prepared study and actual scores remain live."""
        return self._handle.is_live()

    @property
    def identities(self) -> dict[Stage, str]:
        """Declared identities of the last successful run or resume boundary."""
        return _declared(json.loads(self._handle.identities_json()))

    @property
    def capabilities(self) -> Capabilities:
        """Capabilities at the actual native retained-state boundary."""
        return Capabilities.from_wire(json.loads(self._handle.capabilities_json()))

    @property
    def prediction_columns(self) -> tuple[str, ...] | None:
        """Native CATE feature names/order; marginal-only DML retains no predictor."""
        names = self._handle.prediction_columns()
        return None if names is None else tuple(names)

    @property
    def row_ids(self) -> tuple[int, ...] | None:
        """Original input row indices retained by the complete-case score table."""
        rows = self._handle.row_ids()
        return None if rows is None else tuple(rows)

    def score_contrast(self) -> tuple[float, ...] | None:
        """Marginal AIPW score values, independently of the fitted CATE map."""
        scores = self._handle.score_contrast()
        return None if scores is None else tuple(scores)

    def plan(self, request: DrRequest, *, seed: int = 1, threads: int | None = None) -> RecalcPlan:
        """Plan native preparation, fitting, retargeting and utility without running."""
        names, columns = _columns(request.data)
        wire = self._handle.plan(
            names, columns, json.dumps(_spec(request)), seed=_seed(seed), threads=threads
        )
        return RecalcPlan.from_wire(json.loads(wire))

    def execute(
        self, request: DrRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcResult:
        """Run only invalidated stages; seal a shared receipt with actual fit counts."""
        names, columns = _columns(request.data)
        result, artifact, refusal = self._handle.execute(
            names, columns, json.dumps(_spec(request)), seed=_seed(seed), threads=threads
        )
        _raise(refusal)
        if result is None or artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("DR recalculation returned no result")
        wire = json.loads(result)
        return RecalcResult(
            RecalcPlan.from_wire(wire["plan"]),
            RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
            Law(**wire["law"]),
            Decision(**wire["decision"]),
        )

    def predict(
        self,
        rows: Sequence[Sequence[float]],
        *,
        columns: Sequence[str],
        seed: int = 1,
        threads: int | None = None,
    ) -> CatePrediction:
        """Predict named CATE feature rows with the retained final-stage map.

        This distinct operation measures actual zero model fits, supplies no new
        shared-stage receipt, and grants no score-retarget or inference license.
        """
        if (
            len(rows) > MAX_ROWS
            or len(columns) > MAX_COLUMNS
            or any(len(row) > MAX_COLUMNS for row in rows)
            or sum(len(row) for row in rows) > MAX_VALUES
        ):
            raise CausalValueError(
                "recalc.limits_exceeded: predictions exceed size limits",
                reason_code="invalid_argument",
            )
        try:
            features = [[float(value) for value in row] for row in rows]
        except (TypeError, ValueError) as error:
            raise CausalValueError(
                "prediction rows must contain numeric features", reason_code="invalid_argument"
            ) from error
        report, refusal = self._handle.predict(
            features, list(columns), seed=_seed(seed), threads=threads
        )
        _raise(refusal)
        if report is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("DR prediction returned no result")
        wire = json.loads(report)
        return CatePrediction(tuple(wire["values"]), wire["model_fits"])

    def export_scores(self) -> FrozenScores:
        """Verified portable marginal AIPW scores; use existing score resume consumer."""
        return _frozen(self._handle.export_scores())

    def export_predictor(self, *, seed: int = 1, threads: int | None = None) -> FittedEffectModel:
        """Load the retained CATE map from its existing verified parent result artifact."""
        artifact, refusal = self._handle.export_predictor(seed=_seed(seed), threads=threads)
        _raise(refusal)
        if artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("DR predictor export returned no artifact")
        return FittedEffectModel.load(artifact)


__all__ = ["CatePrediction", "DrRequest", "DrSession"]
