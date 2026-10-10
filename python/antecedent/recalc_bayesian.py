"""Native-issued ordinary Gaussian posterior recalculation, with measured work.

Posterior summaries and utility consume retained native draws. These posterior
quantiles describe that model; they supply no new calibration claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import ArrayLike

from . import _native
from ._recalc_bounds import _columns
from .errors import CausalSerializationError
from .inference import Bayesian
from .recalc import Decision, RecalcPlan, RecalcReceipt, Utility, _raise, _seed
from .recalc_static import _Session


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    lower_probability: float = 0.025
    upper_probability: float = 0.975
    threshold: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class BayesianRequest:
    """Checked binary-treatment Gaussian identity model and actual input data.

    Existing Bayesian prior-transfer configurations are checked by the original
    Study engine. Other likelihood/backends remain unsupported by this adapter.
    """

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatment: str
    outcome: str
    utility: Utility
    model: Literal["gaussian", "quadratic_basis"] = "gaussian"
    inference: Bayesian = field(default_factory=lambda: Bayesian(backend="conjugate", n_draws=256))
    summary: PosteriorSummary = field(default_factory=PosteriorSummary)


@dataclass(frozen=True, slots=True)
class BayesianLaw:
    mean: float
    standard_deviation: float
    lower_quantile: float
    upper_quantile: float
    probability_below: float
    draws: int


@dataclass(frozen=True, slots=True)
class BayesianResult:
    plan: RecalcPlan
    receipt: RecalcReceipt
    law: BayesianLaw
    decision: Decision


def _spec(request: BayesianRequest) -> str:
    return json.dumps(
        {
            "edges": request.edges,
            "treatment": request.treatment,
            "outcome": request.outcome,
            "model": request.model,
            "lower_probability": request.summary.lower_probability,
            "upper_probability": request.summary.upper_probability,
            "threshold": request.summary.threshold,
            "benefit_per_unit": request.utility.benefit_per_unit,
            "cost": request.utility.cost,
        }
    )


def _result(payload: tuple[str | None, bytes | None, str | None]) -> BayesianResult:
    result, artifact, error = payload
    _raise(error)
    if result is None or artifact is None:
        raise CausalSerializationError("Bayesian native execution returned no result")
    wire = json.loads(result)
    return BayesianResult(
        RecalcPlan.from_wire(wire["plan"]),
        RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
        BayesianLaw(**wire["law"]),
        Decision(**wire["decision"]),
    )


class BayesianSession(_Session):
    """Actual checked model, producing result and aligned effect posterior rows."""

    __slots__ = ()

    @staticmethod
    def _native_type() -> Any:
        return _native.BayesianSessionHandle

    @property
    def effect_draws(self) -> tuple[float, ...] | None:
        """Copy of actual retained effect rows; never accepted as model state."""
        rows = self._handle.effect_draws()
        return None if rows is None else tuple(rows)

    def export_prior_source(self) -> bytes:
        """Source-bound native posterior for independently checked prior transfer.

        Its raw source and original draws are reexecuted/verified by the target;
        caller posterior metadata and same-source likelihood reuse are refused.
        """
        artifact, error = self._handle.export_prior_source()
        _raise(error)
        if artifact is None:
            raise CausalSerializationError("native posterior source export unavailable")
        return bytes(artifact)

    def plan(
        self, request: BayesianRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        from .estimation import _inference_wire

        names, columns = _columns(request.data)
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(
                    names,
                    columns,
                    _spec(request),
                    _inference_wire(request.inference),
                    seed=_seed(seed),
                    threads=threads,
                )
            )
        )

    def execute(
        self, request: BayesianRequest, *, seed: int = 1, threads: int | None = None
    ) -> BayesianResult:
        from .estimation import _inference_wire

        names, columns = _columns(request.data)
        return _result(
            self._handle.execute(
                names,
                columns,
                _spec(request),
                _inference_wire(request.inference),
                seed=_seed(seed),
                threads=threads,
            )
        )


__all__ = [
    "BayesianRequest",
    "BayesianSession",
    "BayesianLaw",
    "BayesianResult",
    "PosteriorSummary",
]
