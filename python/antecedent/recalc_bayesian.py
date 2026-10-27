"""Native-issued ordinary Gaussian posterior recalculation, with measured work.

Posterior summaries and utility consume retained native draws. These posterior
quantiles describe that model; they supply no new calibration claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from numpy.typing import ArrayLike

from . import _native
from ._artifact_protocol import ExportedArtifact, artifact_export, artifact_from_receipt
from ._recalc_bounds import _columns
from ._refusal import raise_refusal
from .errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from .inference import Bayesian
from .outcomes import outcome_operation
from .recalc import Decision, RecalcPlan, RecalcReceipt, Utility, _refusal_type, _seed
from .recalc_static import _Session

if TYPE_CHECKING:
    from .transport._joint_posterior import GaussianTransportPrior, JointTransportIdentity


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
    raise_refusal(error, _refusal_type)
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

    @artifact_export
    def export(
        self,
        *,
        quantity: Literal["result", "prior_source"] = "result",
        seed: int = 1,
        threads: int | None = None,
    ) -> ExportedArtifact:
        """Source-bound native posterior for independently checked prior transfer.

        Its raw source and original draws are reexecuted/verified by the target;
        caller posterior metadata and same-source likelihood reuse are refused.
        """
        if quantity == "result":
            return _Session.export(self, seed=seed, threads=threads)
        if quantity != "prior_source":
            raise CausalTypeError("quantity must be 'result' or 'prior_source'")
        artifact, error = self._handle.export_prior_source()
        raise_refusal(error, _refusal_type)
        if artifact is None:
            raise CausalSerializationError("native posterior source export unavailable")
        return artifact_from_receipt(self, bytes(artifact))

    def plan(
        self,
        request: BayesianRequest,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> RecalcPlan:
        from .estimation import _inference_wire

        names, columns = _columns(
            request.data, memory_limit_bytes=memory_limit_bytes, cancel=cancel
        )
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(
                    names,
                    columns,
                    _spec(request),
                    _inference_wire(request.inference),
                    seed=_seed(seed),
                    threads=threads,
                    memory_limit_bytes=memory_limit_bytes,
                    cancel=cancel,
                )
            )
        )

    @outcome_operation
    def execute(
        self,
        request: BayesianRequest,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> BayesianResult:
        """Run the invalidated stages under an optional memory limit and cancellation token."""
        from .estimation import _inference_wire

        names, columns = _columns(
            request.data, memory_limit_bytes=memory_limit_bytes, cancel=cancel
        )
        return _result(
            self._handle.execute(
                names,
                columns,
                _spec(request),
                _inference_wire(request.inference),
                seed=_seed(seed),
                threads=threads,
                memory_limit_bytes=memory_limit_bytes,
                cancel=cancel,
            )
        )


@dataclass(frozen=True, slots=True)
class JointRecalcSource:
    """One source population's randomized binary-treatment units with known noise variance.

    ``identity`` names the snapshot and every unit once (``len(identity.datum_ids)`` equals the
    number of units); ``covariates`` holds one column per declared feature.
    """

    id: str
    identity: JointTransportIdentity
    treatment: Sequence[bool]
    outcome: Sequence[float]
    noise_variance: float
    covariates: Sequence[Sequence[float]] = ()

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CausalValueError("source id must be a nonempty string")
        from .transport._joint_posterior import JointTransportIdentity

        if not isinstance(self.identity, JointTransportIdentity):
            raise CausalTypeError("source identity must be a JointTransportIdentity")
        return {
            "id": self.id,
            "identity": self.identity._wire(),
            "treatment": [bool(t) for t in self.treatment],
            "outcome": [float(v) for v in self.outcome],
            "covariates": [[float(v) for v in column] for column in self.covariates],
            "noise_variance": float(self.noise_variance),
        }


@dataclass(frozen=True, slots=True)
class JointRecalcTarget:
    """The target population's covariates; target outcomes are never accepted."""

    identity: JointTransportIdentity
    rows: int
    covariates: Sequence[Sequence[float]] = ()

    def _wire(self) -> dict[str, Any]:
        from .transport._joint_posterior import JointTransportIdentity

        if not isinstance(self.identity, JointTransportIdentity):
            raise CausalTypeError("target identity must be a JointTransportIdentity")
        return {
            "identity": self.identity._wire(),
            "rows": int(self.rows),
            "covariates": [[float(v) for v in column] for column in self.covariates],
        }


@dataclass(frozen=True, slots=True, eq=False)
class JointBayesianRequest:
    """Selection diagram, query, joint model, source units and target covariates.

    ``names``/``edges`` declare the directed selection diagram (a bidirected edge is refused),
    ``selections`` its selection nodes, and the response is the binary-treatment mean curve of
    ``outcome`` over ``treatment`` transported from ``source_population`` to
    ``target_population``. ``invariant`` and ``varying_prior`` are the coefficient priors of the
    shared and the source-varying blocks.
    """

    names: Sequence[str]
    edges: Sequence[tuple[str, str]]
    treatment: str
    outcome: str
    source_population: str
    target_population: str
    sources: Sequence[JointRecalcSource]
    target: JointRecalcTarget
    invariant: GaussianTransportPrior
    varying_prior: GaussianTransportPrior
    utility: Utility
    selections: Sequence[str] = ()
    features: Sequence[str] = ()
    varying: Literal["intercept", "intercept_and_covariates"] = "intercept"
    sharing: Literal["independent_varying_blocks", "shared_varying_block"] = "shared_varying_block"
    dependence: Literal["independent_samples", "overlapping_units", "unknown"] = (
        "independent_samples"
    )
    draws: int = 1024
    seed: int = 1
    max_unsupported_mass: float = 0.0
    conflict_z_threshold: float = 3.0
    summary: PosteriorSummary = field(default_factory=PosteriorSummary)

    def _spec(self) -> str:
        from .transport._joint_posterior import GaussianTransportPrior

        if not all(isinstance(s, JointRecalcSource) for s in self.sources):
            raise CausalTypeError("sources must be JointRecalcSource objects")
        if not isinstance(self.target, JointRecalcTarget):
            raise CausalTypeError("target must be a JointRecalcTarget")
        for prior in (self.invariant, self.varying_prior):
            if not isinstance(prior, GaussianTransportPrior):
                raise CausalTypeError("priors must be GaussianTransportPrior objects")
        return json.dumps(
            {
                "names": list(self.names),
                "edges": [list(edge) for edge in self.edges],
                "selections": list(self.selections),
                "treatment": self.treatment,
                "outcome": self.outcome,
                "source_population": self.source_population,
                "target_population": self.target_population,
                "features": list(self.features),
                "varying": self.varying,
                "sharing": self.sharing,
                "dependence": self.dependence,
                "invariant": self.invariant._wire(),
                "varying_prior": self.varying_prior._wire(),
                "sources": [s._wire() for s in self.sources],
                "target": self.target._wire(),
                "draws": int(self.draws),
                "seed": _seed(self.seed),
                "max_unsupported_mass": float(self.max_unsupported_mass),
                "conflict_z_threshold": float(self.conflict_z_threshold),
                "lower_probability": self.summary.lower_probability,
                "upper_probability": self.summary.upper_probability,
                "threshold": self.summary.threshold,
                "benefit_per_unit": self.utility.benefit_per_unit,
                "cost": self.utility.cost,
            },
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class JointBayesianResult:
    """Plan, measured receipt, target-effect law, decision and the exact joint fit summary."""

    plan: RecalcPlan
    receipt: RecalcReceipt
    law: BayesianLaw
    decision: Decision
    target_effect_mean: float
    target_effect_variance: float
    effect_names: tuple[str, ...]
    effect_means: tuple[float, ...]


def _joint_result(
    payload: tuple[str | None, bytes | None, str | None],
) -> JointBayesianResult:
    result, artifact, error = payload
    raise_refusal(error, _refusal_type)
    if result is None or artifact is None:
        raise CausalSerializationError("joint Bayesian native execution returned no result")
    wire = json.loads(result)
    fit = wire["fit"]
    return JointBayesianResult(
        RecalcPlan.from_wire(wire["plan"]),
        RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
        BayesianLaw(**wire["law"]),
        Decision(**wire["decision"]),
        float(fit["target_effect_mean"]),
        float(fit["target_effect_variance"]),
        tuple(fit["effect_names"]),
        tuple(float(v) for v in fit["effect_means"]),
    )


class JointBayesianSession(_Session):
    """Retained joint checked fit; exported results are independently consumable.

    Part of the calibration-internal candidate family: a build without it refuses to construct
    a session (:class:`~antecedent.errors.CausalUnsupportedError`, ``route_not_supported``).
    """

    __slots__ = ()

    @staticmethod
    def _native_type() -> Any:
        native = getattr(_native, "JointBayesianSessionHandle", None)
        if native is None:
            raise CausalUnsupportedError(
                "joint_bayesian.route_frozen: the joint-transport recalculation is a "
                "calibration-internal candidate with an unmeasured law; this build omits it",
                reason_code="route_not_supported",
            )
        return native

    @outcome_operation
    def execute(
        self,
        request: JointBayesianRequest,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> JointBayesianResult:
        """Run the invalidated stages under an optional memory limit and cancellation token.

        ``seed`` seeds execution; the posterior stream is ``request.seed``. A changed summary or
        utility recomputes only its own stage; a changed unit, prior or posterior seed refits.
        Refusals (a bidirected diagram, an uncertified transport, a prior bank that consumed a
        source unit, oversized work) are typed; resource limits and cancellation raise.
        """
        if not isinstance(request, JointBayesianRequest):
            raise CausalTypeError("request must be a JointBayesianRequest")
        return _joint_result(
            self._handle.execute(
                request._spec(),
                seed=_seed(seed),
                threads=threads,
                memory_limit_bytes=memory_limit_bytes,
                cancel=cancel,
            )
        )


__all__ = [
    "JointBayesianRequest",
    "JointBayesianResult",
    "JointBayesianSession",
    "JointRecalcSource",
    "JointRecalcTarget",
    "BayesianRequest",
    "BayesianSession",
    "BayesianLaw",
    "BayesianResult",
    "PosteriorSummary",
]
