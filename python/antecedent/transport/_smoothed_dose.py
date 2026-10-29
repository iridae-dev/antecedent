"""Smoothed dose-response transport grid.

One cell: a randomized continuous source dose with a **known** conditional density
``pi(a | x)`` on a declared support, a continuous outcome, complete baseline
covariates equal to the certified standardizers, and an overlap-supported target
population. For each declared grid dose ``a``, one declared bandwidth ``h`` and the
Epanechnikov kernel, the estimand is

``psi_h(a) = E_target[ integral K_h(a - t) E(Y | X, A=t, S=1) dt ]``.

The bandwidth and kernel are part of the estimand (a different ``h`` is a different
target, never tuned). Every window ``[a - h, a + h]`` must lie inside the dose support;
extrapolative points are refused. Only the two IID designs ``nested_cohort`` and
``independent_samples`` are licensed.

The outcome curve ``mu(t, x)`` (a regression learner over a row-wise dose-by-covariate
basis) and source membership are learned through ``antecedent.learners`` specs and
cross-fitted on one shared fold assignment. The claim is model double robustness of
each point (outcome curve or participation model correct, density known); no
efficiency, rate, CATE, derivative or simultaneous claim is made. Each grid dose
records its quadrature (numerical) error estimate and a smoothing-bias diagnostic
separately; neither is ever added to the estimate. The smoothing-bias diagnostic is
computed from the fitted curve alone (it is zero for a fit linear in the dose). The whole-estimator interval is not licensed:
its route is closed with ``cell_not_licensed`` until its coverage records are measured.
The derivation is in ``docs/architecture/smoothed-dose-response-transport.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import _native
from .._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from .._posterior_row import PosteriorRowEstimate
from .._scientific import ScientificResult
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..execution import Execution
from ..graph import Admg
from ..learners import Linear, Logistic, _learner_wire
from ..outcomes import Outcome, capture_outcome
from ._controls import call_native
from ._impl import SelectionDiagram
from ._learned_continuous import EstimatorMenu, _frozen, _menu, _thaw

#: The one supported target of a request.
TARGET = "smoothed_dose_response"


@dataclass(frozen=True, slots=True)
class DoseBasis:
    """Row-wise dose-by-covariate basis of the outcome learner.

    ``[1, t .. t^degree, (t - k)_+ for each knot, x, t^j x if interactions]``; no
    statistic is fitted on the rows, so it adds no preprocessing outside the folds.
    """

    degree: int = 2
    knots: Sequence[float] = ()
    interactions: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.degree, int) or isinstance(self.degree, bool) or self.degree < 1:
            raise CausalValueError("degree must be a positive integer")
        if any(not isinstance(k, (int, float)) or not math.isfinite(k) for k in self.knots):
            raise CausalValueError("knots must be finite numbers")

    def _wire(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "knots": [float(k) for k in self.knots],
            "interactions": bool(self.interactions),
        }


@dataclass(frozen=True, slots=True)
class SmoothedDoseOptions:
    """Learners, basis, folds, quadrature, support thresholds and bootstrap request.

    ``quadrature_nodes`` is 16 or 32 per window piece (checked against twice as many);
    every window is split at the basis knots, so a linear learner's fitted curve is
    integrated exactly. ``quadrature_tolerance`` caps the largest target-row disagreement
    of the two rules for any other fitted curve (a tree learner); the disagreement is an
    error estimate, not a bound.
    ``bootstrap`` requests interval replicates (at most 2000, floor 199): the interval
    route is closed, so the request only changes the reported interval status.
    """

    outcome: Any = field(default_factory=Linear)
    membership: Any = field(default_factory=Logistic)
    basis: DoseBasis = field(default_factory=DoseBasis)
    folds: int = 5
    quadrature_nodes: int = 16
    quadrature_tolerance: float = 1e-6
    min_membership_probability: float = 0.05
    min_dose_density: float = 1e-3
    min_local_ess: float = 30.0
    min_distinct_doses: int = 5
    bootstrap: int = 0
    coverage_level: float = 0.95

    def __post_init__(self) -> None:
        for label in ("folds", "quadrature_nodes", "min_distinct_doses", "bootstrap"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CausalTypeError(f"{label} must be a non-negative integer")
        if not isinstance(self.basis, DoseBasis):
            raise CausalTypeError("basis must be a DoseBasis")
        if not 0.0 < float(self.min_membership_probability) < 0.5:
            raise CausalValueError("min_membership_probability must lie strictly between 0 and 0.5")
        if not 0.0 < float(self.coverage_level) < 1.0:
            raise CausalValueError("coverage_level must lie strictly between 0 and 1")

    def _json(self) -> str:
        return json.dumps(
            dict(
                outcome=_learner_wire(self.outcome),
                membership=_learner_wire(self.membership),
                basis=self.basis._wire(),
                folds=self.folds,
                quadrature_nodes=self.quadrature_nodes,
                quadrature_tolerance=float(self.quadrature_tolerance),
                min_membership_probability=float(self.min_membership_probability),
                min_dose_density=float(self.min_dose_density),
                min_local_ess=float(self.min_local_ess),
                min_distinct_doses=self.min_distinct_doses,
                bootstrap=self.bootstrap,
                coverage_level=float(self.coverage_level),
            ),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class SmoothedDoseQuery:
    """The graph, the dose and outcome, and the smoothed estimand's identity.

    ``grid`` holds distinct doses in any order; ``bandwidth`` and ``kernel``
    define the estimand; ``dose_support`` is the randomized dose range; the density
    provenance must be ``known`` (an estimated density is refused).
    """

    graph: Admg
    diagram: SelectionDiagram
    dose: str
    outcome: str
    grid: Sequence[float]
    bandwidth: float
    dose_support: tuple[float, float]
    kernel: str = "epanechnikov"
    density_provenance: str = "known"

    def _json(self) -> str:
        return json.dumps(
            dict(
                source=self.diagram.source,
                target=self.diagram.target,
                dose=self.dose,
                outcome=self.outcome,
                grid=[float(a) for a in self.grid],
                bandwidth=float(self.bandwidth),
                dose_support=[float(self.dose_support[0]), float(self.dose_support[1])],
                kernel=self.kernel,
                density_provenance=self.density_provenance,
            ),
            allow_nan=False,
        )


def _declared_json(data: SmoothedDoseData) -> dict[str, Any]:
    from ._buffers import cluster_ids

    out: dict[str, Any] = {}
    if data.sampling_weights is not None:
        out["sampling_weights"] = [float(w) for w in data.sampling_weights]
    if data.cluster is not None:
        out["cluster"] = cluster_ids(data.cluster)
    if data.serial_lag is not None:
        out["serial_lag"] = int(data.serial_lag)
    return out


@dataclass(frozen=True, slots=True)
class SmoothedDoseData:
    """IID trial and target rows with the known conditional dose density.

    ``dose_density`` is ``pi(dose_i | x_i)`` at each source row. Target outcomes, doses
    and densities are ignored; use finite placeholders.
    """

    covariates: Mapping[str, Sequence[float]]
    outcome: Sequence[float]
    dose: Sequence[float]
    dose_density: Sequence[float]
    source: Sequence[bool]
    sampling: Literal["nested_cohort", "independent_samples"]
    #: Sampling weights, one per row of either sample (finite and non-negative; rows of weight
    #: zero are deleted before preparation, the rest are normalized to mean one).
    sampling_weights: Sequence[float] | None = None
    #: One cluster label per row; clusters stay whole in one cross-fit fold and the influence
    #: standard error diagnostic is cluster-robust. Not combined with ``serial_lag``.
    cluster: Sequence[Any] | None = None
    #: Newey-West lag for rows in time order at unit spacing (folds are contiguous blocks).
    serial_lag: int | None = None

    def _buffers(self, names: Sequence[str]) -> dict[str, Any]:
        from ._buffers import smoothed_buffers

        return smoothed_buffers(self, names)

    def _json(self, names: Sequence[str]) -> str:
        unknown = set(self.covariates) - set(names)
        if unknown:
            raise ValueError(f"Unknown baseline covariates: {sorted(unknown)}")
        features = [i for i, name in enumerate(names) if name in self.covariates]
        return json.dumps(
            dict(
                features=features,
                covariates=[list(map(float, self.covariates[names[i]])) for i in features],
                outcome=list(map(float, self.outcome)),
                dose=list(map(float, self.dose)),
                dose_density=list(map(float, self.dose_density)),
                source=[bool(v) for v in self.source],
                sampling=self.sampling,
                **_declared_json(self),
            ),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class SmoothedDoseGridPoint:
    """One grid dose: the point, its quadrature error, smoothing-bias diagnostic and support.

    ``quadrature`` and ``smoothing_bias`` are separate records, never added to
    ``estimate``; ``influence_se_diagnostic`` is a diagnostic, not a licensed claim.
    """

    dose: float
    estimate: float
    plug_in: float
    augmentation: float
    quadrature: Mapping[str, Any]
    smoothing_bias: Mapping[str, Any]
    support: Mapping[str, Any]
    influence_se_diagnostic: float


@dataclass(frozen=True, slots=True)
class SmoothedDoseEstimate(ScientificResult):
    """The grid with provenance, diagnostics and the interval status.

    ``uncertainty`` is ``point_only`` when no interval was requested and ``withheld``
    otherwise; ``interval`` is always ``None``.
    """

    grid: tuple[SmoothedDoseGridPoint, ...]
    uncertainty: Mapping[str, Any]
    overlap: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    folds: Mapping[str, Any]
    sampling: str
    certificate: Mapping[str, Any]
    query: Mapping[str, Any]
    seed: int
    premises_digest: str
    data_digest: str
    evidence_digest: str
    execution_id: str
    variable_names: tuple[str, ...]
    _artifact_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def interval(self) -> None:
        return None

    def __getitem__(self, dose: float) -> SmoothedDoseGridPoint:
        for point in self.grid:
            if point.dose == dose:
                return point
        raise KeyError(dose)

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """Export the artifact captured from this completed native execution."""
        if self._artifact_bytes is None:
            raise CausalUnsupportedError("no completed native execution artifact is bound")
        return artifact_from_receipt(self, self._artifact_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": [
                {
                    "dose": p.dose,
                    "estimate": p.estimate,
                    "plug_in": p.plug_in,
                    "augmentation": p.augmentation,
                    "quadrature": _thaw(p.quadrature),
                    "smoothing_bias": _thaw(p.smoothing_bias),
                    "support": _thaw(p.support),
                    "influence_se_diagnostic": p.influence_se_diagnostic,
                }
                for p in self.grid
            ],
            "interval": None,
            "uncertainty": _thaw(self.uncertainty),
            "overlap": _thaw(self.overlap),
            "diagnostics": _thaw(self.diagnostics),
            "provenance": [_thaw(p) for p in self.provenance],
            "folds": _thaw(self.folds),
            "sampling": self.sampling,
            "certificate": _thaw(self.certificate),
            "query": _thaw(self.query),
            "seed": self.seed,
            "premises_digest": self.premises_digest,
            "data_digest": self.data_digest,
            "evidence_digest": self.evidence_digest,
            "execution_id": self.execution_id,
        }


def _estimate(payload: str) -> SmoothedDoseEstimate:
    raw = json.loads(payload)
    return SmoothedDoseEstimate(
        tuple(
            SmoothedDoseGridPoint(
                p["dose"],
                p["estimate"],
                p["plug_in"],
                p["augmentation"],
                _frozen(p["quadrature"]),
                _frozen(p["smoothing_bias"]),
                _frozen(p["support"]),
                p["influence_se_diagnostic"],
            )
            for p in raw["grid"]
        ),
        _frozen(raw["uncertainty"]),
        _frozen(raw["overlap"]),
        _frozen(raw["diagnostics"]),
        tuple(_frozen(p) for p in raw["provenance"]),
        _frozen(raw["folds"]),
        raw["sampling"],
        _frozen(raw["certificate"]),
        _frozen(raw["query"]),
        raw["seed"],
        raw["premises_digest"],
        raw["data_digest"],
        raw["evidence_digest"],
        raw["execution_id"],
        tuple(raw["variable_names"]),
    )


def smoothed_dose_estimator_menu(
    query: SmoothedDoseQuery, *, options: SmoothedDoseOptions | None = None
) -> EstimatorMenu:
    """List the estimators for a smoothed dose-response transport query.

    Inspection only: nothing is fitted and nothing is recommended. The smoothed
    estimator's requirements are computed from the certificate, the query and the
    options (release defaults when ``options`` is ``None``); the refused alternatives
    (point curve, estimated density, simultaneous band, conditional group response)
    name their reasons.
    """
    if not isinstance(query, SmoothedDoseQuery):
        raise CausalTypeError("smoothed_dose_estimator_menu requires a SmoothedDoseQuery")
    return _menu(
        _native.smoothed_dose_estimator_menu(
            query.graph,
            list(query.diagram.selections),
            query._json(),
            options._json() if options is not None else None,
        )
    )


@dataclass(frozen=True, slots=True)
class SmoothedDosePosteriorModel:
    """The declared outcome model and posterior sample of the Bayesian smoothed dose response.

    ``mu(t, x) = E(Y | X = x, A = t, S = 1)`` is linear in the declared :class:`DoseBasis` with
    coefficient ``j`` a priori ``N(0, sigma^2 s_j^2)``: ``intercept_scale`` for the intercept and
    ``coefficient_scale`` for every other coefficient (the covariates enter the basis as given,
    so their units set the prior's meaning). Declare the residual variance
    (``residual_variance``) or its inverse-gamma prior (``residual_shape`` and
    ``residual_scale``). ``min_local_rows`` is the fewest source rows inside every grid window.
    """

    basis: DoseBasis = field(default_factory=DoseBasis)
    intercept_scale: float = 10.0
    coefficient_scale: float = 10.0
    residual_variance: float | None = None
    residual_shape: float | None = None
    residual_scale: float | None = None
    min_local_rows: int = 30
    draws: int = 4_000

    def _json(self, *, seed: int) -> str:
        if self.residual_variance is not None and (
            self.residual_shape is None and self.residual_scale is None
        ):
            residual: dict[str, Any] = {"known_variance": float(self.residual_variance)}
        elif (
            self.residual_variance is None
            and self.residual_shape is not None
            and self.residual_scale is not None
        ):
            residual = {
                "inverse_gamma": {
                    "shape": float(self.residual_shape),
                    "scale": float(self.residual_scale),
                }
            }
        else:
            raise CausalValueError(
                "declare either residual_variance, or residual_shape and residual_scale"
            )
        return json.dumps(
            {
                "basis": self.basis._wire(),
                "intercept_scale": float(self.intercept_scale),
                "coefficient_scale": float(self.coefficient_scale),
                "residual": residual,
                "min_local_rows": int(self.min_local_rows),
                "draws": int(self.draws),
                "seed": int(seed),
            },
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class SmoothedDosePosterior(ScientificResult):
    """The joint posterior of ``psi_h`` over the grid under the declared outcome model.

    ``posterior`` holds the posterior mean and equal-tailed 95% credible interval at every grid
    dose and the covariance across the grid from the joint draws (exact Gaussian moments when
    the residual variance is known). The row is unlicensed and its calibration is
    ``"unmeasured"``; the interval route of the point estimate stays closed.
    """

    posterior: PosteriorRowEstimate
    certificate: Mapping[str, Any]
    query: Mapping[str, Any]
    premises_digest: str
    data_digest: str
    variable_names: tuple[str, ...]
    _artifact_bytes: bytes = field(repr=False, compare=False)

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """The posterior as an independently consumable artifact."""
        return artifact_from_receipt(self, self._artifact_bytes)


def _posterior(payload: str, artifact: bytes) -> SmoothedDosePosterior:
    artifact = container_bytes(artifact)
    raw = json.loads(payload)
    return SmoothedDosePosterior(
        posterior=PosteriorRowEstimate._from_mapping(raw["result"]),
        certificate=_frozen(raw["certificate"]),
        query=_frozen(raw["query"]),
        premises_digest=raw["premises_digest"],
        data_digest=raw["data_digest"],
        variable_names=tuple(raw["variable_names"]),
        _artifact_bytes=artifact,
    )


@dataclass(frozen=True, slots=True, repr=False)
class PreparedSmoothedDose(ScientificResult):
    """A prepared smoothed dose transport.

    The certificate is derived once at preparation; ``estimate`` cross-fits the
    nuisances on the retained rows under the frozen seed and never identifies again.
    """

    _native: Any = field(repr=False, compare=False)
    _execution: Execution = field(repr=False, compare=False)

    def __init__(self, native: Any, execution: Execution) -> None:
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_execution", execution)

    def _run(self, operation: str) -> SmoothedDoseEstimate:
        native = self._execution._call(self._native.fork, seed=self._execution.seed)
        result = _estimate(self._execution._call(getattr(native, operation)))
        object.__setattr__(result, "_artifact_bytes", bytes(native.export()))
        return result

    def estimate(self) -> Outcome[SmoothedDoseEstimate]:
        """Execute an independent run, preserving this preparation and earlier answers."""
        return capture_outcome(lambda: self._run("estimate"))

    def reconfigure(self, *, execution: Execution) -> PreparedSmoothedDose:
        """Branch physical policy and stream without fitting or copying raw observations."""
        if not isinstance(execution, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        native = execution._call(self._native.fork, seed=execution.seed)
        return PreparedSmoothedDose(native, execution)

    def bind_snapshot(self, data: SmoothedDoseData) -> PreparedSmoothedDose:
        """Compile a compatible immutable data branch without fitting its learners."""
        if not isinstance(data, SmoothedDoseData):
            raise CausalTypeError("bind_snapshot requires SmoothedDoseData")
        native = self._execution._call(self._native.bind_snapshot, data)
        return PreparedSmoothedDose(native, self._execution)

    def posterior(self, model: SmoothedDosePosteriorModel) -> Outcome[SmoothedDosePosterior]:
        """The unlicensed Bayesian posterior of the same smoothed dose response.

        Runs on the retained rows and the certificate derived at preparation; the model is
        declared, never selected from the rows.
        """
        if not isinstance(model, SmoothedDosePosteriorModel):
            raise CausalTypeError("posterior requires a SmoothedDosePosteriorModel")

        def execute() -> SmoothedDosePosterior:
            payload, artifact = self._execution._call(
                self._native.posterior, model._json(seed=self._execution.seed)
            )
            return _posterior(payload, bytes(artifact))

        return capture_outcome(execute)

    def interval(self) -> Outcome[None]:
        """Return the registered closed interval route as a scientific refusal."""
        return capture_outcome(self._execution._call, self._native.interval)

    def estimator_menu(self) -> EstimatorMenu:
        """The estimator menu for this prepared graph, query and options."""
        return _menu(self._native.estimator_menu())


def prepare_smoothed_dose(
    query: SmoothedDoseQuery,
    data: SmoothedDoseData,
    *,
    options: SmoothedDoseOptions | None = None,
    target: str = TARGET,
    execution: Execution | None = None,
) -> PreparedSmoothedDose:
    """Prepare the certified smoothed dose-response transport grid.

    ``target`` is the fixed-bandwidth smoothed response; point-curve, stochastic,
    coarsened, incremental and derivative targets refuse
    (``dose_response.target_not_smoothed``), conditional and simultaneous ones refuse
    (``dose_response.cate_or_simultaneous``), and any sampling design but the two IID
    ones refuses (``dose_response.non_iid_design``).
    """
    if not isinstance(query, SmoothedDoseQuery) or not isinstance(data, SmoothedDoseData):
        raise CausalTypeError(
            "prepare_smoothed_dose requires SmoothedDoseQuery and SmoothedDoseData"
        )
    options = options or SmoothedDoseOptions()
    execution = Execution() if execution is None else execution
    if not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    native = execution._call(
        _native.prepare_smoothed_dose,
        query.graph,
        list(query.diagram.selections),
        query._json(),
        data,
        options._json(),
        target,
        seed=execution.seed,
        memory_bytes=execution.limits.memory_bytes,
        cancel=execution.cancel,
    )
    return PreparedSmoothedDose(native, execution)


@artifact_consumer(
    proof="PreparedSmoothedDose::consume verifies portable fold predictions and replays every quadrature point without fitting",
    replayable=True,
    resumable=False,
    identity=(),
)
def consume_smoothed_dose(
    artifact: bytes,
    *,
    max_rows: int | None = None,
    max_features: int | None = None,
    execution: Execution | None = None,
) -> SmoothedDoseEstimate:
    """Independently verify an exported grid and replay it.

    The consumer re-derives the certificate, recomputes the folds, re-predicts every
    nuisance from the stored portable fold models, re-integrates the quadrature and
    replays every grid dose bit for bit. It never fits a learner or resamples; it does
    not establish that the stored models were fitted as recorded. Stored premises that
    do not enter the point (the seed, the sampling design, support thresholds the stored
    rows still pass) are bound only against unsealed edits.

    ``max_rows`` and ``max_features`` cap the artifact's size; the execution's memory limit
    lowers the mandatory 512 MiB (536,870,912 bytes) cap on the replay workspace estimate (refused with
    ``CausalResourceError`` before any replay work); its cancellation token is polled before every
    quadrature chunk of the replay.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    for label, value in (("max_rows", max_rows), ("max_features", max_features)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise CausalValueError(f"{label} must be a non-negative integer or None")
    return _estimate(
        call_native(
            execution,
            _native.consume_smoothed_dose,
            artifact,
            max_rows=max_rows,
            max_features=max_features,
            cancel="cancel",
            memory="max_memory_bytes",
        )
    )


@artifact_consumer(
    proof="SmoothedDosePosterior::consume re-derives the certificate and recomputes the posterior from retained rows and seed",
    replayable=True,
    resumable=False,
    identity=(),
)
def consume_smoothed_dose_posterior(
    artifact: bytes, *, max_rows: int = 100_000, max_points: int = 1_024
) -> SmoothedDosePosterior:
    """Independently verify an exported smoothed-dose posterior and replay it.

    The consumer re-derives the transport certificate from the stored graph and query and
    recomputes the posterior from the stored rows and seed; only a bit-identical result is
    accepted, so a changed row, premise, certificate or stored value is refused even when the
    digests were resealed.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _posterior(
        _native.consume_smoothed_dose_posterior(artifact, max_rows=max_rows, max_points=max_points),
        artifact,
    )


__all__ = [
    "DoseBasis",
    "SmoothedDosePosterior",
    "SmoothedDosePosteriorModel",
    "consume_smoothed_dose_posterior",
    "PreparedSmoothedDose",
    "SmoothedDoseData",
    "SmoothedDoseEstimate",
    "SmoothedDoseGridPoint",
    "SmoothedDoseOptions",
    "SmoothedDoseQuery",
    "consume_smoothed_dose",
    "prepare_smoothed_dose",
    "smoothed_dose_estimator_menu",
]
