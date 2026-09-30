"""Smoothed dose-response transport grid (2.2B cell X4).

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
records its quadrature (numerical) error and a smoothing-bias diagnostic separately;
neither is ever added to the estimate. The whole-estimator interval is not licensed:
its route is closed with ``cell_not_licensed`` until its coverage records are measured.
The derivation is in ``docs/smoothed-dose-response-transport.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import _native
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ..learners import Linear, Logistic, _learner_wire
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

    ``quadrature_nodes`` is 16 or 32 (checked against twice as many);
    ``quadrature_tolerance`` bounds their disagreement at every grid dose.
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

    ``grid`` holds at most 16 distinct doses in any order; ``bandwidth`` and ``kernel``
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
class SmoothedDoseEstimate:
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

    @property
    def interval(self) -> None:
        return None

    def __getitem__(self, dose: float) -> SmoothedDoseGridPoint:
        for point in self.grid:
            if point.dose == dose:
                return point
        raise KeyError(dose)

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


class PreparedSmoothedDose:
    """A prepared smoothed dose transport.

    The certificate is derived once at preparation; ``estimate`` cross-fits the
    nuisances on the retained rows under the frozen seed and never identifies again.
    """

    __slots__ = ("_last", "_native")

    def __init__(self, native: Any) -> None:
        self._native = native
        self._last: SmoothedDoseEstimate | None = None

    def estimate(self, *, cancel: Any = None) -> SmoothedDoseEstimate:
        """Cross-fit, integrate and report the grid; the interval is withheld."""
        self._last = _estimate(self._native.estimate(cancel))
        return self._last

    def refresh(self, data: SmoothedDoseData, *, cancel: Any = None) -> None:
        """Replace the rows with a compatible snapshot; the certificate is kept.

        A changed feature schema or sampling design needs a new preparation.
        """
        if not isinstance(data, SmoothedDoseData):
            raise CausalTypeError("refresh requires SmoothedDoseData")
        self._native.refresh(data, cancel)
        self._last = None

    def interval(self, *, cancel: Any = None) -> None:
        """The closed interval route: always refuses with ``cell_not_licensed``."""
        self._native.interval(cancel)

    def estimator_menu(self) -> EstimatorMenu:
        """The estimator menu for this prepared graph, query and options."""
        return _menu(self._native.estimator_menu())

    def export(self) -> bytes:
        """The last estimate as an independently consumable artifact."""
        return bytes(self._native.export())


def prepare_smoothed_dose(
    query: SmoothedDoseQuery,
    data: SmoothedDoseData,
    *,
    options: SmoothedDoseOptions | None = None,
    target: str = TARGET,
    seed: int = 1,
    memory_bytes: int | None = None,
    cancel: Any = None,
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
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise CausalValueError("seed must be a non-negative integer")
    if memory_bytes is not None and (not isinstance(memory_bytes, int) or memory_bytes < 0):
        raise CausalValueError("memory_bytes must be a non-negative integer or None")
    native = _native.prepare_smoothed_dose(
        query.graph,
        list(query.diagram.selections),
        query._json(),
        data,
        options._json(),
        target,
        seed=seed,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedSmoothedDose(native)


def consume_smoothed_dose(
    artifact: bytes, *, max_rows: int | None = None, max_features: int | None = None
) -> SmoothedDoseEstimate:
    """Independently verify an exported grid and replay it.

    The consumer re-derives the certificate, recomputes the folds, re-predicts every
    nuisance from the stored portable fold models, re-integrates the quadrature and
    replays every grid dose bit for bit. It never fits a learner or resamples; it does
    not establish that the stored models were fitted as recorded.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    for label, value in (("max_rows", max_rows), ("max_features", max_features)):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise CausalValueError(f"{label} must be a non-negative integer or None")
    return _estimate(
        _native.consume_smoothed_dose(artifact, max_rows=max_rows, max_features=max_features)
    )


__all__ = [
    "DoseBasis",
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
