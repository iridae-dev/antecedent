"""Exposure-error declarations that the analysis consumes.

A measured exposure is declared on the variable it measures, and ``prepare`` / ``analyze`` carry
the declaration through the whole causal estimate::

    import antecedent as ant
    from antecedent import exposure_measurement as em
    from antecedent import measurement_exposure as mx

    outcome = ant.analyze(
        data,                                   # x is the recorded, misclassified exposure
        graph=[("z", "x"), ("z", "y"), ("x", "y")],
        query=ant.AverageEffect(treatment="x", outcome="y"),
        measurement=em.MisclassifiedExposure(
            "x", mx.ExposureError.by_outcome(cases=..., controls=...), covariates=("z",)
        ),
        measurement_draws=40,
    )
    outcome.answer.value                        # the pooled effect of the true exposure
    outcome.value.interval(0.95)                # Rubin interval over the draws
    outcome.value.exposure.export()             # the draws, as a posterior artifact

Four declarations are offered, each the first-class form of one standalone correction in
:mod:`antecedent.measurement_exposure`:

* :class:`MisclassifiedExposure` -- a recorded 0/1 exposure with a sensitivity and specificity
  that may differ by outcome level (declared, or estimated from validation counts);
* :class:`SoftExposure` -- a column of per-unit probabilities that the exposure is present;
* :class:`DetectedExposure` -- a recorded 0/1 detection whose probability is a declared function
  of a covariate, with an optional false-positive rate;
* :class:`RepeatCodedExposure` -- several conditionally independent codings of one latent binary
  or ordinal facet, the latent class model of :func:`~antecedent.measurement_exposure
  .repeat_codings_latent_class`.

How the declaration reaches the estimate. Each declaration gives a per-unit likelihood of what was
recorded under each latent exposure level. The latent level is imputed ``m`` times from its
posterior given the recorded value, the outcome and the covariates: an exposure model
(softmax in the covariates) and an outcome model (Gaussian, or logistic for a 0/1 outcome) are
fitted by EM on a row bootstrap, and the measurement parameters (validated rates, the detection
function's estimated parameters, the replicate response laws) are redrawn for every imputation, so
the draws carry parameter uncertainty. Each completed table is prepared and estimated by the same
engine the analysis would use for an error-free exposure; the ``m`` estimates are pooled by
Rubin's rules (within, between and total variance; a ``t`` interval with Rubin's degrees of
freedom). The imputation model is part of the assumptions: the outcome model of the imputation
must contain the relations the analysis relies on, and the draws are proper only when it is right.
The pooled interval carries the calibration status ``not_measured``.

The exposure draws are a first-class column of uncertain values: :meth:`ExposureDraws.export`
writes them as a posterior artifact (one quantity per unit, one draw per imputation), and
:func:`consume_exposure_draws` reads them back; :meth:`ExposureDraws.completed_tables` hands
every draw to any downstream stage that takes a table (discovery, other queries).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native
from ._artifact_protocol import container_bytes
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .measurement_exposure import (
    ExposureError,
    LogisticDetection,
    MonotoneDetection,
    TableDetection,
    ValidationCounts,
    correct_exposure_misclassification,
    detection_dependent_prevalence,
    repeat_codings_latent_class,
)
from .measurement_models import (
    BinaryRates,
    MeasurementInterval,
    MeasurementModelRefusal,
    VarianceComponent,
)
from .outcomes import Answer, Outcome

__all__ = [
    "DetectedExposure",
    "ExposureDraws",
    "ExposureMeasurement",
    "MeasuredAnalysis",
    "MisclassifiedExposure",
    "PreparedMeasuredAnalysis",
    "RepeatCodedExposure",
    "RubinPool",
    "SoftExposure",
    "consume_exposure_draws",
    "rubin_pool",
]

_NOT_MEASURED: Mapping[str, Any] = {"status": "not_measured"}
_TINY = 1e-12
_ARTIFACT_BACKEND = "exposure_measurement"


# --------------------------------------------------------------------------- refusals


def _refuse(code: str, detail: str, message: str, remedy: str) -> MeasurementModelRefusal:
    return MeasurementModelRefusal(
        {"code": code, "detail": detail, "message": message, "remedy": remedy}
    )


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a nonempty string")
    return value


def _names(values: Sequence[str] | None, what: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise CausalTypeError(f"{what} must be a sequence of column names")
    return tuple(_name(v, what) for v in values)


# --------------------------------------------------------------------------- columns


def _table_columns(data: Any) -> dict[str, np.ndarray]:
    """The columns of a mapping or data frame as arrays."""
    if isinstance(data, Mapping):
        names = list(data.keys())
    elif hasattr(data, "columns"):
        names = [str(c) for c in data.columns]
    else:
        raise CausalTypeError(
            "a measurement declaration reads a mapping of columns or a data frame; "
            "pass the table, not a path or frame object"
        )
    return {name: np.asarray(data[name]) for name in names}


def _numeric(columns: Mapping[str, np.ndarray], name: str, rows: int | None = None) -> np.ndarray:
    if name not in columns:
        raise CausalValueError(f"column {name!r} is not in the data")
    try:
        array = np.asarray(columns[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"column {name!r} must be numeric") from error
    if array.ndim != 1:
        raise CausalValueError(f"column {name!r} must be one-dimensional")
    if rows is not None and array.shape[0] != rows:
        raise CausalValueError(f"column {name!r} has {array.shape[0]} rows, expected {rows}")
    if not np.all(np.isfinite(array)):
        raise CausalValueError(f"column {name!r} holds a missing or non-finite value")
    return array


def _binary(columns: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    array = _numeric(columns, name)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise CausalValueError(f"column {name!r} must hold 0/1 values")
    return array.astype(np.int64)


def _digest(*parts: Any) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            hasher.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
        else:
            hasher.update(json.dumps(part, sort_keys=True, default=str).encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


# --------------------------------------------------------------------------- parameter draws


def _beta_draw(rng: np.random.Generator, mean: float, variance: float) -> float:
    """A draw of a rate with the stated mean and variance (fixed when the variance is zero)."""
    if variance <= 0.0:
        return mean
    cap = mean * (1.0 - mean)
    if cap <= 0.0 or variance >= cap:
        return mean
    k = cap / variance - 1.0
    return float(rng.beta(max(mean * k, 1e-3), max((1.0 - mean) * k, 1e-3)))


@dataclass(frozen=True, slots=True)
class _Rates:
    """A sensitivity and specificity with the way each is redrawn."""

    sensitivity: float
    specificity: float
    sensitivity_variance: float = 0.0
    specificity_variance: float = 0.0
    counts: tuple[int, int, int, int] | None = None

    @classmethod
    def of(cls, source: BinaryRates | ValidationCounts) -> _Rates:
        if isinstance(source, ValidationCounts):
            tp, fn, tn, fp = source.to_wire()
            if tp + fn == 0 or tn + fp == 0:
                raise _refuse(
                    "measurement_not_identified",
                    "measurement_model.insufficient_replicates",
                    "a validation sample needs truly exposed and truly unexposed units",
                    "Supply validation counts with both margins nonempty.",
                )
            return cls(tp / (tp + fn), tn / (tn + fp), counts=(tp, fn, tn, fp))
        if not isinstance(source, BinaryRates):
            raise CausalTypeError("rates must be BinaryRates or ValidationCounts")
        return cls(
            float(source.sensitivity),
            float(source.specificity),
            float(source.sensitivity_variance),
            float(source.specificity_variance),
        )

    @property
    def youden(self) -> float:
        return self.sensitivity + self.specificity - 1.0

    def draw(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.counts is not None:
            tp, fn, tn, fp = self.counts
            return float(rng.beta(tp + 0.5, fn + 0.5)), float(rng.beta(tn + 0.5, fp + 0.5))
        return (
            _beta_draw(rng, self.sensitivity, self.sensitivity_variance),
            _beta_draw(rng, self.specificity, self.specificity_variance),
        )


# --------------------------------------------------------------------------- imputation core


def _softmax(eta: np.ndarray) -> np.ndarray:
    shifted = eta - eta.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _design(z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(z.shape[0]), z]) if z.shape[1] else np.ones((z.shape[0], 1))


def _fit_exposure_model(
    x: np.ndarray, responsibilities: np.ndarray, phi: np.ndarray, steps: int = 3
) -> np.ndarray:
    """Softmax regression of fractional level responsibilities (Newton, ridge-stabilised)."""
    n, p = x.shape
    k = responsibilities.shape[1] - 1
    for _ in range(steps):
        eta = np.column_stack([np.zeros(n), x @ phi])
        pi = _softmax(eta)
        gradient = x.T @ (responsibilities[:, 1:] - pi[:, 1:])
        w = np.zeros((n, k, k))
        for a in range(k):
            w[:, a, a] = pi[:, a + 1]
            for b in range(k):
                w[:, a, b] -= pi[:, a + 1] * pi[:, b + 1]
        hessian = np.einsum("ia,ib,ikl->akbl", x, x, w).reshape(p * k, p * k)
        hessian += 1e-6 * np.eye(p * k)
        step = np.linalg.solve(hessian, gradient.reshape(p * k))
        phi = phi + step.reshape(p, k)
    return phi


@dataclass(slots=True)
class _OutcomeModel:
    """``E[y | level j, z] = a + b j + g z``: Gaussian, or logistic for a 0/1 outcome."""

    binary: bool
    beta: np.ndarray
    variance: float = 1.0

    def stacked(self, z: np.ndarray, levels: int) -> np.ndarray:
        n = z.shape[0]
        rows = []
        for j in range(levels):
            rows.append(np.column_stack([np.ones(n), np.full(n, float(j)), z]))
        return np.concatenate(rows, axis=0)

    def fit(self, y: np.ndarray, z: np.ndarray, r: np.ndarray, steps: int) -> None:
        n, levels = r.shape
        design = self.stacked(z, levels)
        weight = r.T.reshape(-1)
        response = np.tile(y, levels)
        if not self.binary:
            gram = design.T @ (design * weight[:, None]) + 1e-9 * np.eye(design.shape[1])
            self.beta = np.linalg.solve(gram, design.T @ (weight * response))
            resid = response - design @ self.beta
            floor = 1e-8 * max(float(np.var(y)), 1e-12)
            self.variance = max(float(weight @ (resid**2) / max(weight.sum(), _TINY)), floor)
            return
        beta = self.beta
        for _ in range(steps):
            mu = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0)))
            curvature = weight * mu * (1.0 - mu)
            gram = design.T @ (design * curvature[:, None]) + 1e-6 * np.eye(design.shape[1])
            beta = beta + np.linalg.solve(gram, design.T @ (weight * (response - mu)))
        self.beta = beta

    def log_density(self, y: np.ndarray, z: np.ndarray, levels: int) -> np.ndarray:
        n = z.shape[0]
        out = np.empty((n, levels))
        for j in range(levels):
            row = np.column_stack([np.ones(n), np.full(n, float(j)), z])
            eta = row @ self.beta
            if self.binary:
                out[:, j] = y * (-np.logaddexp(0.0, -eta)) + (1 - y) * (-np.logaddexp(0.0, eta))
            else:
                out[:, j] = -0.5 * (y - eta) ** 2 / self.variance - 0.5 * math.log(self.variance)
        return out


def _standardise(z: np.ndarray) -> np.ndarray:
    if z.shape[1] == 0:
        return z
    sd = z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (z - z.mean(axis=0)) / sd


def _posterior(
    likelihood: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fixed_prior: np.ndarray | None,
    rows: np.ndarray,
    iterations: int = 80,
) -> np.ndarray:
    """Posterior over latent levels for every unit, from models fitted on ``rows``.

    ``likelihood[i, j]`` is the probability of what was recorded for unit ``i`` if its level is
    ``j``. The exposure model (a softmax in ``z``, or the supplied ``fixed_prior``) and the
    outcome model are fitted by EM on the units ``rows``; the posterior of every unit is formed
    from the fitted models.
    """
    n, levels = likelihood.shape
    x = _design(z)
    xb, yb, zb, lb = x[rows], y[rows], z[rows], likelihood[rows]
    fb = None if fixed_prior is None else fixed_prior[rows]
    binary = bool(np.all((y == 0.0) | (y == 1.0)))
    model = _OutcomeModel(binary, np.zeros(2 + z.shape[1]), 1.0)
    phi = np.zeros((x.shape[1], levels - 1))
    r = lb / np.maximum(lb.sum(axis=1, keepdims=True), _TINY)
    if fb is not None:
        r = r * fb
        r /= np.maximum(r.sum(axis=1, keepdims=True), _TINY)
    for _ in range(iterations):
        if fb is None:
            phi = _fit_exposure_model(xb, r, phi)
            prior_b = _softmax(np.column_stack([np.zeros(xb.shape[0]), xb @ phi]))
        else:
            prior_b = fb
        model.fit(yb, zb, r, steps=3)
        logf = model.log_density(yb, zb, levels)
        post = prior_b * lb * np.exp(logf - logf.max(axis=1, keepdims=True))
        post /= np.maximum(post.sum(axis=1, keepdims=True), _TINY)
        change = float(np.max(np.abs(post - r)))
        r = post
        if change < 1e-7:
            break
    prior = (
        _softmax(np.column_stack([np.zeros(n), x @ phi])) if fixed_prior is None else fixed_prior
    )
    logf = model.log_density(y, z, levels)
    full = prior * likelihood * np.exp(logf - logf.max(axis=1, keepdims=True))
    return full / np.maximum(full.sum(axis=1, keepdims=True), _TINY)


def _sample_levels(rng: np.random.Generator, posterior: np.ndarray) -> np.ndarray:
    u = rng.random(posterior.shape[0])[:, None]
    return (u > np.cumsum(posterior, axis=1)[:, :-1]).sum(axis=1).astype(np.float64)


# --------------------------------------------------------------------------- the draws


@dataclass(frozen=True, slots=True, eq=False)
class ExposureDraws:
    """A corrected exposure as an uncertain column: per-unit posterior and ``m`` draws.

    ``probability[i]`` is the posterior mean of unit ``i``'s latent level (the probability the
    exposure is present, for a binary exposure) under the declared measurement and the fitted
    imputation models, and ``draws[k, i]`` the level imputed for it in draw ``k`` (measurement
    parameters and imputation models redrawn per draw). ``declaration_identity`` and
    ``data_identity`` bind the draws to the declaration and the columns it read.
    """

    variable: str
    kind: str
    levels: int
    probability: np.ndarray
    draws: np.ndarray
    declaration_identity: str
    data_identity: str
    seed: int
    notes: tuple[str, ...] = ()

    @property
    def draw_count(self) -> int:
        """The number of imputations ``m``."""
        return int(self.draws.shape[0])

    @property
    def rows(self) -> int:
        """The number of units."""
        return int(self.draws.shape[1])

    @property
    def identity(self) -> str:
        """Digest of the declaration, the data it read, the seed and the draws."""
        return _digest(
            self.kind,
            self.variable,
            self.declaration_identity,
            self.data_identity,
            self.seed,
            self.draws,
        )

    def column(self, draw: int) -> np.ndarray:
        """The exposure column of imputation ``draw``."""
        return np.array(self.draws[draw], dtype=np.float64)

    def completed_tables(self, data: Any) -> list[dict[str, np.ndarray]]:
        """One table per draw: ``data`` with the exposure column replaced (or added).

        Any downstream stage that takes a table (discovery, another query, a bridge) can be run on
        each and its results combined as multiple-imputation results.
        """
        base = _table_columns(data)
        if self.variable in base and base[self.variable].shape[0] != self.rows:
            raise CausalValueError("the data have a different number of rows than the draws")
        tables = []
        for k in range(self.draw_count):
            table = dict(base)
            table[self.variable] = self.column(k)
            tables.append(table)
        return tables

    def export(self) -> bytes:
        """The draws as a posterior artifact: one quantity per unit, one draw per imputation.

        The artifact is the library's posterior record (read it with
        :func:`antecedent.consume_posterior` or :func:`consume_exposure_draws`); its
        ``identification`` field names the declaration and data identities, and the quantity
        ``"{variable}[i]"`` holds unit ``i``'s draws.
        """
        n, m = self.rows, self.draw_count
        draws = self.draws
        names = [f"{self.variable}[{i}]" for i in range(n)]
        mean = [float(v) for v in draws.mean(axis=0)]
        sd = [float(v) for v in draws.std(axis=0)]
        q025 = [float(v) for v in np.quantile(draws, 0.025, axis=0)]
        q975 = [float(v) for v in np.quantile(draws, 0.975, axis=0)]
        artifact = _native.PosteriorArtifact(
            m,
            mean,
            sd,
            q025,
            q975,
            [float(v) for v in draws.T.reshape(-1)],
            f"{_ARTIFACT_BACKEND}.{self.kind}",
            f"{self.declaration_identity}:{self.data_identity}:{self.levels}:{self.seed}",
            names,
        )
        return bytes(_native.encode_posterior_artifact(artifact))


def consume_exposure_draws(
    data: Any, *, variable: str, expected_declaration_identity: str | None = None
) -> ExposureDraws:
    """Read exposure draws back from the posterior artifact :meth:`ExposureDraws.export` wrote.

    Refuses an artifact that does not hold ``variable`` or whose declaration identity differs
    from ``expected_declaration_identity`` (retained out of band).
    """
    _name(variable, "variable")
    native = _native.decode_posterior_artifact(container_bytes(data))
    backend = str(native.backend_id)
    if not backend.startswith(f"{_ARTIFACT_BACKEND}."):
        raise CausalValueError("the artifact is not an exposure-measurement draw record")
    names = [str(n) for n in native.quantity_names]
    prefix = f"{variable}["
    if not names or not all(n.startswith(prefix) for n in names):
        raise CausalValueError(f"the artifact does not hold draws of {variable!r}")
    parts = str(native.identification).split(":")
    if len(parts) != 4:
        raise CausalValueError("the artifact's identification is not an exposure-draw identity")
    declaration_identity, data_identity, levels, seed = parts
    if expected_declaration_identity is not None and declaration_identity != (
        expected_declaration_identity
    ):
        raise CausalValueError("the artifact's declaration identity differs from the expected one")
    n, m = len(names), int(native.n_draws)
    flat = np.asarray(native, dtype=np.float64).reshape(n, m)
    draws = np.ascontiguousarray(flat.T)
    return ExposureDraws(
        variable=variable,
        kind=backend.split(".", 1)[1],
        levels=int(levels),
        probability=draws.mean(axis=0) / max(int(levels) - 1, 1),
        draws=draws,
        declaration_identity=declaration_identity,
        data_identity=data_identity,
        seed=int(seed),
        notes=("probability is the mean of the stored draws, rescaled to [0, 1]",),
    )


# --------------------------------------------------------------------------- declarations


class ExposureMeasurement:
    """The common interface of the exposure declarations.

    ``variable`` is the exposure (or facet) the analysis treats; ``covariates`` name the columns
    the imputation model conditions on in addition to the outcome (``None`` takes the other
    variables of the analysis' graph, an empty tuple none).
    """

    variable: str
    covariates: tuple[str, ...] | None
    kind: str
    levels: int = 2

    def read_columns(self) -> tuple[str, ...]:
        """The data columns the declaration itself reads."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe description, the input of the declaration identity."""
        raise NotImplementedError

    def identity(self) -> str:
        """Digest of the declaration."""
        return _digest(self.kind, self.to_dict())

    def _prepare(self, columns: Mapping[str, np.ndarray], y: np.ndarray, rows: int) -> None:
        raise NotImplementedError

    def _fixed_prior(self) -> np.ndarray | None:
        return None

    def _likelihood(
        self, rng: np.random.Generator | None, boot: np.ndarray | None, y: np.ndarray
    ) -> np.ndarray:
        """Per-unit likelihood of the record under each level; point parameters when ``rng`` is None."""
        raise NotImplementedError

    def realize(
        self,
        data: Any,
        *,
        outcome: str,
        covariates: Sequence[str] = (),
        draws: int = 30,
        seed: int = 1,
    ) -> ExposureDraws:
        """Impute the exposure ``draws`` times and return the draws with their per-unit posterior."""
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
            raise CausalValueError("draws must be an integer of at least 2")
        columns = _table_columns(data)
        y = _numeric(columns, _name(outcome, "outcome"))
        n = int(y.shape[0])
        used = tuple(self.covariates if self.covariates is not None else covariates)
        if self.variable in used or outcome in used:
            raise CausalValueError("covariates must not include the exposure or the outcome")
        z = np.column_stack([_numeric(columns, c, n) for c in used]) if used else np.empty((n, 0))
        z = _standardise(z)
        self._prepare(columns, y, n)
        rng = np.random.default_rng(seed)
        everyone = np.arange(n)
        point = _posterior(self._likelihood(None, None, y), y, z, self._fixed_prior(), everyone)
        values = np.arange(self.levels, dtype=np.float64)
        sampled = np.empty((draws, n))
        for k in range(draws):
            boot = rng.integers(0, n, n)
            likelihood = self._likelihood(rng, boot, y)
            post = _posterior(likelihood, y, z, self._fixed_prior(), boot)
            sampled[k] = _sample_levels(rng, post)
        read = sorted({*self.read_columns(), outcome, *used})
        data_identity = _digest(*[(c, _numeric(columns, c, n)) for c in read if c in columns])
        return ExposureDraws(
            variable=self.variable,
            kind=self.kind,
            levels=self.levels,
            probability=point @ values / max(self.levels - 1, 1),
            draws=sampled,
            declaration_identity=self.identity(),
            data_identity=data_identity,
            seed=int(seed),
            notes=self._notes(),
        )

    def _notes(self) -> tuple[str, ...]:
        return ()

    def core(self, data: Any, *, outcome: str) -> Any:
        """The standalone numerical correction of this declaration (a report; ``None`` if none)."""
        return None


@dataclass(frozen=True, slots=True, eq=False)
class MisclassifiedExposure(ExposureMeasurement):
    """A recorded 0/1 exposure whose sensitivity and specificity may differ by outcome level.

    ``error`` is a :class:`~antecedent.measurement_exposure.ExposureError`. An error declared by
    outcome level needs a 0/1 outcome (cases are the units with outcome 1); a shared error accepts
    any outcome. A Youden index at or below ``min_youden`` in a level is refused as not
    identified.
    """

    variable: str
    error: ExposureError
    covariates: tuple[str, ...] | None = None
    min_youden: float = 0.1
    kind: str = field(default="misclassified_exposure", init=False)
    levels: int = field(default=2, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        if not isinstance(self.error, ExposureError):
            raise CausalTypeError("error must be a measurement_exposure.ExposureError")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def read_columns(self) -> tuple[str, ...]:
        return (self.variable,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "error": self.error.to_wire(),
            "min_youden": self.min_youden,
            "covariates": self.covariates,
        }

    def _pairs(self) -> tuple[_Rates, _Rates]:
        error = self.error
        if error.shared_error is not None:
            shared = _Rates.of(error.shared_error)
            return shared, shared
        assert error.cases is not None and error.controls is not None
        return _Rates.of(error.cases), _Rates.of(error.controls)

    def _prepare(self, columns: Mapping[str, np.ndarray], y: np.ndarray, rows: int) -> None:
        w = _binary(columns, self.variable)
        cases, controls = self._pairs()
        if self.error.shared_error is None and not np.all((y == 0.0) | (y == 1.0)):
            raise _refuse(
                "route_not_supported",
                "measurement_model.binary_outcome_continuous_error",
                "an error declared by outcome level needs a 0/1 outcome",
                "Declare one shared error, or analyse a 0/1 outcome.",
            )
        for label, pair in (("cases", cases), ("controls", controls)):
            if pair.youden <= self.min_youden:
                raise _refuse(
                    "measurement_not_identified",
                    "measurement_model.youden_too_small",
                    f"the Youden index of the {label} rates is {pair.youden:.4g}, at or below "
                    f"the floor {self.min_youden:.4g}",
                    "Declare a test with sensitivity + specificity - 1 above the floor.",
                )
        self._state.update(w=w, cases=cases, controls=controls)

    def _likelihood(
        self, rng: np.random.Generator | None, boot: np.ndarray | None, y: np.ndarray
    ) -> np.ndarray:
        w = self._state["w"]
        cases: _Rates = self._state["cases"]
        controls: _Rates = self._state["controls"]
        shared = self.error.shared_error is not None

        def pair(rates: _Rates) -> tuple[float, float]:
            if rng is None:
                return rates.sensitivity, rates.specificity
            for _ in range(20):
                se, sp = rates.draw(rng)
                if se + sp - 1.0 > self.min_youden:
                    return se, sp
            return rates.sensitivity, rates.specificity

        se1, sp1 = pair(cases)
        se0, sp0 = (se1, sp1) if shared else pair(controls)
        is_case = y == 1.0
        se = np.where(is_case, se1, se0)
        sp = np.where(is_case, sp1, sp0)
        exposed = w == 1
        return np.column_stack([np.where(exposed, 1.0 - sp, sp), np.where(exposed, se, 1.0 - se)])

    def core(self, data: Any, *, outcome: str) -> Any:
        columns = _table_columns(data)
        y = _numeric(columns, outcome)
        if not np.all((y == 0.0) | (y == 1.0)):
            return None
        w = _binary(columns, self.variable)
        case = y == 1.0
        return correct_exposure_misclassification(
            exposed_cases=int(w[case].sum()),
            cases=int(case.sum()),
            exposed_controls=int(w[~case].sum()),
            controls=int((~case).sum()),
            error=self.error,
            min_youden=self.min_youden,
        )


@dataclass(frozen=True, slots=True, eq=False)
class SoftExposure(ExposureMeasurement):
    """A column of per-unit probabilities that the exposure is present.

    ``probability`` names the column. The probabilities are taken as calibrated given the
    covariates, as :func:`~antecedent.measurement_exposure.soft_exposure_effect` does; the
    exposure itself is a new column named ``variable``, imputed from them given the outcome.
    """

    variable: str
    probability: str
    covariates: tuple[str, ...] | None = None
    kind: str = field(default="soft_exposure", init=False)
    levels: int = field(default=2, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        _name(self.probability, "probability")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def read_columns(self) -> tuple[str, ...]:
        return (self.probability,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "probability": self.probability,
            "covariates": self.covariates,
        }

    def _prepare(self, columns: Mapping[str, np.ndarray], y: np.ndarray, rows: int) -> None:
        q = _numeric(columns, self.probability, rows)
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise CausalValueError("exposure probabilities must lie in [0, 1]")
        q = np.clip(q, 1e-9, 1.0 - 1e-9)
        self._state["prior"] = np.column_stack([1.0 - q, q])

    def _fixed_prior(self) -> np.ndarray | None:
        prior: np.ndarray = self._state["prior"]
        return prior

    def _likelihood(
        self, rng: np.random.Generator | None, boot: np.ndarray | None, y: np.ndarray
    ) -> np.ndarray:
        return np.ones_like(self._state["prior"])


def _detection_probability(
    declaration: LogisticDetection | TableDetection, parameters: Sequence[float], c: np.ndarray
) -> np.ndarray:
    if isinstance(declaration, LogisticDetection):
        d = 1.0 / (1.0 + np.exp(-(parameters[0] + parameters[1] * c)))
    else:
        d = np.interp(c, np.asarray(declaration.knots), np.asarray(declaration.values))
    return np.clip(d, 1e-9, 1.0 - 1e-9)


@dataclass(frozen=True, slots=True, eq=False)
class DetectedExposure(ExposureMeasurement):
    """A recorded 0/1 detection whose probability is a declared function of a covariate.

    ``variable`` is the recorded detection column, ``detection_covariate`` the column that drives
    detection (for example a count of records), ``detection`` a
    :class:`~antecedent.measurement_exposure.LogisticDetection` (its parameter covariance is
    redrawn per imputation) or :class:`~antecedent.measurement_exposure.TableDetection`. Presence
    is assumed independent of the detection covariate given the imputation covariates; a partially
    declared :class:`~antecedent.measurement_exposure.MonotoneDetection` identifies only bounds and
    gives no per-unit draws, so it is refused.
    """

    variable: str
    detection_covariate: str
    detection: LogisticDetection | TableDetection
    false_positive_rate: float = 0.0
    false_positive_variance: float = 0.0
    min_net_detection: float = 0.1
    covariates: tuple[str, ...] | None = None
    kind: str = field(default="detected_exposure", init=False)
    levels: int = field(default=2, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        _name(self.detection_covariate, "detection_covariate")
        if isinstance(self.detection, MonotoneDetection):
            raise _refuse(
                "measurement_not_identified",
                "measurement_model.insufficient_replicates",
                "a partially declared monotone detection function identifies bounds on the "
                "prevalence, not per-unit exposure draws",
                "Declare the detection function fully (logistic or table), or use "
                "measurement_exposure.detection_dependent_prevalence for sharp bounds.",
            )
        if not isinstance(self.detection, (LogisticDetection, TableDetection)):
            raise CausalTypeError("detection must be a LogisticDetection or TableDetection")
        if not 0.0 <= self.false_positive_rate < 1.0:
            raise CausalValueError("false_positive_rate must lie in [0, 1)")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def read_columns(self) -> tuple[str, ...]:
        return (self.variable, self.detection_covariate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "detection_covariate": self.detection_covariate,
            "detection": self.detection.to_wire(),
            "false_positive": [self.false_positive_rate, self.false_positive_variance],
            "min_net_detection": self.min_net_detection,
            "covariates": self.covariates,
        }

    def _prepare(self, columns: Mapping[str, np.ndarray], y: np.ndarray, rows: int) -> None:
        d = _binary(columns, self.variable)
        c = _numeric(columns, self.detection_covariate, rows)
        net = float(
            np.mean(_detection_probability(self.detection, self._mean(), c))
            - self.false_positive_rate
        )
        if net <= self.min_net_detection:
            raise _refuse(
                "measurement_not_identified",
                "measurement_model.youden_too_small",
                f"the mean detection net of false positives is {net:.4g}, at or below the "
                f"floor {self.min_net_detection:.4g}",
                "Declare a detection function that detects more of the present exposure.",
            )
        self._state.update(d=d, c=c)

    def _mean(self) -> list[float]:
        if isinstance(self.detection, LogisticDetection):
            return [self.detection.intercept, self.detection.slope]
        return []

    def _draw_parameters(self, rng: np.random.Generator) -> list[float]:
        if not isinstance(self.detection, LogisticDetection):
            return []
        det = self.detection
        cov = np.array(
            [[det.intercept_variance, det.covariance], [det.covariance, det.slope_variance]]
        )
        if not np.any(cov):
            return self._mean()
        return [float(v) for v in rng.multivariate_normal(self._mean(), cov, check_valid="ignore")]

    def _likelihood(
        self, rng: np.random.Generator | None, boot: np.ndarray | None, y: np.ndarray
    ) -> np.ndarray:
        d, c = self._state["d"], self._state["c"]
        if rng is None:
            parameters, f = self._mean(), self.false_positive_rate
        else:
            parameters = self._draw_parameters(rng)
            f = (
                _beta_draw(rng, self.false_positive_rate, self.false_positive_variance)
                if self.false_positive_rate > 0.0
                else 0.0
            )
        det = _detection_probability(self.detection, parameters, c)
        detected = d == 1
        return np.column_stack([np.where(detected, f, 1.0 - f), np.where(detected, det, 1.0 - det)])

    def core(self, data: Any, *, outcome: str) -> Any:
        columns = _table_columns(data)
        return detection_dependent_prevalence(
            _binary(columns, self.variable),
            _numeric(columns, self.detection_covariate),
            self.detection,
            false_positive_rate=self.false_positive_rate,
            false_positive_variance=self.false_positive_variance,
            min_net_detection=self.min_net_detection,
        )


@dataclass(frozen=True, slots=True, eq=False)
class RepeatCodedExposure(ExposureMeasurement):
    """Repeat codings of one latent binary or ordinal facet.

    ``codings`` names the integer columns (category indices ``0..categories-1``) holding the
    codings of each unit; ``variable`` is the facet column the analysis then treats (a new column
    whose values are the imputed latent levels). The latent class model, its identification
    conditions and its refusals are those of
    :func:`~antecedent.measurement_exposure.repeat_codings_latent_class`; the response laws are
    refitted on a row bootstrap for every imputation, so the draws carry their uncertainty.
    """

    variable: str
    codings: tuple[str, ...]
    latent_levels: int = 2
    categories: int | None = None
    exchangeable: bool = False
    lowest_level: tuple[float, ...] | None = None
    starts: int = 8
    covariates: tuple[str, ...] | None = None
    kind: str = field(default="repeat_coded_exposure", init=False)
    levels: int = field(default=2, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        object.__setattr__(self, "codings", _names(self.codings, "codings") or ())
        if not self.codings:
            raise CausalValueError("codings must name at least one column")
        if isinstance(self.latent_levels, bool) or self.latent_levels < 2:
            raise CausalValueError("latent_levels must be at least 2")
        object.__setattr__(self, "levels", int(self.latent_levels))
        if self.lowest_level is not None:
            object.__setattr__(self, "lowest_level", tuple(float(v) for v in self.lowest_level))
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def read_columns(self) -> tuple[str, ...]:
        return self.codings

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "codings": self.codings,
            "latent_levels": self.latent_levels,
            "categories": self.categories,
            "exchangeable": self.exchangeable,
            "lowest_level": self.lowest_level,
            "starts": self.starts,
            "covariates": self.covariates,
        }

    def _fit(self, matrix: np.ndarray, seed: int, categories: int | None = None) -> Any:
        return repeat_codings_latent_class(
            matrix,
            levels=self.latent_levels,
            categories=self.categories if categories is None else categories,
            exchangeable=self.exchangeable,
            lowest_level=self.lowest_level,
            starts=self.starts,
            seed=seed,
        )

    def _prepare(self, columns: Mapping[str, np.ndarray], y: np.ndarray, rows: int) -> None:
        matrix = np.column_stack([_numeric(columns, c, rows) for c in self.codings])
        if not np.all(matrix == np.round(matrix)):
            raise CausalValueError("codings must be integer category indices")
        matrix = matrix.astype(np.int64)
        categories = int(self.categories) if self.categories is not None else int(matrix.max()) + 1
        categories = max(categories, 2)
        self._state.update(
            matrix=matrix, categories=categories, point=self._fit(matrix, 1, categories)
        )

    def _lik_from(self, fit: Any) -> np.ndarray:
        matrix: np.ndarray = self._state["matrix"]
        response = np.asarray(fit.response)
        log_lik = np.zeros((matrix.shape[0], self.levels))
        for k in range(matrix.shape[1]):
            slot = 0 if response.shape[0] == 1 else k
            log_lik += np.log(np.maximum(response[slot][:, matrix[:, k]].T, _TINY))
        return np.exp(log_lik - log_lik.max(axis=1, keepdims=True))

    def _likelihood(
        self, rng: np.random.Generator | None, boot: np.ndarray | None, y: np.ndarray
    ) -> np.ndarray:
        if rng is None or boot is None:
            return self._lik_from(self._state["point"])
        matrix: np.ndarray = self._state["matrix"]
        for attempt in range(5):
            try:
                fit = self._fit(
                    matrix[boot],
                    int(rng.integers(1, 2**31 - 1)) + attempt,
                    self._state["categories"],
                )
            except MeasurementModelRefusal:
                continue
            return self._lik_from(fit)
        raise _refuse(
            "measurement_not_identified",
            "measurement_model.bootstrap_unstable",
            "the latent class model could not be refitted on resampled units",
            "Supply more units or more codings per unit.",
        )

    def _notes(self) -> tuple[str, ...]:
        return ("codings are conditionally independent given the latent level (assumed)",)

    def core(self, data: Any, *, outcome: str) -> Any:
        return self._state.get("point")


# --------------------------------------------------------------------------- pooling


@dataclass(frozen=True, slots=True)
class RubinPool:
    """Rubin's rules over ``m`` estimates and their variances."""

    estimate: float
    within: float
    between: float
    total: float
    degrees_of_freedom: float
    draws: int

    @property
    def standard_error(self) -> float:
        """The square root of the total variance."""
        return math.sqrt(self.total)

    def interval(self, level: float = 0.95) -> MeasurementInterval:
        """The ``t`` interval at ``level``, with calibration ``not_measured``."""
        if not 0.0 < level < 1.0:
            raise CausalValueError("level must lie strictly between 0 and 1")
        t = _student_t_quantile(0.5 + level / 2.0, self.degrees_of_freedom)
        half = t * self.standard_error
        return MeasurementInterval(
            self.estimate - half, self.estimate + half, level, dict(_NOT_MEASURED)
        )


def rubin_pool(estimates: Sequence[float], variances: Sequence[float]) -> RubinPool:
    """Pool ``m >= 2`` estimates with Rubin's rules.

    ``total = within + (1 + 1/m) between`` and the degrees of freedom are Rubin's
    ``(m - 1)(1 + within / ((1 + 1/m) between))^2`` (infinite when there is no between variance).
    """
    q = np.asarray(estimates, dtype=np.float64)
    u = np.asarray(variances, dtype=np.float64)
    if q.ndim != 1 or q.shape != u.shape or q.shape[0] < 2:
        raise CausalValueError("pooling needs at least two estimates, each with a variance")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(u)) and np.all(u >= 0.0)):
        raise CausalValueError("estimates and variances must be finite, variances non-negative")
    m = q.shape[0]
    within = float(u.mean())
    between = float(q.var(ddof=1))
    total = within + (1.0 + 1.0 / m) * between
    if between <= 0.0 or total <= 0.0:
        df = math.inf
    else:
        df = (m - 1.0) * (1.0 + within / ((1.0 + 1.0 / m) * between)) ** 2
    return RubinPool(float(q.mean()), within, between, total, df, int(m))


def _t_cdf(x: float, df: float) -> float:
    if math.isinf(df):
        return 0.5 * math.erfc(-x / math.sqrt(2.0))
    t2 = x * x
    z = df / (df + t2)
    tail = 0.5 * _regularised_beta(df / 2.0, 0.5, z)
    return 1.0 - tail if x > 0 else tail


def _regularised_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _student_t_quantile(p: float, df: float) -> float:
    """Quantile of Student's ``t`` by bisection on its distribution function."""
    if math.isinf(df) or df > 1e7:
        lo, hi = 0.0, 40.0
        df = math.inf
    else:
        lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- the result


@dataclass(frozen=True, slots=True, eq=False)
class MeasuredAnalysis:
    """A causal estimate of a measured exposure, pooled over the exposure draws.

    ``effect`` is the mean of the per-draw estimates, ``variance_components`` Rubin's within-draw
    (the estimator's own variance), between-draw (exposure uncertainty) and total variance, and
    :meth:`interval` the pooled ``t`` interval (calibration ``not_measured``). ``analyses`` holds
    the per-draw typed results, ``exposure`` the draws (export them with
    ``exposure.export()``), ``naive_effect`` the estimate that takes the recorded column at face
    value (``None`` when the exposure has no recorded column).
    """

    effect: float
    pool: RubinPool
    draw_effects: np.ndarray
    draw_variances: np.ndarray
    analyses: tuple[Any, ...]
    exposure: ExposureDraws
    measurement: ExposureMeasurement
    naive_effect: float | None
    query: Any

    @property
    def standard_error(self) -> float:
        """The pooled standard error (uncalibrated)."""
        return self.pool.standard_error

    @property
    def variance_components(self) -> tuple[VarianceComponent, ...]:
        """Within-draw, between-draw and total variance."""
        return (
            VarianceComponent("within_imputation", self.pool.within),
            VarianceComponent(
                "between_imputation", (1.0 + 1.0 / self.pool.draws) * self.pool.between
            ),
            VarianceComponent("total", self.pool.total),
        )

    def interval(self, level: float = 0.95) -> MeasurementInterval:
        """The pooled ``t`` interval at ``level``; calibration is ``not_measured``."""
        return self.pool.interval(level)

    @property
    def answer(self) -> Answer:
        return Answer("point", value=self.effect, detail=f"measured_exposure.{self.exposure.kind}")

    @property
    def identities(self) -> dict[str, Any]:
        return {
            "declaration_identity": self.exposure.declaration_identity,
            "data_identity": self.exposure.data_identity,
            "exposure_draws_identity": self.exposure.identity,
        }

    def inspect(self) -> Any:
        from .results._report import InspectionReport, SlotModel

        interval = self.interval()
        return InspectionReport(
            identification=SlotModel(
                available=True,
                summary="per_draw_identification_of_the_completed_tables",
                reason=None,
                payload={"draws": self.pool.draws},
            ),
            support=SlotModel(available=False, reason="not_measured", summary="not_measured"),
            uncertainty=SlotModel(
                available=True,
                summary="rubin_pooled_interval",
                reason="uncalibrated_no_interval_claim",
                payload={
                    "standard_error": self.standard_error,
                    "interval": (interval.lower, interval.upper),
                    "degrees_of_freedom": self.pool.degrees_of_freedom,
                    "within": self.pool.within,
                    "between": self.pool.between,
                    "total": self.pool.total,
                    "calibration": dict(_NOT_MEASURED),
                },
            ),
            assumptions=SlotModel(
                available=True,
                summary="measurement_declaration_and_imputation_model",
                reason=None,
                payload={"notes": self.exposure.notes, "kind": self.exposure.kind},
            ),
            answer=self.answer,
            family="MeasuredAnalysis",
            stage="executed",
            operations=("inspect",),
        )

    def __repr__(self) -> str:
        interval = self.interval()
        return (
            f"<MeasuredAnalysis {self.exposure.kind} effect={self.effect:.6g} "
            f"se={self.standard_error:.6g} ci95=[{interval.lower:.6g}, {interval.upper:.6g}] "
            f"draws={self.pool.draws}>"
        )


def _per_draw(outcome: Outcome[Any]) -> tuple[float, float]:
    result = outcome.value
    estimate = getattr(result, "estimate", None)
    point = getattr(estimate, "ate", None)
    if point is None:
        raise CausalUnsupportedError(
            "a measured exposure is pooled over scalar effects; this query reports no scalar",
            reason_code="option_not_applicable",
        )
    se = getattr(estimate, "se_analytic", None)
    if se is None or not math.isfinite(float(se)):
        se = getattr(estimate, "se_bootstrap", None)
    if se is None or not math.isfinite(float(se)):
        raise CausalUnsupportedError(
            "a measured exposure is pooled with Rubin's rules, which need each draw's standard error",
            reason_code="option_not_applicable",
        )
    return float(point), float(se) ** 2


# --------------------------------------------------------------------------- preparation


def _graph_covariates(graph: Any, exclude: set[str]) -> tuple[str, ...]:
    if graph is None:
        return ()
    edges = getattr(graph, "edges", graph)
    names: list[str] = []
    try:
        iterator = list(edges() if callable(edges) else edges)
    except TypeError:
        return ()
    for edge in iterator:
        if isinstance(edge, (tuple, list)) and len(edge) >= 2:
            ends = (edge[0], edge[1]) if len(edge) == 2 else (edge[0], edge[2])
            for end in ends:
                if isinstance(end, str) and end not in exclude and end not in names:
                    names.append(end)
    return tuple(names)


@dataclass(frozen=True, slots=True, eq=False)
class PreparedMeasuredAnalysis:
    """A study of a measured exposure: one prepared study per exposure draw.

    Returned by :func:`antecedent.prepare` when ``measurement=`` is declared. ``estimate()`` (or
    :func:`antecedent.estimate`) runs every prepared study and pools the results.
    """

    studies: tuple[Any, ...]
    exposure: ExposureDraws
    measurement: ExposureMeasurement
    declarations: Mapping[str, Any]
    data: Any
    execution: Any
    draws: int
    seed: int

    @property
    def refusal(self) -> Any:
        """The first refusal met while preparing any draw's study, or ``None``."""
        for study in self.studies:
            if study.refusal is not None:
                return study.refusal
        return None

    def estimate(self) -> Outcome[MeasuredAnalysis]:
        """Estimate every completed table and pool the effects by Rubin's rules."""
        from .prepared import estimate

        outcomes = []
        for study in self.studies:
            outcome = estimate(study)
            if outcome.refusal is not None:
                return Outcome(refusal=outcome.refusal)
            outcomes.append(outcome)
        pairs = [_per_draw(o) for o in outcomes]
        pool = rubin_pool([p for p, _ in pairs], [v for _, v in pairs])
        naive = None
        recorded = _table_columns(self.data).get(self.measurement.variable)
        if recorded is not None:
            from .prepared import analyze

            naive_outcome = analyze(self.data, execution=self.execution, **self.declarations)
            if naive_outcome.refusal is None:
                try:
                    naive = _per_draw(naive_outcome)[0]
                except CausalUnsupportedError:
                    naive = None
        return Outcome(
            value=MeasuredAnalysis(
                effect=pool.estimate,
                pool=pool,
                draw_effects=np.array([p for p, _ in pairs]),
                draw_variances=np.array([v for _, v in pairs]),
                analyses=tuple(o.value for o in outcomes),
                exposure=self.exposure,
                measurement=self.measurement,
                naive_effect=naive,
                query=self.declarations["query"],
            )
        )

    def rebind(self, *, data: Any) -> PreparedMeasuredAnalysis:
        """Prepare the same declaration on other data (the exposure is imputed afresh)."""
        return prepare_measured(
            data,
            measurement=self.measurement,
            draws=self.draws,
            seed=self.seed,
            execution=self.execution,
            **self.declarations,
        )

    def reconfigure(self, **declarations: Any) -> PreparedMeasuredAnalysis:
        """Prepare again with some declarations of the analysis replaced."""
        merged = {**self.declarations, **declarations}
        return prepare_measured(
            self.data,
            measurement=self.measurement,
            draws=self.draws,
            seed=self.seed,
            execution=self.execution,
            **merged,
        )

    def inspect(self) -> Any:
        """The inspection report of the first draw's study."""
        return self.studies[0].inspect()


def prepare_measured(
    data: Any,
    *,
    measurement: ExposureMeasurement,
    draws: int = 30,
    seed: int | None = None,
    execution: Any = None,
    query: Any,
    **declarations: Any,
) -> PreparedMeasuredAnalysis:
    """Impute the declared exposure ``draws`` times and prepare the analysis on each completed table.

    ``declarations`` are the arguments of :func:`antecedent.prepare`. The query's treatment must be
    the declared variable and the query must name an outcome.
    """
    from .prepared import prepare

    if not isinstance(measurement, ExposureMeasurement):
        raise CausalTypeError("measurement must be one of the exposure_measurement declarations")
    treatment = getattr(query, "treatment", None)
    outcome = getattr(query, "outcome", None)
    if treatment != measurement.variable:
        raise CausalValueError(
            f"the measurement is declared on {measurement.variable!r} but the query's treatment "
            f"is {treatment!r}; declare it on the exposure the query intervenes on"
        )
    if not isinstance(outcome, str):
        raise CausalValueError("a measured exposure needs a query with a single named outcome")
    if seed is None:
        seed = int(getattr(execution, "seed", 1))
    columns = _table_columns(data)
    covariates = (
        _graph_covariates(declarations.get("graph"), {treatment, outcome})
        if measurement.covariates is None
        else measurement.covariates
    )
    covariates = tuple(c for c in covariates if c in columns)
    exposure = measurement.realize(
        data, outcome=outcome, covariates=covariates, draws=draws, seed=seed
    )
    studies = tuple(
        prepare(table, query=query, execution=execution, **declarations)
        for table in exposure.completed_tables(data)
    )
    return PreparedMeasuredAnalysis(
        studies=studies,
        exposure=exposure,
        measurement=measurement,
        declarations={"query": query, **declarations},
        data=data,
        execution=execution,
        draws=draws,
        seed=seed,
    )
