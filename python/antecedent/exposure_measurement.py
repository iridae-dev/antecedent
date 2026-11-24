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

The semantics are Rust's (``antecedent_estimate::measurement::measured_exposure`` and the facade
``antecedent::measured_exposure``): this module states the declaration as typed values, asks the
native layer to impute the latent exposure and to pool the estimates, and runs the Python
study lifecycle (``prepare`` and ``estimate``) on every completed table.

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

A partially declared monotone detection function identifies only bounds on the prevalence and is
refused: bounds on an effect would need the effect at every detection function inside the
envelope, and the effect of a unit-level exposure is not monotone in the detection function in
general, so the envelope's endpoints do not bound it.

The exposure draws are a first-class column of uncertain values: :meth:`ExposureDraws.export`
writes them as a posterior artifact (one quantity per unit, one draw per imputation), and
:func:`consume_exposure_draws` reads them back; :meth:`ExposureDraws.completed_tables` hands
every draw to any downstream stage that takes a table (discovery, other queries).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native, _reason_details, _retained
from ._artifact_protocol import container_bytes
from ._refusal import raise_refusal
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .measurement_exposure import (
    ExposureError,
    LogisticDetection,
    MonotoneDetection,
    TableDetection,
    ValidationCounts,
    correct_exposure_misclassification,
    detection_dependent_prevalence,
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


def _call(function: Any, declaration: Mapping[str, Any]) -> dict[str, Any]:
    result, refusal = function(json.dumps(declaration, allow_nan=False, separators=(",", ":")))
    raise_refusal(refusal, MeasurementModelRefusal)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


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


def _binary(columns: Mapping[str, np.ndarray], name: str) -> list[int]:
    array = _numeric(columns, name)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise CausalValueError(f"column {name!r} must hold 0/1 values")
    return [int(v) for v in array]


def _rates_wire(source: BinaryRates | ValidationCounts) -> dict[str, Any]:
    if isinstance(source, ValidationCounts):
        return {"validation": source.to_wire()}
    if not isinstance(source, BinaryRates):
        raise CausalTypeError("rates must be BinaryRates or ValidationCounts")
    return {
        "sensitivity": float(source.sensitivity),
        "specificity": float(source.specificity),
        "sensitivity_variance": float(source.sensitivity_variance),
        "specificity_variance": float(source.specificity_variance),
    }


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
    stream: int
    notes: tuple[str, ...] = ()

    @classmethod
    def _from_wire(
        cls, variable: str, wire: Mapping[str, Any], notes: tuple[str, ...] = ()
    ) -> ExposureDraws:
        m, n = int(wire["draw_count"]), int(wire["rows"])
        return cls(
            variable=variable,
            kind=str(wire["kind"]),
            levels=int(wire["levels"]),
            probability=np.asarray(wire["probability"], dtype=np.float64),
            draws=np.asarray(wire["draws"], dtype=np.float64).reshape(m, n),
            declaration_identity=str(wire["declaration_identity"]),
            data_identity=str(wire["data_identity"]),
            stream=int(wire["seed"]),
            notes=notes,
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "levels": self.levels,
            "rows": self.rows,
            "draw_count": self.draw_count,
            "probability": [float(v) for v in self.probability],
            "draws": [float(v) for v in self.draws.reshape(-1)],
            "declaration_identity": self.declaration_identity,
            "data_identity": self.data_identity,
            "seed": self.stream,
        }

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
        """Digest of the declaration, the data it read and the seed."""
        return f"{self.declaration_identity}:{self.data_identity}:{self.stream}"

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
        :func:`antecedent.consume_posterior` or :func:`consume_exposure_draws`); the quantity
        ``"{variable}[i]"`` holds unit ``i``'s draws and the declaration and data identities are
        carried in the record's backend id.
        """
        payload = json.dumps(self._wire(), allow_nan=False, separators=(",", ":"))
        return bytes(_native.measured_exposure_export(payload, self.variable))


def consume_exposure_draws(
    data: Any,
    *,
    variable: str,
    expected_declaration_identity: str | None = None,
    retained_identity: str | None = None,
) -> ExposureDraws:
    """Read exposure draws back from the posterior artifact :meth:`ExposureDraws.export` wrote.

    Refuses an artifact that does not hold ``variable`` or whose declaration identity differs
    from ``expected_declaration_identity`` (retained out of band).
    """
    _retained.admit(
        data,
        retained_identity,
        "antecedent.exposure_measurement.consume_exposure_draws",
        "antecedent.exposure_measurement.consume_exposure_draws(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
        bound=expected_declaration_identity is not None,
    )
    _name(variable, "variable")
    try:
        text = _native.measured_exposure_consume(
            container_bytes(data), variable, expected_declaration_identity
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return ExposureDraws._from_wire(
        variable,
        json.loads(text),
        ("probability is the mean of the stored draws, rescaled to [0, 1]",),
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

    def _wire(self, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """The declaration the native layer reads, with the data columns it covers."""
        raise NotImplementedError

    def realize(
        self,
        data: Any,
        *,
        outcome: str,
        covariates: Sequence[str] = (),
        draws: int = 30,
        execution: Execution | None = None,
    ) -> ExposureDraws:
        """Impute the exposure ``draws`` times and return the draws with their per-unit posterior.

        The imputation randomness is the ``execution``'s seed (default ``Execution()``).
        """
        if execution is not None and not isinstance(execution, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        seed = (Execution() if execution is None else execution).seed
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
            raise CausalValueError("draws must be an integer of at least 2")
        columns = _table_columns(data)
        y = _numeric(columns, _name(outcome, "outcome"))
        n = int(y.shape[0])
        used = tuple(self.covariates if self.covariates is not None else covariates)
        if self.variable in used or outcome in used:
            raise CausalValueError("covariates must not include the exposure or the outcome")
        z = np.column_stack([_numeric(columns, c, n) for c in used]) if used else np.empty((n, 0))
        wire = _call(
            _native.measured_exposure_impute,
            {
                "declaration": self._wire(columns),
                "outcome": [float(v) for v in y],
                "covariates": [float(v) for v in z.T.reshape(-1)],
                "covariate_count": len(used),
                "draws": int(draws),
                "seed": int(seed),
            },
        )
        return ExposureDraws._from_wire(self.variable, wire, self._notes())

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

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        if not isinstance(self.error, ExposureError):
            raise CausalTypeError("error must be a measurement_exposure.ExposureError")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def _wire(self, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
        error = self.error
        if error.shared_error is not None:
            structure: dict[str, Any] = {"kind": "shared", "rates": _rates_wire(error.shared_error)}
        else:
            assert error.cases is not None and error.controls is not None
            structure = {
                "kind": "by_outcome",
                "cases": _rates_wire(error.cases),
                "controls": _rates_wire(error.controls),
            }
        return {
            "kind": "misclassified",
            "recorded": _binary(columns, self.variable),
            "min_youden": float(self.min_youden),
            "error": structure,
        }

    def core(self, data: Any, *, outcome: str) -> Any:
        columns = _table_columns(data)
        y = _numeric(columns, outcome)
        if not np.all((y == 0.0) | (y == 1.0)):
            return None
        w = np.asarray(_binary(columns, self.variable))
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

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        _name(self.probability, "probability")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def _wire(self, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
        q = _numeric(columns, self.probability)
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise CausalValueError("exposure probabilities must lie in [0, 1]")
        return {"kind": "soft", "probability": [float(v) for v in q]}


@dataclass(frozen=True, slots=True, eq=False)
class DetectedExposure(ExposureMeasurement):
    """A recorded 0/1 detection whose probability is a declared function of a covariate.

    ``variable`` is the recorded detection column, ``detection_covariate`` the column that drives
    detection (for example a count of records), ``detection`` a
    :class:`~antecedent.measurement_exposure.LogisticDetection` (its parameter covariance is
    redrawn per imputation) or :class:`~antecedent.measurement_exposure.TableDetection`. Presence
    is assumed independent of the detection covariate given the imputation covariates. A partially
    declared :class:`~antecedent.measurement_exposure.MonotoneDetection` identifies only bounds
    on the prevalence, has no per-unit law and is refused.
    """

    variable: str
    detection_covariate: str
    detection: LogisticDetection | TableDetection
    false_positive_rate: float = 0.0
    false_positive_variance: float = 0.0
    min_net_detection: float = 0.1
    covariates: tuple[str, ...] | None = None
    kind: str = field(default="detected_exposure", init=False)

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        _name(self.detection_covariate, "detection_covariate")
        if isinstance(self.detection, MonotoneDetection):
            raise MeasurementModelRefusal(
                {
                    "code": "measurement_not_identified",
                    "detail": _reason_details.measurement_model.INSUFFICIENT_REPLICATES,
                    "message": "a partially declared monotone detection function identifies "
                    "bounds on the prevalence, not per-unit exposure draws",
                    "remedy": "Declare the detection function fully (logistic or table), or use "
                    "measurement_exposure.detection_dependent_prevalence for sharp bounds.",
                }
            )
        if not isinstance(self.detection, (LogisticDetection, TableDetection)):
            raise CausalTypeError("detection must be a LogisticDetection or TableDetection")
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def _wire(self, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
        n = len(_binary(columns, self.variable))
        return {
            "kind": "detected",
            "detected": _binary(columns, self.variable),
            "covariate": [float(v) for v in _numeric(columns, self.detection_covariate, n)],
            "function": self.detection.to_wire(),
            "false_positive_rate": float(self.false_positive_rate),
            "false_positive_variance": float(self.false_positive_variance),
            "min_net_detection": float(self.min_net_detection),
        }

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

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        object.__setattr__(self, "codings", _names(self.codings, "codings") or ())
        if not self.codings:
            raise CausalValueError("codings must name at least one column")
        if isinstance(self.latent_levels, bool) or self.latent_levels < 2:
            raise CausalValueError("latent_levels must be at least 2")
        if self.lowest_level is not None:
            object.__setattr__(self, "lowest_level", tuple(float(v) for v in self.lowest_level))
        object.__setattr__(self, "covariates", _names(self.covariates, "covariates"))

    def _wire(self, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
        matrix = np.column_stack([_numeric(columns, c) for c in self.codings])
        if not np.all(matrix == np.round(matrix)) or np.any(matrix < 0):
            raise CausalValueError("codings must be non-negative integer category indices")
        categories = int(self.categories) if self.categories is not None else int(matrix.max()) + 1
        return {
            "kind": "repeat_coded",
            "codings": [int(v) for v in matrix.reshape(-1)],
            "coding_count": len(self.codings),
            "categories": max(categories, 2),
            "latent_levels": int(self.latent_levels),
            "exchangeable": bool(self.exchangeable),
            "lowest_level": None if self.lowest_level is None else list(self.lowest_level),
            "starts": int(self.starts),
        }

    def _notes(self) -> tuple[str, ...]:
        return ("codings are conditionally independent given the latent level (assumed)",)


# --------------------------------------------------------------------------- pooling


@dataclass(frozen=True, slots=True)
class RubinPool:
    """Rubin's rules over ``m`` estimates and their variances (computed natively)."""

    estimate: float
    within: float
    between: float
    total: float
    degrees_of_freedom: float
    draws: int
    _intervals: Mapping[float, tuple[float, float]] = field(default_factory=dict, repr=False)
    _points: tuple[float, ...] = field(default=(), repr=False)
    _variances: tuple[float, ...] = field(default=(), repr=False)

    @property
    def standard_error(self) -> float:
        """The square root of the total variance."""
        return math.sqrt(self.total)

    def interval(self, level: float = 0.95) -> MeasurementInterval:
        """The ``t`` interval at ``level``, with calibration ``not_measured``."""
        if not 0.0 < level < 1.0:
            raise CausalValueError("level must lie strictly between 0 and 1")
        wire = _pool_wire(self._points, self._variances, level)
        interval = wire["interval"]
        return MeasurementInterval(
            float(interval["lower"]), float(interval["upper"]), level, dict(_NOT_MEASURED)
        )


def _pool_wire(
    estimates: Sequence[float], variances: Sequence[float], level: float
) -> dict[str, Any]:
    if not all(math.isfinite(float(v)) for v in (*estimates, *variances)):
        raise CausalValueError("estimates and variances must be finite")
    return _call(
        _native.measured_exposure_pool,
        {
            "points": [float(v) for v in estimates],
            "variances": [float(v) for v in variances],
            "level": float(level),
        },
    )


def rubin_pool(estimates: Sequence[float], variances: Sequence[float]) -> RubinPool:
    """Pool ``m >= 2`` estimates with Rubin's rules (the native pooling).

    ``total = within + (1 + 1/m) between`` and the degrees of freedom are Rubin's
    ``(m - 1)(1 + within / ((1 + 1/m) between))^2`` (infinite when there is no between variance).
    """
    wire = _pool_wire(estimates, variances, 0.95)
    return RubinPool(
        estimate=float(wire["estimate"]),
        within=float(wire["within"]),
        between=float(wire["between"]),
        total=float(wire["total"]),
        degrees_of_freedom=float(wire["degrees_of_freedom"])
        if wire["degrees_of_freedom"] is not None
        else math.inf,
        draws=int(wire["draws"]),
        _points=tuple(float(v) for v in estimates),
        _variances=tuple(float(v) for v in variances),
    )


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
    structure: Mapping[str, str] | None = None

    @property
    def standard_error(self) -> float:
        """The pooled standard error (uncalibrated)."""
        return self.pool.standard_error

    @property
    def provenance(self) -> dict[str, Any]:
        """Every stage the pooled result consumed, named by what each declared.

        ``measurement`` is the exposure declaration, its data and the draws; ``structure`` (when a
        posterior search ran on the completed tables) is the accepted pooled posterior's
        provenance: the search, its structure prior and the pooling over the imputations;
        ``declarations`` is every other stage the per-draw results recorded in their assumptions
        (a weighted target population, a composed or selected outcome prior, the structure
        posterior the effect was mixed over), in first-appearance order without repeats.
        """
        stages: dict[str, Any] = {
            "measurement": {
                "kind": self.exposure.kind,
                "draws": self.exposure.draw_count,
                **self.identities,
            }
        }
        if self.structure is not None:
            stages["structure"] = dict(self.structure)
        declared: list[str] = []
        for analysis in self.analyses:
            for item in getattr(analysis, "assumptions", None) or ():
                text = str(item)
                if text not in declared:
                    declared.append(text)
        stages["declarations"] = declared
        target = getattr(self.analyses[0], "weighted_target", None) if self.analyses else None
        if target is not None:
            stages["weighted_target"] = target.to_dict()
        return stages

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
    structure: Mapping[str, str] | None = None

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
                structure=self.structure,
            )
        )

    def rebind(self, *, data: Any) -> PreparedMeasuredAnalysis:
        """Prepare the same declaration on other data (the exposure is imputed afresh)."""
        return prepare_measured(
            data,
            measurement=self.measurement,
            draws=self.draws,
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
    columns = _table_columns(data)
    covariates = (
        _graph_covariates(declarations.get("graph"), {treatment, outcome})
        if measurement.covariates is None
        else measurement.covariates
    )
    covariates = tuple(c for c in covariates if c in columns)
    exposure = measurement.realize(
        data, outcome=outcome, covariates=covariates, draws=draws, execution=execution
    )
    tables = exposure.completed_tables(data)
    structure: Mapping[str, str] | None = None
    study_declarations = declarations
    search = declarations.get("discovery")
    if search is not None and declarations.get("graph") is None:
        from .discovery import _POOLING_SEARCHES, pool_search_over_imputations

        if isinstance(search, _POOLING_SEARCHES):
            # The completed tables are the imputations of the structure search: the engine runs
            # it on each, pools the posteriors once and hands the same accepted posterior (with
            # the search's, the prior's and the pooling's provenance) to every table's study.
            pooled = pool_search_over_imputations(search, tables, execution=execution)
            study_declarations = {**declarations, "discovery": pooled}
            structure = getattr(pooled, "provenance", None)
    studies = tuple(
        prepare(table, query=query, execution=execution, **study_declarations) for table in tables
    )
    return PreparedMeasuredAnalysis(
        studies=studies,
        exposure=exposure,
        measurement=measurement,
        declarations={"query": query, **declarations},
        data=data,
        execution=execution,
        draws=draws,
        structure=structure,
    )
