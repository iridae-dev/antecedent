"""Sequentially randomized unit-period designs and regime survival.

One table row per unit and period. The recorded assignment law (two arms, several arms or
categorical levels, a two-factor factorial, within-cluster interference, or a continuous dose
with its recorded density) identifies the *excursion* effect of a decision on the same unit's
outcome ``lag`` periods later, with carryover inside the target. ``design`` is a mapping whose
``"design"`` key names the design::

    {"design": "two_arm", "assignment": [...], "probability": [...]}
    {"design": "multi_arm", "assignment": [...], "probabilities": [[...], ...],
     "arms": [...], "categorical": False, "ordered": False}
    {"design": "factorial", "first": [...], "first_probability": [...],
     "second": [...], "second_probability": [...]}
    {"design": "interference", "assignment": [...], "probability": [...], "clusters": [...]}
    {"design": "dose", "dose": [...], "density": [...], "bandwidth": 0.3,
     "target": {"kind": "smoothed_curve", "grid": [...]}}
    {"design": "regime_survival", "treatment": "a", "censored": "c",
     "confounders": ["l"], "horizon": 5}

A dose target is ``smoothed_curve`` (``grid``), ``smoothed_contrast`` (``active``,
``control``) or ``local_quadratic`` (``grid``, ``derivative``, ``curvature_bound``).
Regime survival reads observational subject trajectories (outcome = event indicator) and
contrasts always-treat with never-treat survival at ``horizon``.

The same designs are typed queries for :func:`antecedent.analyze` and
:func:`antecedent.prepare`: :class:`TwoArmExcursion`, :class:`MultiArmExcursion`,
:class:`FactorialExcursion`, :class:`InterferenceExcursion`, :class:`DoseExcursion` (with a
:class:`SmoothedDoseCurve`, :class:`SmoothedDoseContrast` or :class:`LocalQuadraticDoseGrid`
target) and :class:`RegimeSurvivalContrast`, each naming columns of the table, with
``quantile_level=`` and ``misclassification=OutcomeMisclassification(...)`` options. The
result is ``result.temporal_design`` (:class:`TemporalDesignEstimate`), which a loaded
artifact reads back equal.

Every result is unlicensed and its calibration is unmeasured. Frequentist intervals are
cluster-robust (a whole-procedure bootstrap for the quantile functional and regime survival,
which need ``bootstrap=``); Bayesian results are a Bayesian bootstrap over independent units
(regime survival: the joint-likelihood parametric g-formula). See
``docs/guides/temporal-designs.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from antecedent import _native
from antecedent._data import ingest_columns, read_columns
from antecedent._temporal_design import (
    DoseExcursion,
    FactorialExcursion,
    InterferenceExcursion,
    LocalQuadraticDoseGrid,
    MultiArmExcursion,
    OutcomeMisclassification,
    RegimeSurvivalContrast,
    SmoothedDoseContrast,
    SmoothedDoseCurve,
    TemporalDesignEstimate,
    TwoArmExcursion,
)
from antecedent.errors import CausalValueError

__all__ = [
    "DoseExcursion",
    "FactorialExcursion",
    "InterferenceExcursion",
    "LocalQuadraticDoseGrid",
    "MultiArmExcursion",
    "OutcomeMisclassification",
    "RegimeSurvivalContrast",
    "SmoothedDoseContrast",
    "SmoothedDoseCurve",
    "TemporalDesignEstimate",
    "TemporalDesignResult",
    "TwoArmExcursion",
    "estimate_temporal_design",
]


@dataclass(frozen=True)
class TemporalDesignResult:
    """One temporal-design result: named coordinates with their uncertainty.

    ``estimate`` is the point estimate (frequentist) or the posterior mean (Bayesian);
    ``standard_error`` is the standard error or the posterior standard deviation. The band
    fields are set when several contrasts or a dose grid carry a simultaneous band.
    """

    estimator: str
    design: str
    functional: str
    coordinates: tuple[str, ...]
    estimate: tuple[float, ...]
    standard_error: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    band_coordinates: tuple[int, ...]
    band_critical_value: float | None
    band_lower: tuple[float, ...] | None
    band_upper: tuple[float, ...] | None
    holm_adjusted_p: tuple[float, ...] | None
    smoothing_bias_bound: tuple[float, ...] | None
    pairs: int
    clusters: int
    unpaired: int
    replicates: tuple[int, int] | None
    draws: int | None
    uncertainty: str
    calibration: str
    licensed: bool

    def coordinate(self, name: str) -> tuple[float, float, float]:
        """``(estimate, lower, upper)`` of the named coordinate."""
        try:
            index = self.coordinates.index(name)
        except ValueError:
            raise CausalValueError(
                f"unknown coordinate {name!r}; the result has {list(self.coordinates)}",
                reason_code="invalid_argument",
            ) from None
        return (self.estimate[index], self.lower[index], self.upper[index])


def _optional(values: Sequence[float] | None) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(value) for value in values)


def estimate_temporal_design(
    data: Any,
    *,
    outcome: str,
    units: Sequence[Any],
    periods: Sequence[int],
    design: Mapping[str, Any],
    lag: int = 0,
    quantile_level: float | None = None,
    min_probability: float = 0.05,
    misclassification: tuple[int, int, int, int] | None = None,
    inference: Literal["frequentist", "bayesian"] = "frequentist",
    n_draws: int | None = None,
    prior_scale: float = 10.0,
    bootstrap: int | None = None,
    seed: int = 1,
    threads: int | None = None,
) -> TemporalDesignResult:
    """Estimate the excursion contrasts of a sequentially randomized design, or regime survival.

    ``misclassification`` is ``(true positive, false negative, true negative, false
    positive)`` from an independent validation sample of a misclassified binary outcome.
    ``bootstrap`` is the whole-procedure replicate count of the quantile functional and of
    frequentist regime survival (at least 99); every other design refuses it.
    """
    if inference not in ("frequentist", "bayesian"):
        raise CausalValueError(
            "inference must be 'frequentist' or 'bayesian'", reason_code="invalid_argument"
        )
    spec = dict(design)
    columns_needed = [outcome]
    if spec.get("design") == "regime_survival":
        columns_needed.extend([str(spec["treatment"]), str(spec["censored"])])
        columns_needed.extend(str(name) for name in spec.get("confounders", ()))
    raw = read_columns(data, tuple(columns_needed), what="temporal design")
    names, columns = ingest_columns({name: raw[name] for name in columns_needed})
    options: dict[str, Any] = {
        "inference": {
            # The design's posterior is its own; the conjugate declaration carries the draw
            # count and prior scale without selecting a regression backend.
            "mode": "conjugate" if inference == "bayesian" else "frequentist",
            "n_draws": n_draws,
            "prior_scale": float(prior_scale),
        }
    }
    if bootstrap is not None:
        options["bootstrap"] = int(bootstrap)
    prepared = _native.PreparedAnalysis.prepare_temporal_design(
        names,
        columns,
        outcome,
        [str(unit) for unit in units],
        [int(period) for period in periods],
        json.dumps(spec),
        lag=int(lag),
        quantile_level=None if quantile_level is None else float(quantile_level),
        min_probability=float(min_probability),
        misclassification=None
        if misclassification is None
        else tuple(int(count) for count in misclassification),
        seed=seed,
        threads=threads,
        options=options,
    )
    result = prepared.estimate(names, columns, seed=seed, threads=threads)
    section = result.temporal_design
    if section is None:
        raise CausalValueError(
            "the temporal design returned no design section", reason_code="invalid_argument"
        )
    return TemporalDesignResult(
        estimator=result.estimator_id,
        design=section.design,
        functional=section.functional,
        coordinates=tuple(section.coordinates),
        estimate=tuple(section.estimate),
        standard_error=tuple(section.standard_error),
        lower=tuple(section.lower),
        upper=tuple(section.upper),
        band_coordinates=tuple(section.band_coordinates),
        band_critical_value=section.band_critical_value,
        band_lower=_optional(section.band_lower),
        band_upper=_optional(section.band_upper),
        holm_adjusted_p=_optional(section.holm_adjusted_p),
        smoothing_bias_bound=_optional(section.smoothing_bias_bound),
        pairs=section.pairs,
        clusters=section.clusters,
        unpaired=section.unpaired,
        replicates=section.replicates,
        draws=section.draws,
        uncertainty=section.uncertainty,
        calibration=section.calibration,
        licensed=section.licensed,
    )
