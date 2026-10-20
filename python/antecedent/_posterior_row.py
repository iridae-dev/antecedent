"""The retained posterior of an unlicensed Bayesian randomized-design row.

``result.posterior_row`` is what the Rust result carries for ``bayesian.piecewise_survival``
(the treated-minus-control survival and restricted-mean contrast of a randomized survival
query) and ``bayesian.continuous_dose_policy`` (the fixed group-dose policy value, its reference
and their paired difference): the posterior mean and equal-tailed 95% credible interval of every
coordinate, the covariance of the joint draws, the exact Gaussian moments when the residual
variance is known, and the named scalar and vector summaries of the row.

The posterior is that of a declared likelihood and prior, the study's
``Bayesian(design_model=...)``; the prior supplies no identifying premise. The row is never
licensed: ``licensed`` is always ``False`` and ``calibration`` is ``"unmeasured"``, so no
coverage claim is made for any interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PosteriorRowEstimate"]


@dataclass(frozen=True, slots=True)
class PosteriorRowEstimate:
    """Posterior of one unlicensed Bayesian randomized-design row."""

    estimator: str
    target: str
    uncertainty: str
    calibration: str
    licensed: bool
    coordinates: tuple[str, ...]
    mean: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    draws: int
    #: Covariance across the coordinates of the retained joint draws, row-major.
    covariance: tuple[float, ...] | None = None
    #: Exact Gaussian posterior mean and covariance (known residual variance only).
    exact_mean: tuple[float, ...] | None = None
    exact_covariance: tuple[float, ...] | None = None
    extras: Mapping[str, float] = field(default_factory=dict)
    vectors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    @classmethod
    def _from_raw(cls, section: Any) -> PosteriorRowEstimate:
        def seq(values: Any) -> tuple[float, ...] | None:
            return None if values is None else tuple(float(v) for v in values)

        return cls(
            estimator=section.estimator,
            target=section.target,
            uncertainty=section.uncertainty,
            calibration=section.calibration,
            licensed=bool(section.licensed),
            coordinates=tuple(section.coordinates),
            mean=tuple(float(v) for v in section.mean),
            lower=tuple(float(v) for v in section.lower),
            upper=tuple(float(v) for v in section.upper),
            draws=int(section.draws),
            covariance=seq(section.covariance),
            exact_mean=seq(section.exact_mean),
            exact_covariance=seq(section.exact_covariance),
            extras={key: float(value) for key, value in section.extras.items()},
            vectors={key: tuple(float(v) for v in value) for key, value in section.vectors.items()},
        )

    @classmethod
    def _from_mapping(cls, wire: Mapping[str, Any]) -> PosteriorRowEstimate:
        """The posterior of a JSON report (a standalone artifact's result body)."""

        def seq(values: Any) -> tuple[float, ...] | None:
            return None if values is None else tuple(float(v) for v in values)

        return cls(
            estimator=str(wire["estimator"]),
            target=str(wire["target"]),
            uncertainty=str(wire["uncertainty"]),
            calibration=str(wire["calibration"]),
            licensed=bool(wire["licensed"]),
            coordinates=tuple(str(name) for name in wire["coordinates"]),
            mean=tuple(float(v) for v in wire["mean"]),
            lower=tuple(float(v) for v in wire["lower"]),
            upper=tuple(float(v) for v in wire["upper"]),
            draws=int(wire["draws"]),
            covariance=seq(wire.get("covariance")),
            exact_mean=seq(wire.get("exact_mean")),
            exact_covariance=seq(wire.get("exact_covariance")),
            extras={key: float(value) for key, value in wire.get("extras", {}).items()},
            vectors={
                key: tuple(float(v) for v in value)
                for key, value in wire.get("vectors", {}).items()
            },
        )

    def covariance_matrix(self) -> np.ndarray | None:
        """The joint-draw covariance as a square array, or ``None`` when none is retained."""
        return self._square(self.covariance)

    def exact_covariance_matrix(self) -> np.ndarray | None:
        """The exact Gaussian posterior covariance, or ``None`` under an unknown variance."""
        return self._square(self.exact_covariance)

    def credible_interval(self, coordinate: str) -> tuple[float, float]:
        """The equal-tailed 95% credible interval of ``coordinate``."""
        index = self.coordinates.index(coordinate)
        return (self.lower[index], self.upper[index])

    def posterior_mean(self, coordinate: str) -> float:
        """The posterior mean of ``coordinate``."""
        return self.mean[self.coordinates.index(coordinate)]

    def _square(self, values: tuple[float, ...] | None) -> np.ndarray | None:
        if values is None:
            return None
        width = len(self.coordinates)
        return np.asarray(values, dtype=float).reshape(width, width)
