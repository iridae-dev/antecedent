"""Observational contrasts between levels of a multi-level treatment.

With three or more treatment levels there is no single average effect, so the population the
contrast speaks for is declared with ``target`` and reported on the result; it has no default::

    from antecedent import multi_level_treatment

    # Level 2 against level 0 over ALL rows: every level is standardized over the whole
    # population with a cross-fitted multinomial-logit generalized propensity.
    full = multi_level_treatment.multi_level_effect(
        outcome, dose, active=2.0, control=0.0,
        target="full_population", levels=[0.0, 1.0, 2.0], adjust={"age": age},
    )
    full.effect, full.se_analytic, full.level_means

    # Level 2 against level 0 on the rows AT level 0 or 2 only: rows at other levels are
    # excluded, counted, and named in the target population.
    pair = multi_level_treatment.multi_level_effect(
        outcome, dose, active=2.0, control=0.0,
        target="pairwise_subpopulation", adjust={"age": age},
    )
    pair.target_population, pair.excluded_rows

The two targets differ whenever the effect varies with the adjustment columns, because they
average it over different populations.

What the answer is **not**:

* No graph is checked: the caller owns that ``adjust`` is a valid adjustment set.
* The standard error is the plug-in influence-function variance of cross-fitted scores. Its
  calibration is **unmeasured** (``calibration == "unmeasured"``): no interval is produced and
  no coverage claim is made.
* With ``target="full_population"`` every row must be at a declared level; a row at an
  undeclared level refuses ``treatment_level_undeclared`` instead of being dropped.

The generalized propensity, the outcome regressions, the cross-fitting, the pairwise
restriction and every refusal are computed in Rust; this module builds the declaration and
presents the result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._native import evaluate_multi_level_treatment as _evaluate
from .errors import CausalTypeError, CausalValueError

__all__ = ["LevelMean", "LevelRows", "MultiLevelEffect", "multi_level_effect"]

MultiLevelTargetName = Literal["pairwise_subpopulation", "full_population"]
_TARGETS = ("pairwise_subpopulation", "full_population")


@dataclass(frozen=True, slots=True)
class LevelRows:
    """Complete-case rows observed at one treatment level."""

    level: float
    rows: int


@dataclass(frozen=True, slots=True)
class LevelMean:
    """The standardized mean outcome ``E[Y(level)]`` over the full population."""

    level: float
    mean: float
    se: float
    #: Smallest and largest out-of-fold generalized propensity of the level.
    propensity_min: float
    propensity_max: float


@dataclass(frozen=True, slots=True)
class MultiLevelEffect:
    """An active-minus-control contrast with the population it speaks for."""

    #: ``"pairwise_subpopulation"`` or ``"full_population_generalized_propensity"``.
    target: str
    #: ``"rows at the two declared levels"`` or ``"all complete-case rows"``.
    target_population: str
    active: float
    control: float
    effect: float
    #: Influence-function standard error of the contrast (calibration unmeasured).
    se_analytic: float
    n_obs: int
    rows_by_level: tuple[LevelRows, ...]
    #: Complete-case rows at no declared level that the target excluded.
    excluded_rows: int
    #: Standardized mean per declared level (full-population target only).
    level_means: tuple[LevelMean, ...]
    calibration: str

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable view."""
        return {
            "target": self.target,
            "target_population": self.target_population,
            "active": self.active,
            "control": self.control,
            "effect": self.effect,
            "se_analytic": self.se_analytic,
            "n_obs": self.n_obs,
            "rows_by_level": [{"level": r.level, "rows": r.rows} for r in self.rows_by_level],
            "excluded_rows": self.excluded_rows,
            "level_means": [
                {
                    "level": m.level,
                    "mean": m.mean,
                    "se": m.se,
                    "propensity_min": m.propensity_min,
                    "propensity_max": m.propensity_max,
                }
                for m in self.level_means
            ],
            "calibration": self.calibration,
        }


def _column(values: object, what: str) -> list[float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    return [float(v) for v in array]


def multi_level_effect(
    outcome: Sequence[float],
    treatment: Sequence[float],
    *,
    active: float,
    control: float,
    target: MultiLevelTargetName,
    levels: Sequence[float] | None = None,
    adjust: Mapping[str, Sequence[float]] | None = None,
    folds: int | None = None,
    seed: int = 0,
) -> MultiLevelEffect:
    """Estimate ``E[Y(active)] - E[Y(control)]`` for a multi-level observational treatment.

    ``treatment`` holds the observed level of each row; ``adjust`` maps column names to the
    covariates the caller declares sufficient for adjustment. ``target`` chooses the population
    (see the module docstring) and has no default. ``levels`` is the declared level set,
    required for ``"full_population"`` and containing ``active`` and ``control``. ``folds`` is
    the cross-fit fold count of the full-population fit (default 5) and ``seed`` seeds the fold
    plan. A ``NaN`` is a missing cell: its row is dropped from the complete cases.

    Raises:
        CausalValueError: A malformed argument.
        CausalUnsupportedError: A refusal with a registered ``reason_code`` such as
            ``treatment_level_undeclared`` or ``arm_not_populated``.
    """
    if target not in _TARGETS:
        raise CausalValueError(f"target must be one of {_TARGETS}, got {target!r}")
    if target == "full_population" and levels is None:
        raise CausalValueError("target='full_population' requires the declared levels")
    declaration: dict[str, Any] = {
        "target": target,
        "active": float(active),
        "control": float(control),
        "seed": int(seed),
    }
    if levels is not None:
        declaration["levels"] = _column(levels, "levels")
    if folds is not None:
        declaration["folds"] = int(folds)
    adjustment = dict(adjust or {})
    report = json.loads(
        _evaluate(
            _column(outcome, "outcome"),
            _column(treatment, "treatment"),
            list(adjustment),
            [_column(values, f"adjust[{name!r}]") for name, values in adjustment.items()],
            json.dumps(declaration),
        )
    )
    return MultiLevelEffect(
        target=report["target"],
        target_population=report["target_population"],
        active=report["active"],
        control=report["control"],
        effect=report["effect"],
        se_analytic=report["se_analytic"],
        n_obs=report["n_obs"],
        rows_by_level=tuple(LevelRows(r["level"], r["rows"]) for r in report["rows_by_level"]),
        excluded_rows=report["excluded_rows"],
        level_means=tuple(
            LevelMean(m["level"], m["mean"], m["se"], m["propensity_min"], m["propensity_max"])
            for m in report["level_means"]
        ),
        calibration=report["calibration"],
    )
