"""Composition across the horizons of one temporal claim.

A quantity built from several horizons of a temporal response (a cumulative effect, a contrast
between horizons, a peak) has the uncertainty of the realizations that refit all its horizons
together. :func:`compose_horizons` takes those joint draws (one column per horizon, row ``r`` of
every column one realization, all under one ``alignment_id``) and composes them at a
horizon-indexed coordinate. Horizons fit one by one are independent marginals and are never
paired by draw index. The result is a labelled spread; no interval or posterior statement is made
without a coverage record at the coordinate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native


@dataclass(frozen=True)
class ComposedHorizons:
    """A composed quantity across horizons and the spread of its joint draws."""

    coordinate: str
    horizons: tuple[int, ...]
    route: str
    alignment_id: str | None
    label: str
    n: int
    mean: float
    variance: float
    standard_deviation: float
    mean_standard_error: float
    draws: tuple[float, ...]


@dataclass(frozen=True)
class JointHorizonDraws:
    """The draws of one dose coupled across the horizons of a temporal response.

    ``columns[k][r]`` is realization ``r`` of horizon ``horizons[k]``; every realization refit
    all horizons together, so the columns share one ``alignment_id`` and may be composed.
    ``center`` is the point value of each horizon's cell.
    """

    horizons: tuple[int, ...]
    center: tuple[float, ...]
    columns: tuple[tuple[float, ...], ...]
    alignment_id: str
    evidence_id: str

    def compose(
        self,
        *,
        operation: str = "weighted_sum",
        coefficients: Sequence[float] | None = None,
        map: Callable[[list[float]], float] | None = None,
        seed: int = 0,
    ) -> ComposedHorizons:
        """Compose these draws with :func:`compose_horizons`."""
        import numpy as np

        return compose_horizons(
            self.horizons,
            [np.asarray(column, dtype=np.float64) for column in self.columns],
            evidence_id=self.evidence_id,
            alignment_id=self.alignment_id,
            operation=operation,
            coefficients=coefficients,
            map=map,
            seed=seed,
        )


def compose_horizons(
    horizons: Sequence[int],
    columns: Sequence[Any],
    *,
    evidence_id: str,
    alignment_id: str,
    operation: str = "weighted_sum",
    coefficients: Sequence[float] | None = None,
    map: Callable[[list[float]], float] | None = None,
    seed: int = 0,
) -> ComposedHorizons:
    """Compose the joint draws of ``horizons`` by ``coefficients`` (linear) or ``map`` (nonlinear).

    ``operation`` names the propagation operation (``sum``, ``contrast``, ``weighted_sum``,
    ``product``, ``nonlinear_functional`` or ``chained_sampling``).
    """
    raw = json.loads(
        _native.compose_horizon_draws(
            [int(h) for h in horizons],
            list(columns),
            evidence_id,
            alignment_id,
            operation,
            coefficients=None if coefficients is None else [float(c) for c in coefficients],
            map=map,
            seed=seed,
        )
    )
    return ComposedHorizons(
        coordinate=raw["coordinate"],
        horizons=tuple(raw["horizons"]),
        route=raw["route"],
        alignment_id=raw["alignment_id"],
        label=raw["label"],
        n=raw["n"],
        mean=raw["mean"],
        variance=raw["variance"],
        standard_deviation=raw["standard_deviation"],
        mean_standard_error=raw["mean_standard_error"],
        draws=tuple(raw["draws"]),
    )


__all__ = ["ComposedHorizons", "JointHorizonDraws", "compose_horizons"]
