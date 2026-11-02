"""Pooled responses of a multiply imputed joint panel.

Rust imputes, fits every completed panel and pools the response vectors; these wrappers
only decode the native record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._panel_execution import run_panel
from .execution import Execution


@dataclass(frozen=True, slots=True)
class PanelImputationResult:
    """The pooled response of every declared program, with how it was pooled.

    ``point`` concatenates the programs' mean responses in ``programs`` order, each over its
    ``horizons`` periods (:meth:`responses` splits it). ``method`` is ``"rubin"`` (Rubin's
    rules over frequentist fits, whose ``total_covariance`` is within + (1 + 1/m) between) or
    ``"stacked"`` (the equal-weight mixture of Bayesian posteriors). ``coordinates`` carries
    each coordinate's ``point``, ``within``, ``between`` and ``total`` variance, and Rubin's
    ``fraction_missing_information`` and ``df`` (``None`` when infinite). ``fits`` holds the
    point response of every imputation, ``imputed_cells`` the cells imputed per imputation
    model variable, and ``assumptions`` the imputation model declarations, including the
    identity ``digest``. Pooling is conditional on the declared missing-at-random model and
    grants no causal license.
    """

    programs: tuple[str, ...]
    horizons: tuple[int, ...]
    method: str
    imputations: int
    point: tuple[float, ...]
    total_covariance: tuple[tuple[float, ...], ...]
    coordinates: tuple[Mapping[str, float | None], ...]
    fits: tuple[tuple[float, ...], ...]
    imputed_cells: Mapping[str, int]
    assumptions: Mapping[str, Any]

    @classmethod
    def _from_json(cls, text: str) -> PanelImputationResult:
        raw = json.loads(text)
        dimension = raw["dimension"]
        flat = raw["total_covariance"]
        return cls(
            programs=tuple(raw["programs"]),
            horizons=tuple(raw["horizons"]),
            method=raw["method"],
            imputations=raw["imputations"],
            point=tuple(raw["point"]),
            total_covariance=tuple(
                tuple(flat[row * dimension : (row + 1) * dimension]) for row in range(dimension)
            ),
            coordinates=tuple(raw["coordinates"]),
            fits=tuple(tuple(fit) for fit in raw["fits"]),
            imputed_cells=dict(raw["imputed_cells"]),
            assumptions=raw["assumptions"],
        )

    def responses(self) -> dict[str, tuple[float, ...]]:
        """Each program's pooled mean response over its periods, keyed by program ID."""
        out: dict[str, tuple[float, ...]] = {}
        start = 0
        for program, horizon in zip(self.programs, self.horizons, strict=True):
            out[program] = self.point[start : start + horizon]
            start += horizon
        return out


@dataclass(frozen=True, slots=True)
class PreparedPanelImputation:
    """A joint panel whose missing outcomes and confounders are multiply imputed."""

    _handle: Any

    def estimate(self, *, execution: Execution | None = None) -> PanelImputationResult:
        """Impute, fit every completed panel and pool, under the caller's ``execution``.

        Imputation table ``j`` opens its own keyed stream from the specification's seed, so
        the result does not depend on threads. Every completed table is admitted against the
        execution's memory limit.
        """
        return PanelImputationResult._from_json(run_panel(execution, self._handle.estimate))
