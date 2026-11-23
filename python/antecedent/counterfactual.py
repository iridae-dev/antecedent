"""GCM counterfactual helpers.

``FittedGcm`` is the fitted graphical causal model every counterfactual, interventional and
attribution method runs on. Each method that draws, resamples or runs in parallel takes
``execution=`` (an :class:`antecedent.Execution`) for its seed, threads and cancellation; no
method takes those controls loose.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from ._native import GcmIteResult
from .errors import CausalTypeError
from .execution import Execution, native_controls


class FittedGcm:
    """A graphical causal model fitted to data: structure, mechanisms and the rows they came from.

    Obtain one from :func:`fit_gcm`. Methods that sample or attribute take ``execution=``;
    ``execution=None`` means the default :class:`antecedent.Execution`.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _native.FittedGcm) -> None:
        if not isinstance(native, _native.FittedGcm):
            raise CausalTypeError("a FittedGcm is created by fit_gcm, not constructed directly")
        self._native = native

    def __repr__(self) -> str:
        return f"FittedGcm(names={list(self._native.names)!r})"

    @property
    def names(self) -> list[str]:
        return self._native.names

    @property
    def n_assignments(self) -> int:
        return self._native.n_assignments

    @property
    def last_route(self) -> Any:
        return self._native.last_route

    def structure_provenance(self) -> dict[str, str] | None:
        return self._native.structure_provenance()

    def mechanism_kinds(self) -> list[tuple[str, str]]:
        return self._native.mechanism_kinds()

    def mechanism_selection(self) -> list[dict[str, Any]]:
        return self._native.mechanism_selection()

    def arrow_strengths(self) -> list[Any]:
        return self._native.arrow_strengths()

    def anomaly_attribution(self, outcomes: list[str], *, max_units: int = 0) -> list[Any]:
        return self._native.anomaly_attribution(outcomes, max_units=max_units)

    def sample_do(
        self,
        interventions: dict[str, float],
        n: int,
        *,
        shifts: dict[str, float] | None = None,
        programs: list[Any] | None = None,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.sample_do,
            interventions,
            n,
            shifts=shifts,
            programs=programs,
            **ctl,
        )

    def sample_noise(
        self,
        interventions: dict[str, float],
        n: int,
        *,
        shifts: dict[str, float] | None = None,
        programs: list[Any] | None = None,
        execution: Execution | None = None,
    ) -> tuple[Any, NDArray[np.float64]]:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.sample_noise,
            interventions,
            n,
            shifts=shifts,
            programs=programs,
            **ctl,
        )

    def replay_noise(
        self,
        noise: NDArray[np.float64],
        interventions: dict[str, float] | None = None,
        *,
        shifts: dict[str, float] | None = None,
        policy_salt: int = 0,
        programs: list[Any] | None = None,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.replay_noise,
            noise,
            interventions,
            shifts=shifts,
            policy_salt=policy_salt,
            programs=programs,
            **ctl,
        )

    def attribute_intrinsic_influence(
        self,
        target: str,
        *,
        independent_noise: bool = False,
        bidirected_edges: list[tuple[str, str]] | None = None,
        allocation: str = "auto",
        n_permutations: int = 200,
        outer_draws: int = 200,
        inner_draws: int = 40,
        replicates: int = 4,
        fit_bootstrap: int = 0,
        posterior_draws: int = 0,
        level: float = 0.95,
        max_sources: int = 64,
        execution: Execution | None = None,
    ) -> str:
        ctl = native_controls(execution, seed="seed", threads="threads", cancel="cancel")
        return _run(
            execution,
            self._native.attribute_intrinsic_influence,
            target,
            independent_noise=independent_noise,
            bidirected_edges=bidirected_edges,
            allocation=allocation,
            n_permutations=n_permutations,
            outer_draws=outer_draws,
            inner_draws=inner_draws,
            replicates=replicates,
            fit_bootstrap=fit_bootstrap,
            posterior_draws=posterior_draws,
            level=level,
            max_sources=max_sources,
            **ctl,
        )

    def population_do_contrast(
        self,
        parent: str,
        child: str,
        delta: float,
        *,
        max_units: int | None = None,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.population_do_contrast,
            parent,
            child,
            delta,
            max_units=max_units,
            **ctl,
        )

    def counterfactual_contrasts(
        self,
        outcomes: list[str],
        active: dict[str, float],
        control: dict[str, float],
        *,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution, self._native.counterfactual_contrasts, outcomes, active, control, **ctl
        )

    def simultaneous_counterfactual(
        self,
        outcome: str,
        outer: dict[str, float],
        inner: dict[str, float],
        *,
        execution: Execution | None = None,
    ) -> float:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution, self._native.simultaneous_counterfactual, outcome, outer, inner, **ctl
        )

    def interventional_mean(
        self,
        outcome: str,
        interventions: dict[str, float],
        n: int,
        *,
        execution: Execution | None = None,
    ) -> float:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(execution, self._native.interventional_mean, outcome, interventions, n, **ctl)

    def do_sample(
        self,
        method: str,
        outcome: str,
        interventions: dict[str, float],
        n: int = 1000,
        *,
        bandwidth: float | None = None,
        proposal_sd: float | None = None,
        burn_in: int | None = None,
        thin: int | None = None,
        root_match_tolerance: float | None = None,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.do_sample,
            method,
            outcome,
            interventions,
            n,
            bandwidth=bandwidth,
            proposal_sd=proposal_sd,
            burn_in=burn_in,
            thin=thin,
            root_match_tolerance=root_match_tolerance,
            **ctl,
        )

    def evaluate(
        self,
        *,
        alpha: float = 0.05,
        n_permutations: int = 20,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.evaluate,
            alpha=alpha,
            n_permutations=n_permutations,
            **ctl,
        )

    def counterfactual_ite(
        self,
        treatment: str,
        outcome: str,
        active: float,
        control: float,
        *,
        execution: Execution | None = None,
    ) -> GcmIteResult:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.counterfactual_ite,
            treatment,
            outcome,
            active,
            control,
            **ctl,
        )

    def attribute_path_specific(
        self,
        treatment: str,
        outcome: str,
        *,
        path_nodes: list[str] | None = None,
        max_paths: int = 64,
        max_len: int = 16,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_path_specific,
            treatment,
            outcome,
            path_nodes=path_nodes,
            max_paths=max_paths,
            max_len=max_len,
            **ctl,
        )

    def attribute_paths(
        self,
        sources: list[str],
        outcome: str,
        *,
        max_paths: int = 64,
        max_len: int = 16,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_paths,
            sources,
            outcome,
            max_paths=max_paths,
            max_len=max_len,
            **ctl,
        )

    def attribute_distribution_change(
        self,
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_distribution_change,
            outcome,
            baseline_start,
            baseline_end,
            comparison_start,
            comparison_end,
            n_samples=n_samples,
            **ctl,
        )

    def attribute_distribution_change_robust(
        self,
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_distribution_change_robust,
            outcome,
            baseline_start,
            baseline_end,
            comparison_start,
            comparison_end,
            n_samples=n_samples,
            **ctl,
        )

    def attribute_structure_change(
        self,
        comparison_edges: list[tuple[str, str]],
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_structure_change,
            comparison_edges,
            outcome,
            baseline_start,
            baseline_end,
            comparison_start,
            comparison_end,
            n_samples=n_samples,
            **ctl,
        )

    def attribute_unit_change(
        self,
        outcome: str,
        *,
        max_units: int = 0,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_unit_change,
            outcome,
            max_units=max_units,
            **ctl,
        )

    def attribute_feature_relevance(
        self,
        outcome: str,
        *,
        delta: float = 1.0,
        n_samples: int = 200,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.attribute_feature_relevance,
            outcome,
            delta=delta,
            n_samples=n_samples,
            **ctl,
        )

    def mechanism_change_detection(
        self,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        execution: Execution | None = None,
    ) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(
            execution,
            self._native.mechanism_change_detection,
            baseline_start,
            baseline_end,
            comparison_start,
            comparison_end,
            **ctl,
        )

    def rank_root_causes(self, attribution: Any, *, execution: Execution | None = None) -> Any:
        ctl = native_controls(execution, seed="seed", threads="threads")
        return _run(execution, self._native.rank_root_causes, attribution, **ctl)


def _run(execution: Execution | None, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run one native operation under ``execution``'s ambient policy (default when ``None``)."""
    execution = Execution() if execution is None else execution
    return execution._call(operation, *args, **kwargs)


def native_gcm(fitted: Any) -> _native.FittedGcm:
    """The native handle of a fitted model, for the native entries that consume one."""
    if isinstance(fitted, FittedGcm):
        return fitted._native
    if isinstance(fitted, _native.FittedGcm):
        return fitted
    raise CausalTypeError("expected a FittedGcm from fit_gcm")


def fit_gcm(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    *,
    candidates: list[str] | None = None,
    constraints: dict[str, dict[str, Any]] | None = None,
    splits: str = "shuffled",
    purge: int = 0,
    execution: Execution | None = None,
) -> FittedGcm:
    """Fit a graphical causal model: one mechanism per node, selected on the data."""
    ctl = native_controls(execution, threads="threads")
    return FittedGcm(
        _run(
            execution,
            _native.fit_gcm,
            names,
            columns,
            edges,
            candidates=candidates,
            constraints=constraints,
            splits=splits,
            purge=purge,
            **ctl,
        )
    )


def _fit_gcm_accepted(
    names: list[str],
    columns: Sequence[Any],
    discovery: Any,
    *,
    candidates: list[str] | None = None,
    constraints: dict[str, dict[str, Any]] | None = None,
    splits: str = "shuffled",
    purge: int = 0,
    execution: Execution | None = None,
) -> tuple[FittedGcm, list[tuple[str, str]]]:
    """Fit a model on the structure a compiled discovery route accepted."""
    ctl = native_controls(execution, threads="threads")
    fitted, edges = _run(
        execution,
        _native.fit_gcm_accepted,
        names,
        columns,
        discovery,
        candidates=candidates,
        constraints=constraints,
        splits=splits,
        purge=purge,
        **ctl,
    )
    return FittedGcm(fitted), edges


def counterfactual_ite(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    active: float,
    control: float,
    *,
    execution: Execution | None = None,
) -> GcmIteResult:
    """Individual counterfactual effects of ``treatment`` on ``outcome`` on a model fitted here."""
    ctl = native_controls(execution, seed="seed", threads="threads")
    return _run(
        execution,
        _native.counterfactual_ite,
        names,
        columns,
        edges,
        treatment,
        outcome,
        active,
        control,
        **ctl,
    )


__all__ = ["FittedGcm", "GcmIteResult", "counterfactual_ite", "fit_gcm"]
