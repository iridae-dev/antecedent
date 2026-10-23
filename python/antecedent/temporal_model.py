"""Standalone queries on the fitted dynamics of a ``TemporalDag``.

* :func:`detect_mechanism_change` tests whether a mechanism changed between two row ranges of one
  series. The null law is a circular-block permutation of the residuals (the series' serial
  dependence stays inside a block), for the likelihood-ratio, mean-difference, kernel and
  classifier statistics. With ``posterior_draws`` the answer is instead the posterior probability
  of a change under the ``change_prior`` the caller declares (never assumed): the residuals are
  modelled as Gaussian with conjugate normal-inverse-gamma priors on the mean and variance, the
  likelihood tempered for the residuals' serial dependence.
* :func:`rollout_dynamics` rolls the fitted dynamics forward from the last rows of the series
  under scheduled interventions, and returns the per-step mean of every variable; with
  ``posterior_draws`` (block-weight posterior) or ``bootstrap_replicates`` (circular-block
  bootstrap) it also returns the uncertainty over refits of the dynamics: for the means, and, in
  the ``path.step[..]`` quantities, for one simulated path per refit, whose summaries are the
  predictive law of a path (refit uncertainty plus the fitted innovation noise).

Both are uncalibrated and unlicensed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _native

LaggedEdge = tuple[str, int, str, int]


@dataclass(frozen=True)
class MechanismChange:
    """The verdict on one target's mechanism."""

    variable: str
    changed: bool
    statistic: float
    p_value: float | None
    adjusted_p_value: float | None
    method: str
    n_permutations: int | None
    posterior_probability: float | None


@dataclass(frozen=True)
class RolloutUncertainty:
    """Posterior (``draws`` set) or bootstrap (``se`` set) of the per-step means."""

    names: tuple[str, ...]
    draws: int | None = None
    mean: tuple[float, ...] = ()
    sd: tuple[float, ...] = ()
    ci_low: tuple[float, ...] = ()
    ci_high: tuple[float, ...] = ()
    se: dict[str, float | None] | None = None


@dataclass(frozen=True)
class Rollout:
    """Per-step means of each variable of a rollout, with their uncertainty when asked for."""

    variables: tuple[str, ...]
    steps: tuple[tuple[float, ...], ...]
    uncertainty: RolloutUncertainty | None

    def mean(self, step: int, variable: str) -> float:
        """Mean of ``variable`` at ``step``."""
        return self.steps[step][self.variables.index(variable)]


def detect_mechanism_change(
    names: Sequence[str],
    columns: Sequence[Any],
    lagged_edges: Sequence[LaggedEdge],
    targets: Sequence[str],
    baseline: tuple[int, int],
    comparison: tuple[int, int],
    *,
    test: str = "likelihood_ratio",
    significance: float = 0.05,
    posterior_draws: int = 0,
    change_prior: float | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> tuple[MechanismChange, ...]:
    """Test the mechanisms of ``targets`` for a change between two row ranges of one series.

    ``lagged_edges`` are ``(source, source_lag, target, target_lag)``. ``test`` is one of
    ``likelihood_ratio``, ``mean_difference``, ``kernel_two_sample``, ``classifier_two_sample`` or ``change_point``
    (``change_point`` is the known-split mean-and-variance likelihood ratio, also with a block null). Under ``posterior_draws`` a ``change_prior`` in (0, 1)
    is required.
    """
    params: dict[str, Any] = {
        "targets": list(targets),
        "baseline_start": baseline[0],
        "baseline_end": baseline[1],
        "comparison_start": comparison[0],
        "comparison_end": comparison[1],
        "significance": significance,
        "test": test,
    }
    if change_prior is not None:
        params["change_prior"] = change_prior
    raw = json.loads(
        _native.temporal_model_query(
            list(names), list(columns), [(str(a), int(b), str(c), int(d)) for a, b, c, d in lagged_edges],
            "mechanism_change", json.dumps(params),
            posterior_draws=posterior_draws, seed=seed, threads=threads,
        )
    )  # fmt: skip
    return tuple(MechanismChange(**entry) for entry in raw["detections"])


def rollout_dynamics(
    names: Sequence[str],
    columns: Sequence[Any],
    lagged_edges: Sequence[LaggedEdge],
    *,
    n_steps: int,
    n_rows: int = 200,
    interventions: Sequence[dict[str, Any]] = (),
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
    seed: int = 0,
    threads: int | None = None,
) -> Rollout:
    """Roll the fitted dynamics forward ``n_steps`` from the last rows of the series.

    Each intervention is ``{"variable": name, "value": v, "from": first_step, "until": last_step}``
    and holds the variable at ``v`` over those steps.
    """
    if posterior_draws and bootstrap_replicates:
        raise ValueError("ask for posterior_draws or bootstrap_replicates, not both")
    params = {"n_steps": n_steps, "n_rows": n_rows, "interventions": list(interventions), "seed": seed}
    raw = json.loads(
        _native.temporal_model_query(
            list(names), list(columns), [(str(a), int(b), str(c), int(d)) for a, b, c, d in lagged_edges],
            "rollout", json.dumps(params),
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        )
    )  # fmt: skip
    uncertainty = None
    if raw.get("posterior"):
        post = raw["posterior"]
        uncertainty = RolloutUncertainty(
            names=tuple(post["names"]), draws=post["draws"], mean=tuple(post["mean"]),
            sd=tuple(post["sd"]), ci_low=tuple(post["ci_low"]), ci_high=tuple(post["ci_high"]),
        )  # fmt: skip
    elif raw.get("bootstrap"):
        se = raw["bootstrap"]["se"]
        uncertainty = RolloutUncertainty(names=tuple(se), se=se)
    return Rollout(
        variables=tuple(raw["variables"]),
        steps=tuple(tuple(step) for step in raw["steps"]),
        uncertainty=uncertainty,
    )


__all__ = ["MechanismChange", "Rollout", "RolloutUncertainty", "detect_mechanism_change", "rollout_dynamics"]
