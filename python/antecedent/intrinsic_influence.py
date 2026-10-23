"""Intrinsic causal influence: how much of a target's variance each noise term explains.

See :func:`intrinsic_influence`. Every interval on the result names the uncertainty component
it covers (Shapley permutation sampling, nested simulation, or mechanism fitting), and a
component that was not measured says so rather than reading zero. With ``fit_bootstrap`` each
source also carries ``whole_procedure``: the spread of the full attribution across row-bootstrap
refits with the fixed permutation variance added. It is uncalibrated and not a licensed
interval.

With ``posterior_draws`` the same attribution is run under Bayesian inference: every draw
refits the mechanisms under one Dirichlet(1, ..., 1) row-weight vector (the mechanism families
held at the ones selected on ``data``) and repeats the whole attribution, so ``posterior``
carries the posterior mean, standard deviation and 95% credible interval of each source's
share given the empirical row law. It is the counterpart of ``fit_bootstrap`` and, like it,
uncalibrated and not a licensed interval.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._data import as_columns
from ._native import attribute_intrinsic_influence as _attribute_intrinsic_influence

Component = Literal["permutation_sampling", "simulation", "mechanism_fit", "whole_procedure"]


@dataclass(frozen=True)
class ComponentUncertainty:
    """One uncertainty component of one contribution.

    ``status`` is ``"quantified"`` (``stderr`` and a Student-t interval ``ci_low`` to
    ``ci_high`` at ``level`` with ``df`` degrees of freedom, for this component only),
    ``"not_applicable"`` (the component does not arise, for example no permutation is sampled
    when every coalition is enumerated, and its ``stderr`` is exactly ``0.0``) or
    ``"unquantified"`` (it arises but was not measured; this is not zero uncertainty).
    ``reason`` explains the last two.
    """

    component: Component
    status: Literal["quantified", "not_applicable", "unquantified"]
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    level: float | None = None
    df: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SourceInfluence:
    """Variance share of one noise term with its per-component uncertainty."""

    source: str
    contribution: float
    uncertainty: tuple[ComponentUncertainty, ...]
    whole_procedure: ComponentUncertainty | None = None

    def component(self, name: Component) -> ComponentUncertainty:
        """The uncertainty entry for ``name``."""
        for entry in self.uncertainty:
            if entry.component == name:
                return entry
        raise KeyError(name)


@dataclass(frozen=True)
class SourcePosterior:
    """Posterior of one source's variance share over the mechanisms' refits."""

    source: str
    mean: float
    sd: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class MechanismPosterior:
    """Posterior over the mechanisms of an intrinsic-influence attribution.

    ``level`` is the nominal level of each source's credible interval (``ci_low`` to
    ``ci_high``); ``draws`` is the number of Dirichlet-weighted refits.
    """

    draws: int
    level: float
    sources: tuple[SourcePosterior, ...]

    def source(self, name: str) -> SourcePosterior:
        """The posterior of ``name``."""
        for entry in self.sources:
            if entry.source == name:
                return entry
        raise KeyError(name)


@dataclass(frozen=True)
class IntrinsicInfluence:
    """Result of :func:`intrinsic_influence`.

    ``allocation_residual`` is ``total_variance`` minus the sum of the contributions. The
    variance game has ``v(empty) = 0`` and ``v(all) = Var(target)``, so the true residual is
    zero and this is the efficiency gap left by estimation error; ``allocation_residual_stderr``
    is its simulation standard error.
    """

    target: str
    sources: tuple[SourceInfluence, ...]
    total_variance: float
    total_variance_stderr: float
    allocation_residual: float
    allocation_residual_stderr: float
    level: float
    method: dict[str, Any]
    budget: dict[str, int]
    posterior: MechanismPosterior | None = None

    @property
    def provenance_id(self) -> str:
        """The provenance record (``provenance/<id>.toml``) the attribution is defined in."""
        return "gcm.attribution.intrinsic_influence"


def intrinsic_influence(
    data: Any,
    edges: Sequence[tuple[str, str]],
    target: str,
    *,
    independent_noise: bool = False,
    bidirected_edges: Sequence[tuple[str, str]] | None = None,
    allocation: Literal["auto", "exact", "permutation"] = "auto",
    n_permutations: int = 200,
    outer_draws: int = 200,
    inner_draws: int = 40,
    replicates: int = 4,
    fit_bootstrap: int = 0,
    posterior_draws: int = 0,
    level: float = 0.95,
    max_sources: int = 64,
    seed: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> IntrinsicInfluence:
    """Attribute the variance of ``target`` to the noise terms of its ancestors and itself.

    The mechanisms are fitted to ``data`` on the DAG ``edges``; the game
    ``v(S) = Var(E[target | noise of S])`` is then estimated by nested Monte Carlo through batch
    noise replay (Janzing et al. 2024, "Quantifying intrinsic causal contributions via structure
    preserving interventions", AISTATS), and split among the sources by Shapley values.

    The model's noise terms must be independent, and the caller says so: pass
    ``independent_noise=True`` or the declared structure's ``bidirected_edges`` (any
    bidirected edge, i.e. a latent common cause, is refused). Without either the call is
    refused.

    ``allocation="auto"`` enumerates every coalition up to ten sources and samples
    ``n_permutations`` permutations above that. ``outer_draws``, ``inner_draws`` and
    ``replicates`` size the nested simulation; ``fit_bootstrap`` (0 for none, otherwise at
    least 10) refits the mechanisms on that many row-bootstrap resamples of ``data`` to
    quantify mechanism-fitting uncertainty, which is otherwise reported ``unquantified``.
    ``posterior_draws`` (0 for none, otherwise at least 10) instead runs that many
    Dirichlet-weighted refits, the Bayesian counterpart, and returns their ``posterior``.
    """
    names, columns = as_columns(data)
    raw = json.loads(
        _attribute_intrinsic_influence(
            names,
            columns,
            [(str(a), str(b)) for a, b in edges],
            target,
            independent_noise=independent_noise,
            bidirected_edges=(
                None
                if bidirected_edges is None
                else [(str(a), str(b)) for a, b in bidirected_edges]
            ),
            allocation=allocation,
            n_permutations=n_permutations,
            outer_draws=outer_draws,
            inner_draws=inner_draws,
            replicates=replicates,
            fit_bootstrap=fit_bootstrap,
            posterior_draws=posterior_draws,
            level=level,
            max_sources=max_sources,
            seed=seed,
            threads=threads,
            cancel=cancel,
        )
    )
    return _influence(raw)


def _influence(raw: dict[str, Any]) -> IntrinsicInfluence:
    sources = tuple(
        SourceInfluence(
            source=s["source"],
            contribution=s["contribution"],
            uncertainty=tuple(ComponentUncertainty(**u) for u in s["uncertainty"]),
            whole_procedure=(
                ComponentUncertainty(component="whole_procedure", **s["whole_procedure"])
                if "whole_procedure" in s
                else None
            ),
        )
        for s in raw.pop("sources")
    )
    posterior_raw = raw.pop("posterior", None)
    posterior = (
        None
        if posterior_raw is None
        else MechanismPosterior(
            draws=posterior_raw["draws"],
            level=posterior_raw["level"],
            sources=tuple(SourcePosterior(**s) for s in posterior_raw["sources"]),
        )
    )
    return IntrinsicInfluence(sources=sources, posterior=posterior, **raw)


def fitted_intrinsic_influence(
    fitted: Any,
    target: str,
    *,
    independent_noise: bool = False,
    bidirected_edges: Sequence[tuple[str, str]] | None = None,
    allocation: Literal["auto", "exact", "permutation"] = "auto",
    n_permutations: int = 200,
    outer_draws: int = 200,
    inner_draws: int = 40,
    replicates: int = 4,
    fit_bootstrap: int = 0,
    posterior_draws: int = 0,
    level: float = 0.95,
    max_sources: int = 64,
    seed: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> IntrinsicInfluence:
    """:func:`intrinsic_influence` on an already fitted model, without refitting it.

    ``fitted`` is the handle ``fit_gcm`` returned: its structure, mechanisms and rows are the
    ones the attribution uses, so the result equals ``intrinsic_influence`` on the data and edges
    it was fitted from. Arguments and refusals are those of :func:`intrinsic_influence`.
    """
    raw = json.loads(
        fitted.attribute_intrinsic_influence(
            target,
            independent_noise=independent_noise,
            bidirected_edges=(
                None
                if bidirected_edges is None
                else [(str(a), str(b)) for a, b in bidirected_edges]
            ),
            allocation=allocation,
            n_permutations=n_permutations,
            outer_draws=outer_draws,
            inner_draws=inner_draws,
            replicates=replicates,
            fit_bootstrap=fit_bootstrap,
            posterior_draws=posterior_draws,
            level=level,
            max_sources=max_sources,
            seed=seed,
            threads=threads,
            cancel=cancel,
        )
    )
    return _influence(raw)


__all__ = [
    "ComponentUncertainty",
    "IntrinsicInfluence",
    "MechanismPosterior",
    "SourceInfluence",
    "SourcePosterior",
    "fitted_intrinsic_influence",
    "intrinsic_influence",
]
