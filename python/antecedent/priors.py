"""External prior bank: catalog metadata and compatibility filtering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ._artifact_protocol import container_bytes
from ._native import (
    beta_binomial_posterior as _beta_binomial_posterior,
)
from ._native import (
    beta_from_mean_and_ess as _beta_from_mean_and_ess,
)
from ._native import (
    beta_from_moments as _beta_from_moments,
)
from ._native import (
    compose_external_priors as _compose_external_priors,
)
from ._native import (
    conflict_shrink_alpha as _shrink_alpha,
)
from ._native import (
    decode_prior_source_meta as _decode_meta,
)
from ._native import (
    encode_prior_source_meta as _encode_meta,
)
from ._native import (
    exceedance_posterior as _exceedance_posterior,
)
from ._native import (
    gamma_from_mean_and_ess as _gamma_from_mean_and_ess,
)
from ._native import (
    gamma_from_moments as _gamma_from_moments,
)
from ._native import (
    gamma_poisson_posterior as _gamma_poisson_posterior,
)
from ._native import (
    mapped_posterior_transfer as _mapped_posterior_transfer,
)
from ._native import (
    prior_catalog_filter as _filter,
)
from ._native import (
    prior_catalog_rank as _rank,
)
from ._native import (
    select_beta_power_prior_alpha as _select_alpha_beta,
)
from ._native import (
    select_gamma_power_prior_alpha as _select_alpha_gamma,
)
from ._native import (
    select_linear_power_prior_alpha as _select_alpha_linear,
)

if TYPE_CHECKING:
    from .joint_distribution import DistributionIdentity, ScientificQuantity

#: Caller convention for population / environment tags on ``PriorSourceMeta.tags``.
POPULATION_TAG_KEY = "population"


@dataclass(frozen=True)
class EstimandFingerprint:
    """Query kind + treatment/outcome names for catalog matching."""

    query_kind: str
    treatment: str
    outcome: str


@dataclass(frozen=True)
class DesignVariable:
    """One variable in a prior-source design summary."""

    name: str
    role: Literal["treatment", "outcome", "covariate", "other"]


@dataclass(frozen=True)
class PriorMapping:
    """Declared bridge from a banked source into a target design prior.

    Use the constructors for the common shapes; hydrate happens inside
    ``Bayesian(prior_from=..., mapping=...)``.
    """

    kind: Literal[
        "identical_coefficient_subspace",
        "effect_functional",
        "named_parameters",
    ]
    source_quantity: str | None = None
    pairs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def identical(cls) -> PriorMapping:
        return cls(kind="identical_coefficient_subspace")

    @classmethod
    def effect_functional(cls, source_quantity: str = "ate") -> PriorMapping:
        return cls(kind="effect_functional", source_quantity=source_quantity)

    @classmethod
    def named_parameters(cls, pairs: Sequence[tuple[str, str]]) -> PriorMapping:
        return cls(kind="named_parameters", pairs=tuple((a, b) for a, b in pairs))

    def to_dict(self) -> dict[str, Any]:
        m: dict[str, Any] = {"kind": self.kind}
        if self.source_quantity is not None:
            m["source_quantity"] = self.source_quantity
        if self.pairs:
            m["pairs"] = [list(p) for p in self.pairs]
        return m


@dataclass(frozen=True)
class ConflictPolicy:
    """Shrink external prior α from prior-PPC / KL conflict signals.

    Applied α is ``α · 1{p > p_min} · exp(−kl_scale · kl)``, never increased.
    Defaults: ``p_min=0.05``, ``kl_scale=1.0``.
    """

    p_min: float = 0.05
    kl_scale: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"p_min": self.p_min, "kl_scale": self.kl_scale}

    def shrink_alpha(
        self,
        alpha: float,
        *,
        p_value: float | None = None,
        kl: float | None = None,
    ) -> float:
        return float(
            _shrink_alpha(
                float(alpha),
                p_value=p_value,
                kl=kl,
                p_min=self.p_min,
                kl_scale=self.kl_scale,
            )
        )


@dataclass(frozen=True)
class TransportPolicy:
    """Explicit invariance claim for cross-population prior transfer.

    Required when source and target ``population`` tags differ. Never inferred.
    """

    kind: Literal[
        "invariant_conditional_outcome",
        "invariant_effect_modifiers",
        "invariant_propensity",
    ]

    @classmethod
    def invariant_conditional_outcome(cls) -> TransportPolicy:
        return cls(kind="invariant_conditional_outcome")

    @classmethod
    def invariant_effect_modifiers(cls) -> TransportPolicy:
        return cls(kind="invariant_effect_modifiers")

    @classmethod
    def invariant_propensity(cls) -> TransportPolicy:
        return cls(kind="invariant_propensity")

    def to_wire(self) -> str:
        return self.kind


@dataclass(frozen=True)
class ExternalPriorWeight:
    """Per-source power-prior α and optional mixture weight."""

    alpha: float = 1.0
    mixture_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "mixture_weight": self.mixture_weight}


@dataclass(frozen=True)
class ExternalPriorSourceSpec:
    """One hydrated Gaussian coefficient prior for composition.

    ``ess`` is an optional caller-declared **prior-strength** sample size for
    this source (e.g. the original study's N, or an effective N after a
    design discount). It is distinct from MCMC effective sample size and from
    Kish importance-weighting ESS (see ``ComposedPrior.kish_ess``) — see
    :mod:`antecedent.priors` module docs.
    """

    id: str
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    weight: ExternalPriorWeight = field(default_factory=ExternalPriorWeight)
    ess: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mean": list(self.mean),
            "variance": list(self.variance),
            "alpha": self.weight.alpha,
            "mixture_weight": self.weight.mixture_weight,
            "ess": self.ess,
        }


@dataclass(frozen=True)
class ComposedPrior:
    """Result of ``compose_external_priors``; usable as ``Bayesian(prior_from=...)``.

    Three distinct effective-sample-size conventions appear across this
    library; none are interchangeable:

    * ``effective_ess`` / ``composed_ess`` below are **prior-strength ESS** —
      the sample size implied by how much precision a source (or the
      composed prior) contributes, via ``alpha * ess``.
    * MCMC / autocorrelation ESS (on posterior diagnostics) measures how many
      effectively independent draws a chain produced.
    * ``kish_ess`` below is a Kish (1965) importance-weighting diagnostic —
      how concentrated a trust/weight vector is, ``(sum(w))**2 / sum(w**2)``.
    """

    mean: tuple[float, ...]
    variance: tuple[float, ...]
    source_ids: tuple[str, ...]
    alphas_requested: tuple[float, ...]
    alphas_applied: tuple[float, ...]
    mixture_weights: tuple[float | None, ...]
    sources: tuple[ExternalPriorSourceSpec, ...]
    conflict: ConflictPolicy | None = None
    conflict_p_values: tuple[float | None, ...] = ()
    conflict_kl_values: tuple[float | None, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    transport: TransportPolicy | None = None
    #: Per-source prior-strength ESS after alpha discount (``alpha_applied *
    #: ess``); ``None`` entries mean that source declared no ``ess``. Zero for
    #: sources dropped from composition regardless of a declared ``ess``.
    effective_ess: tuple[float | None, ...] = ()
    #: Composed prior-strength ESS. On the power path, ``sum(alpha_k * ess_k)``
    #: over contributing sources when *every* contributing source declared an
    #: ``ess`` (``None`` otherwise — a partial sum would misstate it). Always
    #: ``None`` on the mixture path: moment-matching folds in between-component
    #: spread, so the composed prior is weaker than a precision-sum implies and
    #: summing source ESS would overstate composed strength.
    composed_ess: float | None = None
    #: Original Gaussian coefficient baseline, retained for data-bound re-composition.
    baseline_mean: tuple[float, ...] | None = None
    baseline_variance: tuple[float, ...] | None = None
    #: Kish concentration-of-trust diagnostic over the weight vector actually
    #: used in composition (applied alphas on the power path, mixture weights
    #: on the mixture path; dropped sources zeroed). ``None`` only when there
    #: are no sources to diagnose.
    kish_ess: float | None = None

    def to_native_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "mean": list(self.mean),
            "variance": list(self.variance),
            "source_ids": list(self.source_ids),
            "alphas_requested": list(self.alphas_requested),
            "alphas_applied": list(self.alphas_applied),
            "mixture_weights": list(self.mixture_weights),
            "sources": [s.to_dict() for s in self.sources],
            "effective_ess": list(self.effective_ess),
            "composed_ess": self.composed_ess,
            "kish_ess": self.kish_ess,
        }
        if self.baseline_mean is not None:
            d["baseline_mean"] = list(self.baseline_mean)
        if self.baseline_variance is not None:
            d["baseline_variance"] = list(self.baseline_variance)
        if self.conflict is not None:
            d["conflict"] = self.conflict.to_dict()
        if self.conflict_p_values:
            d["conflict_p_values"] = list(self.conflict_p_values)
        if self.conflict_kl_values:
            d["conflict_kl_values"] = list(self.conflict_kl_values)
        if self.assumption_ids:
            d["assumption_ids"] = list(self.assumption_ids)
        if self.transport is not None:
            d["transport"] = self.transport.to_wire()
        return d


@dataclass(frozen=True)
class BetaHyperparameters:
    """``Beta(alpha, beta)`` conjugate hyperparameters.

    Produced by one of two constructors with distinct, honest contracts:

    * :func:`beta_from_moments` matches ``(mean, variance)`` exactly;
      :attr:`ess` is whatever those moments imply, a derived consequence
      rather than a request.
    * :func:`beta_from_mean_and_ess` matches ``mean`` and a caller-declared
      prior-strength ``ess`` exactly; there is no ``variance`` parameter,
      because ``mean`` and ``ess`` alone determine ``alpha`` and ``beta``.

    A Beta prior is the prior of the Beta-Binomial backend
    (:func:`beta_binomial_posterior`, :func:`exceedance_posterior`). It is
    not a coefficient prior, so it is not accepted by
    ``compose_external_priors`` or ``Bayesian(prior_from=...)``.

    ESS convention: ``ess = alpha + beta - 2``, chosen so the flat reference
    prior ``Beta(1, 1)`` maps to ``ess = 0`` — which is what makes
    ``beta_from_mean_and_ess(mean, 0.0)`` degrade to a
    ``Beta(1,1)``-equivalent-strength prior at the requested mean rather
    than to something vanishing or improper. Some references instead report
    ``alpha + beta`` (under which ``Beta(1,1)`` is ``ess = 2``); this module
    never uses that convention. See also the three distinct ESS conventions
    documented on :class:`ComposedPrior` — this is the same prior-strength
    notion, applied to a conjugate family instead of a Gaussian
    coefficient's precision.

    :attr:`ess` can be **negative** on a :func:`beta_from_moments` result:
    any ``(mean, variance)`` pair weaker than the flat reference (total
    concentration ``< 2``) yields ``alpha + beta - 2 < 0`` while ``alpha``
    and ``beta`` stay positive and proper. That is a truthful report of a
    prior weaker than ``Beta(1, 1)``, not an error.
    """

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1.0))

    @property
    def ess(self) -> float:
        return self.alpha + self.beta - 2.0


@dataclass(frozen=True)
class GammaHyperparameters:
    """``Gamma(shape, rate)`` conjugate hyperparameters.

    Produced by one of two constructors with distinct, honest contracts,
    mirroring :class:`BetaHyperparameters`:

    * :func:`gamma_from_moments` matches ``(mean, variance)`` exactly;
      :attr:`ess` is whatever those moments imply.
    * :func:`gamma_from_mean_and_ess` matches ``mean`` and a caller-declared
      prior-strength ``ess`` exactly; no ``variance`` parameter.

    Like :class:`BetaHyperparameters`, this is a count-model prior: it is the
    prior of the Gamma-Poisson backend (:func:`gamma_poisson_posterior`), not a
    coefficient prior accepted by ``compose_external_priors`` or
    ``Bayesian(prior_from=...)``.

    ESS convention: ``ess = shape - 1``, chosen so the reference exponential
    prior ``Gamma(shape=1, ...)`` maps to ``ess = 0``. Some references
    instead report ``shape`` directly (under which ``Gamma(1, ...)`` is
    ``ess = 1``); this module never uses that convention.

    :attr:`ess` can be **negative** on a :func:`gamma_from_moments` result
    (``shape < 1``, i.e. ``variance > mean ** 2``) while ``shape`` and
    ``rate`` stay positive and proper — a truthful report of a prior weaker
    than the reference exponential, not an error.
    """

    shape: float
    rate: float

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        return self.shape / (self.rate * self.rate)

    @property
    def ess(self) -> float:
        return self.shape - 1.0


def beta_from_moments(mean: float, variance: float) -> BetaHyperparameters:
    """Moment-match a Beta prior on a bounded proportion to ``(mean, variance)``.

    Matches both moments exactly. The result's ``ess`` (``alpha + beta -
    2``; see :class:`BetaHyperparameters`) is whatever those moments imply —
    a derived consequence, not a caller input. It can be negative (a proper
    prior weaker than the flat reference ``Beta(1, 1)``); that is a
    truthful report, not an error. Use :func:`beta_from_mean_and_ess` to
    request a specific prior strength instead of reading one back.

    Out-of-support input is rejected, never silently clamped:

    Raises
    ------
    ValueError
        ``mean`` is not finite and strictly inside ``(0, 1)``; or
        ``variance`` is not finite and > 0, or is `>= mean * (1 - mean)`
        (no Beta distribution has those moments — the comparison is exact,
        with no epsilon slack at the boundary).
    """
    alpha, beta = _beta_from_moments(float(mean), float(variance))
    return BetaHyperparameters(alpha=alpha, beta=beta)


def beta_from_mean_and_ess(mean: float, ess: float) -> BetaHyperparameters:
    """Build a Beta prior on a bounded proportion from ``mean`` and a
    caller-declared prior-strength ``ess``.

    Matches ``mean`` and ``ess`` (``alpha + beta - 2``; see
    :class:`BetaHyperparameters`) exactly. There is no ``variance``
    parameter: ``mean`` and ``ess`` alone determine ``alpha`` and ``beta``,
    so a ``variance`` argument would have nothing to do. ``ess=0`` returns a
    prior with the same strength as the flat reference ``Beta(1, 1)`` at
    ``mean`` — never a vanishing or improper one. Every ``(mean, ess >=
    0)`` request is satisfiable; unlike :func:`beta_from_moments`, there is
    no support gate to violate.

    Raises
    ------
    ValueError
        ``mean`` is not finite and strictly inside ``(0, 1)``; or ``ess`` is
        not finite and >= 0.
    """
    alpha, beta = _beta_from_mean_and_ess(float(mean), float(ess))
    return BetaHyperparameters(alpha=alpha, beta=beta)


def gamma_from_moments(mean: float, variance: float) -> GammaHyperparameters:
    """Moment-match a Gamma prior on a non-negative rate to ``(mean, variance)``.

    Matches both moments exactly. The result's ``ess`` (``shape - 1``; see
    :class:`GammaHyperparameters`) is whatever those moments imply — a
    derived consequence, not a caller input. It can be negative (a proper
    prior weaker than the reference exponential ``Gamma(shape=1, ...)``);
    that is a truthful report, not an error. Use
    :func:`gamma_from_mean_and_ess` to request a specific prior strength
    instead of reading one back.

    Out-of-support input is rejected, never silently clamped:

    Raises
    ------
    ValueError
        ``mean`` is not finite and > 0; or ``variance`` is not finite and >
        0.
    """
    shape, rate = _gamma_from_moments(float(mean), float(variance))
    return GammaHyperparameters(shape=shape, rate=rate)


def gamma_from_mean_and_ess(mean: float, ess: float) -> GammaHyperparameters:
    """Build a Gamma prior on a non-negative rate from ``mean`` and a
    caller-declared prior-strength ``ess``.

    Matches ``mean`` and ``ess`` (``shape - 1``; see
    :class:`GammaHyperparameters`) exactly. There is no ``variance``
    parameter: ``mean`` and ``ess`` alone determine ``shape`` and ``rate``
    (``rate = shape / mean``), so a ``variance`` argument would have
    nothing to do. ``ess=0`` returns ``Gamma(shape=1, ...)`` — the reference
    exponential prior — at ``mean``, never a vanishing or improper one.
    Every ``(mean, ess >= 0)`` request is satisfiable.

    Raises
    ------
    ValueError
        ``mean`` is not finite and > 0; or ``ess`` is not finite and >= 0.
    """
    shape, rate = _gamma_from_mean_and_ess(float(mean), float(ess))
    return GammaHyperparameters(shape=shape, rate=rate)


def _normalize_weights(
    sources: Sequence[ExternalPriorSourceSpec],
    weights: Sequence[ExternalPriorWeight | float] | None,
) -> list[ExternalPriorSourceSpec]:
    if weights is None:
        return list(sources)
    if len(weights) != len(sources):
        raise ValueError("weights length must match sources")
    out: list[ExternalPriorSourceSpec] = []
    for src, w in zip(sources, weights, strict=True):
        if isinstance(w, ExternalPriorWeight):
            wt = w
        else:
            wt = ExternalPriorWeight(alpha=src.weight.alpha, mixture_weight=float(w))
        out.append(
            ExternalPriorSourceSpec(
                id=src.id,
                mean=src.mean,
                variance=src.variance,
                weight=wt,
                ess=src.ess,
            )
        )
    return out


def populations_from_prior_sources(
    sources: Sequence[PriorSource],
) -> list[str | None]:
    """Read ``tags["population"]`` from each catalog source (``None`` if unset)."""
    return [s.meta.tags.get(POPULATION_TAG_KEY) for s in sources]


@dataclass(frozen=True)
class MappedPosteriorPrior:
    """A known-variance Gaussian prior transferred from an aligned parameter posterior.

    Coordinates follow the target's design-column order (``target_ids``). ``scale_variance`` is
    the conjugate scale ``V0`` per coefficient and ``correlation`` its dense correlation, so the
    absolute coefficient covariance is ``known_residual_variance * V0`` (see :attr:`covariance`).
    ``assumptions`` records the transfer (source and target snapshot and contract, the complete
    stable-id map and the variance) as ``(id, description)`` pairs.
    """

    target_ids: tuple[str, ...]
    mean: tuple[float, ...]
    scale_variance: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...] | None
    known_residual_variance: float
    assumptions: tuple[tuple[str, str], ...]

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        """Absolute coefficient covariance ``sigma^2 V0`` of the transferred posterior."""
        n = len(self.mean)
        sd = [(self.known_residual_variance * v) ** 0.5 for v in self.scale_variance]
        return tuple(
            tuple(
                sd[i] * sd[j] * (self.correlation[i][j] if self.correlation else float(i == j))
                for j in range(n)
            )
            for i in range(n)
        )


def mapped_posterior_transfer(
    source: bytes,
    *,
    expected_source: DistributionIdentity,
    source_to_target: Mapping[str, str],
    target_quantities: Sequence[ScientificQuantity],
    target_snapshot_id: str,
    target_contract_id: str,
    known_residual_variance: float,
    baseline_variance: float = 1.0,
) -> MappedPosteriorPrior:
    """Transfer an aligned parameter posterior into a known-variance Gaussian prior.

    ``source`` is the exported joint-distribution artifact of the source study and
    ``expected_source`` the identity the consumer retained independently of those bytes.
    ``source_to_target`` maps each source coefficient's stable variable id to the target
    coefficient's; the map must be a complete bijection onto ``target_quantities`` (the target's
    coefficients in design-column order) with identical units, population, regime, horizon,
    conditioning, transform and role. The source must carry complete, unweighted, joint
    parameter-posterior draws whose single residual-variance coordinate is constant and equal to
    ``known_residual_variance``. The source's full covariance is kept under the permutation.

    Raises:
        CausalValueError: an invalid argument.
        CausalError: a refused identity, meaning, map, residual variance or alignment.
    """
    wires = [quantity._wire() for quantity in target_quantities]
    out = _mapped_posterior_transfer(
        container_bytes(source),
        json.dumps(expected_source._wire()),
        [(str(left), str(right)) for left, right in source_to_target.items()],
        json.dumps(wires),
        target_snapshot_id,
        target_contract_id,
        float(baseline_variance),
        float(known_residual_variance),
    )
    correlation = out["correlation"]
    return MappedPosteriorPrior(
        target_ids=tuple(w["variable_id"] for w in wires),
        mean=tuple(out["mean"]),
        scale_variance=tuple(out["scale_variance"]),
        correlation=None if correlation is None else tuple(tuple(row) for row in correlation),
        known_residual_variance=float(out["known_residual_variance"]),
        assumptions=tuple((item["id"], item["description"]) for item in out["restrictions"]),
    )


def compose_external_priors(
    sources: Sequence[ExternalPriorSourceSpec],
    weights: Sequence[ExternalPriorWeight | float] | None = None,
    *,
    baseline: tuple[Sequence[float], Sequence[float]] | None = None,
    conflict: ConflictPolicy | None = None,
    conflict_signals: Sequence[Mapping[str, float | None]] | None = None,
    transport: TransportPolicy | None = None,
    target_population: str | None = None,
    source_populations: Sequence[str | None] | None = None,
    prior_sources: Sequence[PriorSource] | None = None,
    unit_effects: Sequence[float] | None = None,
    transport_weights: Sequence[float] | None = None,
    coef_index: int | None = None,
) -> ComposedPrior:
    """Compose external Gaussian priors with power-prior / mixture weights.

    Parameters
    ----------
    sources:
        Hydrated coefficient priors (same dimension as the target design).
    weights:
        Per-source ``ExternalPriorWeight`` or mixture floats (α kept from source).
    baseline:
        ``(mean, variance)`` for leftover / precision-add baseline. Defaults to
        weakly informative isotropic ``V0=100`` (scale 10) at the source dimension.
    conflict:
        Optional policy; with ``conflict_signals``, shrinks α offline. When used
        as ``Bayesian(prior_from=...)``, the same policy re-evaluates after data bind.
    conflict_signals:
        Optional per-source ``{"p_value": ..., "kl": ...}`` for offline shrink.
    transport:
        Required when ``source_populations`` differ from ``target_population``.
    target_population:
        Target analysis population tag (caller convention; matches meta tag
        ``population``).
    source_populations:
        Per-source population tags (same length as ``sources``). When omitted,
        tags are read from ``prior_sources`` when that argument is provided.
    prior_sources:
        Catalog entries aligned with ``sources``; used to auto-fill
        ``source_populations`` from ``meta.tags["population"]`` when
        ``source_populations`` is unset.
    unit_effects / transport_weights:
        Optional unit-level effect contributions and target-alignment weights
        for importance-weighted moment adjustment. The transported mean is the
        weighted mean; its sampling variance ``Σw²(e−ē)²/(Σw)²`` widens the
        source ``V0``. When the weights' Kish ESS is below 20 the source's α is
        forced to 0 with a recorded reason.
    coef_index:
        Coefficient index rewritten under reweight (default: last).

    Set ``ess`` on each ``ExternalPriorSourceSpec`` to get prior-strength
    effective-sample-size accounting back on the result: per-source
    ``ComposedPrior.effective_ess``, the path-dependent ``composed_ess``, and
    the ``kish_ess`` concentration-of-trust diagnostic. See ``ComposedPrior``
    for what each one means and why they are not interchangeable.
    """
    srcs = _normalize_compose_sources(sources, weights)
    baseline_mean, baseline_var = _compose_baseline(srcs, baseline)
    pop_list = _resolve_source_populations(
        srcs, source_populations=source_populations, prior_sources=prior_sources
    )
    raw = _invoke_compose_native(
        srcs,
        baseline_mean,
        baseline_var,
        conflict=conflict,
        conflict_signals=conflict_signals,
        transport=transport,
        target_population=target_population,
        source_populations=pop_list,
        unit_effects=unit_effects,
        transport_weights=transport_weights,
        coef_index=coef_index,
    )
    return _composed_prior_from_native(
        raw,
        srcs,
        baseline_mean=baseline_mean,
        baseline_variance=baseline_var,
        conflict=conflict,
        transport=transport,
    )


def _normalize_compose_sources(
    sources: Sequence[ExternalPriorSourceSpec],
    weights: Sequence[ExternalPriorWeight | float] | None,
) -> list[ExternalPriorSourceSpec]:
    srcs = _normalize_weights(sources, weights)
    if not srcs:
        raise ValueError("compose_external_priors requires at least one source")
    return srcs


def _compose_baseline(
    srcs: Sequence[ExternalPriorSourceSpec],
    baseline: tuple[Sequence[float], Sequence[float]] | None,
) -> tuple[list[float], list[float]]:
    n = len(srcs[0].mean)
    if baseline is None:
        return [0.0] * n, [100.0] * n
    return list(baseline[0]), list(baseline[1])


def _resolve_source_populations(
    srcs: Sequence[ExternalPriorSourceSpec],
    *,
    source_populations: Sequence[str | None] | None,
    prior_sources: Sequence[PriorSource] | None,
) -> list[str | None] | None:
    if source_populations is not None:
        return [None if p is None else str(p) for p in source_populations]
    if prior_sources is not None:
        if len(prior_sources) != len(srcs):
            raise ValueError("prior_sources length must match sources")
        return populations_from_prior_sources(prior_sources)
    return None


def _invoke_compose_native(
    srcs: Sequence[ExternalPriorSourceSpec],
    baseline_mean: Sequence[float],
    baseline_var: Sequence[float],
    *,
    conflict: ConflictPolicy | None,
    conflict_signals: Sequence[Mapping[str, float | None]] | None,
    transport: TransportPolicy | None,
    target_population: str | None,
    source_populations: Sequence[str | None] | None,
    unit_effects: Sequence[float] | None,
    transport_weights: Sequence[float] | None,
    coef_index: int | None,
) -> dict[str, Any]:
    return _compose_external_priors(
        [s.to_dict() for s in srcs],
        list(baseline_mean),
        list(baseline_var),
        conflict=conflict.to_dict() if conflict is not None else None,
        conflict_signals=[dict(s) for s in conflict_signals]
        if conflict_signals is not None
        else None,
        transport=transport.to_wire() if transport is not None else None,
        target_population=target_population,
        source_populations=list(source_populations) if source_populations is not None else None,
        unit_effects=list(unit_effects) if unit_effects is not None else None,
        transport_weights=list(transport_weights) if transport_weights is not None else None,
        coef_index=coef_index,
    )


def _composed_prior_from_native(
    raw: Mapping[str, Any],
    srcs: Sequence[ExternalPriorSourceSpec],
    *,
    baseline_mean: Sequence[float],
    baseline_variance: Sequence[float],
    conflict: ConflictPolicy | None,
    transport: TransportPolicy | None,
) -> ComposedPrior:
    out_sources = tuple(srcs)
    if raw.get("sources"):
        rebuilt: list[ExternalPriorSourceSpec] = []
        for row in raw["sources"]:
            rebuilt.append(
                ExternalPriorSourceSpec(
                    id=str(row["id"]),
                    mean=tuple(float(x) for x in row["mean"]),
                    variance=tuple(float(x) for x in row["variance"]),
                    weight=ExternalPriorWeight(
                        alpha=float(row["alpha"]),
                        mixture_weight=(
                            None
                            if row.get("mixture_weight") is None
                            else float(row["mixture_weight"])
                        ),
                    ),
                    ess=(None if row.get("ess") is None else float(row["ess"])),
                )
            )
        out_sources = tuple(rebuilt)
    return ComposedPrior(
        mean=tuple(float(x) for x in raw["mean"]),
        variance=tuple(float(x) for x in raw["variance"]),
        source_ids=tuple(str(x) for x in raw["source_ids"]),
        alphas_requested=tuple(float(x) for x in raw["alphas_requested"]),
        alphas_applied=tuple(float(x) for x in raw["alphas_applied"]),
        mixture_weights=tuple(None if w is None else float(w) for w in raw["mixture_weights"]),
        sources=out_sources,
        effective_ess=tuple(
            None if x is None else float(x) for x in raw.get("effective_ess") or ()
        ),
        composed_ess=(None if raw.get("composed_ess") is None else float(raw["composed_ess"])),
        kish_ess=(None if raw.get("kish_ess") is None else float(raw["kish_ess"])),
        conflict=conflict,
        conflict_p_values=tuple(
            None if x is None else float(x) for x in raw.get("conflict_p_values") or ()
        ),
        conflict_kl_values=tuple(
            None if x is None else float(x) for x in raw.get("conflict_kl_values") or ()
        ),
        assumption_ids=tuple(str(x) for x in raw.get("assumption_ids") or ()),
        transport=transport,
        baseline_mean=tuple(float(x) for x in baseline_mean),
        baseline_variance=tuple(float(x) for x in baseline_variance),
    )


@dataclass(frozen=True)
class PriorSourceMeta:
    """Metadata for one prior-bank source."""

    artifact_id: str
    estimand: EstimandFingerprint
    identification: str
    tags: Mapping[str, str] = field(default_factory=dict)
    design: Sequence[DesignVariable] = ()
    contrast: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)
    declared_mapping: PriorMapping | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "estimand": asdict(self.estimand),
            "identification": self.identification,
            "tags": dict(self.tags),
            "design": [asdict(v) for v in self.design],
        }
        if self.contrast is not None:
            d["contrast"] = self.contrast
        if self.provenance:
            d["provenance"] = dict(self.provenance)
        if self.declared_mapping is not None:
            d["declared_mapping"] = self.declared_mapping.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PriorSourceMeta:
        est = d["estimand"]
        mapping = None
        if d.get("declared_mapping"):
            md = d["declared_mapping"]
            pairs = tuple(tuple(p) for p in md.get("pairs", ()))
            mapping = PriorMapping(
                kind=md["kind"],
                source_quantity=md.get("source_quantity"),
                pairs=pairs,
            )
        return cls(
            artifact_id=d["artifact_id"],
            estimand=EstimandFingerprint(
                query_kind=est["query_kind"],
                treatment=est["treatment"],
                outcome=est["outcome"],
            ),
            identification=d["identification"],
            tags=dict(d.get("tags") or {}),
            design=tuple(
                DesignVariable(name=row["name"], role=row["role"]) for row in d.get("design") or ()
            ),
            contrast=d.get("contrast"),
            provenance=dict(d.get("provenance") or {}),
            declared_mapping=mapping,
        )

    def to_cbor(self) -> bytes:
        return bytes(_encode_meta(self.to_dict()))

    @classmethod
    def from_cbor(cls, raw: bytes) -> PriorSourceMeta:
        raw = container_bytes(raw)
        return cls.from_dict(_decode_meta(bytes(raw)))


@dataclass(frozen=True)
class PriorSource:
    """One catalog entry: meta plus optional posterior artifact bytes."""

    meta: PriorSourceMeta
    artifact: bytes | None = None


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of checking one source against a target design."""

    status: Literal["compatible", "partial", "rejected"]
    artifact_id: str
    missing: tuple[str, ...] = ()
    mappable: tuple[str, ...] = ()
    reason: Mapping[str, Any] | None = None

    @property
    def is_usable(self) -> bool:
        return self.status in ("compatible", "partial")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CompatibilityReport:
        return cls(
            status=d["status"],
            artifact_id=d["artifact_id"],
            missing=tuple(d.get("missing") or ()),
            mappable=tuple(d.get("mappable") or ()),
            reason=d.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "artifact_id": self.artifact_id,
        }
        if self.missing:
            out["missing"] = list(self.missing)
        if self.mappable:
            out["mappable"] = list(self.mappable)
        if self.reason is not None:
            out["reason"] = dict(self.reason)
        return out


def _query_estimand(query: Any) -> EstimandFingerprint:
    kind = getattr(query, "kind", None) or "ate"
    query_kind = {
        "average": "ate",
        "pulse": "pulse",
        "sustained": "sustained",
        "response_curve": "response",
        "intervention_response": "response",
        "conditional": "cate",
    }.get(kind, str(kind))
    treatment = getattr(query, "treatment", None)
    outcome = getattr(query, "outcome", None)
    if treatment is None or outcome is None:
        raise TypeError("query must expose treatment and outcome names")
    return EstimandFingerprint(
        query_kind=query_kind,
        treatment=str(treatment),
        outcome=str(outcome),
    )


class PriorCatalog:
    """Catalog of prior sources with compatibility filter and ranking."""

    def __init__(self, sources: Sequence[PriorSource] | None = None) -> None:
        self._sources: list[PriorSource] = list(sources or ())

    @classmethod
    def from_sources(cls, sources: Sequence[PriorSource]) -> PriorCatalog:
        return cls(sources)

    def add(self, source: PriorSource) -> None:
        self._sources.append(source)

    @property
    def sources(self) -> tuple[PriorSource, ...]:
        return tuple(self._sources)

    def compatible_with(
        self,
        *,
        query: Any,
        variables: Sequence[str] = (),
        tags: Mapping[str, str] | None = None,
        allow_unidentified: bool = False,
    ) -> list[CompatibilityReport]:
        """Return one report per source for the target query/design."""
        target = {
            "estimand": asdict(_query_estimand(query)),
            "variables": list(variables),
            "tags": dict(tags or {}),
            "allow_unidentified": allow_unidentified,
        }
        payload = []
        for src in self._sources:
            entry: dict[str, Any] = {"meta": src.meta.to_dict()}
            if src.artifact is not None:
                entry["artifact"] = bytes(src.artifact)
            payload.append(entry)
        raw = _filter(payload, target)
        return [CompatibilityReport.from_dict(r) for r in raw]

    def require_compatible(
        self,
        *,
        query: Any,
        variables: Sequence[str] = (),
        tags: Mapping[str, str] | None = None,
        allow_unidentified: bool = False,
    ) -> PriorSource:
        """Return the first Compatible source, or refuse the catalog."""
        return self._require_source(
            compatible_only=True,
            query=query,
            variables=variables,
            tags=tags,
            allow_unidentified=allow_unidentified,
        )

    def require_usable(
        self,
        *,
        query: Any,
        variables: Sequence[str] = (),
        tags: Mapping[str, str] | None = None,
        allow_unidentified: bool = False,
    ) -> PriorSource:
        """Return the first usable (Compatible or Partial) source, or refuse."""
        return self._require_source(
            compatible_only=False,
            query=query,
            variables=variables,
            tags=tags,
            allow_unidentified=allow_unidentified,
        )

    def _require_source(
        self,
        *,
        compatible_only: bool,
        query: Any,
        variables: Sequence[str],
        tags: Mapping[str, str] | None,
        allow_unidentified: bool,
    ) -> PriorSource:
        from .errors import CausalUnsupportedError

        reports = self.compatible_with(
            query=query,
            variables=variables,
            tags=tags,
            allow_unidentified=allow_unidentified,
        )
        chosen = next(
            (
                report
                for report in reports
                if report.status == "compatible" or (not compatible_only and report.is_usable)
            ),
            None,
        )
        if chosen is None:
            detail = (
                ", ".join(
                    f"{report.artifact_id}:{report.status}"
                    + (f"/{report.reason.get('code')}" if report.reason else "")
                    for report in reports
                )
                or "empty catalog"
            )
            raise CausalUnsupportedError(
                "prior catalog is incompatible with the target cell; "
                f"PriorCatalog.filter_compatible refused every source ({detail})"
            )
        return next(src for src in self._sources if src.meta.artifact_id == chosen.artifact_id)

    def rank(
        self,
        reports: Sequence[CompatibilityReport],
        scores: Mapping[str, float],
    ) -> list[CompatibilityReport]:
        """Stable-rank usable reports by caller similarity scores."""
        payload = [r.to_dict() for r in reports]
        ranked = _rank(payload, dict(scores))
        return [CompatibilityReport.from_dict(r) for r in ranked]


@dataclass(frozen=True)
class PredictiveCounts:
    """Posterior predictive distribution of a future count.

    ``pmf[j]`` is the probability of count ``j`` (for a Beta-Binomial, ``j``
    successes in the future trials; for a Gamma-Poisson, count ``j`` over the
    future exposure).
    """

    mean: float
    variance: float
    pmf: tuple[float, ...]


@dataclass(frozen=True)
class BetaBinomialPosterior:
    """Closed-form posterior of a proportion in one cell.

    ``Beta(alpha, beta)`` is the *posterior* (prior plus data). ``lower`` and
    ``upper`` bound the equal-tailed credible interval at ``level``: a
    posterior statement under the declared prior, not a frequentist
    confidence interval. ``log_marginal_likelihood`` is ``ln p(successes |
    trials, prior)`` for the binomial count.
    """

    alpha: float
    beta: float
    successes: int
    trials: int
    mean: float
    variance: float
    lower: float
    upper: float
    level: float
    log_marginal_likelihood: float
    predictive: PredictiveCounts | None = None


@dataclass(frozen=True)
class GammaPoissonPosterior:
    """Closed-form posterior of a Poisson rate in one cell.

    ``Gamma(shape, rate)`` is the *posterior*: prior shape plus the total
    count, prior rate plus the total exposure. ``lower`` and ``upper`` bound the
    equal-tailed credible interval at ``level`` (a posterior statement under the
    declared prior). ``log_marginal_likelihood`` is ``ln p(counts | prior)``.
    """

    shape: float
    rate: float
    total_count: int
    exposure: float
    mean: float
    variance: float
    lower: float
    upper: float
    level: float
    log_marginal_likelihood: float
    predictive: PredictiveCounts | None = None


def _predictive(d: Mapping[str, Any] | None) -> PredictiveCounts | None:
    if d is None:
        return None
    return PredictiveCounts(
        mean=float(d["mean"]),
        variance=float(d["variance"]),
        pmf=tuple(float(p) for p in d["pmf"]),
    )


def _beta_from_native(d: Mapping[str, Any]) -> BetaBinomialPosterior:
    return BetaBinomialPosterior(
        alpha=float(d["alpha"]),
        beta=float(d["beta"]),
        successes=int(d["successes"]),
        trials=int(d["trials"]),
        mean=float(d["mean"]),
        variance=float(d["variance"]),
        lower=float(d["lower"]),
        upper=float(d["upper"]),
        level=float(d["level"]),
        log_marginal_likelihood=float(d["log_marginal_likelihood"]),
        predictive=_predictive(d.get("predictive")),
    )


def beta_binomial_posterior(
    successes: int,
    trials: int,
    *,
    prior: BetaHyperparameters,
    level: float = 0.95,
    predictive_trials: int | None = None,
) -> BetaBinomialPosterior:
    """Posterior of a proportion after ``successes`` in ``trials`` trials.

    Closed form: ``Beta(prior.alpha + successes, prior.beta + trials -
    successes)``. A prior is an assumption; it never upgrades identification, so
    the proportion is a statement about the cell's observed-data distribution.

    Raises
    ------
    ValueError
        ``successes > trials``, an invalid prior, or ``level`` outside ``(0, 1)``.
    """
    return _beta_from_native(
        _beta_binomial_posterior(
            int(successes),
            int(trials),
            float(prior.alpha),
            float(prior.beta),
            level=float(level),
            predictive_trials=predictive_trials,
        )
    )


def exceedance_posterior(
    outcomes: Sequence[float],
    threshold: float,
    *,
    prior: BetaHyperparameters,
    level: float = 0.95,
    predictive_trials: int | None = None,
) -> BetaBinomialPosterior:
    """Posterior of the exceedance probability ``P(Y > threshold)`` within one cell.

    ``outcomes`` are the outcomes of the units in the cell; a success is an
    outcome strictly above ``threshold``. The exceedance probability is a
    binomial proportion with a ``Beta`` prior, so its posterior, credible
    interval and predictive distribution are exact.

    Raises
    ------
    ValueError
        No outcomes, a non-finite outcome or threshold, an invalid prior, or
        ``level`` outside ``(0, 1)``.
    """
    return _beta_from_native(
        _exceedance_posterior(
            [float(y) for y in outcomes],
            float(threshold),
            float(prior.alpha),
            float(prior.beta),
            level=float(level),
            predictive_trials=predictive_trials,
        )
    )


def gamma_poisson_posterior(
    counts: Sequence[int],
    *,
    prior: GammaHyperparameters,
    exposures: Sequence[float] | None = None,
    level: float = 0.95,
    predictive_exposure: float | None = None,
) -> GammaPoissonPosterior:
    """Posterior of a Poisson rate from per-unit counts within one cell.

    Closed form: ``Gamma(prior.shape + sum(counts), prior.rate + sum(exposures))``
    with unit exposures when ``exposures`` is omitted. The rate is per unit of
    exposure.

    Raises
    ------
    ValueError
        No counts, a negative count, a non-positive exposure, an invalid prior,
        or ``level`` outside ``(0, 1)``.
    """
    d = _gamma_poisson_posterior(
        [int(c) for c in counts],
        float(prior.shape),
        float(prior.rate),
        exposures=None if exposures is None else [float(t) for t in exposures],
        level=float(level),
        predictive_exposure=predictive_exposure,
    )
    return GammaPoissonPosterior(
        shape=float(d["shape"]),
        rate=float(d["rate"]),
        total_count=int(d["total_count"]),
        exposure=float(d["exposure"]),
        mean=float(d["mean"]),
        variance=float(d["variance"]),
        lower=float(d["lower"]),
        upper=float(d["upper"]),
        level=float(d["level"]),
        log_marginal_likelihood=float(d["log_marginal_likelihood"]),
        predictive=_predictive(d.get("predictive")),
    )


@dataclass(frozen=True)
class AlphaSensitivity:
    """The target estimate at one fixed power-prior discount ``alpha``.

    ``score`` is the selection criterion on the original data (log marginal
    likelihood, or the K-fold log predictive density); ``estimate`` is the
    posterior mean and ``lower`` / ``upper`` the equal-tailed credible interval
    at that ``alpha``.
    """

    alpha: float
    score: float
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SelectedAlpha:
    """A power-prior discount chosen from out-of-sample fit, and its estimate.

    ``selected_alpha`` maximizes the criterion over the grid and ``estimate`` is the
    posterior mean there. ``conditional_interval`` treats ``selected_alpha`` as fixed
    in advance and is not the uncertainty of the estimate. Two intervals stay valid
    under the selection:

    * ``uniform_interval`` is the target-only interval widened to contain the
      estimate. It covers at ``level`` whatever the source/target discrepancy, and
      cannot be narrower than the target-only interval (no interval that covers for
      every discrepancy can).
    * ``interval`` is the bias-aware interval around ``estimate`` when a
      ``discrepancy_bound`` was declared: the source's prior means are assumed to lie
      within that bound of the truth, and the worst-case bias of the pooled estimator
      plus a critical value that holds for whichever ``alpha`` the data select covers
      at ``level`` for every discrepancy up to the bound. It is narrower than the
      target-only interval when the bound is small. The bound is an assumption,
      recorded here; beyond it no coverage is claimed. Without a bound,
      ``interval`` is ``uniform_interval``.

    ``sensitivity`` gives the estimate at every grid ``alpha``.
    """

    criterion: Literal["empirical_bayes", "cross_validated"]
    selected_alpha: float
    estimate: float
    level: float
    interval: tuple[float, float]
    conditional_interval: tuple[float, float]
    sensitivity: tuple[AlphaSensitivity, ...]
    uniform_interval: tuple[float, float]
    discrepancy_bound: float | None


def _selected_from_native(d: Mapping[str, Any]) -> SelectedAlpha:
    return SelectedAlpha(
        criterion=d["criterion"],
        selected_alpha=float(d["selected_alpha"]),
        estimate=float(d["estimate"]),
        level=float(d["level"]),
        interval=(float(d["interval"][0]), float(d["interval"][1])),
        conditional_interval=(
            float(d["conditional_interval"][0]),
            float(d["conditional_interval"][1]),
        ),
        sensitivity=tuple(
            AlphaSensitivity(
                alpha=float(r["alpha"]),
                score=float(r["score"]),
                estimate=float(r["estimate"]),
                lower=float(r["lower"]),
                upper=float(r["upper"]),
            )
            for r in d["sensitivity"]
        ),
        uniform_interval=(float(d["uniform_interval"][0]), float(d["uniform_interval"][1])),
        discrepancy_bound=(
            None if d["discrepancy_bound"] is None else float(d["discrepancy_bound"])
        ),
    )


def select_alpha_beta_binomial(
    source_successes: int,
    source_trials: int,
    outcomes: Sequence[bool],
    *,
    base: BetaHyperparameters,
    criterion: Literal["empirical_bayes", "cross_validated"] = "empirical_bayes",
    folds: int = 5,
    grid: Sequence[float] | None = None,
    level: float = 0.95,
    discrepancy_bound: float | None = None,
    seed: int = 0,
) -> SelectedAlpha:
    """Choose the power-prior discount of a source proportion from target fit.

    The power prior is ``Beta(base.alpha + a * k0, base.beta + a * (n0 - k0))``
    for source counts ``k0`` of ``n0`` and discount ``a`` in ``[0, 1]``;
    ``outcomes`` are the target trials. ``criterion`` is ``"empirical_bayes"``
    (maximize the marginal likelihood) or ``"cross_validated"`` (maximize the
    ``folds``-fold log predictive density). ``grid`` defaults to 21 equally
    spaced points on ``[0, 1]`` and must start at ``0``. ``discrepancy_bound``
    declares how far the source proportion is assumed to lie from the truth
    (see :class:`SelectedAlpha`). A hand-set discount needs no selection: build
    the prior directly and call :func:`beta_binomial_posterior`.
    """
    return _selected_from_native(
        _select_alpha_beta(
            int(source_successes),
            int(source_trials),
            [bool(o) for o in outcomes],
            float(base.alpha),
            float(base.beta),
            criterion=criterion,
            folds=int(folds),
            grid=None if grid is None else [float(g) for g in grid],
            level=float(level),
            discrepancy_bound=None if discrepancy_bound is None else float(discrepancy_bound),
            seed=int(seed),
        )
    )


def select_alpha_gamma_poisson(
    source_total_count: int,
    source_exposure: float,
    counts: Sequence[int],
    *,
    base: GammaHyperparameters,
    criterion: Literal["empirical_bayes", "cross_validated"] = "empirical_bayes",
    folds: int = 5,
    grid: Sequence[float] | None = None,
    level: float = 0.95,
    discrepancy_bound: float | None = None,
    seed: int = 0,
) -> SelectedAlpha:
    """Choose the power-prior discount of a source rate from target fit.

    The power prior is ``Gamma(base.shape + a * S0, base.rate + a * E0)`` for a
    source total count ``S0`` over total exposure ``E0``; ``counts`` are the
    target observations over unit exposure. See
    :func:`select_alpha_beta_binomial` for the criteria and the interval.
    """
    return _selected_from_native(
        _select_alpha_gamma(
            int(source_total_count),
            float(source_exposure),
            [int(c) for c in counts],
            float(base.shape),
            float(base.rate),
            criterion=criterion,
            folds=int(folds),
            grid=None if grid is None else [float(g) for g in grid],
            level=float(level),
            discrepancy_bound=None if discrepancy_bound is None else float(discrepancy_bound),
            seed=int(seed),
        )
    )


def select_alpha_linear(
    sources: Sequence[ExternalPriorSourceSpec],
    baseline_mean: Sequence[float],
    baseline_variance: Sequence[float],
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    functional: Sequence[float],
    *,
    known_residual_variance: float | None = None,
    criterion: Literal["empirical_bayes", "cross_validated"] = "empirical_bayes",
    folds: int = 5,
    grid: Sequence[float] | None = None,
    level: float = 0.95,
    discrepancy_bound: float | None = None,
    seed: int = 0,
) -> SelectedAlpha:
    """Choose the power-prior discount of external Gaussian sources.

    Every source takes the same discount from the grid (the power path of
    :func:`compose_external_priors`; each source's own ``alpha`` is ignored and
    a source with a ``mixture_weight`` is refused). ``x`` is the target design
    (one row per unit), ``y`` the outcomes and ``functional`` the weights ``c``
    of the reported linear functional ``c'beta``. The residual variance is
    known when ``known_residual_variance`` is given, else Inv-Gamma with the
    library's weakly informative default. ``discrepancy_bound`` declares how far
    every source's prior means are assumed to lie from the true coefficients, in
    coefficient units (see :class:`SelectedAlpha`).
    """
    return _selected_from_native(
        _select_alpha_linear(
            [{**s.to_dict(), "alpha": 1.0} for s in sources],
            [float(m) for m in baseline_mean],
            [float(v) for v in baseline_variance],
            [[float(v) for v in row] for row in x],
            [float(v) for v in y],
            [float(c) for c in functional],
            known_residual_variance=known_residual_variance,
            criterion=criterion,
            folds=int(folds),
            grid=None if grid is None else [float(g) for g in grid],
            level=float(level),
            discrepancy_bound=None if discrepancy_bound is None else float(discrepancy_bound),
            seed=int(seed),
        )
    )


__all__ = [
    "AlphaSensitivity",
    "BetaBinomialPosterior",
    "BetaHyperparameters",
    "CompatibilityReport",
    "ComposedPrior",
    "ConflictPolicy",
    "DesignVariable",
    "EstimandFingerprint",
    "ExternalPriorSourceSpec",
    "ExternalPriorWeight",
    "GammaHyperparameters",
    "GammaPoissonPosterior",
    "MappedPosteriorPrior",
    "POPULATION_TAG_KEY",
    "PriorCatalog",
    "PriorMapping",
    "PriorSource",
    "PredictiveCounts",
    "PriorSourceMeta",
    "SelectedAlpha",
    "TransportPolicy",
    "beta_binomial_posterior",
    "beta_from_mean_and_ess",
    "beta_from_moments",
    "compose_external_priors",
    "exceedance_posterior",
    "gamma_from_mean_and_ess",
    "gamma_from_moments",
    "gamma_poisson_posterior",
    "mapped_posterior_transfer",
    "populations_from_prior_sources",
    "select_alpha_beta_binomial",
    "select_alpha_gamma_poisson",
    "select_alpha_linear",
]
