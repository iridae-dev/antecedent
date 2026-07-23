"""External prior bank: catalog metadata and compatibility filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ._native import (
    compose_external_priors as _compose_external_priors,
    conflict_shrink_alpha as _shrink_alpha,
    decode_prior_source_meta as _decode_meta,
    encode_prior_source_meta as _encode_meta,
    prior_catalog_filter as _filter,
    prior_catalog_rank as _rank,
)

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
    """One hydrated Gaussian coefficient prior for composition."""

    id: str
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    weight: ExternalPriorWeight = field(default_factory=ExternalPriorWeight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mean": list(self.mean),
            "variance": list(self.variance),
            "alpha": self.weight.alpha,
            "mixture_weight": self.weight.mixture_weight,
        }


@dataclass(frozen=True)
class ComposedPrior:
    """Result of ``compose_external_priors``; usable as ``Bayesian(prior_from=...)``."""

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

    def to_native_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "mean": list(self.mean),
            "variance": list(self.variance),
            "source_ids": list(self.source_ids),
            "alphas_requested": list(self.alphas_requested),
            "alphas_applied": list(self.alphas_applied),
            "mixture_weights": list(self.mixture_weights),
            "sources": [s.to_dict() for s in self.sources],
        }
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


def _normalize_weights(
    sources: Sequence[ExternalPriorSourceSpec],
    weights: Sequence[ExternalPriorWeight] | Sequence[float] | None,
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
            )
        )
    return out


def populations_from_prior_sources(
    sources: Sequence[PriorSource],
) -> list[str | None]:
    """Read ``tags["population"]`` from each catalog source (``None`` if unset)."""
    return [s.meta.tags.get(POPULATION_TAG_KEY) for s in sources]


def compose_external_priors(
    sources: Sequence[ExternalPriorSourceSpec],
    weights: Sequence[ExternalPriorWeight] | Sequence[float] | None = None,
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
        for importance-weighted moment adjustment.
    coef_index:
        Coefficient index rewritten under reweight (default: last).
    """
    srcs = _normalize_weights(sources, weights)
    if not srcs:
        raise ValueError("compose_external_priors requires at least one source")
    n = len(srcs[0].mean)
    if baseline is None:
        baseline_mean = [0.0] * n
        baseline_var = [100.0] * n
    else:
        baseline_mean = list(baseline[0])
        baseline_var = list(baseline[1])
    payload = [s.to_dict() for s in srcs]
    conf_dict = conflict.to_dict() if conflict is not None else None
    sig_list = [dict(s) for s in conflict_signals] if conflict_signals is not None else None
    pop_list = None
    if source_populations is not None:
        pop_list = [None if p is None else str(p) for p in source_populations]
    elif prior_sources is not None:
        if len(prior_sources) != len(srcs):
            raise ValueError("prior_sources length must match sources")
        pop_list = populations_from_prior_sources(prior_sources)
    raw = _compose_external_priors(
        payload,
        baseline_mean,
        baseline_var,
        conflict=conf_dict,
        conflict_signals=sig_list,
        transport=transport.to_wire() if transport is not None else None,
        target_population=target_population,
        source_populations=pop_list,
        unit_effects=list(unit_effects) if unit_effects is not None else None,
        transport_weights=list(transport_weights) if transport_weights is not None else None,
        coef_index=coef_index,
    )
    # Prefer transported source moments when native echoed them.
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
                )
            )
        out_sources = tuple(rebuilt)
    return ComposedPrior(
        mean=tuple(float(x) for x in raw["mean"]),
        variance=tuple(float(x) for x in raw["variance"]),
        source_ids=tuple(str(x) for x in raw["source_ids"]),
        alphas_requested=tuple(float(x) for x in raw["alphas_requested"]),
        alphas_applied=tuple(float(x) for x in raw["alphas_applied"]),
        mixture_weights=tuple(
            None if w is None else float(w) for w in raw["mixture_weights"]
        ),
        sources=out_sources,
        conflict=conflict,
        conflict_p_values=tuple(
            None if x is None else float(x) for x in raw.get("conflict_p_values") or ()
        ),
        conflict_kl_values=tuple(
            None if x is None else float(x) for x in raw.get("conflict_kl_values") or ()
        ),
        assumption_ids=tuple(str(x) for x in raw.get("assumption_ids") or ()),
        transport=transport,
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

    def rank(
        self,
        reports: Sequence[CompatibilityReport],
        scores: Mapping[str, float],
    ) -> list[CompatibilityReport]:
        """Stable-rank usable reports by caller similarity scores."""
        payload = [r.to_dict() for r in reports]
        ranked = _rank(payload, dict(scores))
        return [CompatibilityReport.from_dict(r) for r in ranked]


__all__ = [
    "CompatibilityReport",
    "ComposedPrior",
    "ConflictPolicy",
    "DesignVariable",
    "EstimandFingerprint",
    "ExternalPriorSourceSpec",
    "ExternalPriorWeight",
    "POPULATION_TAG_KEY",
    "PriorCatalog",
    "PriorMapping",
    "PriorSource",
    "PriorSourceMeta",
    "TransportPolicy",
    "compose_external_priors",
    "populations_from_prior_sources",
]
