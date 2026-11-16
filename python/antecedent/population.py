"""Named predicates and custom target-distribution weights for analyze()."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any, TypedDict

from ._validate import count as _count
from .errors import CausalStateError, CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .outcomes import Outcome, capture_outcome

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis
    from .priors import TargetOverlap, TransportKernel
    from .results import AnalysisResult


def replay_retarget(
    study: PreparedAnalysis[Any] | Outcome[Any],
    artifact: Any,
    *,
    execution: Execution | None = None,
) -> Outcome[AnalysisResult]:
    """Replay carried population weights against their exact retained native scores.

    ``study`` may be a completed outcome or its immutable study. ``artifact``
    accepts bytes or a typed exported artifact, including an outcome containing
    an artifact. The independent native consumer verifies the artifact, then
    native replay checks its complete data-snapshot, score-table and dependency
    identity before reweighting. No nuisance model is refitted and no source
    study or earlier result is replaced.

    Independent loads have no live data or scores; they cannot supply ``study``.
    A different snapshot or score identity produces the scientific refusal
    ``row_weights_bound_to_snapshot``. Invalid artifacts and operational failures
    remain typed exceptions. Omitted execution uses the source study's policy.
    """
    from ._artifact_protocol import container_bytes
    from .prepared import PreparedAnalysis, _finish_outcome, _fork

    if isinstance(study, Outcome):
        if study.refusal is not None:
            return Outcome(refusal=study.refusal, _study=study.study)
        retained = study.study
        if retained is None:
            raise CausalStateError(
                "population replay requires a live completed study with retained native scores; "
                "an independently loaded outcome has no applicable replay operation"
            )
        if not isinstance(retained, PreparedAnalysis):
            raise CausalTypeError("population replay requires one scalar completed study")
        study = retained
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("population replay requires one immutable prepared study or outcome")
    policy = study.execution if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if policy.on_stage is not None:
        raise CausalUnsupportedError(
            "population replay reuses completed scores and exposes no causal fitting stages",
            reason_code="stage_stream_unavailable",
        )
    source = replace(study, execution=policy)
    if source._refused is not None:
        return _finish_outcome(source, source._refused)
    encoded = capture_outcome(container_bytes, artifact)
    if encoded.refusal is not None:
        return _finish_outcome(source, Outcome(refusal=encoded.refusal))
    assert encoded.value is not None
    engine = _fork(source)
    policy._report_progress(0.0, "replay_retarget")
    outcome = capture_outcome(
        policy._call,
        engine.reexecute_retarget,
        encoded.value,
        seed=policy.seed,
        threads=policy.threads,
    )
    policy._report_progress(1.0, "replay_retarget")
    return _finish_outcome(replace(source, _engine=engine), outcome)


class DistributionBinding(TypedDict):
    weights: list[float]
    depends_on: list[int]


def _id(value: object, what: str) -> int:
    """An identifier as an integer; a fractional or boolean value is refused, never truncated."""
    return _count(value, what, minimum=0, type_error=CausalValueError)


def _row_indices(rows: Sequence[int]) -> list[int]:
    result: list[int] = []
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, Integral) or row < 0:
            raise CausalValueError("row indices must be non-negative integers, not boolean masks")
        result.append(int(row))
    return result


@dataclass
class PopulationRegistry:
    """Caller bindings for named row predicates and custom target weights.

    Stratification estimators reject ``CustomDistribution``; use IPW/matching
    (``estimator=\"propensity.weighting\"``) for weighted target populations.
    """

    predicates: dict[str, list[int]] = field(default_factory=dict)
    distributions: dict[int, list[float] | DistributionBinding] = field(default_factory=dict)

    def insert_predicate(self, name: str, rows: Sequence[int]) -> None:
        self.predicates[str(name)] = _row_indices(rows)

    def insert_distribution(
        self,
        distribution_id: int,
        weights: Sequence[float],
        *,
        depends_on: Sequence[int] | None = None,
    ) -> None:
        """Bind weights; AIPW requires declared covariate IDs in ``depends_on``."""
        values = [float(w) for w in weights]
        if depends_on is None:
            self.distributions[_id(distribution_id, "distribution_id")] = values
        else:
            self.distributions[_id(distribution_id, "distribution_id")] = {
                "weights": values,
                "depends_on": [_id(v, "depends_on") for v in depends_on],
            }


class Population:
    """Base class for ``AverageEffect.target_population`` specifications.

    Subclasses are frozen, ``slots=True`` dataclasses; each implements
    ``_wire()`` returning the exact wire dict :func:`coerce_target_population`
    has always produced for that spec kind — the wire shape is frozen, only
    the caller-facing constructor changed from a bare-dict-returning function
    to a type. ``__slots__ = ()`` here so a slotted subclass doesn't pick up
    an unwanted ``__dict__`` from this base.
    """

    __slots__ = ()

    def _wire(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AllRows(Population):
    """All observed rows (``kind="all"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "all"}


@dataclass(frozen=True, slots=True)
class Treated(Population):
    """Rows with observed treatment (``kind="treated"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "treated"}


@dataclass(frozen=True, slots=True)
class Untreated(Population):
    """Rows with observed control (``kind="untreated"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "untreated"}


@dataclass(frozen=True, slots=True)
class Named(Population):
    """A caller-registered named predicate (see ``PopulationRegistry``)."""

    name: str

    def _wire(self) -> dict[str, object]:
        return {"kind": "named", "name": str(self.name)}


@dataclass(frozen=True, slots=True)
class Rows(Population):
    """An explicit row-index set."""

    rows: tuple[int, ...]

    def _wire(self) -> dict[str, object]:
        return {"kind": "rows", "rows": _row_indices(self.rows)}


@dataclass(frozen=True, slots=True)
class CustomDistribution(Population):
    """A caller-registered custom weighting (see ``PopulationRegistry``)."""

    distribution_id: int

    def _wire(self) -> dict[str, object]:
        return {"kind": "custom_distribution", "id": _id(self.distribution_id, "distribution_id")}


class WeightedPopulation(Population):
    """Base class of target populations declared as a weighting over the analysed rows.

    A weighted population is resolved against the table at prepare time: its weights become
    the study's custom target distribution (so the target identity and every claim over the
    result name exactly these weights), its overlap with the rows is checked, and the
    declaration is reported on the result as ``result.weighted_target``. It is not a wire
    population by itself; :func:`coerce_target_population` refuses it where no table is
    available to resolve it.
    """

    __slots__ = ()

    def _weights(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        raise NotImplementedError

    def _declaration(self) -> dict[str, Any]:
        raise NotImplementedError

    def _depends_on(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _min_ess_fraction(self) -> float | None:
        raise NotImplementedError

    def _wire(self) -> dict[str, object]:
        raise CausalUnsupportedError(
            "a weighted target population (target_row_weights / target_kernel) is resolved "
            "against the analysed table and applies to AverageEffect through "
            "analyze()/prepare(); this entry point has no table to resolve it against",
            reason_code="option_not_applicable",
        )

    def overlap(self, data: Mapping[str, Any] | Any) -> TargetOverlap:
        """Support and overlap of this target with the rows of ``data``, without analysing.

        The same check ``analyze`` applies before it estimates: positivity, the Kish
        effective-sample-size floor of 20 rows, the ESS fraction floor and the largest
        single-row share. A preview only; it does not refuse.
        """
        from ._data import ingest_columns

        names, arrays = ingest_columns(data)
        weights = self._weights(dict(zip(names, arrays, strict=True)))
        return _overlap_of(weights, self._min_ess_fraction())[0]


@dataclass(frozen=True, slots=True)
class RowWeights(WeightedPopulation):
    """Explicit per-row target weights, one per analysed row in table order.

    ``depends_on`` names the covariates the weights are a function of. They must be members of
    the certified adjustment set (never the treatment or its descendants), so the weighted
    standardization is valid for the declared target. ``min_ess_fraction`` is the floor on the
    weights' Kish effective sample size as a share of the rows (default 0.05); the absolute
    floor of 20 effective rows is never relaxed.
    """

    weights: tuple[float, ...]
    depends_on: tuple[str, ...]
    min_ess_fraction: float | None = None

    def _weights(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        return [float(w) for w in self.weights]

    def _depends_on(self) -> tuple[str, ...]:
        return self.depends_on

    def _min_ess_fraction(self) -> float | None:
        return self.min_ess_fraction

    def _declaration(self) -> dict[str, Any]:
        return {"kind": "row_weights", "depends_on": list(self.depends_on)}


@dataclass(frozen=True, slots=True)
class KernelTarget(WeightedPopulation):
    """A target neighbourhood declared as a kernel over a covariate distance.

    Row ``i`` has distance ``d_i = sqrt(sum_j ((x_ij - center_j) / scale_j) ** 2)`` to the
    declared ``center`` over ``covariates``, and weight ``kernel(d_i)`` (family and bandwidth
    from the declared :class:`~antecedent.priors.TransportKernel`). The covariates are the
    weights' ``depends_on``. Family, bandwidth, center and scale are part of the declaration
    reported on the result; the weights they produce enter the query identity.
    """

    kernel: TransportKernel
    covariates: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    min_ess_fraction: float | None = None

    def _depends_on(self) -> tuple[str, ...]:
        return self.covariates

    def _min_ess_fraction(self) -> float | None:
        return self.min_ess_fraction

    def _distances(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        import numpy as np

        total: Any = None
        for name, centre, scale in zip(self.covariates, self.center, self.scale, strict=True):
            if name not in columns:
                raise CausalValueError(
                    f"kernel target covariate {name!r} is not a column of the analysed table; "
                    f"columns are {sorted(columns)}"
                )
            try:
                values = np.asarray(columns[name], dtype=float)
            except (TypeError, ValueError) as error:
                raise CausalValueError(
                    f"kernel target covariate {name!r} must be numeric"
                ) from error
            if not np.isfinite(values).all():
                raise CausalValueError(
                    f"kernel target covariate {name!r} has missing or non-finite values; the "
                    "distance of such a row is undefined. Next attempt: complete the covariate "
                    "or declare explicit weights with target_row_weights"
                )
            term = ((values - centre) / scale) ** 2
            total = term if total is None else total + term
        return [float(d) for d in np.sqrt(total)]

    def _weights(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        return self.kernel.weights(self._distances(columns))

    def _declaration(self) -> dict[str, Any]:
        return {
            "kind": "kernel",
            "family": str(self.kernel.family),
            "bandwidth": float(self.kernel.bandwidth),
            "covariates": list(self.covariates),
            "center": [float(c) for c in self.center],
            "scale": [float(c) for c in self.scale],
        }


@dataclass(frozen=True)
class WeightedTargetReport:
    """The weighted target population a result is about, with its overlap diagnostics.

    ``declaration`` is the declared kernel (family, bandwidth, covariates, center, scale) or
    the declared covariates of explicit weights; ``identity`` is the canonical digest of the
    declaration and the exact weight bits; ``overlap`` is the support / Kish ESS / largest
    single-row share of the weights over the analysed rows.
    """

    declaration: Mapping[str, Any]
    identity: str
    distribution_id: int
    overlap: TargetOverlap
    min_ess_fraction: float

    def assumptions(self) -> tuple[str, ...]:
        """The assumption statements the target adds to the result's assumptions."""
        decl = self.declaration
        if decl["kind"] == "kernel":
            what = (
                f"the target population is the {decl['family']} kernel neighbourhood "
                f"(bandwidth {decl['bandwidth']}) of {decl['center']} over "
                f"{decl['covariates']} (scale {decl['scale']})"
            )
        else:
            what = f"the target population is the declared row weighting over {decl['depends_on']}"
        return (
            f"weighted target: {what}; the effect is standardized to it, assuming it differs "
            f"from the analysed rows only through those covariates "
            f"(identity {self.identity})",
        )

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data."""
        o = self.overlap
        return {
            "declaration": dict(self.declaration),
            "identity": self.identity,
            "distribution_id": self.distribution_id,
            "min_ess_fraction": self.min_ess_fraction,
            "overlap": {
                "n_rows": o.n_rows,
                "n_support": o.n_support,
                "support_fraction": o.support_fraction,
                "kish_ess": o.kish_ess,
                "ess_fraction": o.ess_fraction,
                "max_weight_share": o.max_weight_share,
            },
        }


def _overlap_of(
    weights: Sequence[float], min_ess_fraction: float | None
) -> tuple[TargetOverlap, str | None, float]:
    """Overlap of ``weights``, the next-attempt refusal text when it fails, and the floor used."""
    from . import _native
    from .priors import TargetOverlap

    raw = _native.target_weights_overlap(
        [float(w) for w in weights], min_ess_fraction=min_ess_fraction
    )
    return TargetOverlap._from_native(raw), raw["refusal"], float(raw["min_ess_fraction"])


def target_row_weights(
    weights: Sequence[float],
    *,
    depends_on: Sequence[str],
    min_ess_fraction: float | None = None,
) -> RowWeights:
    """Declare a target population as explicit per-row weights over named covariates."""
    values = tuple(float(w) for w in weights)
    names = tuple(str(n) for n in depends_on)
    if not values:
        raise CausalValueError("target row weights must be non-empty")
    if not names:
        raise CausalValueError(
            "target row weights need a nonempty depends_on naming the covariates they are a "
            "function of"
        )
    return RowWeights(values, names, min_ess_fraction)


def target_kernel(
    kernel: TransportKernel,
    *,
    center: Mapping[str, float],
    scale: Mapping[str, float] | None = None,
    min_ess_fraction: float | None = None,
) -> KernelTarget:
    """Declare a target neighbourhood as a kernel over the distance to ``center``.

    ``center`` maps each covariate to its coordinate in the target; ``scale`` (default 1 for
    every covariate) divides the coordinate differences, so a bandwidth is in scaled units.
    """
    if not center:
        raise CausalValueError("a kernel target needs a nonempty center naming its covariates")
    names = tuple(str(n) for n in center)
    scales = {str(k): float(v) for k, v in (scale or {}).items()}
    unknown = sorted(set(scales) - set(names))
    if unknown:
        raise CausalValueError(f"scale names covariates the center does not declare: {unknown}")
    centre = tuple(float(center[n]) for n in center)
    scale_values = tuple(scales.get(n, 1.0) for n in names)
    if any(not math.isfinite(c) for c in centre):
        raise CausalValueError("kernel target center coordinates must be finite")
    if any(not math.isfinite(v) or v <= 0.0 for v in scale_values):
        raise CausalValueError("kernel target scales must be finite and > 0")
    return KernelTarget(kernel, names, centre, scale_values, min_ess_fraction)


_WEIGHTED_TARGET_DOMAIN = "antecedent.weighted_target.v1"


def resolve_weighted_population(
    query: Any,
    names: Sequence[str],
    columns: Sequence[Any],
    registry: PopulationRegistry | None,
) -> tuple[Any, PopulationRegistry | None, WeightedTargetReport | None]:
    """Resolve a weighted target population of ``query`` against the analysed table.

    Returns ``(query, registry, report)``. A query without a weighted target comes back
    unchanged with no report, so every existing declaration is untouched. Otherwise the weights
    become a custom target distribution declared over the covariates the weights depend on, the
    query names that distribution, and the overlap check runs here: a weighting without
    positivity, below the Kish ESS floor, or thinner than the declared ESS fraction is refused
    with ``transport_target_overlap_failed`` and a next attempt, before anything is estimated.
    """
    population = getattr(query, "target_population", None)
    if not isinstance(population, WeightedPopulation):
        return query, registry, None
    import numpy as np

    from ._digest import canonical_digest

    if not columns:
        raise CausalUnsupportedError(
            "a weighted target population is declared over the rows of one analysed table; "
            "this route reads its units from a frame or design",
            reason_code="option_not_applicable",
        )
    by_name = dict(zip(names, columns, strict=True))
    depends_on = population._depends_on()
    missing = [n for n in depends_on if n not in by_name]
    if missing:
        raise CausalValueError(
            f"the weighted target depends on {missing}, which are not columns of the analysed "
            f"table; columns are {sorted(by_name)}"
        )
    weights = population._weights(by_name)
    n_rows = len(next(iter(by_name.values())))
    if len(weights) != n_rows:
        raise CausalValueError(
            f"the weighted target has {len(weights)} weights for {n_rows} analysed rows; "
            "declare one weight per row in table order"
        )
    overlap, refusal, fraction = _overlap_of(weights, population._min_ess_fraction())
    if refusal is not None:
        raise CausalUnsupportedError(
            "weighted_target.overlap_failed: " + refusal,
            reason_code="transport_target_overlap_failed",
        )
    declaration = population._declaration()
    identity = canonical_digest(
        _WEIGHTED_TARGET_DOMAIN,
        {
            "declaration": declaration,
            "weights": np.ascontiguousarray(weights, dtype="<f8"),
        },
    )
    distribution_id = int(identity[:8], 16) & 0x7FFFFFFF
    ids = [list(names).index(n) for n in depends_on]
    merged = PopulationRegistry(
        predicates=dict(registry.predicates) if registry is not None else {},
        distributions=dict(registry.distributions) if registry is not None else {},
    )
    if distribution_id in merged.distributions:
        raise CausalValueError(
            f"distribution id {distribution_id} is already registered; the weighted target "
            "derives its id from its identity, so register custom distributions under other ids"
        )
    merged.insert_distribution(distribution_id, weights, depends_on=ids)
    resolved = replace(query, target_population=CustomDistribution(distribution_id))
    report = WeightedTargetReport(
        declaration=declaration,
        identity=identity,
        distribution_id=distribution_id,
        overlap=overlap,
        min_ess_fraction=fraction,
    )
    return resolved, merged, report


def target_all() -> AllRows:
    return AllRows()


def target_treated() -> Treated:
    return Treated()


def target_untreated() -> Untreated:
    return Untreated()


def target_named(name: str) -> Named:
    return Named(str(name))


def target_rows(rows: Sequence[int]) -> Rows:
    return Rows(tuple(_row_indices(rows)))


def target_custom_distribution(distribution_id: int) -> CustomDistribution:
    return CustomDistribution(_id(distribution_id, "distribution_id"))


def coerce_target_population(spec: object) -> dict[str, object] | None:
    """Normalize AverageEffect.target_population / analyze kwargs to a wire dict."""
    if spec is None:
        return None
    if isinstance(spec, Population):
        return spec._wire()
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "_")
        if key in {"all", "all_observed", "observed"}:
            return target_all()._wire()
        if key == "treated":
            return target_treated()._wire()
        if key in {"untreated", "control"}:
            return target_untreated()._wire()
        raise ValueError(
            f"unknown target_population string {spec!r}; "
            "use all|treated|untreated or a target_* helper"
        )
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", "")).lower()
        if kind in {"all", "all_observed"}:
            return target_all()._wire()
        if kind == "treated":
            return target_treated()._wire()
        if kind == "untreated":
            return target_untreated()._wire()
        if kind == "named":
            return target_named(str(spec["name"]))._wire()
        if kind == "rows":
            return target_rows(spec["rows"])._wire()
        if kind in {"custom_distribution", "custom"}:
            return target_custom_distribution(int(spec["id"]))._wire()
        raise ValueError(f"unknown target_population mapping {spec!r}")
    raise TypeError(f"unsupported target_population type: {type(spec)!r}")


def registry_wire(
    registry: PopulationRegistry | None,
) -> tuple[dict[str, list[int]], dict[int, list[float] | DistributionBinding]]:
    if registry is None:
        return {}, {}
    return (
        {name: _row_indices(rows) for name, rows in registry.predicates.items()},
        dict(registry.distributions),
    )


__all__ = [
    "AllRows",
    "CustomDistribution",
    "KernelTarget",
    "Named",
    "Population",
    "PopulationRegistry",
    "RowWeights",
    "Rows",
    "Treated",
    "Untreated",
    "WeightedPopulation",
    "WeightedTargetReport",
    "coerce_target_population",
    "registry_wire",
    "replay_retarget",
    "resolve_weighted_population",
    "target_all",
    "target_custom_distribution",
    "target_kernel",
    "target_named",
    "target_row_weights",
    "target_rows",
    "target_treated",
    "target_untreated",
]
