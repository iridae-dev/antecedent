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

    A weighted population is a declaration only. The native study resolves it against the
    analysed table at build: it computes the weights, applies the overlap gate (positivity,
    the Kish effective-sample-size floor of 20 rows, the ESS fraction floor) and registers the
    weights as the study's custom target distribution. The result reports it as
    ``result.weighted_target``. It is not a wire population by itself;
    :func:`coerce_target_population` refuses it where no table is available to resolve it.
    """

    __slots__ = ()

    def _declaration(self, names: Sequence[str]) -> dict[str, Any]:
        raise NotImplementedError

    def _wire(self) -> dict[str, object]:
        raise CausalUnsupportedError(
            "a weighted target population (target_row_weights / target_kernel) is resolved "
            "against the analysed table and applies to AverageEffect through "
            "analyze()/prepare(); this entry point has no table to resolve it against",
            reason_code="option_not_applicable",
        )


def _column_ids(declared: Sequence[str], names: Sequence[str]) -> list[int]:
    missing = [n for n in declared if n not in names]
    if missing:
        raise CausalValueError(
            f"the weighted target depends on {missing}, which are not columns of the analysed "
            f"table; columns are {sorted(names)}"
        )
    return [list(names).index(n) for n in declared]


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

    def _declaration(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "kind": "row_weights",
            "weights": list(self.weights),
            "covariates": _column_ids(self.depends_on, names),
            "min_ess_fraction": self.min_ess_fraction,
        }


@dataclass(frozen=True, slots=True)
class KernelTarget(WeightedPopulation):
    """A target neighbourhood declared as a kernel over a covariate distance.

    Row ``i`` has distance ``d_i = sqrt(sum_j ((x_ij - center_j) / scale_j) ** 2)`` to the
    declared ``center`` over ``covariates``, and weight ``kernel(d_i)`` (family and bandwidth
    from the declared :class:`~antecedent.priors.TransportKernel`). Family, bandwidth, center
    and scale are part of the declaration reported on the result; the weights they produce
    enter the query identity.
    """

    kernel: TransportKernel
    covariates: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    min_ess_fraction: float | None = None

    def _declaration(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "kind": "kernel",
            "family": str(self.kernel.family),
            "bandwidth": float(self.kernel.bandwidth),
            "covariates": _column_ids(self.covariates, names),
            "center": list(self.center),
            "scale": list(self.scale),
            "min_ess_fraction": self.min_ess_fraction,
        }


@dataclass(frozen=True)
class WeightedTargetReport:
    """The weighted target population a result is about, with its overlap diagnostics.

    ``declaration`` is the declared kernel (family, bandwidth, covariates, center, scale) or
    the declared covariates of explicit weights; ``identity`` is the digest of the declaration
    and the exact weight bits; ``overlap`` is the support / Kish ESS / largest single-row share
    of the weights over the analysed rows. The native study computes all of it.
    """

    declaration: Mapping[str, Any]
    identity: str
    distribution_id: int
    overlap: TargetOverlap
    min_ess_fraction: float
    assumption: str
    #: For a trial-to-target transport, the overlap of the weights the trial rows carry toward
    #: the target; ``None`` for the target of an average effect.
    trial_overlap: TargetOverlap | None = None

    @classmethod
    def _from_native(cls, raw: Mapping[str, Any], names: Sequence[str]) -> WeightedTargetReport:
        from .priors import TargetOverlap

        declaration: dict[str, Any] = {
            "kind": raw["kind"],
            "covariates": [names[i] for i in raw["covariates"]],
        }
        if raw["kind"] == "kernel":
            declaration.update(
                family=raw["family"],
                bandwidth=raw["bandwidth"],
                center=list(raw["center"]),
                scale=list(raw["scale"]),
            )
        return cls(
            declaration=declaration,
            identity=str(raw["identity"]),
            distribution_id=int(raw["distribution_id"]),
            overlap=TargetOverlap._from_native(raw["overlap"]),
            min_ess_fraction=float(raw["min_ess_fraction"]),
            assumption=str(raw["assumption"]),
            trial_overlap=(
                TargetOverlap._from_native(raw["trial_overlap"])
                if raw.get("trial_overlap")
                else None
            ),
        )

    def assumptions(self) -> tuple[str, ...]:
        """The assumption statements the target adds to the result's assumptions."""
        return (self.assumption,)

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


def weighted_target_declaration(
    query: Any, names: Sequence[str]
) -> tuple[Any, dict[str, Any] | None]:
    """Split a weighted target population off ``query`` for the native study.

    Returns ``(query, declaration)``. A query without a weighted target comes back unchanged
    with no declaration. Otherwise the query is returned with the all-observed population and
    the declaration (covariates as column indices) is what the native study resolves: it
    computes the weights, gates the overlap and names the custom distribution itself.
    """
    population = getattr(query, "target_population", None)
    if not isinstance(population, WeightedPopulation):
        return query, None
    return replace(query, target_population=None), population._declaration(names)


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
    "weighted_target_declaration",
    "target_all",
    "target_custom_distribution",
    "target_kernel",
    "target_named",
    "target_row_weights",
    "target_rows",
    "target_treated",
    "target_untreated",
]
