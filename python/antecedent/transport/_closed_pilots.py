"""Advanced measured scalar producers and explicit internal candidate lifecycles.

A standard producer reports measured inference only when its original native source
and every actual scalar basis resolve to current attesting evidence inside the
finite validation protocol. Internal candidate artifacts retain their unmeasured
standing. Other pilot routes retain their own refusal boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from .. import _native
from .._measured_inference import MeasuredInference, _production_limits
from .._native import binary_nested_markov_closed as _binary_nested_markov_closed
from .._native import joint_bayesian_transport_closed as _joint_bayesian_transport_closed
from .._native import (
    sampled_observation_recovery_closed as _sampled_observation_recovery_closed,
)
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ._impl import TransportIdentification, _non_negative
from ._joint_posterior import (
    JointTransportPriors,
    JointTransportSource,
    JointTransportTarget,
    _measured,
)
from ._nested_fisher import NestedFisherCandidate
from ._nested_posterior import NestedMarkovPosteriorCandidate, NestedMarkovPrior
from ._recovery import ObservationRecoveryQuery
from ._sampled_candidate import SampledRecoveryCandidate

__all__ = [
    "NestedFisherCandidate",
    "NestedMarkovPrior",
    "NestedMarkovPosteriorCandidate",
    "binary_nested_markov",
    "binary_nested_markov_fisher_interval",
    "joint_bayesian_transport",
    "sampled_observation_recovery",
]

_GRAPH_CLASSES = ("fixed_dag", "admg", "graph_posterior")
_MAX_BYTE = 255


def _frozen(route: str, detail: str) -> CausalUnsupportedError:
    """The fallback refusal; the native call always raises before it is reached."""
    return CausalUnsupportedError(
        f"{detail}: {route} is a closed route; calibration is unmeasured",
        reason_code="cell_not_licensed",
    )


def _names(name: str, values: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{name} must be a sequence of names")
    names = list(values)
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise CausalValueError(f"{name} must hold non-empty names")
    if len(set(names)) != len(names):
        raise CausalValueError(f"{name} must not repeat a name")
    return names


def _has_bidirected_edge(graph: Admg) -> bool:
    return any(graph.bidirected_neighbors(node) for node in graph.nodes())


def joint_bayesian_transport(
    *,
    sources: Sequence[Mapping[str, Any] | JointTransportSource],
    target: Mapping[str, Any] | JointTransportTarget | None,
    features: Sequence[str],
    graph: Admg | None = None,
    graph_class: str = "fixed_dag",
    dependence: str = "independent_samples",
    varying: str = "intercept",
    sharing: str = "independent_varying_blocks",
    draws: int = 4000,
    seed: int = 0,
    identification: TransportIdentification | None = None,
    priors: JointTransportPriors | None = None,
    treatment: str = "a",
    outcome: str = "y",
    max_unsupported_mass: float = 0.0,
    conflict_z_threshold: float = 3.0,
    level: float = 0.95,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference:
    """Return native-authorized measured inference for the target-effect scalar.

    Requires original identification, typed source/target and complete declared
    Gaussian priors. Native authorization is confined to the measured known-variance
    finite validation design at .95; every scalar retains its own record and actual
    basis. Nearby unsupported protocols refuse. The original complete candidate law
    remains model-conditional and unmeasured; these records are not universal prior
    calibration or confidence guarantees for arbitrary data-generating laws.
    """
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise CausalTypeError("sources must be a sequence of source trials")
    if any(not isinstance(source, (Mapping, JointTransportSource)) for source in sources):
        raise CausalTypeError("every source must be a mapping")
    if target is not None and not isinstance(target, (Mapping, JointTransportTarget)):
        raise CausalTypeError("target must be a mapping or None")
    feature_names = _names("features", features)
    if graph is not None and not isinstance(graph, Admg):
        raise CausalTypeError("graph must be an Admg or None")
    if not isinstance(graph_class, str) or graph_class not in _GRAPH_CLASSES:
        raise CausalValueError(f"graph_class must be one of {_GRAPH_CLASSES}")
    for name, value in (("dependence", dependence), ("varying", varying), ("sharing", sharing)):
        if not isinstance(value, str):
            raise CausalTypeError(f"{name} must be a string")
    declared = "admg" if graph is not None and _has_bidirected_edge(graph) else graph_class
    _non_negative("draws", draws)
    _non_negative("seed", seed)
    if (
        getattr(_native, "joint_transport_measured", None) is not None
        and identification is not None
    ):
        if declared != "fixed_dag":
            raise CausalUnsupportedError(
                "bayesian_transport.unsupported_graph", reason_code="route_not_supported"
            )
        if graph is not None:
            raise CausalValueError(
                "candidate graph is bound by identification; omit the redundant graph argument"
            )
        if priors is None:
            raise CausalTypeError("candidate transport needs explicit complete priors")
        return _measured(
            level=level,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
            sources=sources,
            target=target,
            features=feature_names,
            identification=identification,
            priors=priors,
            treatment=treatment,
            outcome=outcome,
            varying=varying,
            sharing=sharing,
            dependence=dependence,
            basis_degree=1,
            draws=draws,
            seed=seed,
            learned=False,
            max_unsupported_mass=max_unsupported_mass,
            conflict_z_threshold=conflict_z_threshold,
        )
    _joint_bayesian_transport_closed(
        declared,
        dependence,
        varying,
        sharing,
        len(feature_names),
        len(sources),
        target is not None and (isinstance(target, JointTransportTarget) or len(target) > 0),
        draws,
    )
    raise _frozen("antecedent.transport.joint_bayesian", "bayesian_transport.route_frozen")


def _regime_table(
    names: list[str], index: Mapping[str, int], regime: Mapping[str, Any]
) -> tuple[list[int] | None, list[int], list[float]]:
    if not isinstance(regime, Mapping):
        raise CausalTypeError("every regime must be a mapping")
    counts = regime.get("counts")
    if isinstance(counts, (str, bytes)) or not isinstance(counts, Sequence):
        raise CausalTypeError("a regime needs counts, a sequence of cell counts")
    if len(counts) != 16:
        raise CausalUnsupportedError(
            "nested_markov.outside_binary_pilot: the table is not the 2x2x2x2 binary cell table",
            reason_code="route_not_supported",
        )
    cells: list[float] = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise CausalTypeError("cell counts must be numbers")
        try:
            numeric = float(count)
        except OverflowError as error:
            raise CausalValueError(
                "cell counts must be finite", reason_code="invalid_argument"
            ) from error
        if not math.isfinite(numeric):
            raise CausalValueError("cell counts must be finite", reason_code="invalid_argument")
        cells.append(numeric)
    levels = regime.get("levels")
    if levels is not None and (
        isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence)
    ):
        raise CausalTypeError(
            "levels must be a sequence of integer domain sizes", reason_code="invalid_argument"
        )
    if levels is not None and len(levels) != 4:
        raise CausalUnsupportedError(
            "nested_markov.outside_binary_pilot: a variable domain is not binary",
            reason_code="route_not_supported",
        )
    level_list = [2] * len(names) if levels is None else [_non_negative("level", v) for v in levels]
    intervened = regime.get("intervened")
    fixed: list[int] | None = None
    if intervened is not None:
        fixed = []
        for name in _names("intervened", intervened):
            if name not in index:
                raise CausalValueError(f"intervened variable {name!r} is not in the graph")
            fixed.append(index[name])
    return fixed, level_list, cells


def _nested_arguments(
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    max_iterations: int,
    tolerance: float,
) -> tuple[
    list[str],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[list[int] | None, list[int], list[float]]],
]:
    if not isinstance(graph, Admg):
        raise CausalTypeError("binary_nested_markov requires graph=Admg(...)")
    if isinstance(regimes, (str, bytes)) or not isinstance(regimes, Sequence):
        raise CausalTypeError("regimes must be a sequence of regime count tables")
    _non_negative("max_iterations", max_iterations)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise CausalTypeError("tolerance must be a number")
    try:
        finite_tolerance = math.isfinite(tolerance)
    except OverflowError as error:
        raise CausalValueError(
            "tolerance must be finite and positive", reason_code="invalid_argument"
        ) from error
    if not finite_tolerance or tolerance <= 0:
        raise CausalValueError(
            "tolerance must be finite and positive", reason_code="invalid_argument"
        )
    if len(regimes) != 1:
        raise CausalUnsupportedError(
            "nested_markov.outside_binary_pilot: the pilot needs exactly the observational regime",
            reason_code="route_not_supported",
        )
    names = list(graph.nodes())
    if len(names) > 6:
        raise CausalUnsupportedError(
            "nested_markov.outside_binary_pilot: more than 6 observed variables",
            reason_code="route_not_supported",
        )
    index = {name: position for position, name in enumerate(names)}
    directed = [
        (index[parent], index[child]) for parent in names for child in graph.children(parent)
    ]
    bidirected = sorted(
        {
            (min(index[a], index[b]), max(index[a], index[b]))
            for a in names
            for b in graph.bidirected_neighbors(a)
        }
    )
    tables = [_regime_table(names, index, regime) for regime in regimes]
    return names, directed, bidirected, tables


def binary_nested_markov(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
    prior: NestedMarkovPrior | None = None,
    seed: int = 0,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference:
    """Measured mean0, mean1 and contrast95 under the fixed full-11D posterior pilot.

    Native execution checks the frozen prior/sampler/model/count protocol and resolves
    each scalar's original measured basis. Evidence concerns the finite declared
    interior model design; it does not license arbitrary prior or model coverage.
    The underlying continuous posterior artifact retains unmeasured standing.
    """
    return cast(
        MeasuredInference,
        _nested_posterior_entry(
            graph=graph,
            regimes=regimes,
            max_iterations=max_iterations,
            tolerance=tolerance,
            prior=prior,
            seed=seed,
            measured=True,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        ),
    )


def _nested_posterior_candidate(**kwargs: Any) -> NestedMarkovPosteriorCandidate:
    """Explicit internal candidate route, retaining original prior-sensitivity scope."""
    return cast(NestedMarkovPosteriorCandidate, _nested_posterior_entry(**kwargs, measured=False))


def _nested_posterior_entry(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
    prior: NestedMarkovPrior | None = None,
    seed: int = 0,
    measured: bool,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference | NestedMarkovPosteriorCandidate:
    """Shared original-domain validation and separate native authority dispatch."""
    _production_limits(memory_limit_bytes, cancel)
    arguments = _nested_arguments(graph, regimes, max_iterations, tolerance)
    if prior is None:
        prior = NestedMarkovPrior()
    if not isinstance(prior, NestedMarkovPrior):
        raise CausalTypeError("prior must be a NestedMarkovPrior", reason_code="invalid_argument")
    _non_negative("seed", seed)
    if seed > (1 << 64) - 1:
        raise CausalValueError(
            "seed must fit an unsigned 64-bit integer", reason_code="invalid_argument"
        )
    if not 1 <= max_iterations <= 200_000:
        raise CausalValueError(
            "nested_markov.bayesian_invalid_options: max_iterations must lie in 1..=200000",
            reason_code="invalid_argument",
        )
    from .. import _native

    candidate = getattr(
        _native,
        "nested_markov_posterior_measured" if measured else "nested_markov_posterior_candidate",
        None,
    )
    if candidate is not None:
        alpha, beta = prior._wire()
        native = candidate(
            *arguments,
            max_iterations,
            tolerance,
            alpha,
            beta,
            seed,
            **({"memory_limit_bytes": memory_limit_bytes, "cancel": cancel} if measured else {}),
        )
        return (
            MeasuredInference._from_native(native)
            if measured
            else NestedMarkovPosteriorCandidate._from_native(native)
        )
    _binary_nested_markov_closed(*arguments)
    raise _frozen("antecedent.transport.binary_nested_markov", "nested_markov.route_frozen")


def _row(row: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 4:
        raise CausalTypeError("a row is (id, responses, proxies, fully)")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in row):
        raise CausalTypeError("row fields must be integers")
    row_id, responses, proxies, fully = row
    if not 0 <= row_id <= 2**64 - 1 or any(
        not 0 <= value <= _MAX_BYTE for value in (responses, proxies, fully)
    ):
        raise CausalValueError("a row id must be in 0..=2**64-1 and each pattern field in 0..=255")
    return row_id, responses, proxies, fully


def binary_nested_markov_fisher_interval(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    nominal_level: float = 0.95,
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference:
    """Native-authorized Fisher/delta mean0, mean1 and contrast intervals.

    Each scalar resolves its actual construction and current measured record. The
    finite interior IID binary validation protocol and original licensed ID remain
    mandatory; other nominal levels/options/count scopes refuse without licensing
    the complete unmeasured candidate covariance or an arbitrary model.
    """
    return cast(
        MeasuredInference,
        _nested_fisher_entry(
            graph=graph,
            regimes=regimes,
            nominal_level=nominal_level,
            max_iterations=max_iterations,
            tolerance=tolerance,
            measured=True,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        ),
    )


def _nested_fisher_candidate(**kwargs: Any) -> NestedFisherCandidate:
    """Explicit internal Fisher candidate with its original 90/95 numerical scope."""
    return cast(NestedFisherCandidate, _nested_fisher_entry(**kwargs, measured=False))


def _nested_fisher_entry(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    nominal_level: float = 0.95,
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
    measured: bool,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference | NestedFisherCandidate:
    """Shared original-domain validation and separate native authority dispatch."""
    _production_limits(memory_limit_bytes, cancel)
    if isinstance(nominal_level, bool) or not isinstance(nominal_level, (int, float)):
        raise CausalTypeError("nominal_level must be a number")
    try:
        valid_level = math.isfinite(nominal_level) and 0 < nominal_level < 1
    except OverflowError:
        valid_level = False
    if not valid_level:
        raise CausalValueError(
            "nested_markov.fisher_invalid_level: nominal level must lie between zero and one",
            reason_code="invalid_argument",
        )
    arguments = _nested_arguments(graph, regimes, max_iterations, tolerance)
    if not 1 <= max_iterations <= 200_000:
        raise CausalValueError(
            "nested_markov.fisher_invalid_options: max_iterations must lie in 1..=200000",
            reason_code="invalid_argument",
        )
    from .. import _native

    candidate = getattr(
        _native,
        "nested_markov_fisher_measured" if measured else "nested_markov_fisher_candidate",
        None,
    )
    if candidate is not None:
        native = candidate(
            *arguments,
            nominal_level,
            max_iterations,
            tolerance,
            **({"memory_limit_bytes": memory_limit_bytes, "cancel": cancel} if measured else {}),
        )
        return (
            MeasuredInference._from_native(native)
            if measured
            else NestedFisherCandidate._from_native(native)
        )
    _binary_nested_markov_closed(*arguments)
    raise _frozen(
        "antecedent.transport.binary_nested_markov_fisher_interval", "nested_markov.route_frozen"
    )


def sampled_observation_recovery(
    *,
    stage: Any,
    query: ObservationRecoveryQuery,
    rows: Sequence[Sequence[int]],
    snapshot: str,
    replicates: int = 2000,
    seed: int = 0,
    interval_method: Literal["bootstrap_bca", "bootstrap_percentile"] = "bootstrap_bca",
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference:
    """Measured recovered-effect scalar with original native recovery proof.

    The measured finite IID missingness design supports BCa whole-row intervals at
    .95 with exactly 2000 bootstrap replicates and 1000..4000 original rows. Native
    execution and fresh replay retain the original query, proof, data, configuration
    and complete attempted-work receipt. Legacy percentile remains unlicensed; the
    underlying recovery artifact and numerical diagnostics stay unmeasured.
    """
    return cast(
        MeasuredInference,
        _sampled_recovery_entry(
            stage=stage,
            query=query,
            rows=rows,
            snapshot=snapshot,
            replicates=replicates,
            seed=seed,
            interval_method=interval_method,
            measured=True,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        ),
    )


def _sampled_observation_recovery_candidate(**kwargs: Any) -> SampledRecoveryCandidate:
    """Explicit internal original candidate including the historical percentile method."""
    return cast(SampledRecoveryCandidate, _sampled_recovery_entry(**kwargs, measured=False))


def _sampled_recovery_entry(
    *,
    stage: Any,
    query: ObservationRecoveryQuery,
    rows: Sequence[Sequence[int]],
    snapshot: str,
    replicates: int = 2000,
    seed: int = 0,
    interval_method: Literal["bootstrap_bca", "bootstrap_percentile"] = "bootstrap_bca",
    measured: bool,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference | SampledRecoveryCandidate:
    """Original query/row validation with distinct native authority dispatch."""
    _production_limits(memory_limit_bytes, cancel)
    if not isinstance(stage, _native.ObservationRecoveryStage):
        raise CausalTypeError("stage must retain an original native recovery stage")
    if interval_method not in ("bootstrap_bca", "bootstrap_percentile"):
        raise CausalValueError("interval_method must be bootstrap_bca or bootstrap_percentile")
    outcome = getattr(stage, "outcome", None)
    if outcome not in ("recovered", "nonrecoverable"):
        raise CausalTypeError("stage must be the stage of identify_observation_recovery")
    if not isinstance(query, ObservationRecoveryQuery):
        raise CausalTypeError("query must be an ObservationRecoveryQuery")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise CausalValueError("snapshot must name the rows' snapshot")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise CausalTypeError("rows must be a sequence of (id, responses, proxies, fully)")
    if len(rows) > 100_000:
        raise CausalValueError(
            "sampled_recovery.bounds_exceeded: candidate accepts at most 100000 rows",
            reason_code="invalid_argument",
        )
    _non_negative("replicates", replicates)
    if interval_method == "bootstrap_bca" and replicates != 2000:
        raise CausalValueError("BCa requires exactly 2000 whole-row bootstrap replicates")
    _non_negative("seed", seed)
    if seed > 2**64 - 1 or replicates > 2**64 - 1:
        raise CausalValueError("seed and replicates must fit unsigned 64-bit integers")
    checked = [_row(row) for row in rows]
    try:
        _sampled_observation_recovery_closed(
            outcome == "recovered",
            len(query.partially_observed),
            len(query.fully_observed),
            replicates,
            checked,
        )
    except CausalUnsupportedError as error:
        candidate = getattr(stage, "sampled_measured" if measured else "sampled_candidate", None)
        if (
            error.reason_code != "cell_not_licensed"
            or "sampled_recovery.route_frozen" not in str(error)
            or candidate is None
        ):
            raise
        produced = candidate(
            query.population,
            query.observed_regime,
            [(p.variable, p.response, p.proxy) for p in query.partially_observed],
            list(query.fully_observed),
            checked,
            snapshot,
            replicates,
            seed,
            interval_method,
            **({"memory_limit_bytes": memory_limit_bytes, "cancel": cancel} if measured else {}),
        )
        if measured:
            return MeasuredInference._from_native(produced)
        payload, artifact = produced
        return SampledRecoveryCandidate._from_native(payload, bytes(artifact))
    raise _frozen(
        "antecedent.transport.sampled_observation_recovery", "sampled_recovery.route_frozen"
    )
