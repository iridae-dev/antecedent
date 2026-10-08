"""2.3A closed calibrated-interval pilot routes (Python surface).

Three pilots have a Rust core that replays through an independent io artifact, but each
claims a *calibrated* interval whose coverage is measured only at the release cut, so the
public producer of each route is closed:

* :func:`joint_bayesian_transport` (``antecedent.transport.joint_bayesian``, record
  ``2.3A.X4.joint_bayesian_transport``): ``cell_not_licensed`` /
  ``bayesian_transport.route_frozen``.
* :func:`binary_nested_markov` (``antecedent.transport.binary_nested_markov``, record
  ``2.3A.X4.binary_nested_markov_pilot``): ``cell_not_licensed`` /
  ``nested_markov.route_frozen``.
* :func:`sampled_observation_recovery` (``antecedent.transport.sampled_observation_recovery``,
  record ``2.3A.X10.sampled_observation_recovery``): ``cell_not_licensed`` /
  ``sampled_recovery.route_frozen``.

Each function is a real entry: it checks its arguments, then asks the Rust core whether the
request is inside the pilot. A request outside the pilot raises the pilot's own scope
refusal first (``route_not_supported`` with ``bayesian_transport.unsupported_graph``,
``nested_markov.outside_binary_pilot`` or ``sampled_recovery.unrecoverable_pattern``, or an
invalid-argument detail), so a caller learns the real obstruction. Otherwise the call raises
:class:`~antecedent.errors.CausalUnsupportedError` with ``reason_code ==
"cell_not_licensed"`` and the route-frozen detail in the message. The functions never return.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .._native import binary_nested_markov_closed as _binary_nested_markov_closed
from .._native import joint_bayesian_transport_closed as _joint_bayesian_transport_closed
from .._native import (
    sampled_observation_recovery_closed as _sampled_observation_recovery_closed,
)
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ._impl import _non_negative
from ._recovery import ObservationRecoveryQuery

__all__ = [
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
    sources: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any] | None,
    features: Sequence[str],
    graph: Admg | None = None,
    graph_class: str = "fixed_dag",
    dependence: str = "independent_samples",
    varying: str = "intercept",
    sharing: str = "independent_varying_blocks",
    draws: int = 4000,
    seed: int = 0,
) -> NoReturn:
    """Joint Bayesian source-target transport posterior of a target effect: closed.

    ``sources`` are source trials (mappings), ``target`` the target covariate sample and
    ``features`` the covariate names (the certified standardizers). ``graph_class`` is
    ``"fixed_dag"`` (the only supported class), ``"admg"`` or ``"graph_posterior"``; an
    :class:`~antecedent.Admg` ``graph`` with a bidirected edge is an ADMG query. An ADMG or
    graph-posterior query raises ``route_not_supported`` /
    ``bayesian_transport.unsupported_graph``. A draw count outside ``1..=100000`` and a model
    above 256 parameters raise ``invalid_argument``; undeclared or overlapping source
    dependence, no source and no target law raise their own typed refusals. Every other
    request raises ``cell_not_licensed`` / ``bayesian_transport.route_frozen``: the interval
    claims calibration, which is unmeasured.
    """
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise CausalTypeError("sources must be a sequence of source trials")
    if any(not isinstance(source, Mapping) for source in sources):
        raise CausalTypeError("every source must be a mapping")
    if target is not None and not isinstance(target, Mapping):
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
    _joint_bayesian_transport_closed(
        declared,
        dependence,
        varying,
        sharing,
        len(feature_names),
        len(sources),
        target is not None and len(target) > 0,
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
    cells: list[float] = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise CausalTypeError("cell counts must be numbers")
        if not math.isfinite(count):
            raise CausalValueError("cell counts must be finite")
        cells.append(float(count))
    levels = regime.get("levels")
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


def binary_nested_markov(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
) -> NoReturn:
    """Binary nested-Markov likelihood pilot with a calibrated target interval: closed.

    ``graph`` is the declared ADMG over binary observed variables and ``regimes`` are
    regime-specific count tables, each a mapping with ``counts`` (cells, first variable
    most significant), optional ``levels`` (default binary) and optional ``intervened``
    names (default: the observational regime). The pilot is exactly one four-variable
    graph (``X1 -> X2 -> X3 -> X4`` with ``X2 <-> X4``) and the observational regime. Any
    other ADMG, a regime other than the observational one, a non-binary domain or more than
    six observed variables raises ``route_not_supported`` /
    ``nested_markov.outside_binary_pilot`` (never a nonidentification claim); invalid counts
    raise ``invalid_argument``. Every other request raises ``cell_not_licensed`` /
    ``nested_markov.route_frozen``: calibration is unmeasured and no interval or posterior is
    published.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("binary_nested_markov requires graph=Admg(...)")
    if isinstance(regimes, (str, bytes)) or not isinstance(regimes, Sequence):
        raise CausalTypeError("regimes must be a sequence of regime count tables")
    _non_negative("max_iterations", max_iterations)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise CausalTypeError("tolerance must be a number")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise CausalValueError("tolerance must be finite and positive")
    names = list(graph.nodes())
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
    _binary_nested_markov_closed(names, directed, bidirected, tables)
    raise _frozen("antecedent.transport.binary_nested_markov", "nested_markov.route_frozen")


def _row(row: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 4:
        raise CausalTypeError("a row is (id, responses, proxies, fully)")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in row):
        raise CausalTypeError("row fields must be integers")
    row_id, responses, proxies, fully = row
    if row_id < 0 or any(not 0 <= value <= _MAX_BYTE for value in (responses, proxies, fully)):
        raise CausalValueError("a row id must be non-negative and each pattern field in 0..=255")
    return row_id, responses, proxies, fully


def binary_nested_markov_fisher_interval(
    *,
    graph: Admg,
    regimes: Sequence[Mapping[str, Any]],
    nominal_level: float = 0.95,
    max_iterations: int = 50_000,
    tolerance: float = 1e-11,
) -> NoReturn:
    """Expected-Fisher/delta interval for the selected IID binary model: closed.

    The internal method propagates the full eleven-parameter constrained-model
    covariance to both intervention means and their contrast. It assumes positive
    integer IID multinomial counts and a correctly specified interior model. Its
    covariance and artifact replay have independent value evidence, but coverage
    has not been measured. A valid request therefore refuses with
    ``cell_not_licensed`` / ``nested_markov.route_frozen``. This entry does not
    publish an interval or imply a Bayesian posterior.
    """
    if isinstance(nominal_level, bool) or not isinstance(nominal_level, (int, float)):
        raise CausalTypeError("nominal_level must be a number")
    if not (math.isfinite(nominal_level) and 0 < nominal_level < 1):
        raise CausalValueError(
            "nested_markov.fisher_invalid_level: nominal level must lie between zero and one",
            reason_code="invalid_argument",
        )
    binary_nested_markov(
        graph=graph, regimes=regimes, max_iterations=max_iterations, tolerance=tolerance
    )


def sampled_observation_recovery(
    *,
    stage: Any,
    query: ObservationRecoveryQuery,
    rows: Sequence[Sequence[int]],
    snapshot: str,
    replicates: int = 200,
    seed: int = 0,
) -> NoReturn:
    """Whole-method sampling interval of an exactly recovered effect: closed.

    ``stage`` is the decided stage of :func:`identify_observation_recovery` and ``query``
    its :class:`ObservationRecoveryQuery`. ``rows`` are counted observation-pattern rows
    ``(id, responses, proxies, fully)``: bit ``i`` of ``responses`` and ``proxies`` refers to
    the ``i``-th partially observed variable (in the query's order) and bit ``j`` of
    ``fully`` to the ``j``-th fully observed variable; a missing proxy carries no value, so
    its proxy bit is 0. An m-graph with a verified nonrecoverability witness, more than six
    binary variables, a zero complete-case cell (a zero denominator of the recovery formula),
    or a malformed row raises ``route_not_supported`` /
    ``sampled_recovery.unrecoverable_pattern`` or ``sampled_recovery.bounds_exceeded``, or an
    invalid-input refusal. Every other request raises ``cell_not_licensed`` /
    ``sampled_recovery.route_frozen``: the percentile interval's calibration is unmeasured.
    """
    outcome = getattr(stage, "outcome", None)
    if outcome not in ("recovered", "nonrecoverable"):
        raise CausalTypeError("stage must be the stage of identify_observation_recovery")
    if not isinstance(query, ObservationRecoveryQuery):
        raise CausalTypeError("query must be an ObservationRecoveryQuery")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise CausalValueError("snapshot must name the rows' snapshot")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise CausalTypeError("rows must be a sequence of (id, responses, proxies, fully)")
    _non_negative("replicates", replicates)
    _non_negative("seed", seed)
    checked = [_row(row) for row in rows]
    _sampled_observation_recovery_closed(
        outcome == "recovered",
        len(query.partially_observed),
        len(query.fully_observed),
        replicates,
        checked,
    )
    raise _frozen(
        "antecedent.transport.sampled_observation_recovery", "sampled_recovery.route_frozen"
    )
