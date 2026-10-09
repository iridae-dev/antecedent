"""2.3 learned joint source-target transport (Python surface): a closed route.

The learned joint row (``antecedent.transport.learned_joint``) fits the outcome mechanism
through ``antecedent-learn`` (a known-variance Bayesian polynomial-basis regression) over the
fixed DAG and selection assumptions of one identified transport row, and answers the target
average effect. The Rust core and its io artifact replay, but the posterior claims a
*calibrated* interval whose coverage is measured only at the release cut, so the public
producer is closed.

:func:`learned_joint_transport` is a real entry: it checks its arguments, then asks the Rust
core whether the request is inside the row. A request outside the row raises the row's own
scope refusal first (``route_not_supported`` with ``learned_joint_transport.unsupported_graph``,
an invalid-argument refusal for a basis degree outside ``1..=6``, a draw count outside
``1..=100000`` or a model above 256 parameters, ``sampling_dependence_unknown`` for undeclared
or overlapping source dependence, ``transport_missing_evidence`` for no source and
``joint_law_required`` for no target law), so a caller learns the real obstruction. Otherwise
the call raises :class:`~antecedent.errors.CausalUnsupportedError` with ``reason_code ==
"cell_not_licensed"`` and ``learned_joint_transport.route_frozen`` in the message. The function
returns no posterior on a default release wheel. Internal candidate feature builds
can return a numerically replayable, unmeasured posterior when an original native
identification and the complete typed data/prior declarations are supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .. import _native
from .._native import learned_joint_transport_closed as _learned_joint_transport_closed
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ._impl import TransportIdentification, _non_negative
from ._joint_posterior import (
    JointTransportPosterior,
    JointTransportPriors,
    JointTransportSource,
    JointTransportTarget,
    _candidate,
)

__all__ = ["learned_joint_transport"]

GraphClass = Literal["fixed_dag", "admg", "graph_posterior"]
#: Dependence between the source samples (``unknown`` and ``overlapping_units`` refuse).
Dependence = Literal["independent_samples", "overlapping_units", "unknown"]
#: Which coefficients vary by source.
Varying = Literal["intercept", "intercept_and_covariates"]
#: Whether the varying blocks of the sources are independent or one shared block.
Sharing = Literal["independent_varying_blocks", "shared_varying_block"]
_GRAPH_CLASSES = ("fixed_dag", "admg", "graph_posterior")
_ROUTE = "antecedent.transport.learned_joint"
_ROUTE_FROZEN = "learned_joint_transport.route_frozen"


#: What a caller can do instead: the open point-estimate route of the same transport question.
_REMEDY = (
    "use the open learned point-estimate route antecedent.transport.advanced."
    "prepare_learned_continuous (cross-fitted, interval withheld), or an identified closed-form "
    "transport estimate; the learned joint posterior opens only when its coverage is measured"
)


def _frozen() -> CausalUnsupportedError:
    """The fallback refusal; the native call always raises before it is reached."""
    return CausalUnsupportedError(
        f"{_ROUTE_FROZEN}: {_ROUTE} is a closed route; calibration is unmeasured",
        reason_code="cell_not_licensed",
        remedy=_REMEDY,
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


def learned_joint_transport(
    *,
    sources: Sequence[Mapping[str, Any] | JointTransportSource],
    target: Mapping[str, Any] | JointTransportTarget | None,
    features: Sequence[str],
    graph: Admg | None = None,
    graph_class: GraphClass = "fixed_dag",
    dependence: Dependence = "independent_samples",
    varying: Varying = "intercept",
    sharing: Sharing = "independent_varying_blocks",
    basis_degree: int = 2,
    draws: int = 4000,
    seed: int = 0,
    identification: TransportIdentification | None = None,
    priors: JointTransportPriors | None = None,
    treatment: str = "a",
    outcome: str = "y",
    max_unsupported_mass: float = 0.0,
    conflict_z_threshold: float = 3.0,
) -> JointTransportPosterior:
    """Learn-fitted joint transport posterior of the target average effect: closed.

    ``sources`` are source trials (mappings), ``target`` the target covariate sample and
    ``features`` the covariate names (the certified standardizers). ``basis_degree`` is the
    polynomial degree of the basis in every covariate (``1..=6``). ``graph_class`` is
    ``"fixed_dag"`` (the only supported class), ``"admg"`` or ``"graph_posterior"``; an
    :class:`~antecedent.Admg` ``graph`` with a bidirected edge is an ADMG query and raises
    ``route_not_supported`` / ``learned_joint_transport.unsupported_graph``, as does a
    graph-posterior query. A basis degree outside ``1..=6``
    (``learned_joint_transport.invalid_basis``), a draw count outside ``1..=100000``
    (``learned_joint_transport.too_many_draws``) and a model above 256 parameters
    (``learned_joint_transport.too_many_parameters``) raise ``invalid_argument``; undeclared or
    overlapping source dependence, no source and no target law raise their own typed refusals.
    Every other request raises ``cell_not_licensed`` / ``learned_joint_transport.route_frozen``:
    the interval claims calibration, which is unmeasured. An internal candidate
    feature build additionally accepts an original ``identification`` and complete
    ``JointTransportSource``, ``JointTransportTarget`` and ``JointTransportPriors``
    declarations, returning a candidate-only posterior without an interval license.
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
    _non_negative("basis_degree", basis_degree)
    _non_negative("draws", draws)
    _non_negative("seed", seed)
    if (
        getattr(_native, "joint_transport_candidate", None) is not None
        and identification is not None
    ):
        if declared != "fixed_dag":
            raise CausalUnsupportedError(
                "learned_joint_transport.unsupported_graph", reason_code="route_not_supported"
            )
        if graph is not None:
            raise CausalValueError(
                "candidate graph is bound by identification; omit the redundant graph argument"
            )
        if priors is None:
            raise CausalTypeError("candidate transport needs explicit complete priors")
        return _candidate(
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
            basis_degree=basis_degree,
            draws=draws,
            seed=seed,
            learned=True,
            max_unsupported_mass=max_unsupported_mass,
            conflict_z_threshold=conflict_z_threshold,
        )
    try:
        _learned_joint_transport_closed(
            declared,
            dependence,
            varying,
            sharing,
            len(feature_names),
            basis_degree,
            len(sources),
            target is not None and (isinstance(target, JointTransportTarget) or len(target) > 0),
            draws,
        )
    except CausalUnsupportedError as error:
        if getattr(error, "reason_code", None) == "cell_not_licensed" and not getattr(
            error, "remedy", None
        ):
            error.remedy = _REMEDY
        raise
    raise _frozen()
