"""Measured target-effect inference for the known-variance learned joint model.

The standard producer returns native-authorized scalar inference only inside the
finite degree-two validation design. The complete original posterior remains an
unmeasured, model-conditional candidate; its law receives no blanket interval license.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .. import _native
from .._measured_inference import MeasuredInference
from .._native import learned_joint_transport_closed as _learned_joint_transport_closed
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ._impl import TransportIdentification, _non_negative
from ._joint_posterior import (
    JointTransportPriors,
    JointTransportSource,
    JointTransportTarget,
    _measured,
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
    level: float = 0.95,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MeasuredInference:
    """Return the measured target-effect scalar under the declared joint model.

    Requires the original identification and typed source, target and complete priors.
    Native execution checks the finite validation protocol (known variances, declared
    zero-mean Gaussian variance-1000 priors, degree two, fixed target design and .95
    scalar interval), derives the actual basis and resolves its attesting record.
    Nearby unsupported protocols refuse; evidence does not guarantee coverage for
    arbitrary data-generating laws. Original candidate artifacts remain unmeasured.
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
        getattr(_native, "joint_transport_measured", None) is not None
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
