"""Assumption range for a violated transport premise, from a transported response.

A transported response is only as good as the premise the transport rests on. This module
builds, from the response's own point value, coordinate and receipt, the *assumption range* of
that value for a declared violation of the premise, and the sealed
:class:`~antecedent.sensitivity_decision.SensitivityArtifact` of a decision between acting on the
transported quantity and keeping a reference value::

    from antecedent.transport import sensitivity

    result = sensitivity.transport_sensitivity(
        response,
        family="additive_shift",
        bound=0.5,
        grid_points=3,
        reference=reference_quantity,
        reference_value=1.75,
        route="cohort_decision",
    )
    result.lower, result.upper          # the range at each declared grid point
    result.reference_reached_at         # the smallest shift that reaches the reference
    result.artifact                     # decide / export / consume through sensitivity_decision

What the range is: an assumption range at the declared grid points only. The bound is the
caller's declaration of how far the premise may fail; nothing in the data licenses it. It is never
a confidence interval, a credible interval or a posterior, no probability attaches to a shift,
and a sampling interval is withheld (``transport_sensitivity.interval_not_composed``).

**Lifecycle.** Outcome-only data cannot move an assumption range: the range is a function of the
point value, the declared family and bound, and the transport identity. New data produces a new
transported response; calling :func:`transport_sensitivity` again keeps the declared family,
bound and grid and recomputes the baseline and every grid number. The range is bound to the
response's target population and quantity through its identity: the same family applied to a
response about another population is a different range.

Refused: a law-valued response (``transport_sensitivity.law_response_not_scalar``), a structural
premise (``transport_sensitivity.no_scalar_family``: whether an edge holds in the target is not a
shift of a number), a response whose route is ``declared`` with no ``route`` named
(``transport_sensitivity.route_undeclared``), a coordinate outside the response, a proportional
shift of a zero point, and a reference in another population or units.

Rust owns the range, the artifact and every refusal. Refusals raise
:class:`~antecedent.transported_response.TransportedResponseRefusal` with a registered
``reason_code``, a ``transport_sensitivity.*`` ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import _native
from .._refusal import raise_refusal
from .._scientific import ScientificResult
from .._validate import choice, count, real
from ..errors import CausalTypeError
from ..joint_distribution import ScientificQuantity
from ..transported_response import TransportedResponse, TransportedResponseRefusal

__all__ = ["TransportSensitivity", "transport_sensitivity"]

ViolationFamily = Literal["additive_shift", "proportional_shift"]
_FAMILIES = ("additive_shift", "proportional_shift")


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportSensitivity(ScientificResult):
    """The assumption range of a transported point and its sealed decision artifact.

    ``lower`` and ``upper`` are the range of the point value at each ``grid`` shift (from ``0``
    to the declared ``bound``); ``range_at_bound`` is its last entry. ``reference_reached_at``
    is the smallest shift at which the range reaches ``reference_value``, or ``None`` when the
    shift needed exceeds the declared bound. ``statement`` must accompany the range.
    ``identity`` binds route, premise, family, populations, derivation and every grid number.
    """

    route: str
    premise: str
    family: str
    bound: float
    point_value: float
    grid: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    range_at_bound: tuple[float, float]
    reference_value: float
    reference_reached_at: float | None
    statement: str
    identity: str
    _artifact: Any = field(repr=False, compare=False, default=None)

    @property
    def artifact(self) -> Any:
        """The sealed :class:`~antecedent.sensitivity_decision.SensitivityArtifact`: decide,
        export and consume it through :mod:`antecedent.sensitivity_decision`."""
        return self._artifact

    @property
    def label(self) -> str:
        """The standing of the range: an assumption range, never a probability statement."""
        return "assumption_range"


def transport_sensitivity(
    response: TransportedResponse,
    *,
    family: ViolationFamily,
    bound: float,
    grid_points: int,
    reference: ScientificQuantity,
    reference_value: float,
    coordinate: int = 0,
    route: str | None = None,
) -> TransportSensitivity:
    """The assumption range of coordinate ``coordinate`` of a transported point.

    ``family`` is ``additive_shift`` (the quantity lies within ``bound`` of the point value, in
    its units) or ``proportional_shift`` (within ``bound * |value|``; refused at a point of
    zero). ``grid_points`` (at least two) points are placed on ``[0, bound]``. ``route`` names
    the transport route (``cohort_decision``, ``mechanism_transport``, ...); omitted, it is
    taken from the response's own route, and a ``declared`` route must be named.
    ``reference`` is the quantity of the alternative the decision keeps (same population and
    units) and ``reference_value`` its value.
    """
    if not isinstance(response, TransportedResponse):
        raise CausalTypeError("response must be a TransportedResponse")
    if not isinstance(reference, ScientificQuantity):
        raise CausalTypeError("reference must be a ScientificQuantity")
    choice(family, "family", _FAMILIES)
    artifact, summary, refusal = _native.transport_sensitivity_build(
        response._handle,
        family,
        real(bound, "bound"),
        count(grid_points, "grid_points", minimum=0),
        json.dumps(reference._wire()),
        real(reference_value, "reference_value"),
        coordinate=count(coordinate, "coordinate", minimum=0),
        route=route,
    )
    raise_refusal(refusal, TransportedResponseRefusal)
    assert artifact is not None and summary is not None
    from ..sensitivity_decision import SensitivityArtifact

    info: Mapping[str, Any] = json.loads(summary)
    low, high = info["range_at_bound"]
    reached = info["reference_reached_at"]
    return TransportSensitivity(
        route=str(info["route"]),
        premise=str(info["premise"]),
        family=str(info["family"]),
        bound=float(info["bound"]),
        point_value=float(info["point_value"]),
        grid=tuple(float(v) for v in info["grid"]),
        lower=tuple(float(v) for v in info["lower"]),
        upper=tuple(float(v) for v in info["upper"]),
        range_at_bound=(float(low), float(high)),
        reference_value=float(info["reference_value"]),
        reference_reached_at=None if reached is None else float(reached),
        statement=str(info["statement"]),
        identity=str(info["identity"]),
        _artifact=SensitivityArtifact(artifact),
    )
