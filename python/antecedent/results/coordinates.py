"""Scientific coordinate descriptors for the values of a response curve.

A response value is identified by its :class:`ScientificQuantity`, never by its
grid position or a display label. This derives the descriptors exactly as
``antecedent.external.response`` does, so the two can never disagree.
"""

from __future__ import annotations

from typing import Any

from ..errors import CausalValueError
from ..external import _query_quantities
from ..joint_distribution import ScientificQuantity


def response_coordinates(
    query: Any, *, outcome_units: str, population: str = "target", transform: str = "identity"
) -> tuple[ScientificQuantity, ...]:
    """Return one ``do(treatment=dose)`` descriptor per grid point of ``query``.

    ``outcome_units`` is required: units are never inferred or converted.
    """
    if outcome_units is None or not str(outcome_units).strip():
        raise CausalValueError("outcome_units= is required: units are never inferred or converted")
    return _query_quantities(
        query, outcome_units=outcome_units, population=population, transform=transform
    )


__all__ = ["response_coordinates"]
