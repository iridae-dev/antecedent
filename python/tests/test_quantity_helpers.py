"""Quantity selection preserves scientific meaning and resolves close grids safely."""

from types import SimpleNamespace

import pytest
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import QuantityCondition, ScientificQuantity


def _response():
    quantities = tuple(
        ScientificQuantity.outcome("y", units="mmHg", population="target", regime=f"do(a={dose!r})")
        for dose in (1.0, 1.0 + 1e-12)
    )
    return SimpleNamespace(
        quantities=quantities,
        response=SimpleNamespace(points=((1.0, 2.0), (1.0 + 1e-12, 3.0))),
        response_coordinates=lambda **kwargs: quantities,
    )


def test_close_response_grid_preserves_exact_coordinate_and_refuses_ambiguous_rounding():
    response = _response()
    assert (
        ScientificQuantity.from_response_dose(response, 1.0 + 1e-12, outcome_units="mmHg")
        == response.quantities[1]
    )
    with pytest.raises(CausalValueError, match="ambiguously"):
        ScientificQuantity.from_response_dose(response, 1.0 + 5e-13, outcome_units="mmHg")
    with pytest.raises(CausalTypeError, match="number"):
        ScientificQuantity.from_response_dose(response, True, outcome_units="mmHg")


@pytest.mark.parametrize(
    "override", [{"outcome_units": "Pa"}, {"population": "other"}, {"transform": "log"}]
)
def test_retained_response_coordinates_refuse_conflicting_declarations(override):
    with pytest.raises(CausalValueError, match="conflict"):
        ScientificQuantity.from_response(_response(), **{"outcome_units": "mmHg", **override})


def test_quantity_declaration_validates_role_and_conditioning_identity():
    options = {"units": "mmHg", "population": "target", "regime": "observational"}
    with pytest.raises(CausalValueError, match="role"):
        ScientificQuantity.of("unsupported", "y", **options)
    with pytest.raises(CausalTypeError, match="QuantityCondition"):
        ScientificQuantity.outcome("y", conditioning="a=1", **options)
    with pytest.raises(CausalValueError, match="identities"):
        ScientificQuantity.outcome("y", conditioning=(QuantityCondition("", "1"),), **options)
    condition = QuantityCondition("a", "1")
    assert ScientificQuantity.outcome("y", conditioning=(condition,), **options).conditioning == (
        condition,
    )
