"""Joint-law declarations refuse wrongly typed members instead of defaulting them."""

from __future__ import annotations

import json

import pytest
from antecedent import _native


def _mixture(**changes):
    component = {"id": "a", "evidence_id": "e", "weight": 1.0, "mean": 0.0, "variance": 1.0}
    component.update(changes)
    return json.dumps([component])


@pytest.mark.parametrize(
    "changes",
    [{"weight": "1.0"}, {"mean": "0"}, {"variance": [1.0]}, {"id": 5}, {"evidence_id": 7}],
)
def test_mixture_member_of_wrong_type_is_an_invalid_request(changes):
    claim, refusal = _native.joint_law_mixture(_mixture(**changes))
    assert claim is None
    assert json.loads(refusal)["detail"] == "joint_law_propagation.invalid_request"


def test_linear_input_role_must_be_a_string():
    inputs = json.dumps([{"id": "a", "evidence_id": "e", "role": 3, "mean": 1.0}])
    claim, refusal = _native.joint_law_linear_moments(
        "sum", inputs, [1.0], [1.0], "declared_independent"
    )
    assert claim is None
    assert "`role` must be a string" in json.loads(refusal)["message"]
