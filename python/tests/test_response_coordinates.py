"""Response values are identified by scientific descriptor, not grid position."""

from __future__ import annotations

import antecedent as ac
import pytest
from antecedent import external
from antecedent.errors import CausalValueError
from antecedent.results.coordinates import response_coordinates

GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]


def _ident():
    return ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))


def test_helper_matches_external_spec_quantities():
    ident = _ident()
    spec = external.response(ident, outcome_units="mmHg", population="target")
    derived = response_coordinates(ident.query, outcome_units="mmHg", population="target")
    assert derived == spec.quantities


def test_units_are_required():
    ident = _ident()
    with pytest.raises(CausalValueError, match="outcome_units"):
        response_coordinates(ident.query, outcome_units=" ")


def test_descriptors_are_distinct_per_dose_and_never_positional():
    derived = response_coordinates(_ident().query, outcome_units="mmHg")
    assert [q.regime_id for q in derived] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    identities = {(q.variable_id, q.regime_id, q.functional_id) for q in derived}
    assert len(identities) == len(GRID)
    assert {q.role for q in derived} == {"outcome"}
    assert {q.horizon for q in derived} == {0}
