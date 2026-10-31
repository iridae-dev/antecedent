"""Model bundles refuse mechanism declarations that do not match the graph."""

from __future__ import annotations

import pytest
from antecedent import model

_NAN = float("nan")


@pytest.mark.parametrize(
    ("variables", "edges", "mechanisms"),
    [
        (["a", "b"], [(0, 1)], [("linear_gaussian", 0.0, None, _NAN)]),
        (["a", "b"], [(0, 1)], [("constant", 0.0, None, None)]),
        (["a", "b"], [(0, 1)], [("constant", 0.0, None, None), ("mystery", 0.0, None, None)]),
        (
            ["a", "b"],
            [(0, 1)],
            [("constant", 0.0, None, None), ("linear_gaussian", 0.0, [1.0, 2.0], 1.0)],
        ),
        (
            ["a", "b"],
            [(0, 1)],
            [("constant", 0.0, None, None), ("linear_gaussian", 0.0, [_NAN], 1.0)],
        ),
        (
            ["a", "b"],
            [(0, 1)],
            [("constant", 0.0, None, None), ("linear_gaussian", 0.0, [1.0], 0.0)],
        ),
        (
            ["a", "b"],
            [(0, 1)],
            [("constant", 0.0, None, None), ("linear_gaussian", 0.0, [1.0], None)],
        ),
        (["a", "b"], [], [("constant", _NAN, None, None), ("vacant", None, None, None)]),
    ],
)
def test_encode_model_bundle_refuses_inconsistent_model(variables, edges, mechanisms):
    with pytest.raises(ValueError):
        model.encode_model_bundle(variables, edges, mechanisms)


def test_consistent_model_still_encodes():
    mechanisms = [("constant", 1.5, None, None), ("linear_gaussian", 0.5, [2.0], 0.3)]
    names, _, slots = model.decode_model_bundle(
        bytes(model.encode_model_bundle(["a", "b"], [(0, 1)], mechanisms))
    )
    assert list(names) == ["a", "b"] and slots == 2
