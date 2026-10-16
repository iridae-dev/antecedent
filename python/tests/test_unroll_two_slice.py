"""Unrolling a lagged temporal template into the explicit two-slice selection diagram.

Oracle: the template ``l -> a``, ``a -> y``, ``l -> y`` within a step and ``a[t-1] -> l[t]``
across steps, replicated over two steps with the parents before the first step dropped, gives
exactly the edges listed below by hand. The outcome of step 1 is an ordinary covariate of step 2.
"""

from __future__ import annotations

import pytest
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.graph import TemporalDag
from antecedent.transport import advanced as transport

TEMPLATE = TemporalDag.from_lagged_edges(
    ["l", "a", "y"],
    [("l", 0, "a", 0), ("a", 0, "y", 0), ("l", 0, "y", 0), ("a", 1, "l", 0)],
)


def _edges(graph) -> set[tuple[str, str]]:
    return {(p, c) for p in graph.nodes() for c in graph.children(p)}


def test_a_lagged_template_unrolls_into_two_steps_with_places_in_time() -> None:
    unrolled = transport.unroll_two_slice(
        TEMPLATE,
        covariates=["l"],
        action="a",
        outcome="y",
        baseline=["b"],
        baseline_effects=[("b", "l")],
    )
    assert _edges(unrolled.graph) == {
        ("l_1", "a_1"), ("a_1", "y_1"), ("l_1", "y_1"),
        ("l_2", "a_2"), ("a_2", "y_2"), ("l_2", "y_2"),
        ("a_1", "l_2"),
        ("b", "l_1"), ("b", "l_2"),
    }  # fmt: skip
    assert unrolled.baseline == ("b",)
    assert unrolled.covariates == (("l_1",), ("l_2", "y_1"))
    assert unrolled.actions == ("a_1", "a_2")
    assert unrolled.outcome == "y_2"


def test_latent_confounding_is_added_between_named_unrolled_coordinates() -> None:
    unrolled = transport.unroll_two_slice(
        TEMPLATE, covariates=["l"], action="a", outcome="y", latent_pairs=[("a_1", "y_2")]
    )
    assert unrolled.graph.bidirected_neighbors("a_1") == ["y_2"]
    assert unrolled.baseline == ()


def test_the_unrolling_builds_a_spec_once_every_coordinate_is_declared() -> None:
    unrolled = transport.unroll_two_slice(TEMPLATE, covariates=["l"], action="a", outcome="y")
    coordinates = [transport.VariableCoordinate(name, "binary") for name in unrolled.names]
    spec = unrolled.to_spec(coordinates, selections=["l_2"])
    assert spec.actions == ("a_1", "a_2") and spec.selections == ("l_2",)
    with pytest.raises(CausalValueError):
        unrolled.to_spec(coordinates[:-1])


def test_roles_outside_the_template_and_other_types_are_refused() -> None:
    with pytest.raises(CausalValueError, match="temporal_transport.invalid_spec"):
        transport.unroll_two_slice(TEMPLATE, covariates=["l", "z"], action="a", outcome="y")
    with pytest.raises(CausalValueError, match="temporal_transport.invalid_spec"):
        transport.unroll_two_slice(
            TEMPLATE,
            covariates=["l"],
            action="a",
            outcome="y",
            baseline_effects=[("b", "l")],
        )
    with pytest.raises(CausalTypeError):
        transport.unroll_two_slice(object(), covariates=["l"], action="a", outcome="y")
