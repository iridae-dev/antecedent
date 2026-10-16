"""The supplied transport-grid equation is available through both public APIs."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.transport import advanced as transport

from test_transport_interference_execution import _certified_direct_identification


def test_grid_matches_the_rust_hand_calculated_example():
    result = transport.estimate_augmented_response_grid(
        _certified_direct_identification(),
        outcome=[1.0, 3.0, 0.0, 0.0],
        trial=[True, True, False, False],
        selection_probability=[0.5] * 4,
        grid=[0.0, 1.0],
        observed_regression=[1.0, 3.0, 0.0, 0.0],
        target_regression=np.array([[1.0] * 4, [3.0] * 4]),
        response_weight=np.ones((2, 4)),
    )
    assert result.mean == (1.0, 3.0)
    assert result.source_effective_sample_size == (2.0, 2.0)
    assert result.selection_overlap.effective_sample_size == 2
    assert result.to_rows()[1]["grid"] == 1
    assert result.claim == "point_only"


def test_signed_grid_local_weights_and_grid_major_layout():
    result = transport.estimate_augmented_response_grid(
        _certified_direct_identification(),
        outcome=[1.0, 3.0, 0.0, 0.0],
        trial=[True, True, False, False],
        selection_probability=[0.5] * 4,
        grid=[5.0, 2.0],
        observed_regression=[0.0] * 4,
        target_regression=[[10.0] * 4, [20.0] * 4],
        response_weight=[[1.0, -0.25, 9.0, 9.0], [1.0, 1.0, 9.0, 9.0]],
    )
    assert result.grid == (5.0, 2.0)
    assert result.mean == (10.125, 22.0)


@pytest.mark.parametrize(
    "changed",
    [
        {"target_regression": np.ones((4, 2))},
        {"response_weight": [1.0] * 8},
        {"observed_regression": [0.0]},
        {"selection_probability": [0.0] * 4},
        {"target_regression": [[np.nan] * 4] * 2},
    ],
)
def test_invalid_shapes_and_numerics_are_refused(changed):
    kwargs = dict(
        outcome=[1.0, 3.0, 0.0, 0.0],
        trial=[True, True, False, False],
        selection_probability=[0.5] * 4,
        grid=[0.0, 1.0],
        observed_regression=[1.0, 3.0, 0.0, 0.0],
        target_regression=[[1.0] * 4, [3.0] * 4],
        response_weight=np.ones((2, 4)),
    )
    kwargs.update(changed)
    with pytest.raises(ant.CausalError):
        transport.estimate_augmented_response_grid(_certified_direct_identification(), **kwargs)


def test_certificate_for_another_equation_is_refused():
    graph = ant.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
    )
    certificate = transport.identify(graph=graph, query=query)
    assert isinstance(certificate.formula, transport.RecursiveFactorizationFormula)
    with pytest.raises(ant.CausalError):
        transport.estimate_augmented_response_grid(
            certificate,
            outcome=[1.0, 3.0, 0.0, 0.0],
            trial=[True, True, False, False],
            selection_probability=[0.5] * 4,
            grid=[0.0, 1.0],
            observed_regression=[1.0, 3.0, 0.0, 0.0],
            target_regression=[[1.0] * 4, [3.0] * 4],
            response_weight=np.ones((2, 4)),
        )


@pytest.mark.parametrize(
    "changed", [{"outcome": ["bad", 3.0, 0.0, 0.0]}, {"trial": [None, True, False, False]}]
)
def test_array_conversion_errors_name_the_scientific_field(changed):
    arguments = dict(
        outcome=[1.0, 3.0, 0.0, 0.0],
        trial=[True, True, False, False],
        selection_probability=[0.5] * 4,
        grid=[0.0],
        observed_regression=[0.0] * 4,
        target_regression=[[1.0] * 4],
        response_weight=[[1.0] * 4],
    )
    arguments.update(changed)
    with pytest.raises(ant.errors.CausalValueError, match=next(iter(changed))):
        transport.estimate_augmented_response_grid(_certified_direct_identification(), **arguments)
