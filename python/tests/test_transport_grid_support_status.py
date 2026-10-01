"""A transport grid keeps missing evidence apart from a support failure."""

import pytest
from antecedent import Admg, ResponseCurve, analyze, load, transport
from antecedent._transport_results import TransportGridPoint, grid_support_status
from antecedent.results import CausalResponseView
from antecedent.results.response import SupportReport
from antecedent.transport import advanced

from test_transport_meta_grid import fixture


def _curve():
    return transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source", kind="experimental", interventions=["x"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )


def _treated_law_only():
    # No law under do(x = 0): that coordinate lacks evidence.
    return transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.2, 0.8),
                "v1",
                interventions=(("x", 1.0),),
            ),
        )
    )


def test_support_vocabulary_is_extended_additively():
    for status in (
        "supported",
        "weak_overlap",
        "extrapolative",
        "outside_empirical_support",
        "missing_evidence",
    ):
        assert SupportReport(status=status, query_region={}).status == status
    with pytest.raises(ValueError, match="unknown support status"):
        SupportReport(status="support_failure", query_region={})


def test_grid_point_support_status_maps_each_native_status():
    assert grid_support_status("available") == "supported"
    assert grid_support_status("missing_evidence") == "missing_evidence"
    assert grid_support_status("support_failure") == "outside_empirical_support"
    # A payload written before points carried `support_status` reads through the map.
    legacy = TransportGridPoint({"at": {"x": 0.0}, "status": "support_failure"})
    assert legacy.support_status == "outside_empirical_support"


@pytest.mark.parametrize("statistical", [False, True])
def test_native_grid_points_report_missing_evidence(statistical):
    _, identified, catalog, data = fixture(missing=True, statistical=statistical)
    result = advanced.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=19
    ).estimate()
    assert result.points[0]["status"] == "missing_evidence"
    assert [point.support_status for point in result.points] == ["missing_evidence", "supported"]
    slot = result.inspect().support.payload["slot"]
    assert slot["empirical"] == (
        "1 executable; 1 unavailable (1 missing evidence, 0 support failure); "
        "population positivity assumed"
    )
    assert [point.support_status for point in load(result.export()).points] == [
        "missing_evidence",
        "supported",
    ]


def test_transported_curve_does_not_call_missing_evidence_a_support_failure():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    view = analyze(_treated_law_only(), graph=graph, query=_curve())
    assert isinstance(view, CausalResponseView)
    assert tuple(view.support.point_status) == ("missing_evidence", "supported")
    assert view.support.status == "missing_evidence"
