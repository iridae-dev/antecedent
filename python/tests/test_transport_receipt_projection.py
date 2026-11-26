"""Only a transport result carrying its native execution receipt can be projected."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent import transported_response as tr
from antecedent.errors import CausalTypeError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

from _transport_fixtures import effect


def _contract() -> decision.Contract:
    quantity = effect()
    return decision.Contract(
        actions=(
            decision.Action("treat", inputs=(quantity,), utility=decision.x(0)),
            decision.Action("wait", inputs=(quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="clinic_b",
    )


def _ate():
    data = {
        "t": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        "y": np.array([0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0]),
    }
    return require_success(
        ant.analyze(
            data,
            graph=[("t", "y")],
            query=ant.AverageEffect(treatment="t", outcome="y"),
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
        )
    )


def _transport(*, graph_posterior: bool = False):
    names = ["a", "y", "trial", "s", "e"]
    data = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "y": np.array([3.0, 1.0, 0.0, 0.0]),
        "trial": np.array([1.0, 1.0, 0.0, 0.0]),
        "s": np.array([0.5, 0.5, 0.5, 0.5]),
        "e": np.array([0.5, 0.5, 0.5, 0.5]),
    }
    query = transport.TransportQuery(
        ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
        trial="trial",
        selection_probability="s",
        treatment_probability="e",
    )
    graph = ant.Admg.from_edges(names, [("a", "y")])
    structure = (
        {"discovery": ant.discovery.GraphPosterior.from_graphs(names, [1.0], [graph])}
        if graph_posterior
        else {"graph": graph}
    )
    return require_success(ant.analyze(data, query=query, refute="none", **structure))


def test_a_bare_analysis_result_is_not_a_transported_response() -> None:
    result = _ate()
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        tr.project_transported_response(result)
    assert caught.value.detail == reason_details.transported_decision.NOT_TRANSPORTED

    with pytest.raises(CausalTypeError, match="TransportedResponse"):
        tr.evaluate_transported(_contract(), result, analysis=result)


def test_direct_trial_transport_projects_its_execution_receipt() -> None:
    projected = tr.project_transported_response(_transport())
    assert projected.means == pytest.approx((2.0,))
    assert projected.coordinates[0].regime_id == "contrast:do(a=1)-do(a=0)"
    assert projected.receipt.source_populations == ("trial",)
    assert projected.receipt.target_population == "target"
    assert projected.receipt.support == "extrapolative"
    assert projected.receipt.representation == "point_only"


def test_graph_posterior_transport_without_trial_receipt_refuses() -> None:
    result = _transport(graph_posterior=True)
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        tr.project_transported_response(result)
    assert caught.value.reason_code == "transport_missing_evidence"
    assert caught.value.detail == reason_details.transported_decision.TRANSPORT_IDENTITY_INCOMPLETE
