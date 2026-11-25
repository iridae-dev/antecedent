"""A transport result projects to a TransportedResponse only when its receipt is already there.

The success path, including the receipt identity, is the Rust test
``a_transport_receipt_projects_only_when_the_result_already_carries_it``: Python cannot attach
the ``receipt.*`` diagnostic fields. Here the binding refuses a bare analysis result and a
transport result that does not carry them, and ``evaluate_transported`` rejects the bare result.
"""

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


def _transport():
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
    return require_success(
        ant.analyze(
            data,
            graph=ant.Admg.from_edges(names, [("a", "y")]),
            query=query,
            refute="none",
        )
    )


def test_a_bare_analysis_result_is_not_a_transported_response() -> None:
    result = _ate()
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        tr.project_transported_response(result)
    assert caught.value.detail == reason_details.transported_decision.NOT_TRANSPORTED

    with pytest.raises(CausalTypeError, match="TransportedResponse"):
        tr.evaluate_transported(_contract(), result, analysis=result)


def test_a_transport_result_without_receipt_fields_refuses() -> None:
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        tr.project_transported_response(_transport())
    assert (
        caught.value.detail == reason_details.transported_decision.TRANSPORT_IDENTITY_INCOMPLETE
    )
