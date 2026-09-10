"""MAG adjustment needs visibility even when every endpoint is oriented.

Positive cells use R -> T -> Y, where R witnesses T -> Y. The linear
outcome law has effect 2 and response 1 + 2t. An optional independent
S o-o V component exercises retained mixed completions without changing
that functional. Completion counts and retained mass are checked explicitly.
"""

import antecedent as ant
import numpy as np
import pytest


def _case(mixed=False):
    n = 800
    t = np.linspace(0.05, 0.95, n)
    data = {"t": t, "y": 1 + 2 * t, "r": (np.arange(n) % 2).astype(float)}
    edges = [("r", "t", "tail", "arrow"), ("t", "y", "tail", "arrow")]
    if mixed:
        data.update(s=np.sin(np.arange(n)), v=np.cos(np.arange(n)))
        edges.append(("s", "v", "circle", "circle"))
    return data, ant.Pag.from_marked_edges(list(data), edges)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("bayesian", [False, True])
@pytest.mark.parametrize("kind", ["average", "conditional", "response", "curve"])
def test_visible_mag_known_linear_effect(mixed, accepted, bayesian, kind):
    data, graph = _case(mixed)
    if accepted:
        graph = ant.AcceptedGraph.from_graph(graph)
    certificate = ant.identify(graph=graph, query=ant.AverageEffect("t", "y")).certificate
    assert len(certificate["cases"]) == (3 if mixed else 1)
    assert certificate["unidentified_weight"] == 0.0
    query = {
        "average": lambda: ant.AverageEffect("t", "y"),
        "conditional": lambda: ant.ConditionalEffect("t", "y", modifier="r"),
        "response": lambda: ant.InterventionResponse(
            "y", intervention=ant.intervention.Set("t", 0.5)
        ),
        "curve": lambda: ant.ResponseCurve("t", "y", grid=[0.25, 0.75]),
    }[kind]()
    kwargs = {"inference": ant.Bayesian(prior_scale=100.0)} if bayesian else {}
    result = ant.analyze(
        data, graph=graph, query=query, refute=False, bootstrap=0, seed=7, **kwargs
    )
    assert result.certificate is not None
    assert len(result.certificate["cases"]) == (3 if mixed else 1)
    assert result.certificate["unidentified_weight"] == 0.0
    assert result.certificate["graph_class"] == "Pag"
    columns = ant.handoff.econml(result).columns(data)
    np.testing.assert_array_equal(columns["T"], data["t"])
    np.testing.assert_array_equal(columns["Y"], data["y"])
    if kind in ("average", "conditional"):
        assert result.effect == pytest.approx(2.0, abs=0.05)
    else:
        expected = [1.5, 2.5] if kind == "curve" else [2.0]
        np.testing.assert_allclose(
            np.asarray(result.response.values).reshape(-1), expected, atol=0.05
        )


@pytest.mark.parametrize("conditional", [False, True])
def test_invisible_mag_has_no_adjustment_certificate(conditional):
    graph = ant.Pag.from_marked_edges(["t", "y", "r"], [("t", "y", "tail", "arrow")])
    query = (
        ant.ConditionalEffect("t", "y", modifier="r")
        if conditional
        else ant.AverageEffect("t", "y")
    )
    result = ant.identify(graph=graph, query=query)
    assert not result
    assert result.status == "NotIdentified"
    assert not any(case["adjustment_coordinates"] for case in result.certificate["cases"])
