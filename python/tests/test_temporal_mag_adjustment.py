"""Stationary mixed temporal graphs retain latent confounding and lag coordinates."""

import json
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest

PIN = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "conformance/estimate/temporal_class_envelope/latent.json"
    ).read_text()
)


def case(mixed=False):
    indices = np.arange(PIN["n"])
    z = (indices % 2).astype(float)
    t = 0.3 + 0.4 * z + 0.05 * np.sin(indices * 0.017)
    y = np.zeros(len(t))
    y[1:] = 1 + 2 * t[:-1] + 0.5 * z[:-1]
    data = dict(t=t, y=y, z=z, r=((indices // 2) % 2).astype(float))
    edges = [tuple(edge) for edge in PIN["marked_edges"]]
    if mixed:
        data.update(s=np.sin(indices), v=np.cos(indices))
        edges.append(("s", 0, "v", 0, "circle", "circle"))
    graph = ant.graph.TemporalPag.from_marked_lagged_edges(list(data), edges)
    return data, graph


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("sustained", [False, True])
@pytest.mark.parametrize("refute", [False, "cheap", "full"])
def test_temporal_mag_effect_and_prepared_reuse(accepted, mixed, sustained, refute):
    data, graph = case(mixed)
    if accepted:
        graph = ant.AcceptedGraph.from_graph(graph)
    query = (
        ant.SustainedEffect("t", "y", window=(-1, -1), treatment_lag=1)
        if sustained
        else ant.PulseEffect("t", "y", treatment_lag=1)
    )
    identified = ant.identify(graph=graph, query=query)
    cert = identified.certificate
    assert len(cert["cases"]) == (3 if mixed else 1)
    for completion in cert["cases"]:
        assert completion["graph"]["kind"] == "temporal_mag"
        assert [
            (item["name"], item["offset"]) for item in completion["adjustment_coordinates"][0]
        ] == [("z", -1)]
    kwargs = dict(graph=graph, query=query, refute=refute, bootstrap=0, seed=7)
    fresh = ant.analyze(data, **kwargs)
    prepared = ant.estimation.PreparedAnalysis.prepare(data, **kwargs)
    click = prepared.estimate(data, seed=7)
    for result in [fresh, click]:
        assert result.certificate is not None
        assert len(result.certificate["cases"]) == len(cert["cases"])
        for expected, actual in zip(cert["cases"], result.certificate["cases"], strict=True):
            assert actual["adjustment_coordinates"] == expected["adjustment_coordinates"]
            assert actual["indexer"] == expected["indexer"]
            assert actual["identification"]["arena"] == expected["identification"]["arena"]
        if not mixed:
            columns = ant.handoff.econml(result).columns(data)
            np.testing.assert_array_equal(columns["T"], data["t"][:-1])
            np.testing.assert_array_equal(columns["W"][:, 0], data["z"][:-1])
            np.testing.assert_array_equal(columns["Y"], data["y"][1:])
        assert result.effect == pytest.approx(PIN["effect"], abs=PIN["absolute_tolerance"])
    if not mixed:
        cols = ant.handoff.econml(identified).columns(data)
        np.testing.assert_array_equal(cols["T"], data["t"][:-1])
        np.testing.assert_array_equal(cols["W"][:, 0], data["z"][:-1])
        np.testing.assert_array_equal(cols["Y"], data["y"][1:])


def test_fully_oriented_temporal_pag_keeps_mag_certificate():
    data, _ = case()
    edges = [tuple(edge) for edge in PIN["marked_edges"]]
    edges = [
        (*edge[:4], "tail", "arrow") if edge[4:] == ("arrow", "arrow") else edge for edge in edges
    ]
    graph = ant.graph.TemporalPag.from_marked_lagged_edges(list(data), edges)
    query = ant.PulseEffect("t", "y", treatment_lag=1)
    kwargs = dict(graph=graph, query=query, refute=False, bootstrap=0, seed=7)
    fresh = ant.analyze(data, **kwargs)
    prepared = ant.estimation.PreparedAnalysis.prepare(data, **kwargs)
    for result in (fresh, prepared.estimate(data, seed=7)):
        assert result.effect == pytest.approx(2.0, abs=1e-6)
        assert result.certificate["graph_class"] == "TemporalPag"
        assert result.certificate["cases"][0]["graph"]["kind"] == "temporal_mag"
        assert not any("completed_to_dag" in diagnostic for diagnostic in result.diagnostics)
        columns = ant.handoff.econml(result).columns(data)
        np.testing.assert_array_equal(columns["W"][:, 0], data["z"][:-1])
