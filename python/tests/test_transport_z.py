"""The registered zTR specialization stays sound, incomplete and point-only."""

import json

import pytest
from antecedent import Admg
from antecedent.transport import advanced as transport


def fixture(empirical=False):
    names = ["w", "z", "x", "y"]
    graph = Admg.from_edges(
        names,
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "identified"

    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes = (
        transport.EvidenceRegime(
            "do_z_0",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": 0.0},
            measured=names,
        ),
        transport.EvidenceRegime(
            "do_z_1",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": 1.0},
            measured=names,
        ),
    )
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=regimes,
        bindings=(
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
            transport.RegimeBinding(
                "do_z_1",
                "snapshot_z1",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    probabilities = []
    counts = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                p = (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                probabilities.append(p)
                counts.append(round(p * 10_000))
    if empirical:
        total = sum(counts)
        probabilities = [count / total for count in counts]
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
        empirical_counts=tuple(counts) if empirical else None,
    )
    prepared = stage.prepare_exact(catalog, (law,), {"x": 0.0})
    return graph, stage, catalog, prepared, (law,)


@pytest.mark.parametrize("empirical", [False, True])
def test_z_transport_prepared_native_route_matches_independent_truth(empirical):
    _, _, _, prepared, laws = fixture(empirical)
    result = json.loads(prepared.estimate())
    assert result["status"] == "available"
    assert result["scope"] == "registered_surrogate_z_transport_sound_incomplete"
    assert result["interval"] == {"available": False, "reason": "no_interval_reported"}
    true_mass = sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )
    assert true_mass == pytest.approx(0.2, abs=0.002 if empirical else 1e-12)
    prepared.refresh(laws)
    refreshed = json.loads(prepared.estimate())
    assert refreshed["probabilities"] == pytest.approx(result["probabilities"])


def test_z_transport_stage_refuses_unregistered_selection_graph():
    graph = Admg.from_edges(
        ["w", "z", "x", "y"],
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    stage = transport.identify_z_transport(graph=graph, query=query)
    assert stage.outcome == "not_certified"
    assert stage.reason == "z_transport.outside_registered_surrogate_graph"
    with pytest.raises(ValueError, match="zTR refused"):
        stage.prepare_exact(
            None,
            (),
            {},
        )
