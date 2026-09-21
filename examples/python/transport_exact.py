"""Identify, prepare, refresh, and independently consume an exact transported law."""

from dataclasses import replace

from antecedent import Admg, load, prepare, transport


def main() -> None:
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial",
                "source",
                kind="experimental",
                interventions=["x"],
                measured=["y"],
            ),
        ]
    )
    law = transport.ExactDiscreteLaw(
        "source",
        "trial",
        (("y", (0.0, 1.0)),),
        (0.2, 0.8),
        "source-v1",
        interventions=(("x", 1.0),),
    )
    data = transport.ExactTransportData((law,))
    study = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    inspection = study.inspect()  # Metadata only: no evaluation or callbacks.
    result = study.estimate()
    assert abs(result.mean("y") - 0.8) < 1e-12
    assert result.uncertainty is None

    replacement = transport.ExactTransportData(
        (replace(law, probabilities=(0.3, 0.7), snapshot_identity="source-v2"),)
    )
    assert study.preview_transform("compatible_data_replace")["refused"] == "false"
    study.replace_snapshot(replacement)  # Reuse identification; invalidate execution claims.
    assert study.inspect().identification_id == inspection.identification_id
    assert study.inspect().data_snapshot_id != inspection.data_snapshot_id
    refreshed = study.refresh(replacement)  # Explicit, atomic execution.
    consumed = load(study.export())  # Checks proof, bindings and embedded-law numerical truth.
    assert consumed.probabilities == refreshed.probabilities
    assert not consumed.inspect().uncertainty.available
    print(consumed)


if __name__ == "__main__":
    main()
