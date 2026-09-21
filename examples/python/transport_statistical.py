"""Identify, prepare, refresh, and independently consume an empirical transported law."""
from antecedent import Admg, load, prepare, transport


def _sample(snapshot: str, counts: tuple[int, int]) -> transport.RegimeSample:
    y0, y1 = counts
    return transport.RegimeSample(
        "source",
        "trial",
        snapshot,
        {"y": [0.0] * y0 + [1.0] * y1},
        interventions=(("x", 1.0),),
    )


def main() -> None:
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph, transport.SelectionDiagram("source", "target", []),
        outcomes=["y"], treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("source", [
                transport.VariableCoordinate("x", "binary"),
                transport.VariableCoordinate("y", "binary"),
            ]),
            transport.Environment("target", [
                transport.VariableCoordinate("x", "binary"),
                transport.VariableCoordinate("y", "binary"),
            ]),
        ],
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["y"],
            ),
        ],
        bindings=[
            transport.RegimeBinding(
                "trial", "source-v1", sampling="independent", dependence="independent_studies",
            ),
        ],
        target_sampling="representative_sample",
    )
    data = transport.StatisticalTransportData(samples=(_sample("source-v1", (20, 80)),))
    study = prepare(
        data,
        query=transport.StatisticalTransportQuery(identified, catalog, {"x": 1.0}, bootstrap=39, seed=7),
    )
    inspection = study.inspect()
    assert inspection.uncertainty.available
    result = study.estimate()
    assert abs(result.mean("y") - 0.8) < 1e-12
    assert result.uncertainty is not None
    assert result.uncertainty["available"]
    assert result.uncertainty["row"]["interval_scope"] == "pointwise"

    replacement = transport.StatisticalTransportData(samples=(_sample("source-v2", (30, 70)),))
    assert study.preview_transform("compatible_data_replace")["refused"] == "false"
    study.replace_snapshot(replacement)
    assert study.inspect().identification_id == inspection.identification_id
    assert study.inspect().data_snapshot_id != inspection.data_snapshot_id
    refreshed = study.refresh(replacement)
    assert abs(refreshed.mean("y") - 0.7) < 1e-12
    consumed = load(study.export())
    assert consumed.probabilities == refreshed.probabilities
    assert consumed.uncertainty["available"]
    print(consumed)


if __name__ == "__main__":
    main()
