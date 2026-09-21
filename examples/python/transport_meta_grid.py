"""Two complementary experiments; exact finite responses and independent consume.

Run with the locally built antecedent package. No sampling-coverage claim.
"""

from antecedent import Admg, load, prepare, transport


def main():
    graph = Admg.from_edges(
        ["x", "z", "y"], [("x", "z"), ("z", "y")], bidirected=[("x", "z"), ("x", "y")]
    )
    variables = [transport.VariableCoordinate(name, "binary") for name in ("x", "z", "y")]
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("a", variables, selection_targets=["y"]),
            transport.Environment("b", variables, selection_targets=["z"]),
            transport.Environment("target", variables),
        ],
        regimes=[
            transport.EvidenceRegime(
                "x_trial", "a", kind="experimental", interventions=["x"], measured=["z"]
            ),
            transport.EvidenceRegime(
                "z_trial", "b", kind="experimental", interventions=["z"], measured=["y"]
            ),
        ],
    )
    proof = transport.identify_meta(
        graph, catalog, target="target", outcomes=["y"], treatments=["x"]
    )
    laws = []
    for population, regime, treatment, outcome, probabilities in [
        ("a", "x_trial", "x", "z", (0.2, 0.8)),
        ("b", "z_trial", "z", "y", (0.1, 0.9)),
    ]:
        for value, probability in enumerate(probabilities):
            laws.append(
                transport.ExactDiscreteLaw(
                    population,
                    regime,
                    ((outcome, (0.0, 1.0)),),
                    (1 - probability, probability),
                    "supplied-v1",
                    interventions=((treatment, float(value)),),
                )
            )
    data = transport.ExactTransportData(tuple(laws))
    study = prepare(
        data,
        query=transport.TransportResponseGridQuery(proof, catalog, ({"x": 0.0}, {"x": 1.0})),
    )
    print(study.inspect().identification)
    result = study.estimate()
    print([result.mean(i, "y") for i in range(2)])  # 0.26, 0.74
    print(result.contrast(1, 0, "y"))  # 0.48, no fabricated interval
    assert load(result.export()).points == result.points
    # Removing one intervention value retains a visible unsupported point.
    partial = transport.prepare_response_grid(
        proof,
        catalog,
        transport.ExactTransportData(tuple(laws[1:])),
        at=[{"x": 0.0}, {"x": 1.0}],
    ).estimate()
    assert partial.points[0]["status"] == "missing_evidence"
    assert partial.points[1]["status"] == "available"


if __name__ == "__main__":
    main()
