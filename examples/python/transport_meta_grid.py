"""Complementary sources plus a grid. Same verbs as the single-source examples."""

import antecedent as ant


def _slots(result, *, uncertainty: bool) -> None:
    report = result.inspect()
    assert report.identification.available
    assert report.assumptions.available
    assert report.support.summary
    assert report.uncertainty.available is uncertainty
    print(
        f"identification: {report.identification.summary}",
        f"support: {report.support.summary}",
        f"uncertainty: {report.uncertainty.summary}",
        f"assumptions: {report.assumptions.summary}",
        sep="\n",
    )


def main() -> None:
    graph = ant.Admg.from_edges(
        ["x", "z", "y"], [("x", "z"), ("z", "y")], bidirected=[("x", "z"), ("x", "y")]
    )
    evidence = ant.transport.Evidence(
        source=[
            ant.transport.Source(
                "a",
                kind="experimental",
                interventions=["x"],
                sampling="independent",
                selections=["y"],
            ),
            ant.transport.Source(
                "b",
                kind="experimental",
                interventions=["z"],
                sampling="independent",
                selections=["z"],
            ),
        ],
        target_sampling="representative_sample",
    )
    query = ant.transport.Transport(
        ant.ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=evidence,
    )
    laws = []
    for population, regime, treatment, outcome, probabilities in [
        ("a", "x_trial", "x", "z", (0.2, 0.8)),
        ("b", "z_trial", "z", "y", (0.1, 0.9)),
    ]:
        for value, probability in enumerate(probabilities):
            laws.append(
                ant.transport.ExactDiscreteLaw(
                    population,
                    regime,
                    ((outcome, (0.0, 1.0)),),
                    (1 - probability, probability),
                    "supplied-v1",
                    interventions=((treatment, float(value)),),
                )
            )
    data = ant.transport.ExactTransportData(tuple(laws))
    result = ant.analyze(data, graph=graph, query=query)
    _slots(result, uncertainty=False)
    assert result.answer.kind == "response"
    assert [round(row[0], 2) for row in result.response.values] == [0.26, 0.74]
    print(list(result.response.values))
    assert ant.load(result.export()) is not None

    partial = ant.analyze(
        ant.transport.ExactTransportData(tuple(laws[1:])),
        graph=graph,
        query=query,
    )
    assert any(status != "supported" for status in (partial.support.point_status or ()))


if __name__ == "__main__":
    main()
