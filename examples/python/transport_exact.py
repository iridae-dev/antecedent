"""Recursive / exact law. Same verbs as the statistical and complementary examples."""

import antecedent as ant


def main() -> None:
    graph = ant.Admg.from_edges(["x", "y"], [("x", "y")])
    evidence = ant.transport.Evidence(
        source=ant.transport.Source(
            "source", kind="experimental", interventions=["x"], sampling="independent"
        ),
        target_sampling="representative_sample",
    )
    query = ant.transport.Transport(
        ant.ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=evidence,
    )
    data = ant.transport.ExactTransportData(
        (
            ant.transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.5, 0.5),
                "source-v1",
                interventions=(("x", 0.0),),
            ),
            ant.transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.2, 0.8),
                "source-v1",
                interventions=(("x", 1.0),),
            ),
        )
    )
    result = ant.analyze(data, graph=graph, query=query)
    assert result.answer.kind == "response"
    assert [row[0] for row in result.response.values] == [0.5, 0.8]
    print(result)

    replacement = ant.transport.ExactTransportData(
        (
            ant.transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.6, 0.4),
                "source-v2",
                interventions=(("x", 0.0),),
            ),
            ant.transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.3, 0.7),
                "source-v2",
                interventions=(("x", 1.0),),
            ),
        )
    )
    refreshed = result.refresh(replacement)
    assert [row[0] for row in refreshed.response.values] == [0.4, 0.7]
    consumed = ant.load(result.export())
    assert consumed is not None
    print(refreshed)


if __name__ == "__main__":
    main()
