"""Single-source empirical table. Same verbs as the exact and complementary examples."""

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
    data = {
        "x": [0.0] * 50 + [1.0] * 100,
        "y": [0.0] * 25 + [1.0] * 25 + [0.0] * 20 + [1.0] * 80,
    }
    result = ant.analyze(data, graph=graph, query=query)
    assert result.answer.kind == "response"
    assert [row[0] for row in result.response.values] == [0.5, 0.8]
    print(result)

    replacement = {
        "x": [0.0] * 50 + [1.0] * 100,
        "y": [0.0] * 30 + [1.0] * 20 + [0.0] * 30 + [1.0] * 70,
    }
    refreshed = result.refresh(replacement)
    assert [round(row[0], 2) for row in refreshed.response.values] == [0.4, 0.7]
    print(refreshed)


if __name__ == "__main__":
    main()
