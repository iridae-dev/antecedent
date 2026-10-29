//! Frozen clean-room oracle checks for bounded graph-operation motifs.

use antecedent_graph::{
    CompletionSampler, Cpdag, CpdagCompletionSampler, Dag, DenseNodeId, Endpoint, GraphError,
    MarkedEdge, Pag, PagSeparation, is_mag_completion, is_mec_member, latent_project,
};

fn fixture(path: &str) -> serde_json::Value {
    serde_json::from_str(path).unwrap()
}

/// `a--b` undirected or `a->b` directed, as written in the fixture.
fn cpdag_edge(edge: &str) -> (DenseNodeId, DenseNodeId, bool) {
    let (pair, directed) = edge
        .split_once("->")
        .map(|p| (p, true))
        .unwrap_or_else(|| (edge.split_once("--").unwrap(), false));
    let node = |text: &str| DenseNodeId::from_raw(text.parse().unwrap());
    (node(pair.0), node(pair.1), directed)
}

fn cpdag_from_edges(edges: &[&str]) -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    for edge in edges {
        let (a, b, directed) = cpdag_edge(edge);
        if directed {
            graph.insert_directed(a, b).unwrap();
        } else {
            graph.insert_undirected(a, b).unwrap();
        }
    }
    graph
}

fn fixture_strings(value: &serde_json::Value) -> Vec<&str> {
    value.as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect()
}

#[test]
fn cpdag_endpoint_oracle() {
    let expected =
        fixture(include_str!("../../../conformance/graph/cpdag_operations/expected.json"));
    let cases = expected["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        if let Some(size) = case["mec_size"].as_u64() {
            // The class size is the number of streamed DAG completions, each a class member.
            let graph = cpdag_from_edges(&fixture_strings(&case["edges"]));
            let members: Vec<_> = CpdagCompletionSampler::new(graph.clone(), 16).unwrap().collect();
            assert_eq!(members.len() as u64, size, "{id}");
            assert!(members.iter().all(|m| is_mec_member(&graph, &m.graph)), "{id}");
            continue;
        }
        let input = fixture_strings(&case["input"]);
        let mut graph = cpdag_from_edges(&input);
        let operation = case["operation"].as_str().unwrap();
        if let Some(edge) = operation.strip_prefix("orient ") {
            let (a, b, _) = cpdag_edge(edge);
            graph.orient_undirected(a, b).unwrap();
            // Every edge of the declared output is present with the declared orientation, and the
            // edge counts agree so nothing else was oriented or added.
            let output = fixture_strings(&case["output"]);
            for written in &output {
                let (x, y, directed) = cpdag_edge(written);
                let stored = graph.edge_between(x, y).unwrap();
                if directed {
                    assert_eq!(stored.parent_child(), Some((x, y)), "{id}: {written}");
                } else {
                    assert!(stored.is_undirected(), "{id}: {written}");
                }
            }
            assert_eq!(
                graph.undirected_edge_count(),
                output.iter().filter(|e| e.contains("--")).count(),
                "{id}"
            );
        } else if let Some(edge) = operation.strip_prefix("insert ") {
            let (a, b, _) = cpdag_edge(edge);
            assert_eq!(case["error"].as_str(), Some("cycle"), "{id}");
            assert!(matches!(graph.insert_directed(a, b), Err(GraphError::Cycle { .. })), "{id}");
        } else {
            panic!("{id}: unknown operation {operation}");
        }
    }
}

#[test]
fn pag_endpoint_and_definite_status_oracles() {
    let endpoints =
        fixture(include_str!("../../../conformance/graph/pag_operations/expected.json"));
    let cases = endpoints["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let [a, b, c] = [0, 1, 2].map(DenseNodeId::from_raw);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        if let Some(edge) = case["edge"].as_str() {
            let pag = pag_from_marked_edges(&[edge]);
            let stored = pag.edge_between(a, b).unwrap();
            let mark = |name: &str| match name {
                "circle" => Endpoint::Circle,
                "arrow" => Endpoint::Arrow,
                "tail" => Endpoint::Tail,
                other => panic!("{id}: unknown mark {other}"),
            };
            let marks = fixture_strings(&case["marks"]);
            assert_eq!((stored.at_a, stored.at_b), (mark(marks[0]), mark(marks[1])), "{id}");
        } else if case["operation"] == "insert 2->0" {
            let mut pag = Pag::with_variables(3);
            for edge in fixture_strings(&case["input"]) {
                let (x, y) = edge.split_once("->").unwrap();
                let node = |t: &str| DenseNodeId::from_raw(t.parse().unwrap());
                pag.insert_directed(node(x), node(y)).unwrap();
            }
            assert_eq!(case["error"].as_str(), Some("cycle"), "{id}");
            assert!(matches!(pag.insert_directed(c, a), Err(GraphError::Cycle { .. })), "{id}");
        } else {
            assert_eq!(case["operation"].as_str(), Some("insert 0o-o0"), "{id}");
            assert_eq!(case["error"].as_str(), Some("invalid_endpoints"), "{id}");
            let mut pag = Pag::with_variables(3);
            assert!(
                matches!(pag.insert_circle_circle(a, a), Err(GraphError::InvalidEndpoints { .. })),
                "{id}"
            );
        }
    }

    let separation = fixture(include_str!(
        "../../../conformance/graph/definite_status_separation/expected.json"
    ));
    let cases = separation["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let edges: Vec<&str> =
            case["edges"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
        let pag = pag_from_marked_edges(&edges);
        let node = |v: &serde_json::Value| {
            DenseNodeId::from_raw(u32::try_from(v.as_u64().unwrap()).unwrap())
        };
        let z: Vec<DenseNodeId> =
            case["conditioned"].as_array().unwrap().iter().map(node).collect();
        let (x, y) = (node(&case["x"]), node(&case["y"]));
        let expected = match case["status"].as_str().unwrap() {
            "separated" => PagSeparation::Separated,
            "connected" => PagSeparation::Connected,
            "undetermined" => PagSeparation::Undetermined,
            other => panic!("{id}: unknown status {other}"),
        };
        assert_eq!(pag.m_separation_status(x, y, &z, 64, 8).unwrap(), expected, "{id}");
        match expected {
            PagSeparation::Separated => assert!(pag.is_m_separated(x, y, &z, 64, 8).unwrap()),
            PagSeparation::Connected => assert!(!pag.is_m_separated(x, y, &z, 64, 8).unwrap()),
            PagSeparation::Undetermined => assert!(
                matches!(
                    pag.is_m_separated(x, y, &z, 64, 8),
                    Err(GraphError::SeparationUndetermined)
                ),
                "{id}: an undetermined class must never read as separated"
            ),
        }
        if let Some(separated) = case["separated"].as_bool() {
            assert_eq!(separated, expected == PagSeparation::Separated, "{id}");
        }
    }
}

/// `<a><left mark>-<right mark><b>`: `<` / `>` arrowhead, `o` circle, `x`
/// conflict, nothing for a tail.
fn pag_from_marked_edges(edges: &[&str]) -> Pag {
    let parse = |edge: &str| {
        let (left, right) = edge.split_once('-').unwrap();
        let a_end = left.trim_end_matches(['<', 'o', 'x']).len();
        let b_start = right.len() - right.trim_start_matches(['>', 'o', 'x']).len();
        let mark = |m: &str| match m {
            "" => Endpoint::Tail,
            "<" | ">" => Endpoint::Arrow,
            "o" => Endpoint::Circle,
            "x" => Endpoint::Conflict,
            other => panic!("unknown mark {other} in {edge}"),
        };
        let a: u32 = left[..a_end].parse().unwrap();
        let b: u32 = right[b_start..].parse().unwrap();
        (a, mark(&left[a_end..]), mark(&right[..b_start]), b)
    };
    let parsed: Vec<_> = edges.iter().map(|e| parse(e)).collect();
    let n = parsed.iter().map(|&(a, _, _, b)| a.max(b)).max().unwrap() + 1;
    let mut pag = Pag::with_variables(n);
    for (a, at_a, at_b, b) in parsed {
        let (a, b) = (DenseNodeId::from_raw(a), DenseNodeId::from_raw(b));
        if at_a == Endpoint::Conflict || at_b == Endpoint::Conflict {
            pag.insert_circle_circle(a, b).unwrap();
            pag.mark_conflict(a, b).unwrap();
        } else {
            let mut edge = MarkedEdge::directed(a, b);
            edge.at_a = at_a;
            edge.at_b = at_b;
            pag.insert_marked(edge).unwrap();
        }
    }
    pag
}

#[test]
fn latent_projection_oracle() {
    let expected =
        fixture(include_str!("../../../conformance/graph/latent_projection/expected.json"));
    assert_eq!(expected["cases"].as_array().unwrap().len(), 3);
    let [x, latent, y] = [0, 1, 2].map(DenseNodeId::from_raw);

    let mut chain = Dag::with_variables(3);
    chain.insert_directed(x, latent).unwrap();
    chain.insert_directed(latent, y).unwrap();
    let projected = latent_project(&chain, &[x, y]).unwrap();
    let ids = |v: &[DenseNodeId]| v.iter().map(|n| n.raw()).collect::<Vec<_>>();
    // X->L->Y projects to exactly X->Y: no other directed edge and no bidirected edge.
    assert_eq!(ids(projected.children(DenseNodeId::from_raw(0))), [1]);
    assert!(projected.children(DenseNodeId::from_raw(1)).is_empty());
    assert!(projected.bidirected_neighbors(DenseNodeId::from_raw(0)).is_empty());
    assert!(projected.bidirected_neighbors(DenseNodeId::from_raw(1)).is_empty());

    let mut fork = Dag::with_variables(3);
    fork.insert_directed(latent, x).unwrap();
    fork.insert_directed(latent, y).unwrap();
    let projected = latent_project(&fork, &[x, y]).unwrap();
    // X<-L->Y projects to exactly X<->Y and no directed edge.
    assert_eq!(ids(projected.bidirected_neighbors(DenseNodeId::from_raw(0))), [1]);
    assert_eq!(ids(projected.bidirected_neighbors(DenseNodeId::from_raw(1))), [0]);
    assert!(projected.children(DenseNodeId::from_raw(0)).is_empty());
    assert!(projected.children(DenseNodeId::from_raw(1)).is_empty());

    // X->M->L->Y with M observed projects to X->M and M->Y only: the observed mediator blocks
    // the shortcut X->Y.
    let [x, m, l, y] = [0, 1, 2, 3].map(DenseNodeId::from_raw);
    let mut blocked = Dag::with_variables(4);
    blocked.insert_directed(x, m).unwrap();
    blocked.insert_directed(m, l).unwrap();
    blocked.insert_directed(l, y).unwrap();
    let projected = latent_project(&blocked, &[x, m, y]).unwrap();
    assert_eq!(ids(projected.children(DenseNodeId::from_raw(0))), [1]);
    assert_eq!(ids(projected.children(DenseNodeId::from_raw(1))), [2]);
    assert!(projected.children(DenseNodeId::from_raw(2)).is_empty());
    for node in 0..3 {
        assert!(projected.bidirected_neighbors(DenseNodeId::from_raw(node)).is_empty());
    }
}

/// Parses the fixture's `AoB` / `A->B` / `A<->B` edge strings into a PAG.
fn pag_from_edge_strings(edges: &[String]) -> Pag {
    let parsed: Vec<(usize, &str, usize)> = edges
        .iter()
        .map(|token| {
            let digits_a = token.find(|c: char| !c.is_ascii_digit()).expect("edge operator");
            let digits_b = token.rfind(|c: char| !c.is_ascii_digit()).expect("edge operator") + 1;
            (
                token[..digits_a].parse().expect("source node"),
                &token[digits_a..digits_b],
                token[digits_b..].parse().expect("target node"),
            )
        })
        .collect();
    let nodes = parsed.iter().map(|(a, _, b)| (*a).max(*b)).max().expect("an edge") + 1;
    let mut pag = Pag::with_variables(u32::try_from(nodes).expect("node count"));
    for (a, op, b) in parsed {
        let [a, b] = [a, b].map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node fits")));
        match op {
            "o-o" => pag.insert_circle_circle(a, b),
            "o->" => pag.insert_circle_arrow(a, b),
            "<-o" => pag.insert_circle_arrow(b, a),
            "->" => pag.insert_directed(a, b),
            "<-" => pag.insert_directed(b, a),
            "<->" => pag.insert_bidirected(a, b),
            other => panic!("unreadable edge operator {other}"),
        }
        .unwrap();
    }
    pag
}

/// Renders a fully decided completion in the fixture's `A->B` / `A<->B` notation.
fn completion_edge_strings(graph: &Pag) -> Vec<String> {
    let mut rendered = Vec::new();
    for i in 0..graph.node_count() {
        for j in (i + 1)..graph.node_count() {
            let [a, b] = [i, j].map(|k| DenseNodeId::from_raw(u32::try_from(k).expect("node")));
            if let Some(edge) = graph.edge_between(a, b) {
                let (ends_at_i, ends_at_j) = if edge.a == a {
                    (edge.at_a, edge.at_b)
                } else {
                    (edge.at_b, edge.at_a)
                };
                rendered.push(match (ends_at_i, ends_at_j) {
                    (Endpoint::Tail, Endpoint::Arrow) => format!("{i}->{j}"),
                    (Endpoint::Arrow, Endpoint::Tail) => format!("{j}->{i}"),
                    (Endpoint::Arrow, Endpoint::Arrow) => format!("{i}<->{j}"),
                    other => panic!("completion left an undecided or tail-tail edge {other:?}"),
                });
            }
        }
    }
    rendered
}

#[test]
fn pag_completion_oracle() {
    let expected =
        fixture(include_str!("../../../conformance/graph/pag_mag_completion/expected.json"));
    let cases = expected["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 3, "the fixture declares three cases and each is exercised here");
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let input: Vec<String> = serde_json::from_value(case["input"].clone()).expect("input");
        let pag = pag_from_edge_strings(&input);
        if let Some(count) = case["valid_completion_count"].as_u64() {
            let completions: Vec<_> = CompletionSampler::new(pag.clone(), 8).unwrap().collect();
            assert_eq!(completions.len() as u64, count, "{id}: completion count");
            assert!(completions.iter().all(|c| is_mag_completion(&c.graph)), "{id}");
            if let Some(outputs) = case.get("outputs") {
                let mut expected_outputs: Vec<String> =
                    serde_json::from_value(outputs.clone()).expect("outputs");
                let mut actual: Vec<String> = completions
                    .iter()
                    .map(|c| completion_edge_strings(&c.graph).join(" "))
                    .collect();
                expected_outputs.sort();
                actual.sort();
                assert_eq!(actual, expected_outputs, "{id}: completion set");
            }
        }
        if let Some(is_mag) = case["is_mag"].as_bool() {
            assert_eq!(is_mag_completion(&pag), is_mag, "{id}: is_mag");
        }
    }
}
