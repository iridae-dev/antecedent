//! Frozen clean-room oracle checks for bounded graph-operation motifs.

use antecedent_graph::{
    CompletionSampler, Cpdag, Dag, DenseNodeId, Endpoint, GraphError, MarkedEdge, Pag,
    PagSeparation, is_mag_completion, latent_project,
};

fn fixture(path: &str) -> serde_json::Value {
    serde_json::from_str(path).unwrap()
}

#[test]
fn cpdag_endpoint_oracle() {
    let expected =
        fixture(include_str!("../../../conformance/graph/cpdag_operations/expected.json"));
    assert_eq!(expected["cases"][0]["mec_size"].as_u64(), Some(3));
    let [a, b, c] = [0, 1, 2].map(DenseNodeId::from_raw);
    let mut graph = Cpdag::with_variables(3);
    graph.insert_undirected(a, b).unwrap();
    graph.insert_undirected(b, c).unwrap();
    graph.orient_undirected(a, b).unwrap();
    assert_eq!(graph.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
    assert!(graph.edge_between(b, c).unwrap().is_undirected());

    let mut cyclic = Cpdag::with_variables(3);
    cyclic.insert_directed(a, b).unwrap();
    cyclic.insert_directed(b, c).unwrap();
    assert!(matches!(cyclic.insert_directed(c, a), Err(GraphError::Cycle { .. })));
}

#[test]
fn pag_endpoint_and_definite_status_oracles() {
    let endpoints =
        fixture(include_str!("../../../conformance/graph/pag_operations/expected.json"));
    assert_eq!(endpoints["cases"].as_array().unwrap().len(), 4);
    let [a, b, c] = [0, 1, 2].map(DenseNodeId::from_raw);
    let mut marks = Pag::with_variables(3);
    marks.insert_circle_arrow(a, b).unwrap();
    let edge = marks.edge_between(a, b).unwrap();
    assert_eq!((edge.at_a, edge.at_b), (Endpoint::Circle, Endpoint::Arrow));
    assert!(marks.insert_circle_circle(c, c).is_err());

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
    assert!(projected.children(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1)));

    let mut fork = Dag::with_variables(3);
    fork.insert_directed(latent, x).unwrap();
    fork.insert_directed(latent, y).unwrap();
    let projected = latent_project(&fork, &[x, y]).unwrap();
    assert!(
        projected
            .bidirected_neighbors(DenseNodeId::from_raw(0))
            .contains(&DenseNodeId::from_raw(1))
    );
}

#[test]
fn pag_completion_oracle() {
    let expected =
        fixture(include_str!("../../../conformance/graph/pag_mag_completion/expected.json"));
    let expected_count = usize::try_from(
        expected["cases"][0]["valid_completion_count"]
            .as_u64()
            .expect("valid_completion_count is u64"),
    )
    .expect("valid_completion_count fits usize");
    let [a, b, c] = [0, 1, 2].map(DenseNodeId::from_raw);
    let mut one_edge = Pag::with_variables(2);
    one_edge.insert_circle_circle(a, b).unwrap();
    let completions: Vec<_> = CompletionSampler::new(one_edge, 8).unwrap().collect();
    assert_eq!(completions.len(), expected_count);
    assert!(completions.iter().all(|completion| is_mag_completion(&completion.graph)));

    let mut invalid = Pag::with_variables(3);
    invalid.insert_directed(a, b).unwrap();
    invalid.insert_directed(b, c).unwrap();
    invalid.insert_bidirected(a, c).unwrap();
    assert!(!is_mag_completion(&invalid));
}
