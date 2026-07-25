//! Frozen clean-room oracle checks for bounded graph-operation motifs.

use antecedent_graph::{
    CompletionSampler, Cpdag, Dag, DenseNodeId, Endpoint, GraphError, Pag, is_mag_completion,
    latent_project,
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
    assert_eq!(separation["cases"].as_array().unwrap().len(), 4);
    let mut chain = Pag::with_variables(3);
    chain.insert_directed(a, b).unwrap();
    chain.insert_directed(b, c).unwrap();
    assert!(!chain.is_m_separated(a, c, &[], 16, 8).unwrap());
    assert!(chain.is_m_separated(a, c, &[b], 16, 8).unwrap());

    let mut collider = Pag::with_variables(3);
    collider.insert_directed(a, b).unwrap();
    collider.insert_directed(c, b).unwrap();
    assert!(collider.is_m_separated(a, c, &[], 16, 8).unwrap());
    assert!(!collider.is_m_separated(a, c, &[b], 16, 8).unwrap());
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
