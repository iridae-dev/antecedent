//! The class envelope methods of `GeneralizedAdjustmentIdentifier` are the identification the
//! compiled routes publish.
//!
//! Each compiled route that reads a CPDAG or a PAG configures `GeneralizedAdjustmentIdentifier`
//! with the caller's execution context, which bounds storage and cancellation and does not
//! change the answer, and calls one class method. Each test calls the same method directly on
//! the same structure and query and compares the envelope with the one the compiled route
//! retains: the status, the identified and unidentified weights bit for bit, the completions,
//! the truncation count and, where the route retains the whole envelope, each completion's
//! status and estimands.
//!
//! - `identify_cpdag_distribution_envelope` and `identify_pag_distribution_envelope`: the
//!   interventional-distribution route over a supplied CPDAG and PAG (the result's
//!   identification certificate), and the per-atom identification product of a graph
//!   posterior over CPDAG atoms;
//! - `identify_cpdag_path_specific_envelope`: the per-atom identification product of a
//!   path-restricted effect over a graph posterior of CPDAG atoms, the only compiled route
//!   that reads a CPDAG path-specific query;
//! - `identify_cpdag_response_general`, `identify_cpdag_response_envelope`,
//!   `identify_pag_response_general` and `identify_pag_response_envelope`: the class
//!   identification of a single and a joint intervention response.
//!
//! `identify_pag_envelope` and `identify_cpdag_envelope` are compared in
//! `compiled_twins_static.rs`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "the weights are compared by their bits and the fixtures read small literals"
)]

use std::sync::Arc;

use antecedent::identify::{GeneralizedAdjustmentIdentifier, IdentificationEnvelope};
use antecedent::{EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent::{Identification, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery,
    PathSpecificEffectQuery, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::cpdag_from_adjacency_mask;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value as Json;

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn twin thread")
        .join()
        .expect("twin body panicked");
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(9_317)
}

/// Status, weights, completions and truncation of two envelopes, and each completion's status
/// and estimands.
fn assert_same_envelope<G>(
    label: &str,
    direct: &IdentificationEnvelope<G>,
    compiled: &IdentificationEnvelope<G>,
) {
    assert_eq!(direct.status, compiled.status, "{label}: status");
    assert_eq!(
        direct.identified_weight.0.to_bits(),
        compiled.identified_weight.0.to_bits(),
        "{label}: identified weight"
    );
    assert_eq!(
        direct.unidentified_weight.0.to_bits(),
        compiled.unidentified_weight.0.to_bits(),
        "{label}: unidentified weight"
    );
    assert_eq!(direct.cases.len(), compiled.cases.len(), "{label}: completions");
    assert_eq!(direct.truncated_completions, compiled.truncated_completions, "{label}: truncated");
    for (index, (left, right)) in direct.cases.iter().zip(&compiled.cases).enumerate() {
        assert_eq!(left.result.status, right.result.status, "{label}: case {index} status");
        assert_eq!(
            left.result.estimands.len(),
            right.result.estimands.len(),
            "{label}: case {index} estimands"
        );
        for (a, b) in left.result.estimands.iter().zip(&right.result.estimands) {
            assert_eq!(a.method, b.method, "{label}: case {index} method");
            assert_eq!(a.adjustment_set, b.adjustment_set, "{label}: case {index} adjustment set");
        }
    }
}

// --- interventional distribution over a supplied class ------------------------------------

fn pin() -> Json {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/interventional_distribution_classes/expected.json"
    ))
    .unwrap()
}

fn expand(case: &Json) -> TabularData {
    let columns: Vec<&str> =
        case["columns"].as_array().unwrap().iter().map(|column| column.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in case["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn column(case: &Json, name: &str) -> u32 {
    let index = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column.as_str() == Some(name))
        .unwrap();
    u32::try_from(index).unwrap()
}

fn edges(case: &Json, list: &Json) -> Vec<(DenseNodeId, DenseNodeId)> {
    list.as_array()
        .map(|list| {
            list.iter()
                .map(|edge| {
                    (
                        node(column(case, edge[0].as_str().unwrap())),
                        node(column(case, edge[1].as_str().unwrap())),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn variable_count(case: &Json) -> u32 {
    u32::try_from(case["columns"].as_array().unwrap().len()).unwrap()
}

fn distribution_query(case: &Json, pin: &Json) -> InterventionalDistributionQuery {
    InterventionalDistributionQuery::new(
        v(column(case, pin["query"]["outcome"].as_str().unwrap())),
        [Intervention::set(
            v(column(case, "t")),
            Value::f64(pin["query"]["do"]["t"].as_f64().unwrap()),
        )],
    )
}

fn run_distribution(
    data: TabularData,
    query: &InterventionalDistributionQuery,
    graph: impl antecedent::IntoGraphInput,
) -> StudyResult {
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Distribution(query.clone()))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

fn certificate(result: &StudyResult) -> &Identification {
    &result.certificate.as_ref().expect("the class route retains its identification").identification
}

#[test]
fn cpdag_distribution_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let fixture = pin();
        // Completions that disagree on the distribution: a partially identified envelope.
        let case = &fixture["class_disagree"];
        let mut cpdag = Cpdag::with_variables(variable_count(case));
        for (from, to) in edges(case, &case["cpdag"]["directed_edges"]) {
            cpdag.insert_directed(from, to).unwrap();
        }
        for (left, right) in edges(case, &case["cpdag"]["undirected_edges"]) {
            cpdag.insert_undirected(left, right).unwrap();
        }
        let query = distribution_query(case, &fixture);
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_cpdag_distribution_envelope(&cpdag, &CausalQuery::Distribution(query.clone()))
            .unwrap();
        let result = run_distribution(expand(case), &query, cpdag);
        let Identification::CpdagEnvelope { envelope, .. } = certificate(&result) else {
            panic!("a CPDAG distribution retains its completion envelope");
        };
        assert!(direct.cases.len() > 1, "the fixture has several completions");
        assert_same_envelope("cpdag distribution", &direct, envelope);
    });
}

#[test]
fn pag_distribution_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let fixture = pin();
        let case = &fixture["class_point"];
        let mut pag = Pag::with_variables(variable_count(case));
        for (from, to) in edges(case, &case["graph"]["directed_edges"]) {
            pag.insert_directed(from, to).unwrap();
        }
        let query = distribution_query(case, &fixture);
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_pag_distribution_envelope(&pag, &CausalQuery::Distribution(query.clone()))
            .unwrap();
        let result = run_distribution(expand(case), &query, pag);
        let Identification::Envelope { envelope, .. } = certificate(&result) else {
            panic!("a PAG distribution retains its candidate envelope");
        };
        assert_same_envelope("pag distribution", &direct, envelope);
    });
}

// --- path-restricted effect and distribution over a graph posterior of CPDAG atoms ---------

/// `(t, m, y, count)` of the table.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 30),
    (0.0, 0.0, 1.0, 10),
    (0.0, 1.0, 0.0, 5),
    (0.0, 1.0, 1.0, 15),
    (1.0, 0.0, 0.0, 12),
    (1.0, 0.0, 1.0, 8),
    (1.0, 1.0, 0.0, 20),
    (1.0, 1.0, 1.0, 25),
];

fn cells_table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (tv, mv, yv, count) in CELLS {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn class_mask(edges: &[(u32, u32)]) -> u64 {
    let mut dag = Dag::with_variables(3);
    for &(from, to) in edges {
        dag.insert_directed(node(from), node(to)).unwrap();
    }
    adjacency_mask_from_cpdag(&Cpdag::from_dag(&dag)).unwrap()
}

/// A directed cycle: no member of the class, so it identifies nothing.
fn refused_mask() -> u64 {
    [(0, 1), (1, 2), (2, 0)]
        .iter()
        .fold(0_u64, |mask, &(from, to)| set_edge(mask, 3, from, to, true))
}

fn cpdag_posterior(masks: Vec<u64>) -> GraphPosterior {
    let sampling = SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("compiled_twins_class_identifiers"),
        rejected_invalid: 0,
    };
    GraphPosterior::new(3, vec![0.5, 0.3, 0.2], masks, vec![0.0; 9], vec![0.0; 9], sampling)
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

/// The per-atom identification products the contracted result carries, in atom order.
fn atom_products(
    posterior: GraphPosterior,
    query: CausalQuery,
) -> Vec<Option<antecedent_io::IdentificationProductWire>> {
    let built = Study::tabular(cells_table())
        .graph_posterior(posterior)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ctx();
    let prepared = built.prepare(&ctx).unwrap();
    let result = prepared.estimate(&cells_table(), &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "class-identifier-twins", &ctx).unwrap();
    let consumed = antecedent_io::replay_analysis_result(&bytes).unwrap();
    consumed
        .contract
        .as_ref()
        .and_then(|contract| contract.program.as_ref())
        .and_then(|program| program.graph_posterior_programs.as_ref())
        .expect("the contract carries the per-atom programs")
        .atoms
        .iter()
        .map(|atom| atom.identification_product.clone())
        .collect()
}

/// An atom's product carries exactly the envelope the direct method returns on its class.
fn assert_atom_matches(
    label: &str,
    direct: &IdentificationEnvelope<Dag>,
    product: Option<&antecedent_io::IdentificationProductWire>,
) {
    if direct.identified_weight.0 <= 0.0 {
        assert!(product.is_none(), "{label}: an unidentified class carries no product");
        return;
    }
    let product = product.unwrap_or_else(|| panic!("{label}: an identified class has a product"));
    assert_eq!(product.status, direct.status.as_str(), "{label}: status");
    let envelope = product.envelope.as_ref().expect("a class product carries its envelope");
    assert_eq!(envelope.cases, direct.cases.len() as u64, "{label}: completions");
    assert_eq!(
        envelope.identified_weight_bits,
        direct.identified_weight.0.to_bits(),
        "{label}: identified weight"
    );
    assert_eq!(
        envelope.unidentified_weight_bits,
        direct.unidentified_weight.0.to_bits(),
        "{label}: unidentified weight"
    );
    assert_eq!(
        envelope.truncated_completions, direct.truncated_completions as u64,
        "{label}: truncated"
    );
}

fn posterior_twin(label: &str, query: &CausalQuery, distribution: bool) {
    let masks = vec![class_mask(&[(0, 1), (0, 2), (1, 2)]), class_mask(&[(0, 1), (1, 2)])];
    let mut all = masks.clone();
    all.push(refused_mask());
    let products = atom_products(cpdag_posterior(all), query.clone());
    assert_eq!(products.len(), 3, "{label}: three atoms");
    for (index, mask) in masks.iter().enumerate() {
        // The class the route rebuilds from the atom's mask.
        let cpdag = cpdag_from_adjacency_mask(*mask, 3).unwrap();
        let identifier = GeneralizedAdjustmentIdentifier::new();
        let direct = if distribution {
            identifier.identify_cpdag_distribution_envelope(&cpdag, query)
        } else {
            identifier.identify_cpdag_path_specific_envelope(&cpdag, query)
        }
        .unwrap();
        assert_atom_matches(&format!("{label} atom {index}"), &direct, products[index].as_ref());
    }
    assert!(products[2].is_none(), "{label}: the refused atom carries no product");
}

#[test]
fn cpdag_path_specific_envelope_equals_the_posterior_atom_identification() {
    on_large_stack(|| {
        let query = CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]),
        );
        posterior_twin("cpdag path-specific", &query, false);
    });
}

#[test]
fn cpdag_distribution_envelope_equals_the_posterior_atom_identification() {
    on_large_stack(|| {
        let query = CausalQuery::Distribution(InterventionalDistributionQuery::new(
            v(2),
            [Intervention::set(v(0), Value::f64(1.0))],
        ));
        posterior_twin("cpdag distribution", &query, true);
    });
}

// --- single and joint intervention responses over a class ---------------------------------

fn schema_only(variables: usize) -> TabularData {
    let names: Vec<String> = (0..variables).map(|index| format!("v{index}")).collect();
    let empty: &[f64] = &[];
    TabularData::from_f64_columns(names.iter().map(|name| (name.as_str(), empty))).unwrap()
}

/// `E[y | do(set)]` for the given `(variable, level)` interventions, outcome `v3`.
fn response(sets: &[(u32, f64)]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(3),
        interventions: sets
            .iter()
            .map(|&(variable, level)| Intervention::set(v(variable), Value::f64(level)))
            .collect::<Arc<[_]>>(),
    })
}

fn identify_class(graph: impl antecedent::IntoGraphInput, query: &ResponseQuery) -> Identification {
    Study::tabular(schema_only(4))
        .graph(graph)
        .query(CausalQuery::Response(query.clone()))
        .identifier(IdentifierId::GeneralizedAdjustment)
        .identify_structure_within(&ctx())
        .unwrap()
}

/// Treatments `a` (0) and `d` (1), covariate `w` (2), outcome `y` (3): `w -- a` undirected,
/// `w -> y`, `a -> y`, `d -> y`. One completion makes `w` a confounder of `a`, the other a
/// mediator, so the completions adjust differently.
fn response_cpdag() -> Cpdag {
    let mut cpdag = Cpdag::with_variables(4);
    cpdag.insert_undirected(node(2), node(0)).unwrap();
    for (from, to) in [(2, 3), (0, 3), (1, 3)] {
        cpdag.insert_directed(node(from), node(to)).unwrap();
    }
    cpdag
}

/// `w -> a`, `w -> y`, `a -> y`, `d -> y`, every edge visible.
fn response_pag() -> Pag {
    let mut pag = Pag::with_variables(4);
    for (from, to) in [(2, 0), (2, 3), (0, 3), (1, 3)] {
        pag.insert_directed(node(from), node(to)).unwrap();
    }
    pag
}

#[test]
fn cpdag_response_envelopes_equal_the_compiled_identification() {
    on_large_stack(|| {
        let identifier = GeneralizedAdjustmentIdentifier::new();
        let single = response(&[(0, 1.0)]);
        let direct =
            identifier.identify_cpdag_response_general(&response_cpdag(), &single).unwrap();
        let Identification::CpdagEnvelope { envelope, .. } =
            identify_class(response_cpdag(), &single)
        else {
            panic!("a CPDAG response identifies to a completion envelope");
        };
        assert_same_envelope("cpdag single response", &direct, &envelope);

        let joint = response(&[(0, 1.0), (1, 1.0)]);
        let direct =
            identifier.identify_cpdag_response_envelope(&response_cpdag(), &joint).unwrap();
        let Identification::CpdagEnvelope { envelope, .. } =
            identify_class(response_cpdag(), &joint)
        else {
            panic!("a joint CPDAG response identifies to a completion envelope");
        };
        assert_same_envelope("cpdag joint response", &direct, &envelope);
    });
}

#[test]
fn pag_response_envelopes_equal_the_compiled_identification() {
    on_large_stack(|| {
        let identifier = GeneralizedAdjustmentIdentifier::new();
        let single = response(&[(0, 1.0)]);
        let direct = identifier.identify_pag_response_general(&response_pag(), &single).unwrap();
        let Identification::Envelope { envelope, .. } = identify_class(response_pag(), &single)
        else {
            panic!("a PAG response identifies to a candidate envelope");
        };
        assert_same_envelope("pag single response", &direct, &envelope);

        let joint = response(&[(0, 1.0), (1, 1.0)]);
        let direct = identifier.identify_pag_response_envelope(&response_pag(), &joint).unwrap();
        let Identification::Envelope { envelope, .. } = identify_class(response_pag(), &joint)
        else {
            panic!("a joint PAG response identifies to a candidate envelope");
        };
        assert_same_envelope("pag joint response", &direct, &envelope);
    });
}
