//! Held-out mechanism selection publishes a deterministic selection record and no interval,
//! standard error or probability.
//!
//! The record of a node lists every candidate family with its held-out log-likelihood, its
//! complexity penalty and its score, and names the selected family and the rule that picked it.
//! Selection by held-out score is a procedure on the data: the same table, graph, registry,
//! splits and seed select the same family with bit-identical scores, the selected family is the
//! highest-scoring candidate under the best-score rule, and no field of the record states a
//! sampling error or a probability of the selection. Selection uncertainty enters a result only
//! through a whole-procedure interval that selects again inside every refit; this record is not
//! such a statement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_model::{
    CompiledCausalModel, MechanismAssignment, MechanismCandidate, MechanismFamily,
    MechanismRegistry, SelectionOptions, SelectionPolicy, SelectionRule, ValidationSplits,
};

/// The interval a record publishes, read from the only quantities it publishes that could form
/// one.
trait PublishedInterval {
    fn interval(&self) -> Option<(f64, f64)>;
}

/// A candidate names every field it publishes; none is a standard error, an interval or a
/// probability, and a field added later fails to compile here until it is read.
impl PublishedInterval for MechanismCandidate {
    fn interval(&self) -> Option<(f64, f64)> {
        let Self {
            family: _,
            score: _,
            held_out_log_likelihood: _,
            complexity_penalty: _,
            fit_cost: _,
            eval_cost: _,
        } = self;
        None
    }
}

/// A node's record names every field it publishes, and none states an uncertainty.
impl PublishedInterval for MechanismAssignment {
    fn interval(&self) -> Option<(f64, f64)> {
        let Self {
            node: _,
            variable: _,
            candidates,
            selected: _,
            typing: _,
            fitted: _,
            failed_families: _,
            provenance: _,
        } = self;
        assert!(candidates.iter().all(|candidate| candidate.interval().is_none()));
        None
    }
}

/// Two independent parents `x` and `z` of an outcome `y` that is quadratic in `x`, with
/// Gaussian noise. The basis families need at least two parents, so a single-parent chain
/// could never offer a curved candidate.
fn quadratic_chain(n: usize) -> (TabularData, Dag) {
    let mut rng = CausalRng::from_seed(21);
    let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&z)
        .map(|(x, z)| 1.0 + 0.5 * x + 0.8 * x * x + 0.4 * z + 0.3 * standard_normal(&mut rng))
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("z", z.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    (data, graph)
}

fn select(data: &TabularData, graph: &Dag, options: &SelectionOptions) -> Vec<MechanismAssignment> {
    let model = CompiledCausalModel::compile(graph.clone()).unwrap();
    MechanismRegistry::with_heterogeneity_families()
        .assign_and_fit_with(&model, data, SelectionPolicy::BestScore, options)
        .unwrap()
        .1
}

#[test]
fn a_selection_record_publishes_no_interval_standard_error_or_probability() {
    let (data, graph) = quadratic_chain(400);
    let record = select(&data, &graph, &SelectionOptions::default());
    assert_eq!(record.len(), 3);
    for assignment in &record {
        assert!(assignment.interval().is_none());
        assert!(!assignment.candidates.is_empty());
        for candidate in assignment.candidates.iter() {
            assert!(candidate.score.is_finite() && candidate.held_out_log_likelihood.is_finite());
            assert!(candidate.complexity_penalty >= 0.0);
        }
    }
}

#[test]
fn the_same_data_select_the_same_family_with_bit_identical_scores() {
    let (data, graph) = quadratic_chain(400);
    let options = SelectionOptions::default();
    let first = select(&data, &graph, &options);
    let second = select(&data, &graph, &options);
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.selected, b.selected);
        assert_eq!(a.provenance, b.provenance);
        let scores = |r: &MechanismAssignment| {
            r.candidates.iter().map(|c| (c.family, c.score.to_bits())).collect::<Vec<_>>()
        };
        assert_eq!(scores(a), scores(b));
    }
}

#[test]
fn the_best_score_rule_selects_the_top_scoring_candidate_and_the_score_is_held_out_less_the_penalty()
 {
    let (data, graph) = quadratic_chain(400);
    let record = select(&data, &graph, &SelectionOptions::default());
    // Each root has a candidate list of its own; the outcome is the node the quadratic shapes.
    let outcome = record.iter().find(|a| a.node == DenseNodeId::from_raw(2)).unwrap();
    assert_eq!(outcome.provenance.rule, SelectionRule::BestScore);
    assert_eq!(outcome.provenance.splits, ValidationSplits::Shuffled);
    let top = &outcome.candidates[0];
    assert_eq!(outcome.selected, top.family);
    assert!(outcome.candidates.windows(2).all(|pair| pair[0].score >= pair[1].score));
    for candidate in outcome.candidates.iter() {
        if candidate.family == MechanismFamily::Constant {
            continue;
        }
        let rebuilt = candidate.held_out_log_likelihood - candidate.complexity_penalty;
        assert!(
            (candidate.score - rebuilt).abs() <= 1e-9 * (1.0 + rebuilt.abs()),
            "{:?}: score {} against held-out less penalty {rebuilt}",
            candidate.family,
            candidate.score
        );
    }
    // The generating mechanism is curved, so the best score is not a plain linear fit.
    assert_ne!(outcome.selected, MechanismFamily::LinearGaussian);
}
