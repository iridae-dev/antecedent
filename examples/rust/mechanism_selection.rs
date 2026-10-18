//! Mechanism selection: nonlinear and tree families scored on shared held-out folds.
//!
//! `y = 1 + 0.5 x + 0.8 x^2 + noise` is not linear in `x`. Linear, spline and tree-ensemble
//! families all compete on the same folds; the winner is kept together with every candidate's
//! score, the families that could not be fit and why, and the rule that chose. A per-node
//! constraint can require a family whatever its rank, and the other candidates are still scored.
//!
//! Tree ensembles fit through optional providers (the `ml-gbdt` and `ml-forest` features). Without
//! one a tree family is listed as failed with its reason, never silently dropped.
//!
//! Run: `cargo run -p antecedent --example mechanism_selection`
//! Source: `examples/rust/mechanism_selection.rs`

use std::sync::Arc;

use antecedent::gcm::{
    MechanismFamily, NodeFamilyConstraint, SelectionOptions, SelectionRule, ValidationSplits,
    fit_gcm_with, registry_from_candidates,
};
use antecedent_core::{CausalRng, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const X: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = CausalRng::from_seed(11);
    let x: Vec<f64> = (0..600).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> =
        x.iter().map(|x| 1.0 + 0.5 * x + 0.8 * x * x + 0.3 * standard_normal(&mut rng)).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])?;
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    let registry = registry_from_candidates(&[
        "linear",
        "additive_spline",
        "random_forest",
        "gradient_boosted_trees",
    ])?;

    // 1. Best score wins, and the choice is auditable.
    let fitted = fit_gcm_with(graph.clone(), &data, &registry, &SelectionOptions::default())?;
    let node = |assignments: &[antecedent::gcm::MechanismAssignment], v| {
        assignments.iter().position(|a| a.variable == v).ok_or("node is assigned")
    };
    let record = &fitted.assignments[node(&fitted.assignments, Y)?];
    assert_ne!(record.selected, MechanismFamily::LinearGaussian, "a quadratic is not linear");
    assert_eq!(record.provenance.rule, SelectionRule::BestScore);
    assert_eq!(record.provenance.splits, ValidationSplits::Shuffled);
    assert_eq!(record.provenance.n_rows, 600);
    let score_of = |family| record.candidates.iter().find(|c| c.family == family).map(|c| c.score);
    let linear = score_of(MechanismFamily::LinearGaussian).ok_or("linear is scored")?;
    let best = score_of(record.selected).ok_or("the winner is scored")?;
    assert!(best > linear, "the winner beats the linear family on held-out likelihood");
    for c in record.candidates.iter() {
        let recomputed = c.held_out_log_likelihood - c.complexity_penalty;
        assert!((c.score - recomputed).abs() < 1e-12, "score = held-out likelihood - penalty");
    }
    // Every family the caller named is either scored or recorded as failed.
    for family in [
        MechanismFamily::LinearGaussian,
        MechanismFamily::LinearSpline,
        MechanismFamily::TreeBoostedAdditive,
        MechanismFamily::TreeForestAdditive,
    ] {
        let scored = record.candidates.iter().any(|c| c.family == family);
        let failed = record.failed_families.iter().any(|(f, _)| *f == family);
        assert!(scored || failed, "{family:?} must be scored or failed");
    }

    // 2. A root has no parents: the tree families fail there, with the reason recorded.
    let root = &fitted.assignments[node(&fitted.assignments, X)?];
    let (_, reason) = root
        .failed_families
        .iter()
        .find(|(f, _)| *f == MechanismFamily::TreeBoostedAdditive)
        .ok_or("a root cannot be a tree ensemble")?;
    assert!(reason.contains("at least one parent"), "{reason}");

    // 3. Requiring a family selects it whatever its rank; the rest are still scored.
    let options = SelectionOptions {
        constraints: vec![(
            Y,
            NodeFamilyConstraint {
                required: Some(MechanismFamily::LinearGaussian),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let forced = fit_gcm_with(graph.clone(), &data, &registry, &options)?;
    let required = &forced.assignments[node(&forced.assignments, Y)?];
    assert_eq!(required.selected, MechanismFamily::LinearGaussian);
    assert_eq!(required.provenance.rule, SelectionRule::RequiredByNodeConstraint);
    assert!(required.candidates.len() > 1, "the other candidates are still scored");

    // 4. A contradictory constraint is refused before any family is fit.
    let contradictory = SelectionOptions {
        constraints: vec![(
            Y,
            NodeFamilyConstraint {
                required: Some(MechanismFamily::LinearGaussian),
                excluded: Arc::from([MechanismFamily::LinearGaussian]),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let refused = fit_gcm_with(graph, &data, &registry, &contradictory)
        .expect_err("required and excluded contradict");
    assert!(refused.to_string().contains("mechanism_candidates_invalid"), "{refused}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
