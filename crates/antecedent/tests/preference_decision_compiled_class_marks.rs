//! Equivalence-class atoms are told apart by everything that makes them a class, and a
//! sample of a class's members never reads as the class.
//!
//! * Two PAG atoms with one adjacency set and different circle marks (`z o-o t -> y` and
//!   `z <-> t -> y`) are different classes: each is compiled with its own members and keeps
//!   its own posterior weight.
//! * When a class is sampled, the structures' weights are marked as a sampled measure: the
//!   result-level reader refuses to take a minimum or a verdict over every member from them,
//!   and the sampled evaluation carries no verdict over every member.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute,
    CompiledPreferenceEvaluation, DecisionCriterion, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceSet, StructuralPolicy, StructuralPreferenceEvaluation, ValueModel,
};
use antecedent::result::StructuralWeightBasis;
use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, adjacency_masks_from_pag, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn summary() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("preference_class_marks"),
        rejected_invalid: 0,
    }
}

/// The coded three-column table the mixed-graph route evaluates with its discrete plug-in.
fn coded_table() -> TabularData {
    let n = 300_u32;
    let z: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
    let t: Vec<f64> = (0..n)
        .zip(&z)
        .map(|(i, z)| if f64::from((i / 2) % 5) < 2.0 + 2.0 * z { 1.5 } else { 0.5 })
        .collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 0.5 + 1.5 * t + z + 0.3 * (f64::from((i / 10) % 3) - 1.0))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// A confounded table over `t, y, z, w`.
fn table() -> TabularData {
    let n = 300_u32;
    let w: Vec<f64> = (0..n).map(|i| wobble(i, 0.37)).collect();
    let z: Vec<f64> = (0..n).zip(&w).map(|(i, w)| 0.6 * w + 0.5 * wobble(i, 0.91)).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 1.0 + 0.4 * z + 0.5 * wobble(i, 2.3)).collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 0.5 + 1.5 * t + z + 0.3 * wobble(i, 1.3))
        .collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

fn mask(n: usize, edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, n, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

/// `z o-o t -> y` and `z <-> t -> y`, each at half the mass: one adjacency set, two classes.
fn marked_posterior() -> GraphPosterior {
    let mut circles = Pag::with_variables(3);
    circles.insert_circle_circle(DenseNodeId::from_raw(Z), DenseNodeId::from_raw(T)).unwrap();
    circles.insert_directed(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    let (adjacency, marks) = adjacency_masks_from_pag(&circles).unwrap();
    let mut confounded = Pag::with_variables(3);
    confounded.insert_bidirected(DenseNodeId::from_raw(Z), DenseNodeId::from_raw(T)).unwrap();
    confounded.insert_directed(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    let (same, none) = adjacency_masks_from_pag(&confounded).unwrap();
    assert_eq!(same, adjacency, "one adjacency set");
    assert!(none.is_empty());
    GraphPosterior::from_atoms(
        3,
        vec![0.5, 0.5],
        vec![adjacency, same],
        vec![0.0; 9],
        vec![0.0; 9],
        summary(),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms([marks, none])
    .unwrap()
}

/// The path class `w - z - t` with `t -> y` at 0.6 and `w -> z -> t -> y` at 0.4.
fn cpdag_posterior() -> GraphPosterior {
    let path: &[(u32, u32)] = &[(W, Z), (Z, W), (Z, T), (T, Z), (T, Y)];
    let oriented: &[(u32, u32)] = &[(W, Z), (Z, T), (T, Y)];
    GraphPosterior::from_atoms(
        4,
        vec![0.6, 0.4],
        vec![mask(4, path), mask(4, oriented)],
        vec![0.0; 16],
        vec![0.0; 16],
        summary(),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn study(data: &TabularData, posterior: GraphPosterior, prior: Option<ClassPrior>) -> Study {
    let builder = Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(T),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    match prior {
        Some(prior) => builder.class_prior(prior),
        None => builder,
    }
    .build()
    .unwrap()
}

fn quantity(dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "points".into(),
        population_id: "target".into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - price * cost`, the costs of `low` and `high` differing by one.
fn contract(structural_policy: StructuralPolicy) -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower: 0.1,
                upper: 0.5,
                assumption_id: "assumption.price_of_benefit".into(),
            }],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity(dose)),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions: vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("preference-class-marks".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

#[test]
fn pag_atoms_that_differ_only_in_circle_marks_are_different_classes() {
    on_large_stack(marks_body);
}

fn marks_body() {
    let ctx = ExecutionContext::for_tests(61);
    let compiled = study(&coded_table(), marked_posterior(), None)
        .compile_for_preference_decision(
            &contract(StructuralPolicy::RequireInvariantBestAction),
            &ctx,
        )
        .unwrap();
    let structures = compiled.structures();
    assert_eq!(structures.len(), 4, "three members of z o-o t -> y and z <-> t -> y itself");
    let classes: BTreeSet<u64> =
        structures.iter().filter_map(pd::CompiledStructure::class_key).collect();
    assert_eq!(classes.len(), 2, "each mark set is its own class");
    for class in &classes {
        let members: Vec<&pd::CompiledStructure> =
            structures.iter().filter(|s| s.class_key() == Some(*class)).collect();
        close(members.iter().map(|s| s.weight()).sum(), 0.5, 1e-12);
        let share = 0.5 / members.len() as f64;
        for member in members {
            close(member.weight(), share, 1e-12);
        }
    }
    close(structures.iter().map(pd::CompiledStructure::weight).sum(), 1.0, 1e-12);
}

#[test]
fn a_sampled_mixture_is_never_read_as_the_class() {
    on_large_stack(sampled_body);
}

fn sampled_body() {
    let data = table();
    let ctx = ExecutionContext::for_tests(67);
    let prior = ClassPrior::uniform("assumption.uniform_within_class")
        .unwrap()
        .exact_limit(2)
        .unwrap()
        .sample(40, 3)
        .unwrap();
    let compiled = study(&data, cpdag_posterior(), Some(prior))
        .compile_for_preference_decision(&contract(StructuralPolicy::BayesOverStructures), &ctx)
        .unwrap();
    let executed = compiled.execute(None, PreferenceEvaluationOptions::default(), &ctx).unwrap();
    let mixture = executed.mixture.as_ref().expect("a structural mixture");
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::SampledClassMeasure);
    assert_eq!(mixture.weight_basis.as_str(), "sampled_class_measure");

    // The result-level reader takes no minimum and no verdict over every member from a
    // sample; it lists the structures.
    let context = compiled.response_context(0, &ctx).unwrap();
    for policy in [
        StructuralPolicy::Maximin,
        StructuralPolicy::RequireInvariantBestAction,
        StructuralPolicy::BayesOverStructures,
    ] {
        let refused = pd::structure_claims(mixture, &[], data.schema(), &context, policy)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            (refused.code, refused.detail.as_str()),
            (
                "route_not_supported",
                antecedent_core::reason_details::preference_compile::SAMPLED_MIXTURE_NOT_EXACT
            )
        );
    }
    assert!(
        pd::structure_claims(mixture, &[], data.schema(), &context, StructuralPolicy::ReportOnly)
            .is_ok()
    );

    // The sampled evaluation carries no verdict over every member.
    let CompiledPreferenceEvaluation::Sampled(sampled) = &executed.evaluation else {
        panic!("a sampled class is decided on the sampled measure");
    };
    let StructuralPreferenceEvaluation::Exact(result) = &sampled.evaluation else {
        panic!("an affine expectation is answered exactly");
    };
    assert!(!matches!(result.verdict, pd::StructuralPreferenceVerdict::Combined(_)));
    if let Some(combined) = &result.combined {
        assert_eq!(combined.verdict, pd::PreferenceVerdict::ReportOnly);
        assert!(combined.draw_share_best_everywhere.is_none());
        assert!(combined.actions.iter().all(|a| !a.necessarily_best && !a.uniquely_best));
    }
}
