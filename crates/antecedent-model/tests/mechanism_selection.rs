//! Nonlinear and tree-ensemble mechanism selection against known truths, per-node
//! constraints, dependency-aware splits and selection provenance.
//!
//! The tree families fit through `antecedent-learn` providers, so the recovery and
//! round-trip tests are gated on the `ml-gbdt` and `ml-forest` features; everything
//! else (constraints, splits, failure records, capability flags) runs without them.

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismAssignment, MechanismFamily,
    MechanismRegistry, ModelError, NodeFamilyConstraint, NoiseInferenceMode, SelectionOptions,
    SelectionPolicy, SelectionRule, ValidationSplits,
};

const ROOT: u32 = 0;
const CHILD: u32 = 1;

fn uniform(rng: &mut CausalRng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.next_f64()
}

fn gaussian_noise(rng: &mut CausalRng, n: usize, sd: f64) -> Vec<f64> {
    (0..n).map(|_| sd * standard_normal(rng)).collect()
}

/// A known-truth child `y` of one or two parents. Returns the data and the graph.
struct Truth {
    data: TabularData,
    graph: Dag,
    /// Dense node of the outcome `y` (the last column).
    y_node: u32,
}

fn truth(parents: &[(&str, Vec<f64>)], y: &[f64]) -> Truth {
    let mut columns: Vec<(&str, &[f64])> =
        parents.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    columns.push(("y", y));
    let data = TabularData::from_f64_columns(columns).unwrap();
    let child = u32::try_from(parents.len()).unwrap();
    let mut graph = Dag::with_variables(child + 1);
    for p in 0..parents.len() {
        graph
            .insert_directed(
                DenseNodeId::from_raw(u32::try_from(p).unwrap()),
                DenseNodeId::from_raw(child),
            )
            .unwrap();
    }
    Truth { data, graph, y_node: child }
}

fn quadratic(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(11);
    let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 1.0 + 0.5 * x + 0.8 * x * x + e).collect();
    truth(&[("x", x)], &y)
}

fn threshold(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(12);
    let x: Vec<f64> = (0..n).map(|_| uniform(&mut rng, -1.0, 1.0)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> =
        x.iter().zip(&e).map(|(x, e)| 2.0 * f64::from(u8::from(*x > 0.3)) + e).collect();
    truth(&[("x", x)], &y)
}

fn interaction(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(13);
    let x1: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> = x1.iter().zip(&x2).zip(&e).map(|((a, b), e)| a * b + e).collect();
    truth(&[("x1", x1), ("x2", x2)], &y)
}

#[cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]
fn sinusoidal(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(14);
    let x: Vec<f64> = (0..n).map(|_| uniform(&mut rng, -3.0, 3.0)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| (2.0 * x).sin() + e).collect();
    truth(&[("x", x)], &y)
}

#[cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]
fn linear(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(15);
    let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 1.0 + 2.0 * x + e).collect();
    truth(&[("x", x)], &y)
}

/// A binary child with `logit P(t = 1) = 3 x² − 3`.
#[cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]
fn quadratic_logit(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(16);
    let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|x| {
            let p = 1.0 / (1.0 + (-(3.0 * x * x - 3.0)).exp());
            f64::from(u8::from(rng.next_f64() < p))
        })
        .collect();
    truth(&[("x", x)], &y)
}

/// Linear against the tree families, for both outcome types.
#[cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]
fn trees_against_linear() -> MechanismRegistry {
    MechanismRegistry {
        continuous: Arc::from([
            MechanismFamily::LinearGaussian,
            MechanismFamily::TreeBoostedAdditive,
            MechanismFamily::TreeForestAdditive,
            MechanismFamily::Constant,
        ]),
        discrete: Arc::from([
            MechanismFamily::Discrete,
            MechanismFamily::TreeBoostedCategorical,
            MechanismFamily::TreeForestCategorical,
            MechanismFamily::Constant,
        ]),
    }
}

fn fit(
    truth: &Truth,
    registry: &MechanismRegistry,
    policy: SelectionPolicy,
    options: &SelectionOptions,
) -> Result<(CompiledCausalModel, CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
    let compiled = CompiledCausalModel::compile(truth.graph.clone())?;
    let (store, assignments) =
        registry.assign_and_fit_with(&compiled, &truth.data, policy, options)?;
    Ok((compiled, store, assignments))
}

fn child(assignments: &[MechanismAssignment], node: u32) -> &MechanismAssignment {
    assignments.iter().find(|a| a.node == DenseNodeId::from_raw(node)).unwrap()
}

fn score(assignment: &MechanismAssignment, family: MechanismFamily) -> f64 {
    assignment.candidates.iter().find(|c| c.family == family).unwrap().score
}

fn is_tree(family: MechanismFamily) -> bool {
    family.id().starts_with("tree_")
}

#[test]
fn tree_family_ids_and_capability_flags_are_stable() {
    use MechanismFamily as F;
    let ids: Vec<_> = [
        F::TreeBoostedAdditive,
        F::TreeForestAdditive,
        F::TreeBoostedCategorical,
        F::TreeForestCategorical,
    ]
    .iter()
    .map(|f| f.id())
    .collect();
    assert_eq!(
        ids,
        [
            "tree_boosted_additive",
            "tree_forest_additive",
            "tree_boosted_categorical",
            "tree_forest_categorical"
        ]
    );
    for family in [F::TreeBoostedAdditive, F::TreeForestAdditive] {
        assert!(family.is_continuous() && !family.is_categorical());
        assert!(family.has_additive_disturbance());
        assert_eq!(family.noise_inference_mode(), NoiseInferenceMode::Invertible);
        assert!(family.can_modify_effects() && family.attribution_compatible());
    }
    for family in [F::TreeBoostedCategorical, F::TreeForestCategorical] {
        assert!(family.is_categorical() && !family.is_continuous());
        assert!(!family.has_additive_disturbance());
        assert_eq!(family.noise_inference_mode(), NoiseInferenceMode::Posterior);
        assert!(family.can_modify_effects() && family.attribution_compatible());
    }
    // The standard registry is unchanged: no family is added by default.
    let standard = MechanismRegistry::standard();
    assert!(standard.continuous.iter().chain(standard.discrete.iter()).all(|f| !is_tree(*f)));
    let nonlinear = MechanismRegistry::with_nonlinear_families();
    assert!(nonlinear.continuous.contains(&F::TreeBoostedAdditive));
    assert!(nonlinear.discrete.contains(&F::TreeForestCategorical));
}

#[test]
fn a_root_cannot_be_a_tree_ensemble_and_the_failure_is_recorded() {
    let truth = quadratic(300);
    let registry = MechanismRegistry::with_nonlinear_families();
    let (_, _, assignments) =
        fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default()).unwrap();
    let root = child(&assignments, ROOT);
    assert!(!is_tree(root.selected), "{:?}", root.selected);
    let failed: Vec<_> = root.failed_families.iter().map(|(f, _)| *f).collect();
    assert!(failed.contains(&MechanismFamily::TreeBoostedAdditive), "{failed:?}");
    assert!(failed.contains(&MechanismFamily::TreeForestAdditive), "{failed:?}");
    let (_, reason) = root
        .failed_families
        .iter()
        .find(|(f, _)| *f == MechanismFamily::TreeBoostedAdditive)
        .unwrap();
    assert!(reason.contains("at least one parent"), "{reason}");
    // Failed candidates are not among the scored ones.
    assert!(root.candidates.iter().all(|c| !is_tree(c.family)));
    assert_eq!(root.provenance.candidate_set.len(), registry.continuous.len());
}

/// A tree family either scores or is listed as failed with the provider named: it is
/// never silently dropped from a candidate set that names it.
#[test]
fn a_tree_family_is_scored_or_listed_as_failed_never_dropped() {
    let truth = threshold(300);
    let registry = MechanismRegistry::with_nonlinear_families();
    let (_, _, assignments) =
        fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default()).unwrap();
    let node = child(&assignments, CHILD);
    for family in [MechanismFamily::TreeBoostedAdditive, MechanismFamily::TreeForestAdditive] {
        let scored = node.candidates.iter().any(|c| c.family == family);
        let failed = node.failed_families.iter().find(|(f, _)| *f == family);
        assert!(scored || failed.is_some(), "{family:?} vanished from the record");
        if let Some((_, reason)) = failed {
            assert!(!scored && reason.contains("not compiled in"), "{family:?}: {reason}");
        }
    }
}

#[test]
fn per_node_constraints_are_honoured_and_recorded() {
    // Two parents with a pure product effect; no tree families here, so the unconstrained
    // winner is the interaction family (the basis families need two or more parents).
    let truth = interaction(900);
    let target = truth.y_node;
    let registry = MechanismRegistry::with_heterogeneity_families();
    let unconstrained =
        fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default()).unwrap().2;
    // Without a constraint the interaction family wins the product.
    assert_eq!(child(&unconstrained, target).selected, MechanismFamily::LinearInteractions);
    assert_eq!(child(&unconstrained, target).provenance.rule, SelectionRule::BestScore);
    assert!(child(&unconstrained, target).provenance.constraint.is_none());

    let y = antecedent_core::VariableId::from_raw(target);
    // Excluding the winner removes it from the scored set and from the candidate set.
    let excluded = SelectionOptions {
        constraints: vec![(
            y,
            NodeFamilyConstraint {
                excluded: Arc::from([MechanismFamily::LinearInteractions]),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let assignments = fit(&truth, &registry, SelectionPolicy::BestScore, &excluded).unwrap().2;
    let node = child(&assignments, target);
    assert!(node.candidates.iter().all(|c| c.family != MechanismFamily::LinearInteractions));
    assert!(!node.provenance.candidate_set.contains(&MechanismFamily::LinearInteractions));
    assert!(node.provenance.constraint.is_some());
    // The unconstrained root keeps the full list.
    assert_eq!(child(&assignments, ROOT).provenance.candidate_set.len(), registry.continuous.len());

    // `allowed` replaces the list for that node only.
    let allowed = SelectionOptions {
        constraints: vec![(
            y,
            NodeFamilyConstraint {
                allowed: Some(Arc::from([
                    MechanismFamily::LinearGaussian,
                    MechanismFamily::LinearInteractions,
                ])),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let assignments = fit(&truth, &registry, SelectionPolicy::BestScore, &allowed).unwrap().2;
    let node = child(&assignments, target);
    assert_eq!(
        node.provenance.candidate_set.as_ref(),
        [MechanismFamily::LinearGaussian, MechanismFamily::LinearInteractions]
    );
    assert!(node.failed_families.is_empty());

    // `required` overrides the ranking but every other candidate is still scored.
    let required = SelectionOptions {
        constraints: vec![(
            y,
            NodeFamilyConstraint {
                required: Some(MechanismFamily::LinearGaussian),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let assignments = fit(&truth, &registry, SelectionPolicy::BestScore, &required).unwrap().2;
    let node = child(&assignments, target);
    assert_eq!(node.selected, MechanismFamily::LinearGaussian);
    assert_eq!(node.provenance.rule, SelectionRule::RequiredByNodeConstraint);
    assert!(
        score(node, MechanismFamily::LinearInteractions)
            > score(node, MechanismFamily::LinearGaussian)
    );
}

#[test]
fn inconsistent_constraints_are_refused_before_any_fit() {
    let truth = quadratic(200);
    let registry = MechanismRegistry::standard();
    let run = |options: SelectionOptions| {
        fit(&truth, &registry, SelectionPolicy::BestScore, &options).map(|_| ()).unwrap_err()
    };
    let y = antecedent_core::VariableId::from_raw(CHILD);
    let unknown = antecedent_core::VariableId::from_raw(9);
    let any = NodeFamilyConstraint::default();
    assert!(matches!(
        run(SelectionOptions { constraints: vec![(unknown, any.clone())], ..Default::default() }),
        ModelError::InvalidSelection { .. }
    ));
    assert!(matches!(
        run(SelectionOptions {
            constraints: vec![(y, any.clone()), (y, any.clone())],
            ..Default::default()
        }),
        ModelError::InvalidSelection { .. }
    ));
    // A categorical family required of a continuous node.
    assert!(matches!(
        run(SelectionOptions {
            constraints: vec![(
                y,
                NodeFamilyConstraint {
                    required: Some(MechanismFamily::Discrete),
                    ..NodeFamilyConstraint::default()
                }
            )],
            ..Default::default()
        }),
        ModelError::InvalidSelection { .. }
    ));
    // Clustered splits without a UnitId column.
    assert!(matches!(
        run(SelectionOptions { splits: ValidationSplits::Clustered, ..Default::default() }),
        ModelError::InvalidSelection { .. }
    ));
}

#[test]
fn a_required_family_that_cannot_be_fit_names_its_failure() {
    let truth = quadratic(300);
    let registry = MechanismRegistry::standard();
    let options = SelectionOptions {
        constraints: vec![(
            antecedent_core::VariableId::from_raw(ROOT),
            NodeFamilyConstraint {
                required: Some(MechanismFamily::TreeBoostedAdditive),
                ..NodeFamilyConstraint::default()
            },
        )],
        ..SelectionOptions::default()
    };
    let error =
        fit(&truth, &registry, SelectionPolicy::BestScore, &options).map(|_| ()).unwrap_err();
    let ModelError::RequiredFamilyUnavailable { message } = error else {
        panic!("expected a required-family refusal, got {error:?}");
    };
    assert!(message.contains("tree_boosted_additive") && message.contains("at least one parent"));
}

#[test]
fn time_ordered_splits_are_recorded_deterministic_and_seedless() {
    let n = 900;
    let mut rng = CausalRng::from_seed(21);
    let mut x = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.7 * x[t - 1] + standard_normal(&mut rng);
    }
    let y: Vec<f64> = x.iter().map(|v| v.sin() + 0.3 * standard_normal(&mut rng)).collect();
    let truth = truth(&[("x", x)], &y);
    let registry = MechanismRegistry::with_heterogeneity_families();
    let options = SelectionOptions {
        splits: ValidationSplits::TimeOrdered { purge: 4 },
        ..SelectionOptions::default()
    };
    let first = fit(&truth, &registry, SelectionPolicy::BestScore, &options).unwrap().2;
    let second = fit(&truth, &registry, SelectionPolicy::BestScore, &options).unwrap().2;
    let shuffled =
        fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default()).unwrap().2;
    let node = child(&first, CHILD);
    assert_eq!(node.provenance.splits, ValidationSplits::TimeOrdered { purge: 4 });
    assert_eq!(node.provenance.seed, None);
    assert_eq!(node.provenance.folds, 5);
    assert_eq!(node.provenance.n_rows, n);
    // Same splits, same scores, bit for bit.
    for (a, b) in node.candidates.iter().zip(child(&second, CHILD).candidates.iter()) {
        assert_eq!(a.family, b.family);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
    }
    // The shuffled partition is a different, seeded scheme with a different record.
    let reference = child(&shuffled, CHILD);
    assert_eq!(reference.provenance.splits, ValidationSplits::Shuffled);
    assert!(reference.provenance.seed.is_some());
    assert_ne!(
        score(node, MechanismFamily::LinearGaussian).to_bits(),
        score(reference, MechanismFamily::LinearGaussian).to_bits()
    );
}

#[test]
fn every_candidate_is_scored_on_the_same_folds_with_a_comparable_decomposition() {
    let truth = threshold(600);
    let registry = MechanismRegistry::with_heterogeneity_families();
    let assignments =
        fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default()).unwrap().2;
    let node = child(&assignments, CHILD);
    assert!(node.candidates.windows(2).all(|w| w[0].score >= w[1].score));
    for c in node.candidates.iter() {
        assert!((c.score - (c.held_out_log_likelihood - c.complexity_penalty)).abs() < 1e-12);
    }
    assert_eq!(node.provenance.folds, 5);
    assert!(node.provenance.seed.is_some());
}

/// Two parents, `y = x1^2 + sin(2 x2) + e`: additive and smooth, with no cross-parent term.
fn additive_smooth(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(17);
    let x1: Vec<f64> = (0..n).map(|_| uniform(&mut rng, -2.0, 2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| uniform(&mut rng, -2.0, 2.0)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> =
        x1.iter().zip(&x2).zip(&e).map(|((a, b), e)| a * a + (2.0 * b).sin() + e).collect();
    truth(&[("x1", x1), ("x2", x2)], &y)
}

/// Two parents, `y = 1 + 2 x1 - x2 + e`.
fn linear_two_parents(n: usize) -> Truth {
    let mut rng = CausalRng::from_seed(18);
    let x1: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let e = gaussian_noise(&mut rng, n, 0.3);
    let y: Vec<f64> = x1.iter().zip(&x2).zip(&e).map(|((a, b), e)| 1.0 + 2.0 * a - b + e).collect();
    truth(&[("x1", x1), ("x2", x2)], &y)
}

/// Without a tree provider the basis families still recover the truth that suits each one,
/// and the linear family keeps a linear truth: the selected family is the simplest one that
/// earns its held-out score.
#[test]
fn basis_families_recover_the_family_that_suits_each_known_truth() {
    let registry = MechanismRegistry::with_heterogeneity_families();
    let cases: [(&str, Truth, MechanismFamily); 3] = [
        ("product", interaction(900), MechanismFamily::LinearInteractions),
        ("additive smooth", additive_smooth(1500), MechanismFamily::LinearSpline),
        ("linear", linear_two_parents(1500), MechanismFamily::LinearGaussian),
    ];
    for (name, truth, expected) in cases {
        let assignments =
            fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default())
                .unwrap()
                .2;
        let node = child(&assignments, truth.y_node);
        assert_eq!(node.selected, expected, "{name}: {:?}", node.candidates);
        if expected != MechanismFamily::LinearGaussian {
            // The nonlinear truth is worth the extra parameters: it beats the linear family.
            assert!(
                score(node, expected) > score(node, MechanismFamily::LinearGaussian) + 0.1,
                "{name}: {:?}",
                node.candidates
            );
        }
    }
}

#[cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]
mod with_providers {
    use super::*;
    use antecedent_model::{
        MechanismSlot, MechanismWorkspace, ParentBatch, evaluate_column, infer_noise_column,
        infer_noise_column_rng, log_prob_column, sample_noise_column,
    };

    fn parent_values(truth: &Truth, count: usize) -> Vec<f64> {
        use antecedent_data::TableView;
        let mut out = Vec::new();
        for p in 0..count {
            let var = antecedent_core::VariableId::from_raw(u32::try_from(p).unwrap());
            out.extend_from_slice(&truth.data.float64_cow(var).unwrap());
        }
        out
    }

    fn outcome(truth: &Truth) -> Vec<f64> {
        use antecedent_data::TableView;
        let var = antecedent_core::VariableId::from_raw(truth.y_node);
        truth.data.float64_cow(var).unwrap().to_vec()
    }

    /// The selected family is a tree ensemble that beats linear on held-out score.
    fn assert_tree_ensemble_wins(name: &str, truth: &Truth) {
        let assignments = fit(
            truth,
            &trees_against_linear(),
            SelectionPolicy::BestScore,
            &SelectionOptions::default(),
        )
        .unwrap()
        .2;
        let node = child(&assignments, truth.y_node);
        assert!(is_tree(node.selected), "{name}: selected {:?}", node.selected);
        let linear_score = score(node, MechanismFamily::LinearGaussian);
        assert!(
            score(node, node.selected) > linear_score + 0.1,
            "{name}: tree {} vs linear {linear_score}",
            score(node, node.selected)
        );
        // Only the inadmissible constant fails; no tree family does.
        assert!(
            node.failed_families.iter().all(|(f, _)| !is_tree(*f)),
            "{name}: {:?}",
            node.failed_families
        );
        assert_eq!(node.provenance.rule, SelectionRule::BestScore);
        // Both ensembles were scored on the same folds as linear.
        for family in [MechanismFamily::TreeBoostedAdditive, MechanismFamily::TreeForestAdditive] {
            assert!(node.candidates.iter().any(|c| c.family == family), "{name}: {family:?}");
        }
    }

    #[test]
    fn a_tree_ensemble_recovers_a_quadratic_mechanism() {
        assert_tree_ensemble_wins("quadratic", &quadratic(2000));
    }

    #[test]
    fn a_tree_ensemble_recovers_a_threshold_mechanism() {
        assert_tree_ensemble_wins("threshold", &threshold(2000));
    }

    #[test]
    fn a_tree_ensemble_recovers_an_interaction_mechanism() {
        assert_tree_ensemble_wins("interaction", &interaction(3000));
    }

    #[test]
    fn a_tree_ensemble_recovers_a_sinusoidal_mechanism() {
        assert_tree_ensemble_wins("sinusoidal", &sinusoidal(2000));
    }

    #[test]
    fn a_tree_ensemble_beats_the_linear_family_on_a_step() {
        let truth = threshold(2000);
        let registry = MechanismRegistry::with_nonlinear_families();
        let assignments =
            fit(&truth, &registry, SelectionPolicy::BestScore, &SelectionOptions::default())
                .unwrap()
                .2;
        let node = child(&assignments, CHILD);
        assert!(is_tree(node.selected), "selected {:?}", node.selected);
        assert!(score(node, node.selected) > score(node, MechanismFamily::LinearGaussian));
        // The spline-interaction family needs two parents; with one it is recorded as failed.
        assert!(node.failed_families.iter().any(|(family, reason)| {
            *family == MechanismFamily::LinearSpline && reason.contains("at least two parents")
        }));
    }

    #[test]
    fn linear_truth_keeps_the_linear_family() {
        let truth = linear(2000);
        let assignments = fit(
            &truth,
            &trees_against_linear(),
            SelectionPolicy::BestScore,
            &SelectionOptions::default(),
        )
        .unwrap()
        .2;
        let node = child(&assignments, CHILD);
        assert_eq!(node.selected, MechanismFamily::LinearGaussian, "{:?}", node.candidates);
        assert!(node.candidates.iter().filter(|c| is_tree(c.family)).count() == 2);
    }

    #[test]
    fn a_binary_node_with_a_nonlinear_logit_selects_a_tree_classifier() {
        let truth = quadratic_logit(3000);
        let assignments = fit(
            &truth,
            &trees_against_linear(),
            SelectionPolicy::BestScore,
            &SelectionOptions::default(),
        )
        .unwrap()
        .2;
        let node = child(&assignments, CHILD);
        assert!(
            matches!(
                node.selected,
                MechanismFamily::TreeBoostedCategorical | MechanismFamily::TreeForestCategorical
            ),
            "selected {:?}: {:?}",
            node.selected,
            node.candidates
        );
        assert!(score(node, node.selected) > score(node, MechanismFamily::Discrete));
        assert_eq!(node.selected.noise_inference_mode(), NoiseInferenceMode::Posterior);
    }

    #[test]
    fn additive_tree_noise_inversion_round_trips_and_samples_the_truth() {
        let truth = threshold(1500);
        let (_, store, assignments) = fit(
            &truth,
            &trees_against_linear(),
            SelectionPolicy::BestScore,
            &SelectionOptions::default(),
        )
        .unwrap();
        assert!(is_tree(child(&assignments, CHILD).selected));
        let slot = store.get(DenseNodeId::from_raw(CHILD));
        let MechanismSlot::TreeAdditive { sigma, .. } = slot else {
            panic!("tree additive expected, got {slot:?}");
        };
        let n = 1500;
        let x = parent_values(&truth, 1);
        let y = outcome(&truth);
        let batch = ParentBatch { n_rows: n, n_parents: 1, values: &x };
        let mut noise = vec![0.0; n];
        infer_noise_column(slot, &y, batch, &mut noise).unwrap();
        let mut replay = vec![0.0; n];
        evaluate_column(slot, batch, &noise, &mut replay, &mut MechanismWorkspace::default())
            .unwrap();
        for (a, b) in replay.iter().zip(&y) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        let mut mode_noise = vec![0.0; n];
        let mode =
            infer_noise_column_rng(slot, &y, batch, &mut mode_noise, &mut CausalRng::from_seed(1))
                .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Invertible);
        // The noise scale is the out-of-fold residual scale: near the generating 0.3.
        assert!((*sigma - 0.3).abs() < 0.06, "sigma {sigma}");
        // Sampling at a fixed parent value recovers the step's two levels.
        for (at, level) in [(0.8, 2.0), (-0.5, 0.0)] {
            let m = 4000;
            let parents = vec![at; m];
            let mut eps = vec![0.0; m];
            sample_noise_column(slot, m, &mut CausalRng::from_seed(5), &mut eps).unwrap();
            let mut draws = vec![0.0; m];
            evaluate_column(
                slot,
                ParentBatch { n_rows: m, n_parents: 1, values: &parents },
                &eps,
                &mut draws,
                &mut MechanismWorkspace::default(),
            )
            .unwrap();
            let mean = draws.iter().sum::<f64>() / m as f64;
            assert!((mean - level).abs() < 0.12, "x = {at}: mean {mean} vs {level}");
        }
        // The density is the Gaussian around the ensemble mean (the draw at zero noise) with the
        // fitted scale: -ln(sigma) - ln(2 pi) / 2 - (y - mean)^2 / (2 sigma^2) at every row.
        let mut lp = vec![0.0; n];
        log_prob_column(slot, &y, batch, &mut lp).unwrap();
        let mut ensemble_mean = vec![0.0; n];
        evaluate_column(
            slot,
            batch,
            &vec![0.0; n],
            &mut ensemble_mean,
            &mut MechanismWorkspace::default(),
        )
        .unwrap();
        for ((got, y), mean) in lp.iter().zip(&y).zip(&ensemble_mean) {
            let want = -sigma.ln()
                - 0.5 * (2.0 * std::f64::consts::PI).ln()
                - (y - mean).powi(2) / (2.0 * sigma * sigma);
            assert!((got - want).abs() < 1e-9, "log density {got} vs {want}");
        }
    }

    #[test]
    fn categorical_tree_abduction_replays_the_observation_and_normalises() {
        let truth = quadratic_logit(3000);
        let (_, store, _) = fit(
            &truth,
            &trees_against_linear(),
            SelectionPolicy::BestScore,
            &SelectionOptions::default(),
        )
        .unwrap();
        let slot = store.get(DenseNodeId::from_raw(CHILD));
        assert!(matches!(slot, MechanismSlot::TreeCategorical { .. }), "{slot:?}");
        let n = 3000;
        let x = parent_values(&truth, 1);
        let t = outcome(&truth);
        let batch = ParentBatch { n_rows: n, n_parents: 1, values: &x };
        let mut noise = vec![0.0; n];
        let mode =
            infer_noise_column_rng(slot, &t, batch, &mut noise, &mut CausalRng::from_seed(3))
                .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Posterior);
        let mut replay = vec![0.0; n];
        evaluate_column(slot, batch, &noise, &mut replay, &mut MechanismWorkspace::default())
            .unwrap();
        assert_eq!(replay, t);
        // Probabilities over the support sum to one at every row.
        let mut total = vec![0.0; n];
        for level in [0.0, 1.0] {
            let mut lp = vec![0.0; n];
            log_prob_column(slot, &vec![level; n], batch, &mut lp).unwrap();
            for (acc, l) in total.iter_mut().zip(&lp) {
                *acc += l.exp();
            }
        }
        assert!(total.iter().all(|v| (v - 1.0).abs() < 1e-9));
    }

    /// Serial dependence inflates shuffled scores of a flexible family; time-ordered
    /// splits remove the inflation, and agree with shuffled splits on independent rows.
    #[test]
    fn time_ordered_splits_remove_the_inflation_that_serial_dependence_causes() {
        let forest = MechanismFamily::TreeForestAdditive;
        let registry = trees_against_linear();
        let ordered = SelectionOptions {
            splits: ValidationSplits::TimeOrdered { purge: 0 },
            ..SelectionOptions::default()
        };
        let forest_score = |truth: &Truth, options: &SelectionOptions| {
            let assignments = fit(truth, &registry, SelectionPolicy::BestScore, options).unwrap().2;
            score(child(&assignments, truth.y_node), forest)
        };

        // Dependent rows: a slowly moving covariate and an unrelated slowly moving
        // outcome. Neighbouring rows are near copies, so a shuffled fold lets a held-out
        // row's neighbours train its prediction.
        let n = 1500;
        let mut rng = CausalRng::from_seed(31);
        let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
        for t in 1..n {
            x[t] = 0.98 * x[t - 1] + 0.2 * standard_normal(&mut rng);
            y[t] = 0.98 * y[t - 1] + 0.2 * standard_normal(&mut rng);
        }
        let dependent = truth(&[("x", x)], &y);
        let inflated = forest_score(&dependent, &SelectionOptions::default());
        let honest = forest_score(&dependent, &ordered);
        assert!(inflated > honest + 0.1, "shuffled {inflated} vs time-ordered {honest}");

        // Independent rows: the two schemes agree up to the smaller training windows.
        let independent = sinusoidal(1800);
        let shuffled = forest_score(&independent, &SelectionOptions::default());
        let blocked = forest_score(&independent, &ordered);
        assert!((shuffled - blocked).abs() < 0.2, "shuffled {shuffled} vs time-ordered {blocked}");
    }
}
