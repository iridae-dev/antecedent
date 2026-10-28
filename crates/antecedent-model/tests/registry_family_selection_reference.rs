//! Family selection against scores recomputed from exact refits of every candidate.
//!
//! `oracle_data/registry_family_selection_reference.py` (frozen in the JSON beside it) refits each
//! candidate on every training fold with an exact solve (normal equations in 60 digits, Newton on
//! the concave logit likelihoods), evaluates its held-out log density, adds the registry's
//! `k ln n / 2n` parameter-count charge and ranks the candidates; the conditional state space is
//! scored prequentially on the out-of-fold residual series of the *declared* validation scheme.
//! The shuffled partition is the library's seeded kernel: the test regenerates it, requires it to
//! equal the frozen one and prints the replacement when it does not.
//!
//! Covered: continuous and binary outcomes, the shuffled and the time-ordered scheme (with a
//! purge), the recorded split provenance, the inadmissible constant, a required-by-policy choice
//! that leaves the scores alone, and exact ties broken by the candidate listing order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::shuffled_fold_assignment;
use antecedent_model::{
    CompiledCausalModel, MechanismAssignment, MechanismFamily, MechanismRegistry,
    NodeFamilyConstraint, SelectionOptions, SelectionPolicy, SelectionRule, ValidationSplits,
};
use serde_json::Value;

const CV_FOLD_SEED: u64 = 0x4D45_4348_4356_0001;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(columns: &[Vec<f64>], names: &[&str], outcome_type: ValueType) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for (i, name) in names.iter().enumerate() {
        let vt = if i + 1 == names.len() { outcome_type.clone() } else { ValueType::Continuous };
        b.add_variable(
            *name,
            vt,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(columns[0].len());
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.clone()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn model3() -> CompiledCausalModel {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    CompiledCausalModel::compile(g).unwrap()
}

fn roots_linear(splits: ValidationSplits) -> SelectionOptions {
    let linear = || NodeFamilyConstraint {
        allowed: Some(Arc::from([MechanismFamily::LinearGaussian])),
        ..NodeFamilyConstraint::default()
    };
    SelectionOptions {
        constraints: vec![(VariableId::from_raw(0), linear()), (VariableId::from_raw(1), linear())],
        splits,
    }
}

fn assert_score(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn check_candidates(
    assignment: &MechanismAssignment,
    reference: &Value,
    tolerances: &[(&str, f64)],
    what: &str,
) {
    let want = &reference["candidates"];
    let ranking: Vec<&str> =
        reference["ranking"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let got_ranking: Vec<&str> = assignment.candidates.iter().map(|c| c.family.id()).collect();
    assert_eq!(got_ranking, ranking, "{what}: candidate ranking");
    assert_eq!(assignment.selected.id(), reference["selected"].as_str().unwrap(), "{what}");
    for candidate in assignment.candidates.iter() {
        let id = candidate.family.id();
        let tol = tolerances.iter().find(|(f, _)| *f == id).map_or(1e-9, |t| t.1);
        let w = &want[id];
        assert_score(
            candidate.held_out_log_likelihood,
            w["held_out"].as_f64().unwrap(),
            tol,
            &format!("{what}: {id} held-out log likelihood"),
        );
        assert_score(
            candidate.complexity_penalty,
            w["penalty"].as_f64().unwrap(),
            1e-12,
            &format!("{what}: {id} penalty"),
        );
        assert_score(
            candidate.score,
            w["score"].as_f64().unwrap(),
            tol,
            &format!("{what}: {id} score"),
        );
    }
    // The constant claims the variable is deterministic; with any spread it is not a candidate.
    assert!(
        assignment.failed_families.iter().any(|(f, _)| *f == MechanismFamily::Constant),
        "{what}: the constant family is inadmissible"
    );
    assert!(assignment.candidates.iter().all(|c| c.family != MechanismFamily::Constant));
}

fn registry() -> MechanismRegistry {
    use MechanismFamily as F;
    MechanismRegistry {
        continuous: Arc::from([
            F::LinearGaussian,
            F::HierarchicalLinear,
            F::Bvar,
            F::LinearGaussianStateSpace,
            F::Constant,
        ]),
        discrete: Arc::from([F::Discrete, F::HierarchicalGlm, F::Constant]),
    }
}

#[test]
fn registry_family_selection_reference_scores_and_ranking_match_exact_refits() {
    let d: Value =
        serde_json::from_str(include_str!("oracle_data/registry_family_selection_reference.json"))
            .unwrap();
    let (x1, x2) = (floats(&d["x1"]), floats(&d["x2"]));
    let (y, yb) = (floats(&d["y"]), floats(&d["y_binary"]));
    let n = y.len();

    // The shuffled partition is the kernel's seeded one, frozen with the references.
    let strata: Vec<u32> = yb.iter().map(|v| *v as u32).collect();
    let continuous_folds =
        shuffled_fold_assignment(&mut CausalRng::from_seed(CV_FOLD_SEED), n, 5, None).unwrap();
    let binary_folds =
        shuffled_fold_assignment(&mut CausalRng::from_seed(CV_FOLD_SEED), n, 5, Some(&strata))
            .unwrap();
    let frozen = |key: &str| -> Vec<usize> {
        d["folds"][key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    };
    assert!(
        continuous_folds == frozen("continuous") && binary_folds == frozen("binary"),
        "fold assignment changed; regenerate the reference with {}",
        serde_json::json!({ "continuous": continuous_folds, "binary": binary_folds })
    );

    let model = model3();
    let registry = registry();
    let continuous =
        table(&[x1.clone(), x2.clone(), y.clone()], &["x1", "x2", "y"], ValueType::Continuous);
    let binary = table(&[x1, x2, yb], &["x1", "x2", "y"], ValueType::Binary);

    // The conditional state space is fit on the full data (EM); its parameters carry the
    // reference's tolerance on the density.
    let lgssm_tolerance = [("lgssm", 1e-6)];
    let logit_tolerance = [("discrete", 1e-7), ("hierarchical_glm", 1e-7)];

    for (scheme, splits) in [
        ("shuffled", ValidationSplits::Shuffled),
        ("time_ordered", ValidationSplits::TimeOrdered { purge: 2 }),
    ] {
        let (_, assignments) = registry
            .assign_and_fit_with(
                &model,
                &continuous,
                SelectionPolicy::BestScore,
                &roots_linear(splits),
            )
            .unwrap();
        let outcome = &assignments[2];
        let what = format!("continuous/{scheme}");
        check_candidates(outcome, &d["continuous"][scheme], &lgssm_tolerance, &what);
        assert_eq!(outcome.provenance.rule, SelectionRule::BestScore);
        assert_eq!(outcome.provenance.folds, 5, "{what}");
        assert_eq!(outcome.provenance.n_rows, n, "{what}");
        assert_eq!(outcome.provenance.seed.is_some(), scheme == "shuffled", "{what}");
        assert_eq!(
            outcome.provenance.candidate_set.iter().map(|f| f.id()).collect::<Vec<_>>(),
            ["linear_gaussian", "hierarchical_linear", "bvar", "lgssm", "constant"]
        );
    }

    for (scheme, splits) in [
        ("shuffled", ValidationSplits::Shuffled),
        ("time_ordered", ValidationSplits::TimeOrdered { purge: 1 }),
    ] {
        let (_, assignments) = registry
            .assign_and_fit_with(&model, &binary, SelectionPolicy::BestScore, &roots_linear(splits))
            .unwrap();
        check_candidates(
            &assignments[2],
            &d["binary"][scheme],
            &logit_tolerance,
            &format!("binary/{scheme}"),
        );
    }

    // A required family is chosen whatever its rank, and the scores do not move.
    let (_, ranked) = registry
        .assign_and_fit_with(
            &model,
            &continuous,
            SelectionPolicy::BestScore,
            &roots_linear(ValidationSplits::Shuffled),
        )
        .unwrap();
    // The policy applies to every node, so the roots admit the family too.
    let both = || NodeFamilyConstraint {
        allowed: Some(Arc::from([MechanismFamily::LinearGaussian, MechanismFamily::Bvar])),
        ..NodeFamilyConstraint::default()
    };
    let (store, required) = registry
        .assign_and_fit_with(
            &model,
            &continuous,
            SelectionPolicy::RequireFamily(MechanismFamily::Bvar),
            &SelectionOptions {
                constraints: vec![
                    (VariableId::from_raw(0), both()),
                    (VariableId::from_raw(1), both()),
                ],
                splits: ValidationSplits::Shuffled,
            },
        )
        .unwrap();
    assert_eq!(required[2].selected, MechanismFamily::Bvar);
    assert_eq!(required[2].provenance.rule, SelectionRule::RequiredByPolicy);
    assert!(matches!(store.slots[2], antecedent_model::MechanismSlot::Bvar { .. }));
    for (a, b) in ranked[2].candidates.iter().zip(required[2].candidates.iter()) {
        assert_eq!(a.family, b.family);
        assert_eq!(a.score.to_bits(), b.score.to_bits(), "{}", a.family.id());
    }
}

#[test]
fn registry_family_selection_reference_exact_ties_keep_the_listing_order() {
    // A constant outcome: a one-level discrete fit and the constant both score exactly zero (log
    // mass one, no free parameter), so the candidate listed first wins, whichever it is.
    let y = vec![2.5; 30];
    let model = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
    let data = table(&[y], &["y"], ValueType::Continuous);
    let registry = MechanismRegistry::standard();
    let (_, assignments) =
        registry.assign_and_fit(&model, &data, SelectionPolicy::BestScore).unwrap();
    let a = &assignments[0];
    assert_eq!(a.candidates.iter().map(|c| c.score).collect::<Vec<_>>(), vec![0.0, 0.0]);
    assert_eq!(
        a.candidates.iter().map(|c| c.family.id()).collect::<Vec<_>>(),
        ["discrete", "constant"]
    );
    assert_eq!(a.selected, MechanismFamily::Discrete);
    for order in [
        [MechanismFamily::Constant, MechanismFamily::Discrete],
        [MechanismFamily::Discrete, MechanismFamily::Constant],
    ] {
        let options = SelectionOptions {
            constraints: vec![(
                VariableId::from_raw(0),
                NodeFamilyConstraint {
                    allowed: Some(Arc::from(order)),
                    ..NodeFamilyConstraint::default()
                },
            )],
            ..SelectionOptions::default()
        };
        let (_, assignments) = registry
            .assign_and_fit_with(&model, &data, SelectionPolicy::BestScore, &options)
            .unwrap();
        assert_eq!(assignments[0].selected, order[0], "the first listed of an exact tie wins");
    }
}
