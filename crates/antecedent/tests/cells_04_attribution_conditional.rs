//! Anomaly attribution and the static conditional effect against inputs they must refuse or
//! score by definition.
//!
//! The attribution scores a factual row against fitted mechanisms. Its structural model is
//! `z -> t -> y`, `z -> y` with linear mechanisms, so a missing value, a categorical column, a
//! constant column and an exactly duplicated parent each have a definite outcome: a refusal that
//! names the cause (no mechanism can be fitted, the column is not numeric, a value is missing),
//! or, for a constant target, the score that the definition gives (the factual value is the
//! reference, so the tail probability is one and every score and contribution is exactly zero).
//! A declared missing-data policy is refused with `missing_policy_not_applicable`, because the
//! route reads its rows in its own way.
//!
//! The conditional effect `y = 1 + t (1 + 2 z) + 0.5 z` with a balanced binary modifier `z` has
//! the stratum effects 1 and 3 and the modifier-averaged effect 2. A table in which one modifier
//! stratum has no treated rows, or no row is treated at all, has no effect to report and is
//! refused with `design_rank_deficient` rather than answered from the other stratum.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AnomalyAttributionQuery, AverageEffectQuery, CategoryDomainId, CausalQuery,
    CausalSchemaBuilder, ConditionalEffectQuery, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, ImputationSpec,
    MissingDeclaration, MissingPolicy, OwnedColumn, OwnedColumnarStorage, TabularData,
    UnknownCategoryPolicy, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn dag(edges: &[(u32, u32)], variables: u32) -> Dag {
    let mut graph = Dag::with_variables(variables);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn refused(result: Result<StudyResult, CausalError>, what: &str) -> CausalError {
    result.err().unwrap_or_else(|| panic!("{what}: published a result"))
}

// ---------------------------------------------------------------------------------------------
// Anomaly attribution
// ---------------------------------------------------------------------------------------------

const N: usize = 100;

#[derive(Clone, Copy, Default)]
struct Fault {
    constant_root: bool,
    constant_parent: bool,
    constant_target: bool,
    duplicated_parent: bool,
    missing_target_row: Option<usize>,
}

/// Columns `z, t, y` (and `z2 = 2 z + 1`, an exact copy of the root up to scale, when asked).
fn columns(fault: Fault) -> Vec<(&'static str, Vec<f64>)> {
    let at = |i: usize| i as f64;
    let z: Vec<f64> =
        (0..N).map(|i| if fault.constant_root { 2.0 } else { (at(i) * 0.37).sin() }).collect();
    let t: Vec<f64> = (0..N)
        .map(|i| if fault.constant_parent { 1.0 } else { 0.6 * z[i] + (at(i) * 0.71).cos() })
        .collect();
    let mut y: Vec<f64> = (0..N)
        .map(|i| {
            if fault.constant_target {
                5.0
            } else {
                1.0 + 2.0 * t[i] + z[i] + 0.5 * (at(i) * 1.3).sin()
            }
        })
        .collect();
    if let Some(row) = fault.missing_target_row {
        y[row] = f64::NAN;
    }
    let mut columns = vec![("z", z.clone()), ("t", t), ("y", y)];
    if fault.duplicated_parent {
        columns.push(("z2", z.iter().map(|value| 2.0 * value + 1.0).collect()));
    }
    columns
}

fn table(columns: &[(&str, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, c)| (*name, c.as_slice()))).unwrap()
}

fn attribute(
    data: TabularData,
    duplicated: bool,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let graph = if duplicated {
        dag(&[(0, 1), (0, 2), (1, 2), (3, 2)], 4)
    } else {
        dag(&[(0, 1), (0, 2), (1, 2)], 3)
    };
    let mut builder = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(2)], N)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(declaration) = missing {
        builder = builder.missing(declaration);
    }
    builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(3)))
}

/// A table with one categorical column (`t`, three levels) and two continuous ones.
fn with_categorical(column: usize) -> TabularData {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let code: Vec<u32> = (0..N).map(|i| (i % 3) as u32).collect();
    let y: Vec<f64> =
        (0..N).map(|i| 1.0 + 2.0 * f64::from(code[i]) + z[i] + 0.5 * (at(i) * 1.3).sin()).collect();
    let mut builder = CausalSchemaBuilder::new();
    let mut built = Vec::new();
    for (index, name) in ["z", "t", "y"].iter().enumerate() {
        let id = v(u32::try_from(index).unwrap());
        if index == column {
            builder
                .add_variable(
                    *name,
                    ValueType::Categorical,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    Some(CategoryDomainId::from_raw(1)),
                    MeasurementSpec::default(),
                )
                .unwrap();
            let domain = CategoryDomain::try_new(
                CategoryDomainId::from_raw(1),
                ["a", "b", "c"]
                    .into_iter()
                    .map(|label| CategoryLevel { label: Arc::from(label) })
                    .collect::<Vec<_>>(),
                false,
                Some(CategoryCode::from_raw(0)),
                UnknownCategoryPolicy::Fail,
            )
            .unwrap();
            built.push(OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    id,
                    code.iter().map(|c| CategoryCode::from_raw(*c)).collect::<Vec<_>>(),
                    ValidityBitmap::all_valid(N),
                    Arc::new(domain),
                )
                .unwrap(),
            ));
        } else {
            builder
                .add_variable(
                    *name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
            let values: Vec<f64> = match index {
                0 => z.clone(),
                1 => code.iter().map(|c| f64::from(*c)).collect(),
                _ => y.clone(),
            };
            built.push(OwnedColumn::Float64(
                Float64Column::new(
                    id,
                    Arc::<[f64]>::from(values.as_slice()),
                    ValidityBitmap::all_valid(N),
                )
                .unwrap(),
            ));
        }
    }
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), built, None, None).unwrap(),
    )
}

#[test]
fn the_anomaly_attribution_refuses_a_missing_value_a_categorical_column_and_every_missing_policy() {
    // The attribution works on the complete numeric table, and its contributions add up to the
    // score above the baseline, row by row (the efficiency of the noise Shapley value).
    let result = attribute(table(&columns(Fault::default())), false, None).unwrap();
    let scores = &result.anomaly().expect("anomaly payload")[0];
    let players = scores.noise_components.len();
    assert_eq!(players, 3);
    for row in 0..scores.rows.len() {
        let total: f64 =
            scores.noise_contributions[row * players..(row + 1) * players].iter().sum();
        assert!(
            (total - (scores.scores[row] - scores.baseline_scores[row])).abs() < 1e-9,
            "row {row}"
        );
    }

    // A missing target value is refused whatever policy is declared: the route reads its rows in
    // its own way and executes no policy.
    let holey = || table(&columns(Fault { missing_target_row: Some(5), ..Fault::default() }));
    let error = refused(attribute(holey(), false, None), "undeclared");
    assert!(error.to_string().contains("non-finite factual value"), "{error}");
    for (name, policy) in [
        ("complete case", MissingPolicy::CompleteCase),
        ("error", MissingPolicy::ErrorOnMissing),
        ("multiple imputation", MissingPolicy::MultipleImputation(ImputationSpec::new())),
        ("inverse probability", MissingPolicy::InverseProbabilityOfObservation),
    ] {
        let error = refused(attribute(holey(), false, Some(MissingDeclaration::new(policy))), name);
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{name}: {error}");
        assert!(
            error.to_string().contains("does not execute a missing-data policy"),
            "{name}: {error}"
        );
    }

    // A categorical column is not read as the numbers of its codes.
    for (column, what) in [(1, "categorical parent"), (2, "categorical target")] {
        let error = refused(attribute(with_categorical(column), false, None), what);
        assert!(error.to_string().contains("is not of type float64"), "{what}: {error}");
    }
}

#[test]
fn the_anomaly_attribution_refuses_constant_and_duplicated_parents_and_scores_a_constant_target_zero()
 {
    for (what, fault, duplicated) in [
        ("constant root", Fault { constant_root: true, ..Fault::default() }, false),
        ("constant parent", Fault { constant_parent: true, ..Fault::default() }, false),
        ("duplicated parent", Fault { duplicated_parent: true, ..Fault::default() }, true),
    ] {
        let error = refused(attribute(table(&columns(fault)), duplicated, None), what);
        let text = error.to_string();
        assert!(text.contains("no mechanism candidates"), "{what}: {text}");
        assert!(text.contains("rank deficient"), "{what}: {text}");
    }
    // A constant target equals its own reference: the two-sided tail probability is one, so the
    // score is -ln 1 = 0 for every row and no noise contributes to it.
    let result = attribute(
        table(&columns(Fault { constant_target: true, ..Fault::default() })),
        false,
        None,
    )
    .unwrap();
    let scores = &result.anomaly().expect("anomaly payload")[0];
    assert!(scores.scores.iter().all(|score| *score == 0.0), "{:?}", &scores.scores[..4]);
    assert!(scores.baseline_scores.iter().all(|score| *score == 0.0));
    assert!(scores.noise_contributions.iter().all(|share| *share == 0.0));
}

// ---------------------------------------------------------------------------------------------
// Conditional effect
// ---------------------------------------------------------------------------------------------

/// `t` alternates within each modifier stratum; `treated_in_one` leaves the stratum `z = 1`
/// untreated and `never_treated` leaves every row untreated.
fn conditional_table(treated_in_one: bool, never_treated: bool) -> TabularData {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
    let t: Vec<f64> =
        (0..n)
            .map(|i| {
                if never_treated || (treated_in_one && z[i] == 1.0) { 0.0 } else { (i % 2) as f64 }
            })
            .collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + t[i] * (1.0 + 2.0 * z[i]) + 0.5 * z[i]).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn conditional(data: TabularData) -> Result<StudyResult, CausalError> {
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers(vec![v(2)]),
    )
    .unwrap();
    Study::tabular(data)
        .graph(dag(&[(2, 0), (2, 1), (0, 1)], 3))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .query(CausalQuery::ConditionalEffect(query))
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn the_conditional_effect_refuses_a_modifier_stratum_or_a_table_with_no_treated_rows() {
    // Strata effects 1 and 3 on equal shares of the modifier: the conditional effect is 2.
    let effect =
        conditional(conditional_table(false, false)).unwrap().estimate().as_effect().unwrap().ate;
    assert!((effect - 2.0).abs() < 1e-9, "{effect}");
    for (what, data) in [
        ("a stratum without treated rows", conditional_table(true, false)),
        ("no treated rows at all", conditional_table(false, true)),
    ] {
        let error = refused(conditional(data), what);
        assert_eq!(error.reason_code(), Some("design_rank_deficient"), "{what}: {error}");
    }
}
