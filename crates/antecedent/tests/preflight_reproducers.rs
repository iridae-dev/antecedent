//! 2.2 E0 reproducers and E1 preflight, rank-drop and cost checks on bounded synthetic data.
//!
//! The reproducer tests document what the branch does today for the reported workloads
//! (rank 174 of 175, duplicate columns, near-separation, a retargeted overlap refusal) and
//! the E1 tests pin what the new fit-free diagnostics add. Every numeric expectation is
//! derived by hand in the test or from an independent implementation, not from the code
//! under test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact counts and exact copies of columns"
)]

use antecedent::{
    BatchStudy, CausalError, ColumnPriority, EstimatorId, FindingSeverity, PreflightInput,
    PropensityOutcome, RankDropPolicy, RefuteSuite, Study, fit_diagnostics_design, plan_rank_drop,
    preflight_design,
};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, StatsError, fit_propensity,
};

fn stream(seed: u64, index: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, index)
}

fn normals(n: usize, rng: &mut CausalRng) -> Vec<f64> {
    (0..n).map(|_| standard_normal(rng)).collect()
}

fn frame(columns: &[(String, Vec<f64>)]) -> TabularData {
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn id(data: &TabularData, name: &str) -> VariableId {
    data.schema().id_of(name).unwrap()
}

fn named(columns: &[(&str, &Vec<f64>)]) -> TabularData {
    let owned: Vec<(String, Vec<f64>)> =
        columns.iter().map(|(name, values)| ((*name).to_string(), (*values).clone())).collect();
    frame(&owned)
}

fn coin(n: usize, rng: &mut CausalRng) -> Vec<f64> {
    (0..n).map(|_| f64::from(rng.next_f64() < 0.5)).collect()
}

fn codes(report: &antecedent::PreflightReport, code: &str) -> Vec<antecedent::PreflightFinding> {
    report.findings.iter().filter(|f| f.code == code).cloned().collect()
}

/// E0, rank 174 of 175: one exact duplicate among 175 design columns. The QR backend refuses
/// it with the rank and column count only (no column name); preflight names the dependent
/// column and its exact relation, and the two ranks agree.
#[test]
fn rank_174_of_175_is_refused_by_the_backend_and_named_by_preflight() {
    let n = 400;
    let mut rng = stream(1, 1);
    let mut covariates: Vec<Vec<f64>> = (0..173).map(|_| normals(n, &mut rng)).collect();
    covariates.push(covariates[5].clone());
    let t = coin(n, &mut rng);
    let y: Vec<f64> = (0..n).map(|i| t[i] + covariates[0][i] + standard_normal(&mut rng)).collect();

    // Independent oracle: the backend's own column-pivoted QR on the same 175 columns.
    let mut design = vec![1.0; n];
    for column in &covariates {
        design.extend_from_slice(column);
    }
    let backend_error = FaerBackend
        .least_squares(&design, n, 175, &y, &mut LeastSquaresWorkspace::default())
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(backend_error, StatsError::RankDeficient { rank: 174, ncols: 175 }),
        "{backend_error}"
    );

    let mut columns = vec![("t".to_string(), t), ("y".to_string(), y)];
    columns.extend(covariates.into_iter().enumerate().map(|(i, c)| (format!("x{i}"), c)));
    let data = frame(&columns);
    let adjustment: Vec<VariableId> = (0..174).map(|i| id(&data, &format!("x{i}"))).collect();
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(1);
    let report = preflight_design(&input, &ctx).unwrap();

    assert_eq!(report.rows_complete, n);
    assert_eq!(report.adjustment_set.len(), 174, "the adjustment set is reported unchanged");
    let rank = report.rank.as_ref().unwrap();
    assert_eq!((rank.design_columns, rank.numerical_rank), (175, 174));
    assert_eq!(rank.dependent.len(), 1);
    assert_eq!(rank.dependent[0].column, "x173");
    let relation = &rank.dependent[0].explained_by;
    assert_eq!(relation.len(), 1, "{relation:?}");
    assert_eq!(relation[0].column, "x5");
    assert!((relation[0].coefficient - 1.0).abs() < 1e-8);
    assert_eq!(report.duplicates.len(), 1);
    assert_eq!(report.duplicates[0].columns, vec!["x5".to_string(), "x173".to_string()]);
    assert!(report.is_blocked());

    // The blocking finding becomes a typed refusal with structured fields.
    let refusal = report.refusal().expect("a rank-deficient design refuses");
    assert_eq!(refusal.reason_code(), Some("design_rank_deficient"));
    let fields = refusal.refusal_fields().expect("structured fields ride on the refusal");
    assert_eq!(fields.stage.as_deref(), Some("preflight"));
    assert_eq!((fields.numerical_rank, fields.design_columns), (Some(174), Some(175)));
    assert_eq!(fields.implicated_columns, vec!["x173".to_string()]);
    assert!(fields.remedy.is_some());
    // Diagnostics that need a fitted nuisance stay absent: nothing was fit.
    assert!(fields.arm_ess.is_empty() && fields.propensity_min.is_none());
    assert!(fields.propensity_quantiles.is_empty() && fields.cluster_count.is_none());
    // The message is the registered coded text, byte for byte what the typed code promises.
    assert!(refusal.to_string().starts_with("reason=design_rank_deficient: "), "{refusal}");

    // The fit that would follow fails before any score exists: absent, with the rank.
    let fitted = fit_diagnostics_design(&input, &ctx).unwrap();
    match fitted.propensity {
        PropensityOutcome::Absent { numerical_rank, design_columns, .. } => {
            assert_eq!((numerical_rank, design_columns), (Some(174), 175));
        }
        PropensityOutcome::Fitted(_) => panic!("a rank-deficient propensity design cannot fit"),
    }
    let fields = fitted.refusal_fields("propensity_fit", "design rank deficient");
    assert_eq!(fields.numerical_rank, Some(174));
    assert!(fields.propensity_min.is_none() && fields.arm_ess.is_empty());
}

/// E0, 175 columns with near-separation: with more design columns than half the rows, random
/// arm labels are linearly separable (Cover), so the strict propensity fit refuses, while the
/// fit-free preflight sees a full-rank design with no single separating column. Only the fit
/// shows the joint separation, and it is reported flagged rather than as a clean fit.
#[test]
fn a_175_column_propensity_separates_jointly_and_only_the_fit_shows_it() {
    let n = 240;
    let p = 174;
    let mut rng = stream(2, 1);
    let covariates: Vec<Vec<f64>> = (0..p).map(|_| normals(n, &mut rng)).collect();
    let t = coin(n, &mut rng);
    let y = normals(n, &mut rng);

    let mut design = vec![1.0; n];
    for column in &covariates {
        design.extend_from_slice(column);
    }
    let mut workspace = antecedent_stats::PropensityWorkspace::default();
    let strict = fit_propensity(
        &design,
        n,
        p + 1,
        &t,
        &FaerBackend,
        &mut workspace,
        &antecedent_stats::GlmOptions::default(),
    );
    assert!(strict.is_err(), "no maximum-likelihood propensity exists for 175 columns on 240 rows");

    let mut columns = vec![("t".to_string(), t), ("y".to_string(), y)];
    columns.extend(covariates.into_iter().enumerate().map(|(i, c)| (format!("z{i}"), c)));
    let data = frame(&columns);
    let adjustment: Vec<VariableId> = (0..p).map(|i| id(&data, &format!("z{i}"))).collect();
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(2);
    let report = preflight_design(&input, &ctx).unwrap();
    assert!(!report.is_blocked(), "{:?}", report.findings);
    assert_eq!(report.rank.as_ref().unwrap().numerical_rank, 175);
    assert!(codes(&report, "column_separates_arms").is_empty());

    let fitted = fit_diagnostics_design(&input, &ctx).unwrap();
    match fitted.propensity {
        PropensityOutcome::Fitted(fit) => {
            assert!(fit.separated || !fit.converged || fit.boundary_saturated, "{fit:?}");
        }
        PropensityOutcome::Absent { .. } => {}
    }
}

/// E0, near-separation by one column: a single covariate that splits the arms is flagged for
/// review by the fit-free check (a positivity warning, not a verdict), and the diagnostic fit
/// reports flagged scores instead of hiding the saturation behind a ridge.
#[test]
fn one_column_that_separates_the_arms_is_flagged_without_a_fit() {
    let n = 300;
    let mut rng = stream(3, 1);
    let z = normals(n, &mut rng);
    let other = normals(n, &mut rng);
    let t: Vec<f64> = z.iter().map(|v| f64::from(*v > 0.0)).collect();
    let y: Vec<f64> = (0..n).map(|i| t[i] + other[i]).collect();
    let data = named(&[("t", &t), ("y", &y), ("z", &z), ("w", &other)]);
    let adjustment = [id(&data, "z"), id(&data, "w")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(3);
    let report = preflight_design(&input, &ctx).unwrap();
    let flags = codes(&report, "column_separates_arms");
    assert_eq!(flags.len(), 1, "{:?}", report.findings);
    assert_eq!(flags[0].columns, vec!["z".to_string()]);
    assert_eq!(flags[0].severity, FindingSeverity::Review);
    assert!(flags[0].detail.contains("completely"), "{}", flags[0].detail);
    assert!(!report.is_blocked(), "a review flag does not block and changes no set");

    let fitted = fit_diagnostics_design(&input, &ctx).unwrap();
    match fitted.propensity {
        PropensityOutcome::Fitted(fit) => {
            assert!(fit.separated || !fit.converged || fit.boundary_saturated, "{fit:?}");
        }
        PropensityOutcome::Absent { .. } => {}
    }
}

/// E1, duplicate aliases and nearly deterministic but nonidentical columns stay separate: an
/// exact alias is blocking and named with its relation; a column that differs from another by
/// `1e-4` noise is only a review flag, and neither changes the adjustment set.
#[test]
fn exact_aliases_block_and_near_duplicates_only_flag() {
    let n = 300;
    let mut rng = stream(4, 1);
    let t = coin(n, &mut rng);
    let z0 = normals(n, &mut rng);
    let noise = normals(n, &mut rng);
    let z2: Vec<f64> = (0..n).map(|i| z0[i] + 1e-4 * noise[i]).collect();
    let z3 = normals(n, &mut rng);
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + z3[i] + 0.5 * normals(1, &mut rng)[0]).collect();
    let ctx = ExecutionContext::for_tests(4);

    // Near-duplicate only.
    let data = named(&[("t", &t), ("y", &y), ("z0", &z0), ("z2", &z2), ("z3", &z3)]);
    let adjustment = [id(&data, "z0"), id(&data, "z2"), id(&data, "z3")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let report = preflight_design(&input, &ctx).unwrap();
    assert!(!report.is_blocked());
    assert!(report.duplicates.is_empty());
    assert!(report.rank.as_ref().unwrap().dependent.is_empty());
    let near = codes(&report, "near_collinear_column");
    assert_eq!(near.len(), 1, "{:?}", report.findings);
    assert_eq!(near[0].columns, vec!["z2".to_string()]);
    assert_eq!(near[0].severity, FindingSeverity::Review);
    let ratio = near[0].measure.unwrap();
    assert!(ratio > 1e-6 && ratio < 1e-3, "{ratio}");
    assert_eq!(report.adjustment_set, vec!["z0", "z2", "z3"]);

    // Exact alias added.
    let alias = z0.clone();
    let data = named(&[("t", &t), ("y", &y), ("z0", &z0), ("alias", &alias), ("z3", &z3)]);
    let adjustment = [id(&data, "z0"), id(&data, "alias"), id(&data, "z3")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let report = preflight_design(&input, &ctx).unwrap();
    assert!(report.is_blocked());
    let rank = report.rank.as_ref().unwrap();
    assert_eq!((rank.design_columns, rank.numerical_rank), (4, 3));
    assert_eq!(rank.dependent[0].column, "alias");
    assert_eq!(codes(&report, "duplicate_adjustment_columns").len(), 1);
    assert_eq!(report.adjustment_set, vec!["z0", "alias", "z3"]);
}

/// E1, exact collinearity: a column `3a - 2b + 5` is named dependent with coefficients
/// computed by hand, a constant covariate is explained by the intercept, and the rank
/// matches the backend QR on the same columns.
#[test]
fn exact_collinearity_names_the_relation_in_original_units() {
    let n = 200;
    let mut rng = stream(5, 1);
    let t = coin(n, &mut rng);
    let a = normals(n, &mut rng);
    let b = normals(n, &mut rng);
    let combo: Vec<f64> = (0..n).map(|i| 3.0 * a[i] - 2.0 * b[i] + 5.0).collect();
    let constant = vec![7.0; n];
    let y: Vec<f64> = (0..n).map(|i| t[i] + a[i]).collect();
    let data = named(&[
        ("t", &t),
        ("y", &y),
        ("a", &a),
        ("b", &b),
        ("combo", &combo),
        ("constant", &constant),
    ]);
    let adjustment: Vec<VariableId> =
        ["a", "b", "combo", "constant"].iter().map(|n| id(&data, n)).collect();
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let report = preflight_design(&input, &ExecutionContext::for_tests(5)).unwrap();
    let rank = report.rank.as_ref().unwrap();
    assert_eq!((rank.design_columns, rank.numerical_rank), (5, 3));
    let names: Vec<&str> = rank.dependent.iter().map(|d| d.column.as_str()).collect();
    assert_eq!(names, vec!["combo", "constant"]);
    let weight = |column: &str, of: &str| {
        rank.dependent
            .iter()
            .find(|d| d.column == column)
            .and_then(|d| d.explained_by.iter().find(|w| w.column == of))
            .map_or(f64::NAN, |w| w.coefficient)
    };
    assert!((weight("combo", "(intercept)") - 5.0).abs() < 1e-8);
    assert!((weight("combo", "a") - 3.0).abs() < 1e-8);
    assert!((weight("combo", "b") + 2.0).abs() < 1e-8);
    assert!((weight("constant", "(intercept)") - 7.0).abs() < 1e-8);

    // Independent rank oracle: the backend's QR on [1, a, b, combo, constant] reports rank 3.
    let mut design = vec![1.0; n];
    for column in [&a, &b, &combo, &constant] {
        design.extend_from_slice(column);
    }
    let error = FaerBackend
        .least_squares(&design, n, 5, &y, &mut LeastSquaresWorkspace::default())
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(error, StatsError::RankDeficient { rank: 3, ncols: 5 }), "{error}");
}

/// E1, near-determinism review flags: a covariate that near-deterministically predicts the
/// outcome, and one that is an exact copy of the treatment, are flagged for human review (with
/// the measure and threshold) and neither blocks nor changes the adjustment set.
#[test]
fn leakage_like_columns_are_flagged_for_review_only() {
    let n = 400;
    let mut rng = stream(6, 1);
    let t = coin(n, &mut rng);
    let z = normals(n, &mut rng);
    let leak: Vec<f64> = (0..n).map(|i| 3.0 * z[i] + 1e-3 * standard_normal(&mut rng)).collect();
    let y = leak.clone();
    let treated_copy = t.clone();
    let data = named(&[("t", &t), ("y", &y), ("z", &z), ("treated_copy", &treated_copy)]);
    let adjustment = [id(&data, "z"), id(&data, "treated_copy")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let report = preflight_design(&input, &ExecutionContext::for_tests(6)).unwrap();

    let outcome = codes(&report, "adjustment_column_tracks_outcome");
    assert_eq!(outcome.len(), 1, "{:?}", report.findings);
    assert_eq!(outcome[0].columns, vec!["z".to_string(), "y".to_string()]);
    assert!(outcome[0].measure.unwrap() >= outcome[0].threshold.unwrap());
    let span = codes(&report, "outcome_near_determined_by_adjustment");
    assert_eq!(span.len(), 1);
    assert!(span[0].detail.contains("not a causal statement"));
    let treatment = codes(&report, "adjustment_column_tracks_treatment");
    assert_eq!(treatment.len(), 1);
    assert_eq!(treatment[0].columns, vec!["treated_copy".to_string(), "t".to_string()]);
    assert_eq!(codes(&report, "adjustment_duplicates_query_variable").len(), 1);
    assert!(!report.is_blocked(), "review flags never block: {:?}", report.findings);
    assert!(report.refusal().is_none());
    assert_eq!(report.adjustment_set, vec!["z", "treated_copy"]);
}

/// E1, zero overlap: an arm with no rows blocks with `arm_not_populated` and structured
/// per-arm counts; no propensity fit is attempted (absent, with the reason).
#[test]
fn an_empty_arm_is_a_typed_refusal_with_arm_counts() {
    let n = 120;
    let mut rng = stream(7, 1);
    let t = vec![0.0; n];
    let z = normals(n, &mut rng);
    let y = normals(n, &mut rng);
    let data = named(&[("t", &t), ("y", &y), ("z", &z)]);
    let adjustment = [id(&data, "z")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(7);
    let report = preflight_design(&input, &ctx).unwrap();
    assert_eq!(report.arms[0].rows, n);
    assert_eq!(report.arms[1].rows, 0);
    assert!(report.is_blocked());
    let refusal = report.refusal().unwrap();
    assert_eq!(refusal.reason_code(), Some("arm_not_populated"));
    let fields = refusal.refusal_fields().unwrap();
    assert_eq!(fields.arm_ess.len(), 2);
    assert_eq!((fields.arm_ess[1].1).0, 0.0);
    assert_eq!((fields.arm_ess[0].1).0, 120.0);
    assert!(fields.numerical_rank.is_none(), "rank was not part of this refusal");

    let fitted = fit_diagnostics_design(&input, &ctx).unwrap();
    assert!(matches!(
        fitted.propensity,
        PropensityOutcome::Absent { ref reason, .. } if reason.contains("no complete-case rows")
    ));
}

/// E1, missingness: rows with a missing cell leave the complete-case set, per-column counts
/// are exact, and the arm counts are over complete rows only.
#[test]
fn missing_cells_are_counted_per_column_and_leave_the_complete_cases() {
    let n = 100;
    let mut rng = stream(8, 1);
    let t: Vec<f64> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
    let mut z = normals(n, &mut rng);
    let y = normals(n, &mut rng);
    for i in [3, 10, 11] {
        z[i] = f64::NAN;
    }
    let data = named(&[("t", &t), ("y", &y), ("z", &z)]);
    let adjustment = [id(&data, "z")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let report = preflight_design(&input, &ExecutionContext::for_tests(8)).unwrap();
    assert_eq!((report.rows_total, report.rows_complete), (100, 97));
    let missing = |name: &str| {
        report.missingness.iter().find(|m| m.column == name).map(|m| m.non_finite).unwrap()
    };
    assert_eq!((missing("t"), missing("y"), missing("z")), (0, 0, 3));
    // Rows 3, 11 are odd (control), row 10 is even (active): 50 - 1 active, 50 - 2 control.
    assert_eq!(report.arms[1].rows, 49);
    assert_eq!(report.arms[0].rows, 48);
    assert_eq!(report.arms.iter().map(|a| a.rows).sum::<usize>(), 97);
    assert_eq!(report.rows_other_levels, 0);
}

/// E1, the opt-in rank drop declares a priority first: with `b = 2a` the column that goes is
/// the lower-priority one, with the exact relation recorded, and the same data under the other
/// priority drops the other column. The design identity records the outcome.
#[test]
fn the_rank_drop_follows_the_declared_priority_and_records_the_relation() {
    let n = 150;
    let mut rng = stream(9, 1);
    let t = coin(n, &mut rng);
    let a = normals(n, &mut rng);
    let b: Vec<f64> = a.iter().map(|v| 2.0 * v).collect();
    let c = normals(n, &mut rng);
    let y: Vec<f64> = (0..n).map(|i| t[i] + a[i] + c[i]).collect();
    let data = named(&[("t", &t), ("y", &y), ("a", &a), ("b", &b), ("c", &c)]);
    let (ia, ib, ic) = (id(&data, "a"), id(&data, "b"), id(&data, "c"));
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[ia, ib, ic],
        0.0,
        1.0,
    );
    let ctx = ExecutionContext::for_tests(9);

    let adjustment_order = RankDropPolicy { priority: ColumnPriority::AdjustmentOrder };
    let plan = plan_rank_drop(&input, &adjustment_order, &ctx).unwrap();
    assert_eq!(plan.dropped.len(), 1);
    assert_eq!(plan.dropped[0].column, "b");
    assert_eq!(plan.dropped[0].explained_by.len(), 1);
    assert_eq!(plan.dropped[0].explained_by[0].column, "a");
    assert!((plan.dropped[0].explained_by[0].coefficient - 2.0).abs() < 1e-8);
    assert_eq!(plan.kept_adjustment, vec!["a", "c"]);
    assert_eq!(plan.original_adjustment, vec!["a", "b", "c"]);
    assert_eq!(plan.numerical_rank, 3);
    assert_eq!(plan.design_identity, "adjustment=[a,c];dropped=[b];priority=[a,b,c]");

    let declared = RankDropPolicy { priority: ColumnPriority::Declared(vec![ib, ic, ia]) };
    let plan = plan_rank_drop(&input, &declared, &ctx).unwrap();
    assert_eq!(plan.dropped[0].column, "a");
    assert!((plan.dropped[0].explained_by[0].coefficient - 0.5).abs() < 1e-8);
    assert_eq!(plan.kept_adjustment, vec!["b", "c"]);
    assert_eq!(plan.priority, vec!["b", "c", "a"]);
    // Deterministic: the same inputs give the same plan.
    assert_eq!(plan, plan_rank_drop(&input, &declared, &ctx).unwrap());
}

/// E1, the rank drop refuses (and drops nothing) when the priority does not cover the
/// adjustment set, when a dependent column is a protected query column, and when an
/// adjustment column aliases the treatment.
#[test]
fn the_rank_drop_refuses_when_a_query_column_or_the_priority_is_at_stake() {
    let n = 150;
    let mut rng = stream(10, 1);
    let t = coin(n, &mut rng);
    let a = normals(n, &mut rng);
    let b: Vec<f64> = a.iter().map(|v| -v).collect();
    let y = normals(n, &mut rng);
    let data = named(&[("t", &t), ("y", &y), ("a", &a), ("b", &b)]);
    let (ia, ib) = (id(&data, "a"), id(&data, "b"));
    let ctx = ExecutionContext::for_tests(10);
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &[ia, ib], 0.0, 1.0);

    let incomplete = RankDropPolicy { priority: ColumnPriority::Declared(vec![ia]) };
    let refusal = plan_rank_drop(&input, &incomplete, &ctx).unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));
    let fields = refusal.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("rank_drop"));
    assert_eq!(fields.implicated_columns, vec!["b".to_string()]);

    // An effect modifier that the drop would remove is protected.
    let mut protected = input.clone();
    protected.protected = vec![ib];
    let adjustment_order = RankDropPolicy { priority: ColumnPriority::AdjustmentOrder };
    let refusal = plan_rank_drop(&protected, &adjustment_order, &ctx).unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));
    assert_eq!(refusal.refusal_fields().unwrap().implicated_columns, vec!["b".to_string()]);

    // An adjustment column that is a copy of the treatment is a treatment-definition question.
    let alias = t.clone();
    let data = named(&[("t", &t), ("y", &y), ("a", &a), ("alias", &alias)]);
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[id(&data, "a"), id(&data, "alias")],
        0.0,
        1.0,
    );
    let refusal = plan_rank_drop(&input, &adjustment_order, &ctx).unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));
    assert!(refusal.refusal_fields().unwrap().implicated_columns.contains(&"alias".to_string()));
}

/// The checks observe cancellation: a cancelled context stops preflight and the fit
/// diagnostics with a typed cancellation rather than a partial report.
#[test]
fn preflight_and_fit_diagnostics_observe_cancellation() {
    let n = 60;
    let mut rng = stream(11, 1);
    let t = coin(n, &mut rng);
    let z = normals(n, &mut rng);
    let y = normals(n, &mut rng);
    let data = named(&[("t", &t), ("y", &y), ("z", &z)]);
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[id(&data, "z")],
        0.0,
        1.0,
    );
    let ctx = ExecutionContext::for_tests(11);
    ctx.cancellation.cancel();
    assert!(matches!(preflight_design(&input, &ctx), Err(CausalError::Cancelled { .. })));
    assert!(matches!(fit_diagnostics_design(&input, &ctx), Err(CausalError::Cancelled { .. })));
}

/// The three new reason codes are registered runtime refusals.
#[test]
fn the_new_reason_codes_are_registered() {
    for code in ["design_rank_deficient", "rank_drop_not_licensed", "arm_not_populated"] {
        assert!(antecedent_core::reason_code::is_runtime_refusal(code), "{code}");
    }
}

fn confounded(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = stream(seed, 0xE1);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut t2 = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-0.9 * zi).exp()));
        t2[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (0.5 * zi).exp()));
        y[i] = t[i] + 0.5 * t2[i] + zi + 0.4 * standard_normal(&mut rng);
    }
    let data = named(&[("t", &t), ("y", &y), ("z", &z), ("t2", &t2)]);
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(2, 0), (2, 1), (0, 1), (2, 3), (3, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn kish(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    sum * sum / weights.iter().map(|w| w * w).sum::<f64>()
}

/// E1 on a prepared plan: the report reads the certified adjustment set and the retained
/// table, the fitted diagnostics match an independent propensity fit and hand-computed Kish
/// ESS, and the cost counts match the hand derivation for cross-fitted AIPW.
#[test]
fn a_prepared_plan_diagnoses_fit_free_then_fitted_and_counts_its_cost() {
    let n = 500;
    let (data, graph) = confounded(n, 12);
    let ctx = ExecutionContext::for_tests(12);
    let query = AverageEffectQuery::binary_ate(id(&data, "t"), id(&data, "y"));
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();

    let report = prepared.diagnose(&ctx).unwrap();
    assert!(report.adjustment_set.contains(&"z".to_string()), "{:?}", report.adjustment_set);
    assert!(!report.adjustment_set.iter().any(|name| name == "t" || name == "y"));
    let design_columns = report.adjustment_set.len() + 1;
    assert_eq!((report.rows_total, report.rows_complete), (n, n));
    assert_eq!(report.arms.iter().map(|a| a.rows).sum::<usize>(), n);
    assert_eq!(report.rows_other_levels, 0);
    assert!(!report.is_blocked(), "{:?}", report.findings);
    assert_eq!(report.rank.as_ref().unwrap().numerical_rank, design_columns);

    // Fitted diagnostics against an independent fit of the same design.
    let fitted = prepared.diagnose_fit(&ctx).unwrap();
    let PropensityOutcome::Fitted(fit) = fitted.propensity else { panic!("a clean design fits") };
    assert!(fit.converged && !fit.separated && !fit.boundary_saturated);
    let t = data.float64_values(id(&data, "t")).unwrap();
    let mut design = vec![1.0; n];
    for name in &report.adjustment_set {
        design.extend(data.float64_values(id(&data, name)).unwrap());
    }
    let oracle = fit_propensity(
        &design,
        n,
        design_columns,
        &t,
        &FaerBackend,
        &mut antecedent_stats::PropensityWorkspace::default(),
        &antecedent_stats::GlmOptions::default(),
    )
    .unwrap();
    let lo = oracle.scores.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = oracle.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((fit.min - lo).abs() < 1e-12 && (fit.max - hi).abs() < 1e-12);
    assert_eq!(fit.quantiles.len(), 7);
    assert!(fit.quantiles.windows(2).all(|w| w[0].value <= w[1].value));
    assert!(fit.min <= fit.quantiles[0].value && fit.quantiles[6].value <= fit.max);
    let treated: Vec<f64> =
        oracle.scores.iter().zip(&t).filter(|(_, ti)| **ti > 0.5).map(|(e, _)| 1.0 / e).collect();
    let control: Vec<f64> = oracle
        .scores
        .iter()
        .zip(&t)
        .filter(|(_, ti)| **ti < 0.5)
        .map(|(e, _)| 1.0 / (1.0 - e))
        .collect();
    let ess = |label: &str| fit.arm_ess.iter().find(|a| a.label.starts_with(label)).unwrap();
    assert_eq!(ess("active").rows, treated.len());
    assert_eq!(ess("control").rows, control.len());
    assert!((ess("active").ess - kish(&treated)).abs() < 1e-8);
    assert!((ess("control").ess - kish(&control)).abs() < 1e-8);

    // A clean design drops nothing.
    let plan = prepared
        .plan_rank_drop(&RankDropPolicy { priority: ColumnPriority::AdjustmentOrder }, &ctx)
        .unwrap();
    assert!(plan.dropped.is_empty());
    let listed = report.adjustment_set.join(",");
    assert_eq!(
        plan.design_identity,
        format!("adjustment=[{listed}];dropped=[];priority=[{listed}]")
    );

    // Cost: 5 folds x (1 propensity + 2 outcome) fits in the single pass of 0 replicates.
    let cost = prepared.estimate_cost().unwrap();
    assert!(cost.planning_hint);
    assert_eq!(cost.estimator, "aipw");
    assert_eq!(cost.crossfit_folds, Some(5));
    assert_eq!(cost.nuisance_fits_per_pass, Some(15));
    assert_eq!(cost.passes_upper_bound, 1);
    assert_eq!(cost.nuisance_fits_upper_bound, Some(15));
    assert_eq!(cost.design_matrix_bytes, Some(u64::try_from(n * design_columns * 8).unwrap()));
    assert_eq!(cost.inference.mode, "frequentist");
    assert!(cost.inference.refit_warning.is_none());
    assert!(cost.seconds.is_none() && !cost.seconds_basis.is_empty());
}

/// E1 on a prepared batch: one report and cost per claim, the shared covariate design is
/// reported, and cost grows with the number of claims.
#[test]
fn a_prepared_batch_diagnoses_and_costs_every_claim() {
    let n = 500;
    let (data, graph) = confounded(n, 13);
    let ctx = ExecutionContext::for_tests(13);
    let q1 = AverageEffectQuery::binary_ate(id(&data, "t"), id(&data, "y"));
    let q2 = AverageEffectQuery::binary_ate(id(&data, "t2"), id(&data, "y"));
    let batch = BatchStudy::new(data.clone(), graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(10);
    let both = batch.prepare(&[q1.clone(), q2], &ctx).unwrap();
    let one = batch.prepare(&[q1], &ctx).unwrap();

    let report = both.diagnose(&ctx).unwrap();
    assert_eq!(report.reports.len(), 2);
    let same_sets = report.reports[0].adjustment_set == report.reports[1].adjustment_set;
    assert_eq!(report.shares_covariates, same_sets);
    assert!(report.blocked_plans().is_empty());
    assert_eq!(report.reports[0].treatment, vec!["t"]);
    assert_eq!(report.reports[1].treatment, vec!["t2"]);
    assert!(report.reports.iter().all(|r| r.adjustment_set.contains(&"z".to_string())));

    let cost_both = both.estimate_cost().unwrap();
    let cost_one = one.estimate_cost().unwrap();
    assert_eq!((cost_both.claims, cost_one.claims), (2, 1));
    // 5 folds x 3 fits x (1 + 10 replicates) per claim (an upper bound).
    assert_eq!(cost_one.nuisance_fits_upper_bound, Some(165));
    assert_eq!(cost_both.nuisance_fits_upper_bound, Some(330));
    assert_eq!(cost_both.bootstrap_replicates_total, 20);
    assert!(cost_both.plans[0].inference.refit_warning.is_some());
    let columns = report.reports[0].adjustment_set.len() + 1;
    assert_eq!(cost_both.plans[0].shared_covariate_bytes.is_some(), report.shares_covariates);
    assert_eq!(
        cost_both.plans[0].design_matrix_bytes,
        Some(u64::try_from(n * columns * 8).unwrap())
    );
    assert_eq!(both.diagnose_fit(&ctx).unwrap().len(), 2);
}

/// E0, retargeted overlap refusal (current behavior): a row-weight retarget onto a region the
/// treated arm never reaches refuses as a support failure. The refusal carries no structured
/// fields today; the E1 container exists but this estimator-owned site is not wired to it.
#[test]
fn a_retarget_onto_unsupported_rows_refuses_as_support_without_structured_fields() {
    let mut rng = stream(14, 1);
    let n = 800usize;
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t[i] = f64::from(zi < 0.0 && rng.next_f64() < 0.7);
        y[i] = t[i] + 0.2 * standard_normal(&mut rng);
        w[i] = f64::from(zi > 1.2);
    }
    let data = named(&[("t", &t), ("y", &y), ("z", &z)]);
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(14);
    let prepared = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let error = prepared.retarget(&w, &[VariableId::from_raw(2)], &ctx).unwrap_err();
    assert!(matches!(error, CausalError::Support { .. }), "{error}");
    assert!(error.refusal_fields().is_none());
}
