#![cfg_attr(
    test,
    allow(
        clippy::many_single_char_names,
        clippy::float_cmp,
        reason = "this unit-test module compares floats that are copied, clamped or hand-set without rounding, so exact equality is intended"
    )
)]

use antecedent_core::StreamDomain;

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
    TargetPopulation, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_expr::ExprId;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

use super::*;
use crate::overlap::OverlapPolicy;
use antecedent_kernels::standard_normal;

/// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
/// True ATE = 2. Matches the propensity-estimator test fixture (`crate::propensity`).
fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let (t, y, z) = confounded_columns(n, seed);
    build_dataset(t, y, z)
}

fn confounded_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);

    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.5 + zi;
        let p = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    (t, y, z)
}

fn build_dataset(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn identified_aipw(data_estimand: &IdentifiedEstimand) -> IdentificationResult {
    let query = antecedent_core::AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    );
    let mut arena = antecedent_expr::CausalExprArena::new();
    let functional = arena.backdoor_ate(
        query.treatment,
        query.outcome,
        &[VariableId::from_raw(2)],
        antecedent_core::Value::f64(1.0),
        antecedent_core::Value::f64(0.0),
    );
    IdentificationResult::identified(
        antecedent_core::CausalQuery::average_effect(query),
        vec![IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::clone(&data_estimand.adjustment_set),
            functional,
        )],
        arena,
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    )
}

#[test]
fn aipw_runs_the_complete_interval_procedure_on_both_bootstrap_routes() {
    use crate::util::interval_test_support::{assert_procedure, met_early, never_met};
    let (data, estimand) = confounded_scm(300, 12);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (cap, wave) = (90, 30);
    let est = AipwAte::new().with_bootstrap_replicates(cap);
    let fit = |est: &AipwAte| {
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        est.fit(
            &prep,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(3),
            AssumptionSet::new(),
        )
        .unwrap()
    };
    // The cross-fitted route (untrimmed AllObserved).
    assert_procedure(
        &fit(&est),
        &fit(&est.clone().with_bootstrap_interval(never_met(wave))),
        &fit(&est.clone().with_bootstrap_interval(met_early(wave, 60))),
        cap,
        wave,
    );
    // The penalized refit bootstrap keeps its penalties beside the same procedure.
    let ridge = ridge_estimator(&[0.5, 5.0], 3).with_bootstrap_replicates(cap);
    let fixed = fit(&ridge);
    let never = fit(&ridge.clone().with_bootstrap_interval(never_met(wave)));
    let early = fit(&ridge.clone().with_bootstrap_interval(met_early(wave, 60)));
    assert_procedure(&fixed, &never, &early, cap, wave);
    let kept = |e: &EffectEstimate| {
        e.diagnostics
            .penalized
            .as_ref()
            .unwrap()
            .variance
            .as_ref()
            .unwrap()
            .replicate_penalties
            .len()
    };
    assert_eq!(kept(&never), kept(&fixed));
    assert_eq!(
        kept(&early),
        usize::try_from(early.uncertainty.bootstrap.replicates_ok.unwrap()).unwrap()
    );
}

#[test]
fn checked_aipw_executes_retained_target_and_rebinds_rows() {
    let (data, estimand) = confounded_scm(2400, 9182);
    let identification = identified_aipw(&estimand);
    let estimator = AipwAte::new().with_bootstrap_replicates(0);
    let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
    assert_eq!(checked.lowering().functional, checked.target().functional);
    assert_eq!(checked.lowering().treatment, VariableId::from_raw(0));
    assert_eq!(checked.lowering().outcome, VariableId::from_raw(1));
    assert_eq!(checked.lowering().adjustment.as_ref(), &[VariableId::from_raw(2)]);
    assert_eq!(checked.lowering().procedure, CheckedAipwProcedure::CrossFittedLogisticOls);
    assert_eq!(checked.lowering().rows.as_ref(), checked.problem().row_index.as_ref());
    let rebound = estimator.rebind_checked(&checked, &data).unwrap();
    assert_eq!(rebound.program().mapping(), checked.program().mapping());
    assert_eq!(rebound.lowering().rows.as_ref(), checked.lowering().rows.as_ref());

    let mut workspace = AipwWorkspace::default();
    let estimate =
        estimator.fit_checked(&checked, &mut workspace, &ExecutionContext::for_tests(441)).unwrap();
    assert!((estimate.ate - 2.0).abs() < 0.15, "AIPW estimate {}", estimate.ate);
}

#[test]
fn checked_aipw_refuses_target_with_wrong_roles() {
    let (data, estimand) = confounded_scm(300, 33);
    let mut identification = identified_aipw(&estimand);
    identification.estimands[0] = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(0)]),
        identification.estimands[0].functional,
    );
    let estimator = AipwAte::new().with_bootstrap_replicates(0);
    assert!(estimator.prepare_checked(&data, &identification, 0).is_err());
}

/// A trimming policy lowers to the full-sample common-support procedure and
/// the checked fit is the estimator's own trimmed fit, bit for bit.
#[test]
fn checked_trimmed_aipw_retains_its_policy_and_matches_the_plain_fit() {
    let (data, estimand) = confounded_scm(1200, 4171);
    let identification = identified_aipw(&estimand);
    let overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) };
    let estimator = AipwAte { overlap, ..AipwAte::new() }.with_bootstrap_replicates(6);
    let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
    assert_eq!(checked.lowering().procedure, CheckedAipwProcedure::TrimmedLogisticOls);
    assert_eq!(checked.lowering().overlap, overlap);
    assert_eq!(checked.lowering().folds, 0);
    assert_eq!(checked.problem().overlap, overlap);
    let rebound = estimator.rebind_checked(&checked, &data).unwrap();
    assert_eq!(rebound.lowering().procedure, CheckedAipwProcedure::TrimmedLogisticOls);

    let mut workspace = AipwWorkspace::default();
    let sealed =
        estimator.fit_checked(&checked, &mut workspace, &ExecutionContext::for_tests(441)).unwrap();
    let query = identification.average_effect().unwrap();
    let problem = estimator.prepare(&data, &identification.estimands[0], query).unwrap();
    let plain = estimator
        .fit(&problem, &mut workspace, &ExecutionContext::for_tests(441), AssumptionSet::new())
        .unwrap();
    assert_eq!(sealed.ate.to_bits(), plain.ate.to_bits());
    assert_eq!(sealed.se_analytic().to_bits(), plain.se_analytic().to_bits());
    assert_eq!(sealed.se_bootstrap().map(f64::to_bits), plain.se_bootstrap().map(f64::to_bits));
    assert!(sealed.scores.table.is_none(), "the trimmed fit has no cross-fitted score table");
    assert!(
        sealed
            .diagnostics
            .overlap_report
            .as_ref()
            .is_some_and(|report| report.excluded_fraction > 0.0)
    );

    // An estimator whose policy differs from the receipt cannot execute it.
    let untrimmed = AipwAte::new().with_bootstrap_replicates(6);
    assert!(untrimmed.fit_checked(&checked, &mut workspace, &ctx()).is_err());
    assert!(untrimmed.rebind_checked(&checked, &data).is_err());
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

/// The facade attaches the bootstrap to a replicate-free point fit; every AIPW route
/// (cross-fitted untrimmed ATE, trimmed/ATT nuisance refit) must publish the same estimate.
#[test]
fn aipw_attached_bootstrap_equals_the_one_shot_fit_on_every_route() {
    let (data, estimand) = confounded_scm(400, 3);
    let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let att = ate.clone().with_target_population(TargetPopulation::Treated);
    for query in [ate, att] {
        let boot = AipwAte::new().with_bootstrap_replicates(30);
        let prep = boot.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let full = boot.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let point_only = boot.clone().with_bootstrap_replicates(0);
        let point = point_only.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(point.se_bootstrap().is_none());
        let attached = boot.attach_bootstrap(&prep, &mut ws, &ctx(), point).unwrap();
        assert_eq!(attached.ate.to_bits(), full.ate.to_bits());
        assert!(full.se_bootstrap().is_some());
        assert_eq!(attached.se_bootstrap(), full.se_bootstrap());
        assert_eq!(
            attached.uncertainty.bootstrap.replicates_ok,
            full.uncertainty.bootstrap.replicates_ok
        );
    }
}

#[test]
fn aipw_recovers_ate_two() {
    let (data, estimand) = confounded_scm(800, 1);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = AipwAte::new().with_bootstrap_replicates(30);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = AipwWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
    assert!(effect.se_bootstrap().is_some());
    assert!(effect.diagnostics.overlap_report.is_some());
}

#[test]
fn aipw_rejects_explicit_override() {
    let (data, estimand) = confounded_scm(200, 2);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = AipwAte { overlap: OverlapPolicy::ExplicitOverride, ..AipwAte::new() };
    let err = est.prepare(&data, &estimand, &query).unwrap_err();
    assert!(matches!(err, EstimationError::Overlap { .. }));
}

#[test]
fn aipw_recovers_att_two() {
    let (data, estimand) = confounded_scm(1_200, 3);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let est = AipwAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = AipwWorkspace::default();
    let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!((fit.ate - 2.0).abs() < 0.35, "att={}", fit.ate);
}

/// ATT IF remains unbiased when μ₁ is misspecified but propensity is correct.
/// The old formula divided the treated residual by `e`, breaking double robustness.
#[test]
fn att_if_doubly_robust_under_mu1_misspecification() {
    let n = 4_000usize;
    let mut rng = ExecutionContext::for_tests(42).rng.stream_for(StreamDomain::Estimate, 0xA11u64);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    for i in 0..n {
        let z = standard_normal(&mut rng);
        let logit = -0.5 + z;
        let pi_i = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        t[i] = ti;
        y[i] = 2.0 * ti + z + noise;
        e[i] = pi_i;
        // Correct μ₀; deliberately wrong μ₁ (constant 0 instead of 2+z).
        mu0[i] = z;
        mu1[i] = 0.0;
    }
    let mut psi = Vec::new();
    aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Treated, &mut psi).unwrap();
    let att = psi.iter().sum::<f64>() / psi.len() as f64;
    assert!(
        (att - 2.0).abs() < 0.15,
        "ATT IF mean under μ₁ misspecification should stay near 2; got {att}"
    );
}

/// Guide MATH-003 deterministic counterexample: correct e, wrong m0 → ATC = 2.
#[test]
fn atc_if_deterministic_counterexample() {
    // P(T)=0.5, e=0.5, Y(0)=1, Y(1)=3, m1=3, m0=0 → true ATC = 2.
    let t = vec![1.0, 1.0, 0.0, 0.0];
    let y = vec![3.0, 3.0, 1.0, 1.0];
    let e = vec![0.5, 0.5, 0.5, 0.5];
    let mu0 = vec![0.0, 0.0, 0.0, 0.0];
    let mu1 = vec![3.0, 3.0, 3.0, 3.0];
    let mut psi = Vec::new();
    aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
    let atc = psi.iter().sum::<f64>() / psi.len() as f64;
    assert!((atc - 2.0).abs() < 1e-12, "deterministic ATC IF mean should be 2; got {atc}");
}

/// ATC IF remains unbiased when μ₀ is misspecified but propensity is correct.
#[test]
fn atc_if_doubly_robust_under_mu0_misspecification() {
    let n = 4_000usize;
    let mut rng = ExecutionContext::for_tests(42).rng.stream_for(StreamDomain::Estimate, 0xA7Cu64);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    for i in 0..n {
        let z = standard_normal(&mut rng);
        let logit = -0.5 + z;
        let pi_i = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        t[i] = ti;
        y[i] = 2.0 * ti + z + noise;
        e[i] = pi_i;
        // Correct μ₁; deliberately wrong μ₀ (constant 0 instead of z).
        mu0[i] = 0.0;
        mu1[i] = 2.0 + z;
    }
    let mut psi = Vec::new();
    aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
    let atc = psi.iter().sum::<f64>() / psi.len() as f64;
    assert!(
        (atc - 2.0).abs() < 0.15,
        "ATC IF mean under μ₀ misspecification should stay near 2; got {atc}"
    );
}

/// ATC IF remains unbiased when propensity is misspecified but outcomes are correct.
#[test]
fn atc_if_doubly_robust_under_propensity_misspecification() {
    let n = 4_000usize;
    let mut rng = ExecutionContext::for_tests(43).rng.stream_for(StreamDomain::Estimate, 0xBEEFu64);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    for i in 0..n {
        let z = standard_normal(&mut rng);
        let logit = -0.5 + z;
        let pi_i = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        t[i] = ti;
        y[i] = 2.0 * ti + z + noise;
        // Deliberately wrong propensity (constant 0.5).
        e[i] = 0.5;
        mu0[i] = z;
        mu1[i] = 2.0 + z;
    }
    let mut psi = Vec::new();
    aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
    let atc = psi.iter().sum::<f64>() / psi.len() as f64;
    assert!(
        (atc - 2.0).abs() < 0.15,
        "ATC IF mean under propensity misspecification should stay near 2; got {atc}"
    );
}

/// ATC IF unbiased when both propensity and outcome models are correct.
#[test]
fn atc_if_doubly_robust_when_both_correct() {
    let n = 4_000usize;
    let mut rng = ExecutionContext::for_tests(44).rng.stream_for(StreamDomain::Estimate, 0xCAFEu64);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    for i in 0..n {
        let z = standard_normal(&mut rng);
        let logit = -0.5 + z;
        let pi_i = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        t[i] = ti;
        y[i] = 2.0 * ti + z + noise;
        e[i] = pi_i;
        mu0[i] = z;
        mu1[i] = 2.0 + z;
    }
    let mut psi = Vec::new();
    aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
    let atc = psi.iter().sum::<f64>() / psi.len() as f64;
    assert!(
        (atc - 2.0).abs() < 0.15,
        "ATC IF mean with both models correct should stay near 2; got {atc}"
    );
}

#[test]
fn aipw_hc1_multiway_newey_west_finite_se() {
    let (data, estimand) = confounded_scm(400, 8);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let n = 400;
    let dim_a: Vec<u32> = (0..n).map(|i| u32::try_from(i % 20).unwrap_or(0)).collect();
    let dim_b: Vec<u32> = (0..n).map(|i| u32::try_from(i % 15).unwrap_or(0)).collect();
    for kind in
        [AnalyticSeKind::Hc1, AnalyticSeKind::Multiway, AnalyticSeKind::NeweyWest { lag: 2 }]
    {
        let est = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_se_kind(kind)
            .with_multiway_ids(vec![dim_a.clone(), dim_b.clone()]);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(fit.se_analytic().is_finite() && fit.se_analytic() > 0.0, "kind={kind:?}");
    }
}

#[test]
fn aipw_trim_excludes_extreme_propensity_unit() {
    // One treated outlier with z = -8 (raw propensity ≈ 2e-4) and y = 1000: its clipped
    // T/e term is ~100 · (1000 − μ1), which wrecks the untrimmed ψ average. Trimming on
    // the raw score must exclude it.
    let (mut t, mut y, mut z) = confounded_columns(800, 5);
    t.push(1.0);
    y.push(1000.0);
    z.push(-8.0);
    let (data, estimand) = build_dataset(t, y, z);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));

    let untrimmed = AipwAte::new().with_bootstrap_replicates(0);
    let trimmed = AipwAte {
        overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.02) },
        ..untrimmed.clone()
    };
    let mut ws = AipwWorkspace::default();
    let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
    let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
    let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

    assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed ate={}", raw.ate);
    assert!((clean.ate - 2.0).abs() < 0.35, "trimmed ate={}", clean.ate);
    let report = clean.diagnostics.overlap_report.as_ref().unwrap();
    assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
}

#[test]
fn aipw_works_with_efficient_backdoor_estimand() {
    let (data, mut estimand) = confounded_scm(800, 4);
    estimand.method = Arc::from("backdoor.efficient");
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = AipwAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = AipwWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
}

/// Leave-one-out jackknife SE of the whole estimator (logistic + both OLS refit each time):
/// an SE that needs no influence-function algebra, so it is an independent yardstick.
fn jackknife_se(t: &[f64], y: &[f64], z: &[f64], est: &AipwAte) -> f64 {
    let n = t.len();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut ws = AipwWorkspace::default();
    let mut loo = Vec::with_capacity(n);
    for drop in 0..n {
        let keep = |v: &[f64]| -> Vec<f64> {
            v.iter().enumerate().filter(|(i, _)| *i != drop).map(|(_, x)| *x).collect()
        };
        let (data, estimand) = build_dataset(keep(t), keep(y), keep(z));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        loo.push(est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap().ate);
    }
    let mean = loo.iter().sum::<f64>() / n as f64;
    let ss: f64 = loo.iter().map(|v| (v - mean).powi(2)).sum();
    ((n as f64 - 1.0) / n as f64 * ss).sqrt()
}

#[test]
fn ate_analytic_se_matches_jackknife_under_misspecified_propensity() {
    // True propensity depends on z²; the fitted logistic is linear in z (misspecified).
    // Outcome per arm is linear in z (correct) with CATE 2 + 1.5 z, so the estimator is
    // consistent for ATE = 2, E[∂ψ/∂γ] = 0, but E[(τ(Z)−τ)·Z(e₀−e*)] ≠ 0: projecting ψ
    // off the logistic score would delete real variance. The stacked linearisation must
    // agree with the delete-one jackknife of the full estimator.
    let n = 300usize;
    let mut rng = ExecutionContext::for_tests(61).rng.stream_for(StreamDomain::Estimate, 0x77);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.5 + 0.3 * zi + 1.2 * (zi * zi - 1.0);
        let p = antecedent_kernels::sigmoid(logit);
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + 1.5 * ti * zi + 0.5 * standard_normal(&mut rng);
    }
    // A trim below every score keeps all rows but routes through the full-sample path.
    let est = AipwAte {
        overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(1e-9) },
        ..AipwAte::new()
    }
    .with_bootstrap_replicates(0);
    let (data, estimand) = build_dataset(t.clone(), y.clone(), z.clone());
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let fit = est.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new());
    let fit = fit.unwrap();
    let jk = jackknife_se(&t, &y, &z, &est);
    let rel = (fit.se_analytic() - jk).abs() / jk;
    assert!(rel < 0.15, "analytic {} vs jackknife {jk}", fit.se_analytic());
}

#[test]
fn point_estimate_does_not_depend_on_the_requested_se_kind() {
    let (data, estimand) = confounded_scm(400, 12);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut ws = AipwWorkspace::default();
    let base = AipwAte::new().with_bootstrap_replicates(0);
    let prep = base.prepare(&data, &estimand, &query).unwrap();
    let iid = base.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    let clusters: Vec<u32> = (0..400).map(|i| u32::try_from(i / 4).unwrap_or(0)).collect();
    for est in [
        base.clone().with_se_kind(AnalyticSeKind::Hc1),
        base.clone().with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(clusters.clone()),
    ] {
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert_eq!(fit.ate.to_bits(), iid.ate.to_bits(), "{:?}", est.inference.se_kind);
        assert!(fit.scores.table.is_some());
        assert!(fit.joint.covariance.is_none() && fit.scores.inference.is_none());
    }
}

fn ridge_estimator(grid: &[f64], inner_folds: usize) -> AipwAte {
    AipwAte {
        propensity: PropensityNuisance::ridge_logistic(
            crate::propensity::RidgeTuning::new(grid, inner_folds).unwrap(),
        ),
        ..AipwAte::new()
    }
    .with_bootstrap_replicates(0)
}

/// A resample that lists every row once, in order, is the original problem: the same score
/// table and the same selections, to the bit.
#[test]
fn the_identity_resample_is_the_original_problem() {
    let (data, estimand) = confounded_scm(300, 71);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = ridge_estimator(&[0.5, 5.0], 3);
    let original =
        AipwAte::crossfit_problem(&est.prepare(&data, &estimand, &query).unwrap(), &ctx());
    let idx: Vec<usize> = (0..original.nrows).collect();
    let same = resampled_problem(&original, &idx, None);
    let (a, fit_a) = est.crossfit_table(&original, false, &ctx(), None).unwrap();
    let (b, fit_b) = est.crossfit_table(&same, false, &ctx(), None).unwrap();
    let bits = |t: &crate::scores::ScoreTable| -> Vec<u64> {
        t.scores.iter().map(|v| v.to_bits()).collect()
    };
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(fit_a.selections, fit_b.selections);
}

/// Per-replicate draws on the penalized stream match the replicate-major batch.
#[test]
fn penalized_bootstrap_indexes_match_the_batch_stream() {
    let n = 9usize;
    let replicates = 6usize;
    let ctx = ExecutionContext::for_tests(23);
    let mut batch = vec![antecedent_data::GlobalId::new(0); n * replicates];
    antecedent_data::fill_resample_index_batch(
        antecedent_data::ResamplingPlan::IidBootstrap,
        n,
        replicates,
        None,
        &ctx,
        super::PENALIZED_BOOTSTRAP_STREAM,
        &mut batch,
    )
    .unwrap();
    let mut ids = Vec::new();
    let mut idx = Vec::new();
    let (mut held_ids, mut held_idx) = (0usize, 0usize);
    for replicate in 0..replicates {
        super::fill_penalized_bootstrap_indexes(
            n,
            u64::try_from(replicate).unwrap(),
            &ctx,
            &mut ids,
            &mut idx,
            None,
            None,
            crate::util::SamplingDeclaration::Undeclared,
        )
        .unwrap();
        let expect: Vec<usize> = batch[replicate * n..(replicate + 1) * n]
            .iter()
            .map(|id| usize::try_from(id.get()).unwrap())
            .collect();
        assert_eq!(idx, expect);
        if replicate == 0 {
            held_ids = ids.capacity();
            held_idx = idx.capacity();
        } else {
            assert_eq!(ids.capacity(), held_ids);
            assert_eq!(idx.capacity(), held_idx);
        }
    }
}

/// The refit bootstrap clones only to install a different fold seed or propensity.
#[test]
fn penalized_bootstrap_base_clones_only_when_the_seed_or_propensity_differs() {
    let (data, estimand) = confounded_scm(40, 3);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = ridge_estimator(&[0.5, 5.0], 2);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let nuisance = est.propensity.clone();
    assert_eq!(problem.fold_seed, 0);
    assert_eq!(problem.propensity, nuisance);

    let changed = super::penalized_bootstrap_base(&problem, 7, nuisance.clone());
    assert!(matches!(changed, std::borrow::Cow::Owned(_)));
    assert_eq!(changed.fold_seed, 7);
    assert_eq!(changed.propensity, nuisance);

    let mut seeded = problem.clone();
    seeded.fold_seed = 7;
    let same = super::penalized_bootstrap_base(&seeded, 7, nuisance);
    assert!(matches!(same, std::borrow::Cow::Borrowed(_)));
    assert!(std::ptr::eq(same.as_ref(), &seeded));

    let swapped = super::penalized_bootstrap_base(&seeded, 7, PropensityNuisance::default());
    assert!(matches!(swapped, std::borrow::Cow::Owned(_)));
    assert_eq!(swapped.fold_seed, 7);
    assert_eq!(swapped.propensity, PropensityNuisance::default());
}

/// The refit bootstrap's replicate selects every fold's penalty on the resample only.
///
/// The resample draws only from the first half of the rows, repeating them. (1) The
/// penalty of each fold equals one re-derived here from that fold's training rows of the
/// resample alone (the resample's own fold plan, designs, treatment and unit identities).
/// (2) Rewriting every row the resample did not draw (covariates, treatment, outcome)
/// leaves every selection, and therefore the replicate, unchanged.
#[test]
fn a_bootstrap_replicate_selects_its_penalty_on_the_resample_only() {
    use crate::propensity::RidgeFoldInput;
    let (data, estimand) = confounded_scm(300, 73);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let tuning = crate::propensity::RidgeTuning::new(&[0.1, 1.0, 10.0, 100.0], 3).unwrap();
    let est = ridge_estimator(&[0.1, 1.0, 10.0, 100.0], 3);
    let original =
        AipwAte::crossfit_problem(&est.prepare(&data, &estimand, &query).unwrap(), &ctx());
    let n = original.nrows;
    let idx: Vec<usize> = (0..n).map(|k| (k * 7) % (n / 2)).collect();
    let replicate = resampled_problem(&original, &idx, None);
    let (_, fit) = est.crossfit_table(&replicate, false, &ctx(), None).unwrap();
    assert_eq!(fit.selections.len(), 5);

    let arms: Vec<u32> =
        replicate.treatment.iter().map(|&t| u32::from(TreatmentArm::of(t).is_treated())).collect();
    let folds = crate::crossfit_plan::crossfit_fold_plan(
        &arms,
        &replicate.row_index,
        5,
        replicate.fold_seed,
    )
    .unwrap();
    for selection in &fit.selections {
        let (train, valid): (Vec<usize>, Vec<usize>) =
            (0..n).partition(|&i| folds[i] as usize != selection.fold);
        let (mut design_train, mut design_valid) = (Vec::new(), Vec::new());
        let ncols = replicate.design_ncols;
        select_rows_colmajor(&replicate.design_matrix, n, ncols, &train, &mut design_train);
        select_rows_colmajor(&replicate.design_matrix, n, ncols, &valid, &mut design_valid);
        let t_train = gather(&replicate.treatment, &train);
        let units: Vec<_> = train.iter().map(|&i| replicate.row_index[i]).collect();
        let (_, again) = crate::propensity::fit_ridge_fold(
            &tuning,
            &RidgeFoldInput {
                design_train: &design_train,
                n_train: train.len(),
                design_valid: &design_valid,
                n_valid: valid.len(),
                ncols,
                t_train: &t_train,
                units: &units,
                fold: selection.fold,
                seed: replicate.fold_seed,
                weights: None,
            },
            &ctx(),
        )
        .unwrap();
        assert_eq!(again.lambda.to_bits(), selection.lambda.to_bits(), "fold {}", selection.fold);
    }

    let ncols = original.design_ncols;
    let mut design = original.design_matrix.to_vec();
    for c in 1..ncols {
        for i in n / 2..n {
            design[c * n + i] += 5.0;
        }
    }
    let mut rewritten = original.clone();
    rewritten.design_matrix = design.into();
    rewritten.treatment = original
        .treatment
        .iter()
        .enumerate()
        .map(|(i, &t)| if i >= n / 2 { 1.0 - t } else { t })
        .collect::<Vec<_>>()
        .into();
    rewritten.outcome = original
        .outcome
        .iter()
        .enumerate()
        .map(|(i, &y)| if i >= n / 2 { y + 40.0 } else { y })
        .collect::<Vec<_>>()
        .into();
    let replicate_b = resampled_problem(&rewritten, &idx, None);
    let (_, fit_b) = est.crossfit_table(&replicate_b, false, &ctx(), None).unwrap();
    assert_eq!(fit_b.selections, fit.selections);
}

#[test]
fn aipw_bootstrap_follows_the_declared_sampling_design() {
    use antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories;
    let (data, estimand) = confounded_scm(400, 12);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let base = AipwAte::new().with_bootstrap_replicates(6);

    // Undeclared, the rows are one-row subjects and the row draw is the default.
    let prep = base.prepare(&data, &estimand, &query).unwrap();
    base.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap();

    // A declared trajectory design with no subject labels is a row-level draw.
    let declared = base.clone().with_sampling_design(IndependentSubjectTrajectories);
    let prep = declared.prepare(&data, &estimand, &query).unwrap();
    let refused = declared.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new());
    assert!(
        matches!(
            refused,
            Err(EstimationError::Data(antecedent_data::DataError::SamplingDesign(_)))
        ),
        "{refused:?}"
    );

    // With subject labels the whole trajectory is the draw, and the design allows it.
    let subjects: Vec<u32> = (0..400).map(|i| u32::try_from(i / 4).unwrap_or(0)).collect();
    let trajectories = declared.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(subjects);
    let prep = trajectories.prepare(&data, &estimand, &query).unwrap();
    let fit = trajectories
        .fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    assert!(fit.se_bootstrap().is_some());
}

/// The refit the bootstrap replaced: gather the resampled design and outcome, refit the logistic
/// propensity by IRLS on the repeated rows (weighted by the drawn rows' sampling weights, when
/// declared), and score the contrast over every drawn row.
fn gathered_replicate_ate(
    est: &AipwAte,
    problem: &PreparedPropensityProblem,
    idx: &[usize],
) -> Option<f64> {
    let clip = clip_of(problem.overlap);
    let trim = trim_of(problem.overlap);
    let n = problem.nrows;
    let ncols = problem.design_ncols;
    let drawn = idx.len();
    let mut design = vec![0.0; drawn * ncols];
    crate::util::gather_cols(&mut design, &problem.design_matrix, n, ncols, idx);
    let treatment = gather(&problem.treatment, idx);
    let outcome = gather(&problem.outcome, idx);
    let w_boot: Option<Vec<f64>> =
        problem.sampling_weights.as_ref().map(|s| idx.iter().map(|&i| s.weights[i]).collect());
    let mut workspace = AipwWorkspace::default();
    let fit = match w_boot.as_deref() {
        Some(w) => antecedent_stats::fit_propensity_weighted_in_place(
            &design,
            drawn,
            ncols,
            &treatment,
            w,
            &est.backend,
            &mut workspace.propensity,
            &est.glm_options,
        ),
        None => antecedent_stats::fit_propensity_in_place(
            &design,
            drawn,
            ncols,
            &treatment,
            &est.backend,
            &mut workspace.propensity,
            &est.glm_options,
        ),
    };
    fit.ok()?;
    let raw = workspace.propensity.scores[..drawn].to_vec();
    let mut scores = raw.clone();
    if let Some(c) = clip {
        clamp_scores(&mut scores, c);
    }
    let retained = trim_retained_rows(&raw, trim).ok()?;
    let keep = |values: &[f64]| -> Vec<f64> {
        match &retained {
            Some(kept) => gather(values, kept),
            None => values.to_vec(),
        }
    };
    let mut design_used = Vec::new();
    match &retained {
        Some(kept) => select_rows_colmajor(&design, drawn, ncols, kept, &mut design_used),
        None => design_used.clone_from(&design),
    }
    let (t_used, y_used, e_used) = (keep(&treatment), keep(&outcome), keep(&scores));
    let nrows = t_used.len();
    let w_used: Option<Vec<f64>> = w_boot.as_ref().map(|w| {
        let mut used = keep(w);
        let scale = used.len() as f64 / used.iter().sum::<f64>();
        for value in &mut used {
            *value *= scale;
        }
        used
    });
    let (beta0, beta1) = fit_outcome_models_weighted(
        &design_used,
        nrows,
        ncols,
        &t_used,
        &y_used,
        w_used.as_deref(),
        est.backend,
        &mut workspace,
    )
    .ok()?;
    let (mut mu0, mut mu1) = (Vec::new(), Vec::new());
    predict_colmajor(&design_used, nrows, ncols, &beta0, &mut mu0);
    predict_colmajor(&design_used, nrows, ncols, &beta1, &mut mu1);
    let mut psi = Vec::new();
    aipw_psi_weighted(
        &t_used,
        &y_used,
        &e_used,
        &mu0,
        &mu1,
        &problem.target_population,
        w_used.as_deref(),
        &mut psi,
    )
    .ok()?;
    if let Some(w) = w_used.as_deref() {
        for (value, &m) in psi.iter_mut().zip(w) {
            *value *= m;
        }
    }
    Some(psi.iter().sum::<f64>() / psi.len() as f64)
}

/// A replicate refit on the distinct drawn rows, each carrying its multiplicity as a case
/// weight, is the refit of the gathered resample: the weighted likelihood and squared residuals
/// are sums over the repeated rows, and the ATT / ATC arm share divides by the number of draws.
#[test]
fn the_count_weighted_replicate_matches_the_gathered_resample() {
    let n = 300;
    let (data, estimand) = confounded_scm(n, 21);
    let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let trimmed = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) };
    let weights: Vec<f64> = (0..n).map(|i| 0.4 + (i % 5) as f64 * 0.35).collect();
    let cases = [
        (ate.clone(), trimmed, false),
        (ate.clone(), trimmed, true),
        (ate.clone().with_target_population(TargetPopulation::Treated), trimmed, false),
        (ate.clone().with_target_population(TargetPopulation::Treated), trimmed, true),
        (
            ate.clone().with_target_population(TargetPopulation::Untreated),
            crate::propensity::default_propensity_overlap(),
            false,
        ),
        (
            ate.with_target_population(TargetPopulation::Untreated),
            crate::propensity::default_propensity_overlap(),
            true,
        ),
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for (query, overlap, weighted) in cases {
        let mut est = AipwAte::new().with_bootstrap_replicates(0);
        est.overlap = overlap;
        if weighted {
            est = est.with_sampling_weights(
                crate::sampling_weights::SamplingWeights::new(weights.clone()).unwrap(),
            );
        }
        let problem = est.prepare(&data, &estimand, &query).unwrap();
        let mut scratch = ReplicateScratch::new(problem.nrows, problem.design_ncols);
        for _ in 0..4 {
            let idx: Vec<usize> = (0..problem.nrows)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    usize::try_from(state >> 33).unwrap_or(0) % problem.nrows
                })
                .collect();
            let counted = est.replicate_ate(&problem, &idx, &mut scratch).expect("counted fit");
            let gathered = gathered_replicate_ate(&est, &problem, &idx).expect("gathered fit");
            assert!(
                (counted - gathered).abs() <= 1e-8 * (1.0 + gathered.abs()),
                "{:?} weighted={weighted}: counted {counted} vs gathered {gathered}",
                query.target_population
            );
        }
    }
}
