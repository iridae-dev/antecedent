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

use super::family::{FISHER_MEAN_FLOOR, SCORE_PROBABILITY_CLAMP};
use super::*;
use crate::overlap::OverlapPolicy;

/// Binary-outcome SCM: `Z ~ U(-0.5, 0.5)`, `T ∈ {0,1}`, `logit(Y=1) = -0.5 + 2T + Z`.
fn binary_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xABCD_u64);
    let mut t = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ti = (i % 2) as f64;
        let zi = (i as f64) / (n as f64) - 0.5;
        let logit = -0.5 + 2.0 * ti + zi;
        let p = antecedent_kernels::sigmoid(logit);
        let yi = if rng.next_f64() < p { 1.0 } else { 0.0 };
        t[i] = ti;
        z[i] = zi;
        y[i] = yi;
    }

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

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn oracle_cell_table() -> (TabularData, IdentifiedEstimand) {
    let mut t = Vec::with_capacity(400);
    let mut z = Vec::with_capacity(400);
    let mut y = Vec::with_capacity(400);
    for (zi, ti, successes) in [(0.0, 0.0, 20), (0.0, 1.0, 50), (1.0, 0.0, 40), (1.0, 1.0, 70)] {
        t.extend(std::iter::repeat_n(ti, 100));
        z.extend(std::iter::repeat_n(zi, 100));
        y.extend(std::iter::repeat_n(1.0, successes));
        y.extend(std::iter::repeat_n(0.0, 100 - successes));
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(t),
                ValidityBitmap::all_valid(400),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(y),
                ValidityBitmap::all_valid(400),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(z),
                ValidityBitmap::all_valid(400),
            )
            .unwrap(),
        ),
    ];
    let data =
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (data, estimand)
}

#[test]
fn glm_attached_bootstrap_equals_the_one_shot_fit() {
    let (data, estimand) = binary_scm(300, 5);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let boot = GlmAdjustmentAte::new().with_bootstrap_replicates(30);
    let prep = boot.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
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

#[test]
fn glm_adjustment_matches_pinned_statsmodels_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/estimate/glm_adjustment_grid/expected.json"
    ))
    .unwrap();
    let (data, estimand) = oracle_cell_table();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let estimator = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let mut workspace = GlmAdjustmentWorkspace::default();
    let effect = estimator.fit(&problem, &mut workspace, &ctx(), AssumptionSet::new()).unwrap();
    let ate_target = fixture["reference"]["all_observed_ate"].as_f64().unwrap();
    let ate_tolerance = fixture["acceptance"]["ate_atol"].as_f64().unwrap();
    assert!((effect.ate - ate_target).abs() <= ate_tolerance);

    let mut direct_workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef {
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: &problem.design.outcome,
        },
        &FaerBackend,
        &mut direct_workspace,
        &GlmOptions::default(),
    )
    .unwrap();
    fit.require_ok().unwrap();
    let coefficient_tolerance = fixture["acceptance"]["coefficient_atol"].as_f64().unwrap();
    for (actual, expected) in fit
        .coefficients
        .iter()
        .zip(fixture["reference"]["coefficients_intercept_t_z"].as_array().unwrap())
    {
        assert!((*actual - expected.as_f64().unwrap()).abs() <= coefficient_tolerance);
    }
}

#[test]
fn recovers_positive_ate_on_binary_outcome() {
    let (data, estimand) = binary_scm(4000, 1);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(30);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!(effect.ate > 0.0, "ate={}", effect.ate);
    assert!(effect.ate < 1.0, "ate={}", effect.ate);
    crate::prepare::assert_records_preparation(&effect, 4000);
    assert!(effect.se_bootstrap().is_some());
}

#[test]
fn glm_runs_the_complete_interval_procedure() {
    use crate::util::interval_test_support::{assert_procedure, met_early, never_met};
    let (data, estimand) = binary_scm(600, 6);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (cap, wave) = (120, 30);
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(cap);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let fit = |est: &GlmAdjustmentAte| {
        est.fit(&prep, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap()
    };
    assert_procedure(
        &fit(&est),
        &fit(&est.clone().with_bootstrap_interval(never_met(wave))),
        &fit(&est.clone().with_bootstrap_interval(met_early(wave, 60))),
        cap,
        wave,
    );
}

#[test]
fn works_with_efficient_backdoor_estimand() {
    let (data, mut estimand) = binary_scm(2000, 2);
    estimand.method = Arc::from("backdoor.efficient");
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!(effect.ate > 0.0, "ate={}", effect.ate);
}

#[test]
fn rejects_require_diagnostics_overlap() {
    let (data, estimand) = binary_scm(200, 3);
    let est = GlmAdjustmentAte {
        overlap: OverlapPolicy::require_diagnostics(),
        ..GlmAdjustmentAte::new()
    };
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let err = est.prepare(&data, &estimand, &query).unwrap_err();
    assert!(matches!(err, EstimationError::Overlap { .. }));
}

/// Gaussian SCM with homogeneous contrasts: `Y = 1 + 2T + Z + noise` (no interactions).
fn gaussian_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xFEED_u64);
    let mut t = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ti = (i % 2) as f64;
        let zi = (i as f64) / (n as f64) - 0.5;
        let noise = rng.next_f64() - 0.5;
        t[i] = ti;
        z[i] = zi;
        y[i] = 1.0 + 2.0 * ti + zi + noise;
    }

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

#[test]
fn gaussian_delta_method_se_positive_and_near_bootstrap() {
    // Homogeneous contrasts: every per-row contrast equals β_T exactly, so the old
    // spread-based formula (sample_std(diffs)/√n) returned ≈0 regardless of coefficient
    // noise. The delta-method SE must be positive and on the bootstrap's scale.
    let (data, estimand) = gaussian_scm(400, 6);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = GlmAdjustmentAte { family: GlmFamily::GaussianIdentity, ..GlmAdjustmentAte::new() }
        .with_bootstrap_replicates(200);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!(effect.se_analytic() > 0.0, "se_analytic={}", effect.se_analytic());
    let boot = effect.se_bootstrap().unwrap();
    assert!(
        effect.se_analytic() < 3.0 * boot && effect.se_analytic() > boot / 3.0,
        "se_analytic={} se_bootstrap={boot}",
        effect.se_analytic()
    );
}

#[test]
fn probit_adjustment_fits_binary_outcome() {
    let (data, estimand) = binary_scm(200, 7);
    let est = GlmAdjustmentAte { family: GlmFamily::BinomialProbit, ..GlmAdjustmentAte::new() }
        .with_bootstrap_replicates(0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    // The data are logistic, logit(P) = -0.5 + 2t + z with z uniform on (-0.5, 0.5), so the
    // risk difference is the mean over z of sigma(1.5 + z) - sigma(-0.5 + z) = 0.4338
    // (numerical quadrature); the probit fit approximates it to sampling error at n = 200
    // (SD about 0.05). A constant or sign-flipped effect would miss it.
    assert!((effect.ate - 0.4338).abs() < 0.15, "probit ATE {} vs 0.4338", effect.ate);
    assert!(effect.se_analytic().is_finite() && effect.se_analytic() > 0.0);
}

#[test]
fn rejects_non_binary_outcome() {
    let (data, estimand) = binary_scm(200, 4);
    // Replace outcome with a non-binary value to trigger the validation path.
    let (data, _) = data.with_appended_float("dummy", Arc::from(vec![0.0; 200])).unwrap();
    let bad_y = (0..200).map(f64::from).collect::<Vec<_>>();
    let data = data.with_replaced_float(VariableId::from_raw(1), Arc::from(bad_y)).unwrap();
    let est = GlmAdjustmentAte::new();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let err = est.prepare(&data, &estimand, &query).unwrap_err();
    assert!(matches!(err, EstimationError::Unsupported { .. }));
}

#[test]
fn recovers_att_via_gcomp() {
    let (data, estimand) = binary_scm(800, 5);
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    // T alternates independently of Z, so the ATT equals the ATE: the mean over Z ~ U(-0.5, 0.5)
    // of sigmoid(1.5 + z) - sigmoid(-0.5 + z) = 0.43378 (numerical integration). The tolerance is
    // the one the probit ATE check above uses, about four standard errors at n = 800.
    assert!((effect.ate - 0.43378).abs() < 0.15, "logit ATT {} vs 0.43378", effect.ate);
}

#[test]
fn att_se_differs_from_ate_se_under_logit() {
    let (data, estimand) = binary_scm(800, 5);
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let ate_q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let att_q = ate_q.clone().with_target_population(TargetPopulation::Treated);
    let mut ws = GlmAdjustmentWorkspace::default();
    let ate = est
        .fit(&est.prepare(&data, &estimand, &ate_q).unwrap(), &mut ws, &ctx(), AssumptionSet::new())
        .unwrap();
    let att = est
        .fit(&est.prepare(&data, &estimand, &att_q).unwrap(), &mut ws, &ctx(), AssumptionSet::new())
        .unwrap();
    assert!(ate.se_analytic().is_finite() && att.se_analytic().is_finite());
    assert!(
        (ate.se_analytic() - att.se_analytic()).abs() > 1e-8,
        "logit ATT SE must not reuse the ATE gradient; ate_se={} att_se={}",
        ate.se_analytic(),
        att.se_analytic()
    );
}

#[test]
fn predicate_restricts_prepared_rows_and_shifts_logit_ate() {
    use antecedent_core::PredicateExpr;
    let (data, estimand) = binary_scm(400, 7);
    let est = GlmAdjustmentAte::new().with_bootstrap_replicates(0);
    let all = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let even = all.clone().with_target_population(TargetPopulation::Predicate(
        PredicateExpr::rows((0..200).collect::<Vec<_>>()),
    ));
    let prep_all = est.prepare(&data, &estimand, &all).unwrap();
    let prep_even = est.prepare(&data, &estimand, &even).unwrap();
    assert_eq!(prep_all.design.nrows, 400);
    assert_eq!(prep_even.design.nrows, 200);
    let named =
        all.with_target_population(TargetPopulation::Predicate(PredicateExpr::named("cohort")));
    let err = est.prepare(&data, &estimand, &named).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("PopulationRegistry") || msg.contains("named"), "{msg}");
}

#[test]
fn negbin_gcomp_accepts_nonnegative_counts() {
    let n = 200usize;
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
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([]), ExprId::from_raw(0));
    let est = GlmAdjustmentAte { family: GlmFamily::NegativeBinomial, ..GlmAdjustmentAte::new() }
        .with_bootstrap_replicates(0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!(effect.ate.is_finite() && effect.ate > 0.0);
}

#[test]
fn nb_fisher_weight_and_gcomp_se_respect_alpha() {
    // Closed form: NB2 log-link Fisher weight is μ/(1+αμ), not the Poisson μ.
    let mu = 4.0_f64;
    let alpha = 2.0_f64;
    let eta = mu.ln();
    let w_nb = fisher_weight(GlmFamily::NegativeBinomial, eta, alpha);
    let w_pois = fisher_weight(GlmFamily::PoissonLog, eta, 0.0);
    assert!((w_nb - mu / (1.0 + alpha * mu)).abs() < 1e-14);
    assert!((w_pois - mu).abs() < 1e-14);
    assert!(w_nb < w_pois);

    // Tiny deterministic design: intercept + T, constant mean contrast.
    // n=4 rows, T = [0,0,1,1], X col-major [1|T].
    let nrows = 4usize;
    let ncols = 2usize;
    let x = vec![
        1.0, 1.0, 1.0, 1.0, // intercept
        0.0, 0.0, 1.0, 1.0, // treatment
    ];
    // β = (ln μ, 0) ⇒ μ(T) = μ for both arms; g-comp ATE gradient still
    // probes the treatment column through μ'·ΔT.
    let coefficients = [mu.ln(), 0.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let population = TargetPopulation::AllObserved;
    let target = GcompTarget { treatment: &treatment, population: &population, weights: None };
    let contrast = |family| GcompContrast {
        family,
        x_colmajor: &x,
        nrows,
        ncols,
        t_col: 1,
        active: 1.0,
        control: 0.0,
    };
    let se_nb = gcomp_delta_method_se(
        &contrast(GlmFamily::NegativeBinomial),
        &coefficients,
        0.0,
        alpha,
        &target,
    );
    let se_pois =
        gcomp_delta_method_se(&contrast(GlmFamily::PoissonLog), &coefficients, 0.0, 0.0, &target);
    assert!(se_nb.is_finite() && se_pois.is_finite());
    // At a common mean the NB2 covariance is the Poisson one scaled by 1+αμ.
    let ratio = se_nb / se_pois;
    assert!(
        (ratio - (1.0 + alpha * mu).sqrt()).abs() < 1e-10,
        "NB/Poisson SE ratio {ratio} must be sqrt(1+αμ); nb={se_nb} pois={se_pois}"
    );
}

#[test]
fn hc1_sandwich_se_finite_without_bootstrap() {
    let (data, estimand) = binary_scm(800, 9);
    let est =
        GlmAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(AnalyticSeKind::Hc1);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = GlmAdjustmentWorkspace::default();
    let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
    assert!(effect.se_analytic().is_finite() && effect.se_analytic() > 0.0);
}

/// `Φ(eta)` at 60 digits (mpmath `ncdf`), rounded once, for the `etas` of the test below.
const PROBIT_MPMATH: [f64; 9] = [
    0.0,
    0.0,
    0.15865525393145705,
    0.49999999999960104,
    0.5,
    0.5000000000003989,
    0.8413447460685429,
    1.0,
    1.0,
];

#[test]
fn family_mean_matches_the_historical_link_expressions_bit_for_bit() {
    // The adjustment-resume mean used these expressions. The logit is the same evaluation
    // (stable sigmoid) and stays bit-exact; the probit is held to a tolerance, see below.
    let etas = [-700.0, -40.0, -1.0, -1e-12, 0.0, 1e-12, 1.0, 40.0, 700.0];
    for eta in etas {
        let historical_logit = if eta >= 0.0 {
            antecedent_kernels::sigmoid(eta)
        } else {
            let e = eta.exp();
            e / (1.0 + e)
        };
        assert_eq!(
            historical_logit.to_bits(),
            GlmFamily::BinomialLogit.mean_from_eta(eta).to_bits(),
            "logit {eta}"
        );
        // The probit mean is the kernel normal CDF, which since the tail-precision fix is
        // correctly rounded to a few ulps; the old `0.5 * erfc(-eta / sqrt2)` expression is
        // only a close neighbour of it (it differs by one ulp at eta = -1). Both are held to
        // the 60-digit mpmath value instead of to each other bit for bit.
        let historical_probit = 0.5 * antecedent_kernels::erfc(-eta / std::f64::consts::SQRT_2);
        let probit = GlmFamily::BinomialProbit.mean_from_eta(eta);
        let exact = PROBIT_MPMATH[etas.iter().position(|e| *e == eta).unwrap()];
        for (name, got) in [("product", probit), ("historical", historical_probit)] {
            let tol = 4.0 * f64::EPSILON * exact.abs().max(f64::MIN_POSITIVE);
            assert!((got - exact).abs() <= tol, "probit {name} {eta}: {got:e} vs {exact:e}");
        }
        assert_eq!(eta.to_bits(), GlmFamily::GaussianIdentity.mean_from_eta(eta).to_bits());
        assert_eq!(eta.exp().to_bits(), GlmFamily::PoissonLog.mean_from_eta(eta).to_bits());
        assert_eq!(eta.exp().to_bits(), GlmFamily::NegativeBinomial.mean_from_eta(eta).to_bits());
    }
}

#[test]
fn fisher_and_score_clamps_stay_on_different_paths() {
    assert_eq!(FISHER_MEAN_FLOOR.to_bits(), (1e-12_f64).to_bits());
    assert_eq!(SCORE_PROBABILITY_CLAMP.to_bits(), (1e-9_f64).to_bits());
    let kernel = FamilyKernel::new(GlmFamily::BinomialProbit);
    let fisher = kernel.fisher_weight(20.0, 0.0);
    let (_score, weight) = kernel.score_at(20.0, 1.0, 0.0);
    assert_ne!(fisher.to_bits(), weight.to_bits());
}

#[test]
fn routed_glm_sandwich_matches_the_direct_covariance_bit_for_bit() {
    let x = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y = [0.0, 1.0, 0.0, 1.0];
    let coef = [0.0, 0.5];
    let (score_u, fisher_w) =
        glm_score_components(GlmFamily::BinomialLogit, &x, 4, 2, &coef, &y, 0.0);
    let spec = |kind| SeSpec { kind, cluster_ids: None, multiway_ids: None, panel_times: None };
    assert!(
        sandwich_cov_matrix(spec(AnalyticSeKind::Homoskedastic), &x, 4, 2, &score_u, &fisher_w)
            .unwrap()
            .is_none()
    );
    for kind in [antecedent_stats::SandwichKind::Hc0, antecedent_stats::SandwichKind::Hc1] {
        let direct = score_coefficient_covariance(&x, 4, 2, &score_u, &fisher_w, kind).unwrap();
        let analytic = match kind {
            antecedent_stats::SandwichKind::Hc0 => AnalyticSeKind::Hc0,
            antecedent_stats::SandwichKind::Hc1 => AnalyticSeKind::Hc1,
            _ => unreachable!(),
        };
        let routed =
            sandwich_cov_matrix(spec(analytic), &x, 4, 2, &score_u, &fisher_w).unwrap().unwrap();
        assert!(direct.iter().zip(&routed).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let groups = [0_u32, 0, 1, 1];
    let direct = score_coefficient_covariance(
        &x,
        4,
        2,
        &score_u,
        &fisher_w,
        antecedent_stats::SandwichKind::Cluster { groups: &groups },
    )
    .unwrap();
    let routed = sandwich_cov_matrix(
        SeSpec {
            kind: AnalyticSeKind::Cluster,
            cluster_ids: Some(&groups),
            multiway_ids: None,
            panel_times: None,
        },
        &x,
        4,
        2,
        &score_u,
        &fisher_w,
    )
    .unwrap()
    .unwrap();
    assert!(direct.iter().zip(&routed).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn gcomp_gradient_matches_the_inline_row_etas_bit_for_bit() {
    let nrows = 5;
    let ncols = 3;
    let x: Vec<f64> = (0..nrows * ncols).map(|i| 0.1 * i as f64).collect();
    let beta = [0.2, -0.4, 0.3];
    let treatment = [0.0, 1.0, 0.0, 1.0, 1.0];
    let contrast = GcompContrast {
        family: GlmFamily::BinomialLogit,
        x_colmajor: &x,
        nrows,
        ncols,
        t_col: 1,
        active: 1.0,
        control: 0.0,
    };
    let population = TargetPopulation::AllObserved;
    let target = GcompTarget { treatment: &treatment, population: &population, weights: None };
    let grad = gcomp_gradient(&contrast, &beta, &target);
    let mut reference = vec![0.0; ncols];
    let mut count = 0usize;
    for r in 0..nrows {
        count += 1;
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for (c, &coef) in beta.iter().enumerate() {
            if c == 1 {
                eta_active += coef;
            } else {
                let val = x[c * nrows + r];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        let d1 = mean_derivative(GlmFamily::BinomialLogit, eta_active);
        let d0 = mean_derivative(GlmFamily::BinomialLogit, eta_control);
        for c in 0..ncols {
            let (x1, x0) = if c == 1 { (1.0, 0.0) } else { (x[c * nrows + r], x[c * nrows + r]) };
            reference[c] += d1 * x1 - d0 * x0;
        }
    }
    let n = count as f64;
    for (got, want) in grad.iter().zip(&reference) {
        assert_eq!(got.to_bits(), (want / n).to_bits());
    }
}

/// Row-by-row references: the loops the column-outer kernels replaced.
fn reference_covariance(
    family: GlmFamily,
    x: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    nb_alpha: f64,
) -> Option<Vec<f64>> {
    let mut x_w = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            eta += x[c * nrows + r] * coefficients[c];
        }
        let sqrt_w = fisher_weight(family, eta, nb_alpha).max(0.0).sqrt();
        for c in 0..ncols {
            x_w[c * nrows + r] = x[c * nrows + r] * sqrt_w;
        }
    }
    crate::util::xtx_inverse(&x_w, nrows, ncols)
}

#[test]
fn column_outer_covariance_and_scores_are_bit_identical_to_the_row_first_loops() {
    let (nrows, ncols) = (1_500usize, 5usize);
    let mut state = 0x1234_5678_9ABC_DEF0_u64;
    let mut x = vec![0.0; nrows * ncols];
    for (index, value) in x.iter_mut().enumerate() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *value = if index < nrows { 1.0 } else { (state >> 33) as f64 / 2f64.powi(31) - 0.5 };
    }
    let y: Vec<f64> = (0..nrows).map(|i| f64::from(u8::from(i % 4 == 0))).collect();
    let beta = [0.2, -0.4, 0.7, 0.1, -0.3];
    for family in [
        GlmFamily::GaussianIdentity,
        GlmFamily::BinomialLogit,
        GlmFamily::BinomialProbit,
        GlmFamily::PoissonLog,
        GlmFamily::NegativeBinomial,
    ] {
        let covariance = adjustment_glm_covariance(family, &x, nrows, ncols, &beta, 3.0, 0.4);
        let reference = reference_covariance(family, &x, nrows, ncols, &beta, 0.4);
        let dispersion = if matches!(family, GlmFamily::GaussianIdentity) {
            3.0 / (nrows as f64 - ncols as f64).max(1.0)
        } else {
            1.0
        };
        let reference: Option<Vec<f64>> =
            reference.map(|cov| cov.into_iter().map(|c| c * dispersion).collect());
        match (&covariance, &reference) {
            (Some(a), Some(b)) => {
                assert!(a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()), "{family:?}");
            }
            (None, None) => {}
            _ => panic!("covariance availability differs for {family:?}"),
        }
        let (u, w) = glm_score_components(family, &x, nrows, ncols, &beta, &y, 0.4);
        assert_eq!(u.len(), nrows);
        assert_eq!(w.len(), nrows);
    }
}

#[test]
#[ignore = "timing harness: cargo test --release -p antecedent-estimate --lib timing_ -- --ignored --nocapture"]
fn timing_glm_row_kernels() {
    let (nrows, ncols) = (200_000usize, 10usize);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut x = vec![0.0; nrows * ncols];
    for (index, value) in x.iter_mut().enumerate() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *value = if index < nrows { 1.0 } else { (state >> 33) as f64 / 2f64.powi(31) - 0.5 };
    }
    let y: Vec<f64> = (0..nrows).map(|i| f64::from(u8::from(i % 3 == 0))).collect();
    let treatment: Vec<f64> = (0..nrows).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let beta: Vec<f64> = (0..ncols).map(|c| 0.05 * (c as f64 + 1.0)).collect();
    let family = GlmFamily::BinomialLogit;
    let time = |label: &str, reps: u32, mut work: Box<dyn FnMut()>| {
        let start = std::time::Instant::now();
        for _ in 0..reps {
            work();
        }
        println!("{label}: {:.3} ms/call", start.elapsed().as_secs_f64() * 1e3 / f64::from(reps));
    };
    {
        let (x, beta) = (&x, &beta);
        time(
            "adjustment_glm_covariance",
            5,
            Box::new(|| {
                std::hint::black_box(adjustment_glm_covariance(
                    family, x, nrows, ncols, beta, 0.0, 0.0,
                ));
            }),
        );
    }
    {
        let (x, beta, y) = (&x, &beta, &y);
        time(
            "glm_score_components",
            5,
            Box::new(|| {
                std::hint::black_box(glm_score_components(family, x, nrows, ncols, beta, y, 0.0));
            }),
        );
    }
    {
        let (x, beta, treatment) = (&x, &beta, &treatment);
        let contrast = GcompContrast {
            family,
            x_colmajor: x,
            nrows,
            ncols,
            t_col: 1,
            active: 1.0,
            control: 0.0,
        };
        let population = TargetPopulation::AllObserved;
        let target = GcompTarget { treatment, population: &population, weights: None };
        time(
            "gcomp_gradient",
            5,
            Box::new(|| {
                std::hint::black_box(gcomp_gradient(&contrast, beta, &target));
            }),
        );
    }
}

#[test]
fn glm_bootstrap_follows_the_declared_sampling_design() {
    use antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories;
    let (data, estimand) = binary_scm(400, 5);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let base = GlmAdjustmentAte::new().with_bootstrap_replicates(6);

    let prep = base.prepare(&data, &estimand, &query).unwrap();
    base.fit(&prep, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap();

    // A declared trajectory design with no subject labels is a row-level draw.
    let declared = base.clone().with_sampling_design(IndependentSubjectTrajectories);
    let prep = declared.prepare(&data, &estimand, &query).unwrap();
    let refused =
        declared.fit(&prep, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new());
    assert!(
        matches!(
            refused,
            Err(EstimationError::Data(antecedent_data::DataError::SamplingDesign(_)))
        ),
        "{refused:?}"
    );

    // With subject labels the whole trajectory is the draw, and the design allows it.
    let subjects: Vec<u32> = (0..400).map(|i| u32::try_from(i / 4).unwrap_or(0)).collect();
    let trajectories = declared.with_cluster_ids(subjects);
    let prep = trajectories.prepare(&data, &estimand, &query).unwrap();
    let fit = trajectories
        .fit(&prep, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    assert!(fit.se_bootstrap().is_some());
}
