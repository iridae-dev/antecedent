use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::backend_boundary::PreparationBackend;
use antecedent_data::materialize::{AdmissionRecord, SchemaValueKind};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_expr::{Assignment, DomainRef, EmpiricalTableProvider, ExprId, FactorSpec};
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};
use antecedent_stats::{form_xtx, invert_square};

use crate::overlap::OverlapReport;

use super::*;

fn toy() -> (TabularData, IdentifiedEstimand) {
    toy_with(|_, t, z| 1.0 + 2.0 * t + z)
}

fn toy_with(outcome: impl Fn(usize, f64, f64) -> f64) -> (TabularData, IdentifiedEstimand) {
    let n = 100usize;
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
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| outcome(i, t[i], z[i])).collect();
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

fn checked_result(query: AverageEffectQuery, adjustment: &[VariableId]) -> IdentificationResult {
    let mut arena = CausalExprArena::new();
    let active = intervention_f64(&query.active).unwrap();
    let control = intervention_f64(&query.control).unwrap();
    let functional = arena.backdoor_ate(
        query.treatment,
        query.outcome,
        adjustment,
        antecedent_core::Value::f64(active),
        antecedent_core::Value::f64(control),
    );
    IdentificationResult::identified(
        antecedent_core::CausalQuery::average_effect(query),
        vec![IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment.to_vec()),
            functional,
        )],
        arena,
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    )
}

#[test]
fn checked_linear_backends_execute_equal_designs_and_refuse_unavailable_routes() {
    use antecedent_core::physical_choice::PhysicalAlternative;
    let (data, _) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let identification = checked_result(query, &[VariableId::from_raw(2)]);
    let estimator = LinearAdjustmentAte::new();
    let direct = estimator
        .prepare_checked_with_backend(
            &data,
            &identification,
            0,
            PhysicalAlternative::DirectPreparedNumericViews,
        )
        .unwrap();
    let materialized = estimator
        .prepare_checked_with_backend(
            &data,
            &identification,
            0,
            PhysicalAlternative::MaterializedDesign,
        )
        .unwrap();
    assert_eq!(direct.problem().design.matrix, materialized.problem().design.matrix);
    assert_eq!(direct.problem().design.outcome, materialized.problem().design.outcome);
    for unavailable in
        [PhysicalAlternative::StreamedReduction, PhysicalAlternative::OptionalLazyPreparation]
    {
        let refused = estimator
            .prepare_checked_with_backend(&data, &identification, 0, unavailable)
            .err()
            .expect("an unavailable physical route is refused");
        assert!(
            refused.to_string().contains("streamed and lazy preparation are unavailable"),
            "{refused}"
        );
    }
}

#[test]
fn checked_linear_adjustment_retains_selected_target_and_assumptions() {
    let (data, _) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let identification = checked_result(query, &[VariableId::from_raw(2)]);
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
    assert_eq!(checked.source_functional, checked.target.functional);
    assert_ne!(checked.executable_functional, checked.source_functional);
    assert_eq!(checked.lowering.source, checked.source_functional);
    assert_eq!(checked.lowering.executable, checked.executable_functional);
    assert_eq!(checked.lowering.population, TargetPopulation::AllObserved);
    assert!(checked.factor_requirements.iter().all(|f| f.intervention.is_empty()));
    assert_eq!(
        checked.program.arena().node(checked.source_functional),
        identification.arena.node(checked.source_functional)
    );
    assert_eq!(checked.problem.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    assert!(checked.required_assumptions.is_empty());
    let rebound = estimator.rebind_checked(&checked, &data).unwrap();
    assert_eq!(rebound.program.mapping(), checked.program.mapping());
    assert_eq!(rebound.problem.design.nrows, checked.problem.design.nrows);

    let t = [0.0, 1.0];
    let y = [0.0, 1.0];
    let z = [0.0, 1.0];
    let changed_schema = TabularData::from_f64_columns([
        ("renamed_t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let refused = estimator.rebind_checked(&checked, &changed_schema).err().expect("renamed schema");
    assert!(refused.to_string().contains("no longer agree"), "{refused}");
}

#[test]
fn checked_observational_program_evaluates_with_arm_bindings() {
    let (data, _) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let identification = checked_result(query, &[VariableId::from_raw(2)]);
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();

    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let mut provider = EmpiricalTableProvider::new();
    provider.set_domain(t, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
    provider.set_domain(y, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
    provider.set_domain(z, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
    for (zv, probability) in [(0.0, 0.5), (1.0, 0.5)] {
        let spec = FactorSpec {
            variables: &[z],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        provider
            .insert_probability(
                &spec,
                &Assignment::from_pairs([(z, antecedent_core::Value::f64(zv))]),
                probability,
            )
            .unwrap();
    }
    for (tv, zv, py1) in [(0.0, 0.0, 0.3), (0.0, 1.0, 0.2), (1.0, 0.0, 0.8), (1.0, 1.0, 0.6)] {
        let spec = FactorSpec {
            variables: &[y],
            conditioned_on: &[t, z],
            intervention: &[],
            domain: DomainRef::Observational,
            population: "",
            regime: None,
        };
        for (yv, probability) in [(0.0, 1.0 - py1), (1.0, py1)] {
            provider
                .insert_probability(
                    &spec,
                    &Assignment::from_pairs([
                        (t, antecedent_core::Value::f64(tv)),
                        (z, antecedent_core::Value::f64(zv)),
                        (y, antecedent_core::Value::f64(yv)),
                    ]),
                    probability,
                )
                .unwrap();
        }
    }
    let evaluator = checked.program.compile().unwrap();
    let context = antecedent_expr::EvalContext::default();
    let active = evaluator
        .evaluate_with(
            &provider,
            &context,
            &Assignment::from_pairs([(t, antecedent_core::Value::f64(1.0))]),
        )
        .unwrap();
    let control = evaluator
        .evaluate_with(
            &provider,
            &context,
            &Assignment::from_pairs([(t, antecedent_core::Value::f64(0.0))]),
        )
        .unwrap();
    assert!((active - 0.7).abs() < 1e-12);
    assert!((control - 0.25).abs() < 1e-12);
    assert!((active - control - 0.45).abs() < 1e-12);
}

#[test]
fn checked_linear_adjustment_rejects_semantic_tampering() {
    let (data, _) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut reversed = checked_result(query.clone(), &[VariableId::from_raw(2)]);
    let mut arena = reversed.arena.clone();
    let functional = arena.backdoor_ate(
        query.treatment,
        query.outcome,
        &[VariableId::from_raw(2)],
        antecedent_core::Value::f64(0.0),
        antecedent_core::Value::f64(1.0),
    );
    reversed.arena = arena;
    reversed.estimands[0].functional = functional;
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let mismatch = "selected functional does not match the query contrast";
    let refused = estimator.prepare_checked(&data, &reversed, 0).err().expect("reversed arms");
    assert!(refused.to_string().contains(mismatch), "{refused}");

    let mut changed_arms = checked_result(query.clone(), &[VariableId::from_raw(2)]);
    let mut different_query = query.clone();
    different_query.active = antecedent_core::Intervention::set(
        different_query.treatment,
        antecedent_core::Value::f64(2.0),
    );
    changed_arms.query = antecedent_core::CausalQuery::average_effect(different_query);
    let refused = estimator.prepare_checked(&data, &changed_arms, 0).err().expect("changed arms");
    assert!(refused.to_string().contains(mismatch), "{refused}");

    let mut changed_covariate = checked_result(query, &[VariableId::from_raw(2)]);
    changed_covariate.estimands[0].adjustment_set = Arc::from([]);
    let refused =
        estimator.prepare_checked(&data, &changed_covariate, 0).err().expect("changed covariate");
    assert!(refused.to_string().contains(mismatch), "{refused}");

    let mut changed_population =
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    changed_population.target_population = TargetPopulation::Treated;
    let changed_population = checked_result(changed_population, &[VariableId::from_raw(2)]);
    let refused =
        estimator.prepare_checked(&data, &changed_population, 0).err().expect("treated population");
    assert!(refused.to_string().contains("supports AllObserved population averaging"), "{refused}");
}

#[test]
fn overlap_report_from_propensities() {
    let ps = [0.1, 0.5, 0.9];
    let ws = [10.0, 2.0, 1.111];
    let report = OverlapReport::from_propensities(
        &ps,
        Some(&ws),
        OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.05) },
        Some(&[0.0, 0.0, 1.0]),
        Some(crate::overlap::IpwTarget::Ate),
        None,
    );
    assert!((report.propensity_min - 0.1).abs() < 1e-12);
    assert!((report.propensity_max - 0.9).abs() < 1e-12);
    assert_eq!(report.extreme_weight_count, 0); // none strictly > 10
    assert_eq!(report.clip, Some(0.05));
    assert!((report.excluded_fraction - 0.0).abs() < 1e-12);
    assert!((report.target_population_support - 1.0).abs() < 1e-12);
    assert_eq!(report.excluded_regions.len(), 2);
    assert!((report.excluded_regions[0].high - 0.05).abs() < 1e-12);
    let sens = report.clip_sensitivity.as_ref().expect("clip sensitivity");
    assert!(sens.thresholds.len() >= 2);
    assert_eq!(sens.ess.len(), sens.thresholds.len());
}

#[test]
fn rejects_require_diagnostics_on_linear_path() {
    let (data, estimand) = toy();
    let est = LinearAdjustmentAte {
        overlap: OverlapPolicy::require_diagnostics(),
        ..LinearAdjustmentAte::new()
    };
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let err = est.prepare(&data, &estimand, &query).unwrap_err();
    assert!(matches!(err, EstimationError::Overlap { .. }));
}

/// The residuals the point fit hands back are the ones a separate least-squares solve of
/// the same design produces, so the normal-equation scores built from them are identical
/// and no second solve is needed; a shrinkage fit publishes no OLS residuals at all.
#[test]
fn point_fit_residuals_give_the_normal_equation_scores_without_a_second_solve() {
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + ((i * i) % 7) as f64 - 3.0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let (point, residuals) = est
        .fit_point_with_ols_residuals(
            &prep,
            &mut EstimationWorkspace::default(),
            AssumptionSet::new(),
        )
        .unwrap();
    let residuals = residuals.expect("OLS fit carries residuals");
    let (m, n, p) = (&prep.design.matrix, prep.design.nrows, prep.design.ncols);
    let from_residuals = crate::normal_equation_scores_of_residuals(m, n, p, &residuals);
    let refit = crate::normal_equation_scores(m, n, p, &prep.design.outcome).unwrap();
    assert_eq!(from_residuals.len(), refit.len());
    for (a, b) in from_residuals.iter().flatten().zip(refit.iter().flatten()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    // The residuals are non-trivial, so the comparison above is not 0 == 0.
    assert!(residuals.iter().any(|e| e.abs() > 0.5));
    assert_eq!(point.provenance.n_obs, Some(n as u64));

    let ridge = LinearAdjustmentAte {
        fit_kind: LinearFitKind::Ridge { lambda: 1.0 },
        ..LinearAdjustmentAte::new()
    }
    .with_bootstrap_replicates(0);
    let (_, none) = ridge
        .fit_point_with_ols_residuals(
            &prep,
            &mut EstimationWorkspace::default(),
            AssumptionSet::new(),
        )
        .unwrap();
    assert!(none.is_none());
}

#[test]
fn bootstrap_replicate_failures_stay_inside_the_replicate() {
    assert!(bootstrap_fit_failed(&EstimationError::stats_msg("singular design")));
    assert!(bootstrap_fit_failed(&EstimationError::unsupported(LASSO_UNCONVERGED)));
    assert!(bootstrap_fit_failed(&EstimationError::unsupported(HUBER_UNCONVERGED)));
    assert!(bootstrap_fit_failed(&EstimationError::data_msg(
        "target population left no rows for g-computation",
    )));
    assert!(!bootstrap_fit_failed(&EstimationError::TargetPopulation));
}

#[test]
fn recovers_ate_two() {
    let (data, estimand) = toy();
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(50);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!((effect.ate - 2.0).abs() < 1e-8);
    assert!(effect.se_bootstrap().is_some());
    let identity: Vec<usize> = (0..prep.design.nrows).collect();
    let resampled = est.ate_on_row_indices(&prep, &mut ws, &identity).unwrap();
    assert!((resampled - effect.ate).abs() < 1e-12);
}

/// A requested interval is produced by the whole procedure on the production estimator: a
/// request whose tolerances are never met runs the declared replicate count and reproduces
/// the fixed-count bootstrap's standard error bit for bit, while a met tolerance stops on a
/// wave boundary below the cap. The waves, Monte Carlo error and stop reason are reported
/// beside the interval, and a fit with no request reports none.
#[test]
fn a_requested_interval_runs_the_complete_resampling_procedure() {
    use crate::adaptive_resample_waves::StopReason;
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + ((i * i) % 7) as f64 - 3.0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(1);
    let fit = |est: &LinearAdjustmentAte| {
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        est.fit(&prep, &mut EstimationWorkspace::default(), &ctx, AssumptionSet::new()).unwrap()
    };
    let fixed_est = LinearAdjustmentAte::new().with_bootstrap_replicates(240);
    let fixed = fit(&fixed_est);
    assert!(fixed.diagnostics.resampling.is_none());

    let never = crate::util::IntervalRequest {
        level: 0.9,
        wave_size: 60,
        min_replicates: 0,
        se_rel_tol: Some(1e-12),
        half_width_rel_tol: Some(1e-12),
    };
    let full = fit(&fixed_est.clone().with_bootstrap_interval(never));
    let report = full.diagnostics.resampling.as_ref().expect("the request is reported");
    assert_eq!(report.stop, StopReason::MaxReplicates);
    assert_eq!((report.attempted, report.waves.len()), (240, 4));
    assert_eq!(
        full.uncertainty.bootstrap.se.map(f64::to_bits),
        fixed.uncertainty.bootstrap.se.map(f64::to_bits)
    );
    assert_eq!(full.uncertainty.bootstrap.replicates_ok, fixed.uncertainty.bootstrap.replicates_ok);
    assert_eq!(full.ate.to_bits(), fixed.ate.to_bits());
    let (lower, upper) = report.interval.expect("the interval is licensed");
    assert!(lower < full.ate && full.ate < upper);
    assert!(report.se_rel_mc_error.is_some() && report.half_width_rel_mc_error.is_some());

    let loose = crate::util::IntervalRequest {
        level: 0.9,
        wave_size: 30,
        min_replicates: 60,
        se_rel_tol: Some(0.5),
        half_width_rel_tol: Some(0.9),
    };
    let early = fit(&fixed_est.clone().with_bootstrap_interval(loose));
    let report = early.diagnostics.resampling.as_ref().unwrap();
    assert_eq!(report.stop, StopReason::Converged);
    assert!(report.attempted < 240 && report.attempted % 30 == 0);
    assert!(early.uncertainty.bootstrap.early_stopped);
    assert_eq!(
        usize::try_from(report.attempted).unwrap(),
        report.waves.iter().map(|w| usize::try_from(w.attempted).unwrap()).sum::<usize>()
    );
}

/// The shared Gram inverse and the column-wise leverage pass reproduce the
/// row-by-row Frisch–Waugh influence `δ·n·[(XᵀX)⁻¹x_i]_T·e_i` and the
/// homoskedastic SE `√(σ²·[(XᵀX)⁻¹]_TT)` exactly.
#[test]
fn influence_and_homoskedastic_se_match_the_row_wise_gram_formula() {
    let (data, estimand) = toy();
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let t_col = prep.design.treatment_column().unwrap();
    let fit = prep.design.fit_ols(&est.backend, &mut ws.ols).unwrap();
    let mut xtx = vec![0.0; p * p];
    form_xtx(&prep.design.matrix, n, p, &mut xtx);
    let inv = invert_square(&xtx, p).unwrap();
    let sigma2 = fit.rss / (n - p) as f64;
    let se = (sigma2 * inv[t_col * p + t_col]).sqrt() * prep.treatment_delta.abs();
    // The production inverse comes from the QR factor and never forms XᵀX, so it agrees with
    // the Gram inverse to rounding rather than bit for bit.
    let reported = effect.se_analytic();
    assert!((reported - se).abs() <= 1e-10 * se.abs(), "{reported} vs {se}");
    let influence = effect.scores.influence.as_ref().unwrap();
    let row = &inv[t_col * p..(t_col + 1) * p];
    let scale = prep.treatment_delta * n as f64;
    for i in 0..n {
        let leverage: f64 = (0..p).map(|c| row[c] * prep.design.matrix[c * n + i]).sum();
        let expected = scale * leverage * fit.residuals[i];
        assert!((influence[i] - expected).abs() <= 1e-10 * (1.0 + expected.abs()), "row {i}");
    }
}

#[test]
fn scales_ate_by_level_delta() {
    let (data, estimand) = toy();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 2.0);
    // The 0/1 treatment never takes level 2, so the arm reading refuses the contrast instead
    // of extrapolating the slope.
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let refused = est.prepare(&data, &estimand, &query).err().expect("level 2 has no rows");
    assert!(refused.to_string().starts_with("reason=arm_not_populated:"), "{refused}");
    // Read as a numeric dose, the contrast is the slope times the level distance.
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0).with_dose_reading(true);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    // β_T ≈ 2, delta = 2 → ATE ≈ 4
    assert!((effect.ate - 4.0).abs() < 1e-8);
}

#[test]
fn recovers_att_via_gcomp() {
    let (data, estimand) = toy();
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!((effect.ate - 2.0).abs() < 1e-8, "att={}", effect.ate);
}

#[test]
fn predicate_restricts_prepared_rows() {
    use antecedent_core::PredicateExpr;
    let (data, estimand) = toy();
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let all = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let half = all.clone().with_target_population(TargetPopulation::Predicate(
        PredicateExpr::rows((0..10).collect::<Vec<_>>()),
    ));
    let prep_all = est.prepare(&data, &estimand, &all).unwrap();
    let prep_half = est.prepare(&data, &estimand, &half).unwrap();
    assert_eq!(prep_all.design.nrows, 100);
    assert_eq!(prep_half.design.nrows, 10);
    let named =
        all.with_target_population(TargetPopulation::Predicate(PredicateExpr::named("cohort")));
    let err = est.prepare(&data, &estimand, &named).unwrap_err();
    assert!(err.to_string().contains("PopulationRegistry") || err.to_string().contains("named"));
}

#[test]
fn source_cluster_labels_follow_their_rows_through_the_population_mask() {
    use antecedent_core::PredicateExpr;
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Predicate(PredicateExpr::rows(
            (10..70).collect::<Vec<_>>(),
        )));
    // Not an arithmetic progression mod 7: for those, every shifted or reversed copy is the same
    // partition under renamed ids (same cluster SE), so it would not detect misalignment.
    let labels: Vec<u32> = (0..100u32).map(|i| (i * i / 3 + i) % 7).collect();
    let se_of = |est: LinearAdjustmentAte| {
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        est.fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
            .unwrap()
            .se_analytic()
    };
    let base = LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_se_kind(AnalyticSeKind::Cluster);
    let by_source = se_of(base.clone().with_source_cluster_ids(labels.clone()));
    let aligned = se_of(base.clone().with_cluster_ids(labels[10..70].to_vec()));
    let misaligned = se_of(base.clone().with_cluster_ids(labels[..60].to_vec()));
    assert!(by_source.is_finite() && by_source > 0.0);
    assert_eq!(by_source.to_bits(), aligned.to_bits());
    assert_ne!(by_source.to_bits(), misaligned.to_bits());
}

#[test]
fn source_cluster_labels_must_cover_the_source_rows_and_exclude_complete_case_labels() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let base = LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_se_kind(AnalyticSeKind::Cluster);
    let short = base.clone().with_source_cluster_ids(vec![0; 99]);
    assert!(short.prepare(&data, &estimand, &query).unwrap_err().to_string().contains("99"));
    let both = base.with_source_cluster_ids(vec![0; 100]).with_cluster_ids(vec![0; 100]);
    assert!(both.prepare(&data, &estimand, &query).unwrap_err().to_string().contains("not both"));
}

#[test]
fn hc_sandwich_kinds_yield_finite_se() {
    // Noisy outcome so the residuals are non-zero. The reference forms the sandwich for the
    // treatment coefficient from the explicit Gram inverse: with u_i = [(XᵀX)⁻¹x_i]_T,
    // HC_k is Σ e_i² u_i² / (1−h_i)^k (k = 0, 1 for HC1-free HC0, 2, 3 as named) and
    // Newey-West is the Bartlett-weighted long-run sum of e_t u_t.
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let probe = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = probe.prepare(&data, &estimand, &query).unwrap();
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let t_col = prep.design.treatment_column().unwrap();
    let fit = prep.design.fit_ols(&probe.backend, &mut EstimationWorkspace::default().ols).unwrap();
    let mut xtx = vec![0.0; p * p];
    form_xtx(&prep.design.matrix, n, p, &mut xtx);
    let inv = invert_square(&xtx, p).unwrap();
    let u: Vec<f64> =
        (0..n).map(|i| (0..p).map(|c| inv[t_col * p + c] * prep.design.matrix[c * n + i]).sum()).collect();
    let leverage: Vec<f64> = (0..n)
        .map(|i| {
            (0..p)
                .map(|a| {
                    (0..p).map(|b| prep.design.matrix[a * n + i] * inv[a * p + b] * prep.design.matrix[b * n + i]).sum::<f64>()
                })
                .sum()
        })
        .collect();
    let e = &fit.residuals;
    let hc = |power: i32| -> f64 {
        (0..n).map(|i| e[i] * e[i] * u[i] * u[i] / (1.0 - leverage[i]).powi(power)).sum::<f64>().sqrt()
            * prep.treatment_delta.abs()
    };
    let newey_west = |lag: usize| -> f64 {
        let mut total: f64 = (0..n).map(|i| e[i] * e[i] * u[i] * u[i]).sum();
        for l in 1..=lag {
            let weight = 1.0 - l as f64 / (lag as f64 + 1.0);
            let cross: f64 = (l..n).map(|t| e[t] * u[t] * e[t - l] * u[t - l]).sum();
            total += 2.0 * weight * cross;
        }
        total.sqrt() * prep.treatment_delta.abs()
    };
    let mut seen = Vec::new();
    for (kind, expected) in [
        (AnalyticSeKind::Hc0, hc(0)),
        (AnalyticSeKind::Hc2, hc(1)),
        (AnalyticSeKind::Hc3, hc(2)),
        (AnalyticSeKind::NeweyWest { lag: 2 }, newey_west(2)),
    ] {
        let effect = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, kind);
        let got = effect.se_analytic();
        assert!(got.is_finite() && got > 0.0, "{kind:?}");
        let tolerance = 1e-9 * expected;
        assert!((got - expected).abs() <= tolerance, "{kind:?}: {got} vs {expected}");
        seen.push(got);
    }
    // The four estimators are different sandwiches, not one value reported four times.
    assert!(seen[0] < seen[1] && seen[1] < seen[2], "{seen:?}");
}

#[test]
fn ridge_lasso_huber_fit_kinds_recover_ate() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    // The data are exact (`y = 1 + 2t + z`), so OLS and Huber recover 2 to rounding; the
    // penalised fits carry only their own shrinkage bias, of the order of the penalty.
    for (kind, tolerance) in [
        (LinearFitKind::Ridge { lambda: 1e-3 }, 3e-4),
        (LinearFitKind::Lasso { lambda: 1e-4 }, 2e-5),
        (LinearFitKind::Huber { c: 1.345 }, 1e-9),
    ] {
        let est = LinearAdjustmentAte { fit_kind: kind, ..LinearAdjustmentAte::new() }
            .with_bootstrap_replicates(0);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let effect =
            est.fit(&prep, &mut ws, &ExecutionContext::for_tests(2), AssumptionSet::new()).unwrap();
        assert!(effect.ate.is_finite(), "{kind:?}");
        assert!((effect.ate - 2.0).abs() < tolerance, "ate={} kind={kind:?}", effect.ate);
    }
}

#[test]
fn lasso_analytic_se_nan_bootstrap_finite() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let no_boot = LinearAdjustmentAte {
        fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
        ..LinearAdjustmentAte::new()
    }
    .with_bootstrap_replicates(0);
    let prep = no_boot.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let effect =
        no_boot.fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new()).unwrap();
    assert!(effect.se_analytic().is_nan(), "lasso se_analytic={}", effect.se_analytic());
    assert!(effect.se_bootstrap().is_none());

    let with_boot = LinearAdjustmentAte {
        fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
        ..LinearAdjustmentAte::new()
    }
    .with_bootstrap_replicates(40);
    let effect = with_boot
        .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
        .unwrap();
    assert!(effect.se_analytic().is_nan());
    let boot = effect.se_bootstrap().expect("bootstrap SE");
    assert!(boot.is_finite() && boot >= 0.0, "se_bootstrap={boot}");

    // With noise the near-unpenalised lasso is OLS, so its pairs bootstrap SE estimates the
    // OLS standard error; 40 replicates pin it to well within a factor of two (the sampling
    // error of an SE from 40 replicates is about 11%).
    let (noisy, noisy_estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
    let ols_se =
        fit_kind_effect(&noisy, &noisy_estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic)
            .se_analytic();
    let prep = with_boot.prepare(&noisy, &noisy_estimand, &query).unwrap();
    let boot = with_boot
        .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
        .unwrap()
        .se_bootstrap()
        .expect("bootstrap SE");
    assert!(boot > 0.6 * ols_se && boot < 1.6 * ols_se, "bootstrap {boot} vs OLS {ols_se}");
}

/// Deterministic bounded noise in `[-0.1, 0.1]`.
fn wiggle(i: usize) -> f64 {
    0.1 * (7.31 * i as f64).sin()
}

fn fit_kind_effect(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    fit_kind: LinearFitKind,
    se_kind: AnalyticSeKind,
) -> EffectEstimate {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte { fit_kind, ..LinearAdjustmentAte::new() }
        .with_bootstrap_replicates(0)
        .with_se_kind(se_kind);
    let prep = est.prepare(data, estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    est.fit(&prep, &mut ws, &ExecutionContext::for_tests(5), AssumptionSet::new()).unwrap()
}

#[test]
fn huber_with_unclipped_scores_reproduces_ols_variance_and_influence() {
    // With c so large that no residual is clipped, ψ(u)=u and ψ′=1, so the Huber
    // sandwich collapses to the OLS one: τ² = RSS/(n−p), κ = 1, bread (XᵀX)⁻¹.
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
    for se_kind in [AnalyticSeKind::Homoskedastic, AnalyticSeKind::Hc0, AnalyticSeKind::Hc3] {
        let ols = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, se_kind);
        let hub = fit_kind_effect(&data, &estimand, LinearFitKind::Huber { c: 1e9 }, se_kind);
        assert!((hub.ate - ols.ate).abs() < 1e-8, "{se_kind:?}");
        let rel = (hub.se_analytic() - ols.se_analytic()).abs() / ols.se_analytic();
        assert!(rel < 1e-7, "{se_kind:?}: huber {} ols {}", hub.se_analytic(), ols.se_analytic());
        let (hi, oi) =
            (hub.scores.influence.as_ref().unwrap(), ols.scores.influence.as_ref().unwrap());
        for (a, b) in hi.iter().zip(oi.iter()) {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
        }
    }
}

#[test]
fn huber_se_ignores_gross_outliers_the_estimator_downweights() {
    // 10% of rows (one odd, one even index per 20, so both arms) carry a +50 shock. OLS
    // RSS/(n−p) is dominated by them; Huber's τ² = E[ψ²]/E[ψ′]² only sees them clipped
    // at c·s, with s ≈ MAD-scale of the ±0.1 noise, so its SE must be far below the OLS
    // one and the effect stays near 2.
    let (data, estimand) = toy_with(|i, t, z| {
        let shock = if i % 20 == 3 || i % 20 == 8 { 50.0 } else { 0.0 };
        1.0 + 2.0 * t + z + wiggle(i) + shock
    });
    let ols = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
    let hub = fit_kind_effect(
        &data,
        &estimand,
        LinearFitKind::Huber { c: 1.345 },
        AnalyticSeKind::Homoskedastic,
    );
    assert!((hub.ate - 2.0).abs() < 0.1, "huber ate {}", hub.ate);
    assert!(
        hub.se_analytic() < 0.1 * ols.se_analytic(),
        "huber se {} vs ols se {}",
        hub.se_analytic(),
        ols.se_analytic()
    );
}

#[test]
fn ridge_and_lasso_publish_no_analytic_se_and_no_ols_influence() {
    let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
    let ols = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
    let ridge0 = fit_kind_effect(
        &data,
        &estimand,
        LinearFitKind::Ridge { lambda: 0.0 },
        AnalyticSeKind::Homoskedastic,
    );
    assert!(ridge0.se_analytic().is_nan());
    // λ = 0 ridge is OLS, so its influence must be the OLS one.
    let (ri, oi) =
        (ridge0.scores.influence.as_ref().unwrap(), ols.scores.influence.as_ref().unwrap());
    for (a, b) in ri.iter().zip(oi.iter()) {
        assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
    }
    // Heavy ridge shrinks the treatment coefficient, so its bread differs from (XᵀX)⁻¹.
    let ridge = fit_kind_effect(
        &data,
        &estimand,
        LinearFitKind::Ridge { lambda: 20.0 },
        AnalyticSeKind::Homoskedastic,
    );
    assert!(ridge.se_analytic().is_nan());
    assert!(ridge.ate < ols.ate - 1e-3);
    let heavy = ridge.scores.influence.as_ref().unwrap();
    assert!(heavy.iter().zip(oi.iter()).any(|(a, b)| (a - b).abs() > 1e-3));
    let lasso = fit_kind_effect(
        &data,
        &estimand,
        LinearFitKind::Lasso { lambda: 1e-4 },
        AnalyticSeKind::Homoskedastic,
    );
    assert!(lasso.se_analytic().is_nan());
    assert!(lasso.scores.influence.is_none());
}

#[test]
fn unconverged_iterative_fits_are_refused() {
    assert!(require_converged(true, LASSO_UNCONVERGED).is_ok());
    for msg in [LASSO_UNCONVERGED, HUBER_UNCONVERGED] {
        let err = require_converged(false, msg).unwrap_err();
        assert!(err.to_string().contains("did not converge"), "{err}");
        assert!(bootstrap_fit_failed(&err));
    }
}

#[test]
fn bootstrap_skips_singular_fits_and_empty_arms_only() {
    assert!(bootstrap_fit_failed(&EstimationError::stats_msg("singular design")));
    assert!(bootstrap_fit_failed(&EstimationError::data_msg(
        "target population left no rows for g-computation",
    )));
    assert!(!bootstrap_fit_failed(&EstimationError::data_msg(
        "row resample buffers must match the row map",
    )));
}

#[test]
fn the_estimate_records_the_preparation_it_was_fitted_on() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte::new();
    let prepared = est.prepare(&data, &estimand, &query).unwrap();
    let point = est
        .fit_point(&prepared, &mut EstimationWorkspace::default(), AssumptionSet::new())
        .unwrap();
    crate::prepare::assert_records_preparation(&point, 100);
}

#[test]
fn prepare_records_order_schema_missing_values_and_a_proved_bound() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prepared = LinearAdjustmentAte::new().prepare(&data, &estimand, &query).unwrap();
    let receipt = &prepared.preparation;
    assert_eq!(receipt.path(), PreparationBackend::DirectArrow);
    assert_eq!(
        receipt.schema().columns.iter().map(|column| column.name.as_ref()).collect::<Vec<_>>(),
        ["t", "y", "z"]
    );
    assert_eq!(receipt.schema().columns[0].value_kind, SchemaValueKind::Continuous);
    assert!(receipt.categories().is_empty());
    assert_eq!(receipt.missing().unwrap().proved_rows_kept(), Some(data.row_count()));
    assert_eq!(receipt.proved_rows(), Some(u64::try_from(data.row_count()).unwrap()));
    assert!(!receipt.estimator_sees_unbounded_stream());
    // No budget was declared, so the payload is unmetered and the route is not bounded.
    assert!(matches!(receipt.admission(), AdmissionRecord::Unmetered { .. }));
    assert!(!receipt.is_bounded_end_to_end_route());

    let budgeted = LinearAdjustmentAte::new()
        .with_preparation_budget(PreparationBudget::new(1 << 20))
        .prepare(&data, &estimand, &query)
        .unwrap();
    assert!(budgeted.preparation.is_bounded_end_to_end_route());
    assert_eq!(budgeted.preparation.proved_rows(), receipt.proved_rows());
}

#[test]
fn a_tight_budget_refuses_preparation() {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let estimator = LinearAdjustmentAte::new().with_preparation_budget(PreparationBudget::new(0));
    let message = estimator.prepare(&data, &estimand, &query).expect_err("budget").to_string();
    assert!(message.contains("exceeds the budget 0"), "{message}");
}

#[test]
fn linear_adjustment_refuses_a_scan_that_is_not_a_bounded_route() {
    use antecedent_data::materialize::{
        EstimatorStreamNeed, PreparationBudget, StreamDelivery, StreamQuote, deliver_stream,
    };

    let quote =
        StreamQuote { scratch_bytes: Some(8), materialize_bytes: Some(8), chunk_rows: vec![4, 4] };
    let mut controls = PreparationControls::finite_resident();
    controls.stream = EstimatorStreamNeed::CanStream;
    controls.budget = Some(PreparationBudget::new(8));
    let StreamDelivery::Scan(scan) =
        deliver_stream(&quote, PreparationBudget::new(8), controls).unwrap()
    else {
        panic!("scan");
    };
    let refused = LinearAdjustmentAte::reject_stream(&scan);
    assert!(refused.to_string().contains("cannot stream"), "{refused}");
    assert!(scan.receipt().estimator_sees_unbounded_stream());
    assert!(!scan.receipt().is_bounded_end_to_end_route());
}

fn categorical_adjustment_table() -> (TabularData, IdentifiedEstimand) {
    use antecedent_core::CategoryDomainId;
    use antecedent_data::{
        CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
        ValidityBitmap,
    };

    let domain = CategoryDomain::try_new(
        CategoryDomainId::from_raw(2),
        vec![CategoryLevel { label: Arc::from("a") }, CategoryLevel { label: Arc::from("b") }],
        false,
        Some(CategoryCode::from_raw(0)),
        UnknownCategoryPolicy::Fail,
    )
    .unwrap();
    let n = 4usize;
    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "z",
            ValueType::Categorical,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            Some(CategoryDomainId::from_raw(2)),
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::<[f64]>::from(vec![0.0, 1.0, 0.0, 1.0]),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::<[f64]>::from(vec![0.0, 1.0, 1.0, 2.0]),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(
                VariableId::from_raw(2),
                vec![
                    CategoryCode::from_raw(0),
                    CategoryCode::from_raw(0),
                    CategoryCode::from_raw(1),
                    CategoryCode::from_raw(1),
                ],
                ValidityBitmap::all_valid(n),
                Arc::new(domain),
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
fn a_categorical_adjustment_is_expanded_before_the_design() {
    use antecedent_data::{CategoryCode, Contrast};
    let (data, estimand) = categorical_adjustment_table();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let estimator = LinearAdjustmentAte::new().with_contrast(
        VariableId::from_raw(2),
        Contrast::Treatment { reference: CategoryCode::from_raw(0) },
    );
    let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
    assert_eq!(prepared.design.ncols, 3);
    assert_eq!(prepared.preparation.categories().len(), 1);
    assert_eq!(prepared.preparation.categories()[0].coding, "treatment");
    assert_eq!(prepared.preparation.categories()[0].n_columns, 1);
}

#[test]
fn a_repeated_categorical_is_expanded_and_recorded_once() {
    use antecedent_data::{CategoryCode, Contrast};
    let (data, _) = categorical_adjustment_table();
    let contrast = Contrast::Treatment { reference: CategoryCode::from_raw(0) };
    let ids = [
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        VariableId::from_raw(2),
    ];
    let product = prepare_table(
        &data,
        &ids,
        MissingPolicy::CompleteCase,
        &[(VariableId::from_raw(2), contrast)],
        PreparationControls::finite_resident(),
    )
    .unwrap();
    assert_eq!(product.expanded.len(), 1);
    assert_eq!(product.receipt.categories().len(), 1);
    assert_eq!(product.receipt.schema().columns.len(), 3);
    assert_eq!(product.receipt.category_levels().len(), 1);
    assert_eq!(product.row_mask, vec![true; data.row_count()]);
}
