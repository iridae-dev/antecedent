//! Multiple imputation: Rubin's rules against a hand computation, recovery of a known linear
//! truth with valid coverage on missing-at-random data where complete-case analysis is biased,
//! the stacked posterior, refusals, determinism and admission.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "imputed binary cells and bitwise replays compare exactly")]

use std::sync::Arc;

use antecedent_core::execution::AdmissionPool;
use antecedent_core::{CausalRng, DeclaredSamplingDesign, ExecutionContext};
use antecedent_data::{ImputationModel, ImputationSpec, MissingnessAssumption};
use antecedent_estimate::multiple_imputation::{
    self, ImputationColumn, ImputationFit, ImputationUnits, Imputer, PooledEstimate, rubin_pool,
    stack_posterior,
};
use antecedent_kernels::standard_normal;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn sparse_group_ids_are_refused_before_sizing_imputation_draws() {
    for labels in [vec![0, 4_000_000_000], vec![0, 2, 2]] {
        let rows = labels.len();
        let groups: Arc<[u32]> = labels.into();
        let error =
            multiple_imputation::bind_units(None, Some(Arc::clone(&groups)), rows).unwrap_err();
        assert_eq!(error.detail(), "multiple_imputation.units_labels");
        // Directly constructing Groups must not bypass bind_units' validation.
        let values: Vec<f64> = (0..rows).map(|i| i as f64 + 1.0).collect();
        let error = Imputer::new(
            vec![ImputationColumn::new("x", values)],
            ImputationSpec::new(),
            ImputationUnits::Groups(groups),
        )
        .unwrap_err();
        assert_eq!(error.detail(), "multiple_imputation.units_labels");
    }
    let groups =
        multiple_imputation::bind_units(None, Some(Arc::from([0_u32, 0, 1, 1])), 4).unwrap();
    assert_eq!(groups.count(4), 2);
}

#[test]
fn interactions_recheck_the_observed_conditional_model_width() {
    let columns = || {
        (0..4)
            .map(|j| {
                let values: Vec<f64> = (0..7)
                    .map(|i| if j == 0 && i == 0 { f64::NAN } else { ((i + 1) as f64).powi(j + 1) })
                    .collect();
                ImputationColumn::new(format!("x{j}"), values)
            })
            .collect()
    };
    let model = Imputer::new(columns(), ImputationSpec::new(), ImputationUnits::Rows).unwrap();
    let error = model
        .with_interactions(&[
            (Arc::from("x1"), Arc::from("x2")),
            (Arc::from("x1"), Arc::from("x3")),
            (Arc::from("x2"), Arc::from("x3")),
        ])
        .unwrap_err();
    assert_eq!(error.detail(), "multiple_imputation.not_identified");
}

#[test]
fn interaction_and_dynamic_width_checks_are_independent_of_builder_order() {
    let model = || {
        let columns = (0..3)
            .map(|j| {
                let values: Vec<f64> = (0..14)
                    .map(|i| if j == 0 && i == 7 { f64::NAN } else { ((i + 1) as f64).powi(j + 1) })
                    .collect();
                ImputationColumn::new(format!("x{j}"), values)
            })
            .collect();
        Imputer::new(columns, ImputationSpec::new(), ImputationUnits::Rows).unwrap()
    };
    let pair = [(Arc::from("x1"), Arc::from("x2"))];
    // Eleven observed interior rows identify nine base dynamic coefficients,
    // but not the tenth interaction coefficient with the required two-row margin.
    let error = model().with_dynamics(1).unwrap().with_interactions(&pair).unwrap_err();
    assert_eq!(error.detail(), "multiple_imputation.not_identified");
    let error = model().with_interactions(&pair).unwrap().with_dynamics(1).unwrap_err();
    assert_eq!(error.detail(), "multiple_imputation.not_identified");
}

#[test]
fn rubin_pooling_matches_a_hand_computation() {
    // m = 3: Q = (1, 2, 4), U = (0.5, 0.6, 0.7).
    let pooled = rubin_pool(&[1.0, 2.0, 4.0], &[0.5, 0.6, 0.7], None).unwrap();
    let q_bar = 7.0 / 3.0;
    let u_bar = 0.6;
    let b = ((1.0 - q_bar) * (1.0 - q_bar)
        + (2.0 - q_bar) * (2.0 - q_bar)
        + (4.0 - q_bar) * (4.0 - q_bar))
        / 2.0;
    let t = u_bar + (1.0 + 1.0 / 3.0) * b;
    assert!(close(pooled.point, q_bar, 1e-12));
    assert!(close(pooled.within, u_bar, 1e-12));
    assert!(close(pooled.between, b, 1e-12));
    assert!(close(pooled.total, t, 1e-12));
    let r = (1.0 + 1.0 / 3.0) * b / u_bar;
    assert!(close(pooled.relative_increase, r, 1e-12));
    let df_rubin = 2.0 * (1.0 + 1.0 / r).powi(2);
    assert!(close(pooled.df_rubin, df_rubin, 1e-9));
    assert!(close(pooled.df, df_rubin, 1e-9), "large-sample df without a complete-data df");
    let gamma = (r + 2.0 / (df_rubin + 3.0)) / (r + 1.0);
    assert!(close(pooled.fraction_missing_information, gamma, 1e-12));
    assert!(close(pooled.standard_error(), t.sqrt(), 1e-12));

    // Barnard-Rubin with a complete-data df of 20.
    let br = rubin_pool(&[1.0, 2.0, 4.0], &[0.5, 0.6, 0.7], Some(20.0)).unwrap();
    let lambda = (1.0 + 1.0 / 3.0) * b / t;
    let df_obs = 21.0 / 23.0 * 20.0 * (1.0 - lambda);
    let df_br = 1.0 / (1.0 / df_rubin + 1.0 / df_obs);
    assert!(close(br.df, df_br, 1e-9));
    assert!(br.df < 20.0 && br.df <= df_rubin);
    let (lo, hi) = br.interval(0.95);
    let half = antecedent_stats::student_t_ppf(0.975, df_br) * t.sqrt();
    assert!(close(lo, q_bar - half, 1e-9) && close(hi, q_bar + half, 1e-9));

    // No between-imputation variance: infinite Rubin df, Barnard-Rubin falls back to df_obs.
    let flat = rubin_pool(&[1.0, 1.0], &[0.2, 0.2], Some(10.0)).unwrap();
    assert_eq!(flat.between, 0.0);
    assert!(flat.df_rubin.is_infinite());
    assert!(close(flat.df, 11.0 / 13.0 * 10.0, 1e-12));

    assert_eq!(
        rubin_pool(&[1.0], &[0.1], None).unwrap_err().detail(),
        "multiple_imputation.too_few_imputations"
    );
    assert_eq!(
        rubin_pool(&[1.0, 2.0], &[0.1, -0.1], None).unwrap_err().detail(),
        "multiple_imputation.pool_non_finite"
    );
    assert_eq!(
        rubin_pool(&[1.0, 2.0], &[0.1, 0.1], Some(0.0)).unwrap_err().detail(),
        "multiple_imputation.pool_df"
    );
}

#[test]
fn the_stacked_posterior_keeps_the_between_axis() {
    let a = [0.0, 1.0, 2.0];
    let b = [10.0, 11.0, 12.0];
    let stacked = stack_posterior(&[&a, &b]).unwrap();
    assert_eq!(stacked.draws, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    assert!(close(stacked.mean, 6.0, 1e-12));
    assert!(close(stacked.within, 1.0, 1e-12));
    assert!(close(stacked.between, 50.0, 1e-12));
    assert!(stacked.between_share() > 0.8);
    assert_eq!(
        stack_posterior(&[&a, &b[..2]]).unwrap_err().detail(),
        "multiple_imputation.stack_shape",
        "unequal draws would weight unequally"
    );
    assert_eq!(
        stack_posterior(&[&a]).unwrap_err().detail(),
        "multiple_imputation.too_few_imputations"
    );
}

/// `y = 1 + 2 x + 1.5 z + e` with `corr(x, z) = 0.6`: the slope of `y` on `x` alone is
/// `2 + 1.5 · 0.6 = 2.9`. `y` is missing with probability rising in `z` (missing at random
/// given `z`, which is not in the analysis model), and `x` is missing for a few rows at
/// random. Complete cases select on `z` and bias the slope; imputation with `z` in the model
/// recovers it.
const TRUE_SLOPE: f64 = 2.9;

fn simulate(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let xi = 0.6 * zi + 0.8 * standard_normal(&mut rng);
        let yi = 1.0 + 2.0 * xi + 1.5 * zi + standard_normal(&mut rng);
        let p_missing_y = 1.0 / (1.0 + (-(2.0 * zi - 0.5)).exp());
        let y_obs = if rng.next_f64() < p_missing_y { f64::NAN } else { yi };
        let x_obs = if rng.next_f64() < 0.1 { f64::NAN } else { xi };
        x.push(x_obs);
        y.push(y_obs);
        z.push(zi);
    }
    (x, y, z)
}

/// Least-squares slope of `y` on `x` with an intercept, its variance and `n − 2`.
fn ols_slope(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let rows: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();
    let n = rows.len() as f64;
    let mx = rows.iter().map(|r| r.0).sum::<f64>() / n;
    let my = rows.iter().map(|r| r.1).sum::<f64>() / n;
    let sxx = rows.iter().map(|r| (r.0 - mx).powi(2)).sum::<f64>();
    let sxy = rows.iter().map(|r| (r.0 - mx) * (r.1 - my)).sum::<f64>();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let rss = rows.iter().map(|r| (r.1 - intercept - slope * r.0).powi(2)).sum::<f64>();
    let s2 = rss / (n - 2.0);
    (slope, s2 / sxx, n - 2.0)
}

fn imputer(x: &[f64], y: &[f64], z: &[f64], spec: ImputationSpec) -> Imputer {
    Imputer::new(
        vec![
            ImputationColumn::new("x", x.to_vec()),
            ImputationColumn::new("y", y.to_vec()),
            ImputationColumn::new("z", z.to_vec()),
        ],
        spec,
        ImputationUnits::Rows,
    )
    .unwrap()
}

fn pooled_slope(imputer: &Imputer, ctx: &ExecutionContext) -> multiple_imputation::RubinPooled {
    let run = multiple_imputation::run(imputer, ctx, |table, _| {
        let (slope, var, df) = ols_slope(&table.columns()[0], &table.columns()[1]);
        Ok::<_, ()>(ImputationFit::frequentist(slope, var, Some(df)))
    })
    .unwrap();
    match run.pooled {
        PooledEstimate::Rubin(pooled) => pooled,
        other => panic!("expected Rubin pooling, got {other:?}"),
    }
}

#[test]
fn imputation_recovers_a_linear_truth_with_valid_coverage_where_complete_cases_are_biased() {
    let ctx = ExecutionContext::for_tests(1);
    let replications = 200_u64;
    let mut covered = 0_u32;
    let mut mi_sum = 0.0;
    let mut cc_sum = 0.0;
    for seed in 0..replications {
        let (x, y, z) = simulate(400, 1_000 + seed);
        let (cc, _, _) = ols_slope(&x, &y);
        cc_sum += cc;
        let spec = ImputationSpec::new().with_imputations(10).with_iterations(5).with_seed(seed);
        let pooled = pooled_slope(&imputer(&x, &y, &z, spec), &ctx);
        mi_sum += pooled.point;
        let (lo, hi) = pooled.interval(0.95);
        if lo <= TRUE_SLOPE && TRUE_SLOPE <= hi {
            covered += 1;
        }
    }
    let n = replications as f64;
    let mi_mean = mi_sum / n;
    let cc_mean = cc_sum / n;
    let coverage = f64::from(covered) / n;
    assert!(close(mi_mean, TRUE_SLOPE, 0.05), "imputation mean slope {mi_mean}");
    assert!((cc_mean - TRUE_SLOPE).abs() > 0.1, "complete-case mean slope {cc_mean} is not biased");
    // 200 replications: the binomial standard error at 0.95 is about 0.015.
    assert!((0.90..=0.99).contains(&coverage), "coverage {coverage}");
}

#[test]
fn the_joint_normal_model_recovers_the_truth_and_refuses_an_incomplete_binary() {
    let ctx = ExecutionContext::for_tests(1);
    let mut sum = 0.0;
    for seed in 0..20 {
        let (x, y, z) = simulate(600, 5_000 + seed);
        let spec = ImputationSpec::new()
            .with_model(ImputationModel::JointNormal)
            .with_imputations(10)
            .with_iterations(10)
            .with_seed(seed);
        sum += pooled_slope(&imputer(&x, &y, &z, spec), &ctx).point;
    }
    assert!(close(sum / 20.0, TRUE_SLOPE, 0.08), "joint normal mean slope {}", sum / 20.0);

    let binary = vec![0.0, 1.0, f64::NAN, 1.0, 0.0, 1.0, 0.0, 1.0];
    let other: Vec<f64> = (0..8).map(f64::from).collect();
    let refused = Imputer::new(
        vec![ImputationColumn::new("b", binary), ImputationColumn::new("o", other)],
        ImputationSpec::new().with_model(ImputationModel::JointNormal),
        ImputationUnits::Rows,
    )
    .unwrap_err();
    assert_eq!(refused.code(), "missing_policy_not_applicable");
    assert_eq!(
        refused.next_attempt().map(antecedent_core::NextAttempt::name),
        Some("chained_equations")
    );
}

#[test]
fn chained_equations_draw_binary_cells_as_zero_or_one() {
    let mut rng = CausalRng::from_seed(3);
    let n = 300;
    let mut treat = Vec::with_capacity(n);
    let mut cov = Vec::with_capacity(n);
    for i in 0..n {
        let c = standard_normal(&mut rng);
        let t = if rng.next_f64() < 1.0 / (1.0 + (-c).exp()) { 1.0 } else { 0.0 };
        cov.push(c);
        treat.push(if i % 7 == 0 { f64::NAN } else { t });
    }
    let imputer = Imputer::new(
        vec![ImputationColumn::new("t", treat.clone()), ImputationColumn::new("c", cov)],
        ImputationSpec::new().with_imputations(3),
        ImputationUnits::Rows,
    )
    .unwrap();
    let variables = imputer.variables();
    assert_eq!(variables[0].kind, multiple_imputation::VariableKind::Binary);
    let table = imputer.impute(0, &ExecutionContext::for_tests(1)).unwrap();
    for (i, value) in table.columns()[0].iter().enumerate() {
        assert!(*value == 0.0 || *value == 1.0, "row {i} drew {value}");
        if !treat[i].is_nan() {
            assert_eq!(*value, treat[i], "an observed cell changed");
        }
    }
}

#[test]
fn missing_not_at_random_is_refused_with_a_sensitivity_analysis() {
    let spec = ImputationSpec::new().with_assumption(MissingnessAssumption::MissingNotAtRandom);
    let refused = multiple_imputation::check_spec(&spec).unwrap_err();
    assert_eq!(refused.code(), "missing_policy_not_applicable");
    assert_eq!(refused.detail(), "multiple_imputation.not_at_random");
    assert_eq!(
        refused.next_attempt().map(antecedent_core::NextAttempt::name),
        Some("sensitivity_analysis")
    );
    assert!(refused.message().contains("next attempt"));
    for spec in
        [ImputationSpec::new().with_imputations(1), ImputationSpec::new().with_iterations(0)]
    {
        let refused = multiple_imputation::check_spec(&spec).unwrap_err();
        assert_eq!(refused.code(), "invalid_argument");
        assert_eq!(refused.detail(), "multiple_imputation.spec_range");
    }
}

#[test]
fn the_sampling_design_governs_the_imputation_unit() {
    let groups: Arc<[u32]> = Arc::from(vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(multiple_imputation::bind_units(None, None, 6).unwrap(), ImputationUnits::Rows);
    assert_eq!(
        multiple_imputation::bind_units(
            Some(DeclaredSamplingDesign::ClusteredPanel),
            Some(Arc::clone(&groups)),
            6
        )
        .unwrap()
        .count(6),
        3
    );
    let refused =
        multiple_imputation::bind_units(Some(DeclaredSamplingDesign::ClusteredPanel), None, 6)
            .unwrap_err();
    assert_eq!(
        refused.next_attempt().map(antecedent_core::NextAttempt::name),
        Some("supply_cluster_labels")
    );
    let series =
        multiple_imputation::bind_units(Some(DeclaredSamplingDesign::DependentTimeSeries), None, 6)
            .unwrap_err();
    assert_eq!(
        series.next_attempt().map(antecedent_core::NextAttempt::name),
        Some("temporal_complete_case")
    );
    assert_eq!(
        multiple_imputation::bind_units(None, Some(groups), 5).unwrap_err().detail(),
        "multiple_imputation.units_shape"
    );

    // Whole-cluster draws: rows of one cluster share one bootstrap weight, and the result
    // differs from row-level draws of the same data.
    let (x, y, z) = simulate(120, 77);
    let clusters: Arc<[u32]> = (0..120_u32).map(|i| i / 4).collect();
    let spec = ImputationSpec::new().with_imputations(2).with_seed(9);
    let columns = || {
        vec![
            ImputationColumn::new("x", x.clone()),
            ImputationColumn::new("y", y.clone()),
            ImputationColumn::new("z", z.clone()),
        ]
    };
    let ctx = ExecutionContext::for_tests(1);
    let by_cluster = Imputer::new(columns(), spec, ImputationUnits::Groups(clusters)).unwrap();
    assert_eq!(by_cluster.unit_count(), 30);
    let by_row = Imputer::new(columns(), spec, ImputationUnits::Rows).unwrap();
    let a = by_cluster.impute(0, &ctx).unwrap();
    let b = by_row.impute(0, &ctx).unwrap();
    assert_ne!(a.columns()[1], b.columns()[1]);
}

#[test]
fn completed_tables_are_deterministic_across_threads_and_draw_order() {
    let (x, y, z) = simulate(300, 42);
    let spec = ImputationSpec::new().with_imputations(4).with_seed(11);
    let imputer = imputer(&x, &y, &z, spec);
    let serial = ExecutionContext::for_tests(1);
    let parallel = ExecutionContext::production(1, 4);
    let forward: Vec<Vec<Arc<[f64]>>> =
        (0..4).map(|j| imputer.impute(j, &serial).unwrap().columns().to_vec()).collect();
    let backward: Vec<Vec<Arc<[f64]>>> =
        (0..4).rev().map(|j| imputer.impute(j, &parallel).unwrap().columns().to_vec()).collect();
    for j in 0..4 {
        for (a, b) in forward[j].iter().zip(&backward[3 - j]) {
            let bits_a: Vec<u64> = a.iter().map(|v| v.to_bits()).collect();
            let bits_b: Vec<u64> = b.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits_a, bits_b, "imputation {j}");
        }
    }
    assert_ne!(forward[0][1], forward[1][1], "imputations draw distinct tables");
    let one = pooled_slope(&imputer, &serial);
    let many = pooled_slope(&imputer, &parallel);
    assert_eq!(one.point.to_bits(), many.point.to_bits());
    assert_eq!(one.total.to_bits(), many.total.to_bits());
    // Another seed draws other tables.
    let other = Imputer::new(
        vec![
            ImputationColumn::new("x", x.clone()),
            ImputationColumn::new("y", y.clone()),
            ImputationColumn::new("z", z.clone()),
        ],
        spec.with_seed(12),
        ImputationUnits::Rows,
    )
    .unwrap();
    assert_ne!(other.impute(0, &serial).unwrap().columns()[1], forward[0][1]);
}

#[test]
fn a_tiny_admission_pool_refuses_before_a_table_is_drawn() {
    let (x, y, z) = simulate(200, 5);
    let imputer = imputer(&x, &y, &z, ImputationSpec::new().with_imputations(2));
    let pool = AdmissionPool::fixed(1, 64);
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let refused = imputer.impute(0, &ctx).unwrap_err();
    assert_eq!(refused.detail(), "multiple_imputation.admission_refused");
    assert_eq!(pool.available().bytes(), 64, "nothing stays charged");
    let failure = multiple_imputation::run(&imputer, &ctx, |_, _| {
        Ok::<_, ()>(ImputationFit::frequentist(0.0, 1.0, None))
    })
    .unwrap_err();
    assert!(matches!(failure, multiple_imputation::MultipleImputationFailure::Imputation(_)));

    // A pool that funds one table at a time is enough: each table releases its lease.
    let bytes = imputer.imputation_bytes();
    let funded = AdmissionPool::fixed(1, bytes);
    let ctx = ExecutionContext::for_tests(1).with_admission(funded.clone());
    let table = imputer.impute(0, &ctx).unwrap();
    assert_eq!(funded.available().bytes(), 0);
    drop(table);
    assert_eq!(funded.available().bytes(), bytes);
    assert!(pooled_slope(&imputer, &ctx).total > 0.0);
}

#[test]
fn bayesian_fits_are_stacked_into_one_posterior() {
    let (x, y, z) = simulate(300, 8);
    let imputer = imputer(&x, &y, &z, ImputationSpec::new().with_imputations(5).with_seed(2));
    let run = multiple_imputation::run(&imputer, &ExecutionContext::for_tests(1), |table, _| {
        // A normal approximation to the slope's posterior, drawn deterministically per table.
        let (slope, var, _) = ols_slope(&table.columns()[0], &table.columns()[1]);
        let mut rng = CausalRng::from_seed(u64::from(table.index()));
        let draws: Arc<[f64]> =
            (0..400).map(|_| slope + var.sqrt() * standard_normal(&mut rng)).collect();
        Ok::<_, ()>(ImputationFit::bayesian(draws))
    })
    .unwrap();
    let PooledEstimate::Stacked(stacked) = run.pooled else {
        panic!("expected a stacked posterior")
    };
    assert_eq!(stacked.imputations, 5);
    assert_eq!(stacked.draws.len(), 2_000);
    assert!(stacked.between > 0.0);
    assert!(stacked.variance > stacked.within, "the mixture is wider than one table's posterior");
    // The stacked mean is the mean of the five tables' own slopes (equal weights), up to the
    // Monte Carlo error of 2000 draws (observed 0.003), and sits near the truth 2.9.
    let slopes: Vec<f64> = (0..5)
        .map(|j| {
            let table = imputer.impute(j, &ExecutionContext::for_tests(1)).unwrap();
            ols_slope(&table.columns()[0], &table.columns()[1]).0
        })
        .collect();
    let mean_slope = slopes.iter().sum::<f64>() / 5.0;
    assert!(close(stacked.mean, mean_slope, 0.01), "{} vs {mean_slope}", stacked.mean);
    assert!(close(stacked.mean, TRUE_SLOPE, 0.15), "{}", stacked.mean);
    assert!(run.imputed_cells.iter().any(|(name, cells)| name.as_ref() == "y" && *cells > 0));
}

#[test]
fn unidentified_conditional_models_are_refused() {
    let x = vec![1.0, f64::NAN, f64::NAN, f64::NAN];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    // Declare the single observed 0/1 value so this tests model identification,
    // independently of the ambiguous-domain refusal.
    let refused = Imputer::new_with_domains(
        vec![ImputationColumn::new("x", x), ImputationColumn::new("y", y)],
        ImputationSpec::new(),
        ImputationUnits::Rows,
        &[(Arc::from("x"), antecedent_data::ImputationDomain::Continuous)],
    )
    .unwrap_err();
    assert_eq!(refused.detail(), "multiple_imputation.not_identified");
    assert_eq!(
        refused.next_attempt().map(antecedent_core::NextAttempt::name),
        Some("complete_case")
    );
}

#[test]
fn constant_zero_one_requires_domain_and_binary_boundary_preserves_support() {
    use antecedent_data::ImputationDomain;
    use antecedent_estimate::multiple_imputation::VariableKind;
    for value in [0.0, 1.0] {
        let columns = || vec![ImputationColumn::new("x", vec![value, value, value, value, f64::NAN])];
        let error = Imputer::new(columns(), ImputationSpec::new(), ImputationUnits::Rows).unwrap_err();
        assert_eq!(error.code(), "invalid_argument");
        assert_eq!(error.detail(), "multiple_imputation.ambiguous_variable_domain");
        let binary = [(Arc::from("x"), ImputationDomain::Binary)];
        let imputer = Imputer::new_with_domains(columns(), ImputationSpec::new(), ImputationUnits::Rows, &binary).unwrap();
        assert_eq!(imputer.variables()[0].kind, VariableKind::Binary);
        assert_eq!(imputer.binary_boundary_variables().as_slice(), &[(Arc::<str>::from("x"), if value == 1.0 { 1 } else { 0 })]);
        let table = imputer.impute(0, &ExecutionContext::for_tests(1)).unwrap();
        assert_eq!(table.column("x").unwrap()[4], value);
        let normal = Imputer::new_with_domains(columns(), ImputationSpec::new().with_model(ImputationModel::JointNormal), ImputationUnits::Rows, &binary).unwrap_err();
        assert_eq!(normal.detail(), "multiple_imputation.binary_under_joint_normal");
        let continuous = [(Arc::from("x"), ImputationDomain::Continuous)];
        let continuous = Imputer::new_with_domains(columns(), ImputationSpec::new(), ImputationUnits::Rows, &continuous).unwrap();
        assert_eq!(continuous.variables()[0].kind, VariableKind::Continuous);
        assert!(continuous.impute(0, &ExecutionContext::for_tests(1)).unwrap().column("x").unwrap()[4].is_finite());
        let binary_digest = multiple_imputation::ImputationAssumptions::of(&imputer, None).digest();
        assert_ne!(binary_digest, multiple_imputation::ImputationAssumptions::of(&continuous, None).digest());
    }
    let ordered_columns = || vec![
        ImputationColumn::new("a", vec![0.0, 1.0, 0.0, 1.0, f64::NAN]),
        ImputationColumn::new("b", vec![2.0, 3.0, 4.0, 5.0, 6.0]),
    ];
    let first_domains = [(Arc::from("a"), ImputationDomain::Binary), (Arc::from("b"), ImputationDomain::Continuous)];
    let second_domains = [(Arc::from("b"), ImputationDomain::Continuous), (Arc::from("a"), ImputationDomain::Binary)];
    let first = Imputer::new_with_domains(ordered_columns(), ImputationSpec::new(), ImputationUnits::Rows, &first_domains).unwrap();
    let second = Imputer::new_with_domains(ordered_columns(), ImputationSpec::new(), ImputationUnits::Rows, &second_domains).unwrap();
    assert_eq!(multiple_imputation::ImputationAssumptions::of(&first, None).digest(),
        multiple_imputation::ImputationAssumptions::of(&second, None).digest());
    let columns = || vec![ImputationColumn::new("x", vec![0.0, 1.0, 0.0, 1.0, f64::NAN])];
    let inferred = Imputer::new(columns(), ImputationSpec::new(), ImputationUnits::Rows).unwrap();
    assert_eq!(inferred.variables()[0].kind, VariableKind::Binary);
    let empty = Imputer::new_with_domains(columns(), ImputationSpec::new(), ImputationUnits::Rows, &[]).unwrap();
    assert_eq!(multiple_imputation::ImputationAssumptions::of(&inferred, None).digest(),
        multiple_imputation::ImputationAssumptions::of(&empty, None).digest());
    for domains in [vec![(Arc::from("missing"), ImputationDomain::Binary)],
        vec![(Arc::from("x"), ImputationDomain::Binary), (Arc::from("x"), ImputationDomain::Binary)]] {
        let error = Imputer::new_with_domains(columns(), ImputationSpec::new(), ImputationUnits::Rows, &domains).unwrap_err();
        assert_eq!(error.code(), "invalid_argument");
        assert_eq!(error.detail(), "multiple_imputation.variable_domain");
    }
    let bad = vec![ImputationColumn::new("x", vec![0.0, 0.0, 2.0, 0.0, f64::NAN])];
    let error = Imputer::new_with_domains(bad, ImputationSpec::new(), ImputationUnits::Rows,
        &[(Arc::from("x"), ImputationDomain::Binary)]).unwrap_err();
    assert_eq!(error.code(), "invalid_argument");
    assert_eq!(error.detail(), "multiple_imputation.variable_domain");
}

#[test]
fn lagged_binary_boundary_uses_fit_rows_not_excluded_endpoints() {
    let mut values = vec![0.0; 12];
    values[0] = 1.0; // The opposite class is only at an excluded lag boundary.
    values[6] = f64::NAN;
    let model = Imputer::new(
        vec![ImputationColumn::new("x", values)], ImputationSpec::new(), ImputationUnits::Rows,
    ).unwrap().with_dynamics(1).unwrap();
    assert_eq!(model.binary_boundary_variables().as_slice(), &[(Arc::<str>::from("x"), 0)]);
    let table = model.impute(0, &ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(table.column("x").unwrap()[6], 0.0);
}
