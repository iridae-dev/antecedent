//! Joint pooling of vector-valued fits, the pooled Wald statistic, stacked joint draws, and the
//! assumptions record every impute-fit-pool run carries.
//!
//! References are computed in this file from the stated points and covariances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, DeclaredSamplingDesign, ExecutionContext};
use antecedent_data::{ImputationModel, ImputationSpec, MissingnessAssumption};
use antecedent_estimate::multiple_imputation::{
    self, ImputationColumn, ImputationFit, ImputationUnits, Imputer, PooledEstimate,
    PooledVectorEstimate, VectorImputationFit, pool_vector_fits, rubin_pool, rubin_pool_vector,
    stack_posterior_quantities,
};
use antecedent_kernels::standard_normal;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn vector_pooling_matches_a_hand_computation_and_the_scalar_pooling_per_coordinate() {
    // m = 3, p = 2.
    let points = vec![vec![1.0, 0.5], vec![2.0, 0.1], vec![4.0, 0.3]];
    let covariances =
        vec![vec![0.5, 0.1, 0.1, 0.2], vec![0.6, 0.0, 0.0, 0.25], vec![0.7, 0.2, 0.2, 0.3]];
    let pooled = rubin_pool_vector(&points, &covariances, Some(30.0)).unwrap();
    let q0 = (1.0 + 2.0 + 4.0) / 3.0;
    let q1 = (0.5 + 0.1 + 0.3) / 3.0;
    assert!(close(pooled.point[0], q0, 1e-12) && close(pooled.point[1], q1, 1e-12));
    let u = [[0.6, 0.1], [0.1, 0.25]];
    let dev = |j: usize| [points[j][0] - q0, points[j][1] - q1];
    let mut b = [[0.0; 2]; 2];
    for j in 0..3 {
        for r in 0..2 {
            for c in 0..2 {
                b[r][c] += dev(j)[r] * dev(j)[c] / 2.0;
            }
        }
    }
    for r in 0..2 {
        for c in 0..2 {
            assert!(close(pooled.within[r * 2 + c], u[r][c], 1e-12), "within {r}{c}");
            assert!(close(pooled.between[r * 2 + c], b[r][c], 1e-12), "between {r}{c}");
            let total = u[r][c] + (1.0 + 1.0 / 3.0) * b[r][c];
            assert!(close(pooled.total[r * 2 + c], total, 1e-12), "total {r}{c}");
        }
    }
    for (i, coordinate) in pooled.coordinates.iter().enumerate() {
        let column: Vec<f64> = points.iter().map(|q| q[i]).collect();
        let variances: Vec<f64> = covariances.iter().map(|c| c[i * 2 + i]).collect();
        assert_eq!(coordinate, &rubin_pool(&column, &variances, Some(30.0)).unwrap());
    }
    assert!(close(pooled.standard_error(0).unwrap(), pooled.total[0].sqrt(), 1e-12));
    assert!(pooled.standard_error(2).is_none());

    // r-bar = (1 + 1/m) tr(B U^-1) / p, with the 2 x 2 inverse written out.
    let det = u[0][0] * u[1][1] - u[0][1] * u[1][0];
    let inv = [[u[1][1] / det, -u[0][1] / det], [-u[1][0] / det, u[0][0] / det]];
    let mut trace = 0.0;
    for r in 0..2 {
        for k in 0..2 {
            trace += b[r][k] * inv[k][r];
        }
    }
    let r_bar = (1.0 + 1.0 / 3.0) * trace / 2.0;
    assert!(close(pooled.average_relative_increase, r_bar, 1e-12));

    let null = [1.5, 0.0];
    let wald = pooled.wald(&null).unwrap();
    let gap = [q0 - null[0], q1 - null[1]];
    let quadratic = gap[0] * (inv[0][0] * gap[0] + inv[0][1] * gap[1])
        + gap[1] * (inv[1][0] * gap[0] + inv[1][1] * gap[1]);
    assert!(close(wald.statistic, quadratic / (2.0 * (1.0 + r_bar)), 1e-12));
    assert_eq!(wald.df_numerator, 2.0);
    // t = p (m - 1) = 4, not above 4, so the small-sample denominator applies.
    let t = 4.0_f64;
    let nu2 = t * (1.0 + 1.0 / 2.0) * (1.0 + 1.0 / r_bar).powi(2) / 2.0;
    assert!(close(wald.df_denominator, nu2, 1e-12));
    assert_eq!(pooled.wald(&pooled.point).unwrap().statistic, 0.0);
    assert!(pooled.wald(&[0.0]).is_err());
}

#[test]
fn one_coordinate_vector_pooling_is_the_scalar_pooling() {
    let q = [1.0, 2.0, 4.0, 3.5];
    let u = [0.5, 0.6, 0.7, 0.4];
    let scalar = rubin_pool(&q, &u, None).unwrap();
    let vector = rubin_pool_vector(
        &q.iter().map(|v| vec![*v]).collect::<Vec<_>>(),
        &u.iter().map(|v| vec![*v]).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    assert!(close(vector.total[0], scalar.total, 1e-14));
    assert!(close(vector.average_relative_increase, scalar.relative_increase, 1e-12));
    let wald = vector.wald(&[0.0]).unwrap();
    let expected = (scalar.point * scalar.point / scalar.within) / (1.0 + scalar.relative_increase);
    assert!(close(wald.statistic, expected, 1e-12));
}

#[test]
fn vector_pooling_refuses_malformed_fits() {
    let ok = vec![vec![1.0, 2.0], vec![1.5, 2.5]];
    let eye = vec![vec![1.0, 0.0, 0.0, 1.0]; 2];
    assert!(rubin_pool_vector(&ok, &eye, None).is_ok());
    assert!(rubin_pool_vector(&ok[..1], &eye[..1], None).is_err(), "one imputation");
    assert!(rubin_pool_vector(&ok, &eye[..1], None).is_err(), "covariance count");
    let ragged = vec![vec![1.0, 2.0], vec![1.5]];
    assert!(rubin_pool_vector(&ragged, &eye, None).is_err());
    let wrong_size = vec![vec![1.0, 0.0, 0.0, 1.0], vec![1.0]];
    assert!(rubin_pool_vector(&ok, &wrong_size, None).is_err());
    let asymmetric = vec![vec![1.0, 0.4, 0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0]];
    assert!(rubin_pool_vector(&ok, &asymmetric, None).is_err());
    let negative = vec![vec![-1.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0]];
    assert!(rubin_pool_vector(&ok, &negative, None).is_err());
    let nan = vec![vec![1.0, f64::NAN], vec![1.5, 2.5]];
    assert!(rubin_pool_vector(&nan, &eye, None).is_err());
    assert!(rubin_pool_vector(&ok, &eye, Some(0.0)).is_err());
    // A singular within covariance pools but cannot form the joint statistic.
    let singular = vec![vec![1.0, 1.0, 1.0, 1.0]; 2];
    let pooled = rubin_pool_vector(&ok, &singular, None).unwrap();
    assert!(pooled.average_relative_increase.is_infinite());
    assert!(pooled.wald(&[0.0, 0.0]).is_err());
}

#[test]
fn stacked_quantities_keep_the_joint_dependence_between_coordinates() {
    // Coordinate 1 is exactly twice coordinate 0 within every imputation: correlation 1.
    let mut rng = CausalRng::from_seed(5);
    let per_imputation: Vec<Vec<Vec<f64>>> = (0..4)
        .map(|j| {
            let shift = f64::from(j);
            let first: Vec<f64> =
                (0..50).map(|_| shift + 0.5 * standard_normal(&mut rng)).collect();
            let second = first.iter().map(|v| 2.0 * v).collect();
            vec![first, second]
        })
        .collect();
    let stacked = stack_posterior_quantities(&per_imputation).unwrap();
    assert_eq!(stacked.quantities.len(), 2);
    assert_eq!(stacked.quantities[0].imputations, 4);
    let cov = &stacked.covariance;
    assert!(close(cov[1], 2.0 * cov[0], 1e-10), "cov(0,1) = 2 var(0)");
    assert!(close(cov[3], 4.0 * cov[0], 1e-10), "var(1) = 4 var(0)");
    assert!(close(cov[0], stacked.quantities[0].variance, 1e-12));
    assert!(stacked.quantities[0].between > 0.0);
    // Imputations that report different numbers of quantities are not stacked.
    let mut ragged = per_imputation.clone();
    ragged[1].pop();
    assert!(stack_posterior_quantities(&ragged).is_err());
    assert!(stack_posterior_quantities(&[]).is_err());
    // Fits that mix covariances and draws are refused.
    let mixed = vec![
        VectorImputationFit::frequentist(vec![1.0], vec![1.0], None),
        VectorImputationFit::bayesian(vec![vec![1.0, 2.0, 3.0]]),
    ];
    assert!(pool_vector_fits(&mixed).is_err());
    assert!(pool_vector_fits(&[]).is_err());
}

fn simulate(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let xi = 0.6 * zi + 0.8 * standard_normal(&mut rng);
        let yi = 1.0 + 2.0 * xi + 1.5 * zi + standard_normal(&mut rng);
        let y_obs =
            if rng.next_f64() < 1.0 / (1.0 + (-(2.0 * zi - 0.5)).exp()) { f64::NAN } else { yi };
        x.push(if rng.next_f64() < 0.1 { f64::NAN } else { xi });
        y.push(y_obs);
        z.push(zi);
    }
    (x, y, z)
}

/// Intercept and slope of `y` on `x` with their covariance and `n - 2`.
fn ols(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
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
    let s2 = rows.iter().map(|r| (r.1 - intercept - slope * r.0).powi(2)).sum::<f64>() / (n - 2.0);
    let cov = vec![s2 * (1.0 / n + mx * mx / sxx), -s2 * mx / sxx, -s2 * mx / sxx, s2 / sxx];
    (vec![intercept, slope], cov, n - 2.0)
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

#[test]
fn a_vector_run_pools_the_coordinates_jointly_and_agrees_with_the_scalar_run() {
    let ctx = ExecutionContext::for_tests(1);
    let (x, y, z) = simulate(400, 77);
    let spec = ImputationSpec::new().with_imputations(8).with_iterations(5).with_seed(3);
    let imp = imputer(&x, &y, &z, spec);
    let vector = multiple_imputation::run_vector(&imp, &ctx, |table, _| {
        let (beta, cov, df) = ols(&table.columns()[0], &table.columns()[1]);
        Ok::<_, ()>(VectorImputationFit::frequentist(beta, cov, Some(df)))
    })
    .unwrap();
    let scalar = multiple_imputation::run(&imp, &ctx, |table, _| {
        let (beta, cov, df) = ols(&table.columns()[0], &table.columns()[1]);
        Ok::<_, ()>(ImputationFit::frequentist(beta[1], cov[3], Some(df)))
    })
    .unwrap();
    let PooledVectorEstimate::Rubin(joint) = &vector.pooled else { panic!("Rubin pooling") };
    let PooledEstimate::Rubin(single) = &scalar.pooled else { panic!("Rubin pooling") };
    assert_eq!(joint.coordinates[1], *single, "the slope coordinate is the scalar pooling");
    assert!(joint.total[3] >= joint.within[3] && joint.between[3] > 0.0);
    assert!((joint.point[1] - 2.9).abs() < 0.5, "slope {}", joint.point[1]);
    assert_eq!(vector.fits.len(), 8);
    let wald = joint.wald(&[1.0, 2.9]).unwrap();
    assert!(wald.statistic.is_finite() && wald.statistic >= 0.0);
    assert!(wald.df_denominator > 0.0);
    assert_eq!(joint.wald(&joint.point).unwrap().statistic, 0.0);
    assert_eq!(vector.pooled.total_covariance(), joint.total.as_slice());
    assert_eq!(vector.assumptions, scalar.assumptions, "both runs rest on the same assumptions");
}

#[test]
fn a_bayesian_vector_run_stacks_joint_draws() {
    let ctx = ExecutionContext::for_tests(1);
    let (x, y, z) = simulate(300, 8);
    let imp = imputer(&x, &y, &z, ImputationSpec::new().with_imputations(5).with_seed(2));
    let run = multiple_imputation::run_vector(&imp, &ctx, |table, _| {
        let (beta, cov, _) = ols(&table.columns()[0], &table.columns()[1]);
        let mut rng = CausalRng::from_seed(u64::from(table.index()));
        // Joint normal draws through the Cholesky factor of the 2 x 2 covariance.
        let l00 = cov[0].sqrt();
        let l10 = cov[2] / l00;
        let l11 = (cov[3] - l10 * l10).sqrt();
        let mut intercept = Vec::new();
        let mut slope = Vec::new();
        for _ in 0..300 {
            let (a, b) = (standard_normal(&mut rng), standard_normal(&mut rng));
            intercept.push(beta[0] + l00 * a);
            slope.push(beta[1] + l10 * a + l11 * b);
        }
        Ok::<_, ()>(VectorImputationFit::bayesian(vec![intercept, slope]))
    })
    .unwrap();
    let PooledVectorEstimate::Stacked(stacked) = &run.pooled else { panic!("stacked") };
    assert_eq!(stacked.quantities.len(), 2);
    assert_eq!(stacked.quantities[1].draws.len(), 1_500);
    assert!(stacked.covariance[1] < 0.0, "intercept and slope draws are negatively dependent");
    assert!(stacked.quantities[1].variance > stacked.quantities[1].within);
    assert_eq!(run.pooled.point().len(), 2);
}

#[test]
fn the_assumptions_record_binds_model_design_unit_and_every_variable() {
    let ctx = ExecutionContext::for_tests(1);
    let (x, y, z) = simulate(300, 11);
    let spec = ImputationSpec::new().with_imputations(4).with_iterations(3).with_seed(9);
    let imp = imputer(&x, &y, &z, spec);
    let run = |imp: &Imputer| {
        multiple_imputation::run(imp, &ctx, |table, _| {
            let (beta, cov, df) = ols(&table.columns()[0], &table.columns()[1]);
            Ok::<_, ()>(ImputationFit::frequentist(beta[1], cov[3], Some(df)))
        })
        .unwrap()
    };
    let base = run(&imp);
    let record = &base.assumptions;
    assert_eq!(record.spec, spec);
    assert_eq!((record.unit, record.units, record.rows), ("row", 300, 300));
    assert_eq!(record.variables.len(), 3);
    let missing: Vec<usize> = record.variables.iter().map(|v| v.missing).collect();
    assert!(missing[0] > 0 && missing[1] > 0 && missing[2] == 0, "{missing:?}");
    assert!(record.valid_under_declared_assumption());
    assert_eq!(record.design, None);
    assert_eq!(base.assumptions_digest(), run(&imp).assumptions_digest(), "replay");

    // Each assumption moves the digest.
    let digest = base.assumptions_digest();
    let with = |spec: ImputationSpec| run(&imputer(&x, &y, &z, spec)).assumptions_digest();
    assert_ne!(digest, with(spec.with_seed(10)));
    assert_ne!(digest, with(spec.with_imputations(5)));
    assert_ne!(digest, with(spec.with_iterations(4)));
    assert_ne!(digest, with(spec.with_model(ImputationModel::JointNormal)));
    let designed =
        run(&imp).with_design(Some(DeclaredSamplingDesign::IndependentSubjectTrajectories));
    assert_ne!(digest, designed.assumptions_digest());
    assert_eq!(
        designed.assumptions.design,
        Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)
    );
    let mut x_more = x.clone();
    x_more[0] = f64::NAN;
    x_more[1] = f64::NAN;
    let more = imputer(&x_more, &y, &z, spec);
    assert_ne!(digest, run(&more).assumptions_digest(), "imputed-cell counts bind");

    // Not-at-random is refused before any table is drawn, with the sensitivity analysis next.
    let refused = multiple_imputation::check_spec(
        &spec.with_assumption(MissingnessAssumption::MissingNotAtRandom),
    )
    .unwrap_err();
    assert!(refused.next_attempt().is_some());
}
