//! Ridge-logistic propensity for cross-fitted AIPW (2.2 E2): the route fits and replays, tunes
//! its penalty on training rows only, keeps the score table, row identity, overlap report and
//! retargeting, publishes no interval, and refuses lasso, an unlicensed fallback, an interval
//! request, an invalid penalty and a design the outcome stage cannot fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals and build dense designs element by element"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, EffectEstimate, EstimationError, NuisanceFallback,
    OverlapPolicy, PropensityNuisance, RidgeTuning, ScoreTable, build_binary_scores,
    provenance_withholds_interval, retarget,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;
use antecedent_stats::{FaerBackend, GlmOptions, StatsError};

/// Raw columns of one synthetic data set.
#[derive(Clone)]
struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
enum Regime {
    /// Moderate overlap: `P(T = 1 | z) = expit(0.8 z0 - 0.5 z1)`.
    Moderate,
    /// `T = 1{z0 > 0}`: complete separation, so the unpenalized logistic has no MLE.
    Separated,
    /// Moderate overlap, with the last covariate an exact copy of the second.
    Duplicated,
}

/// `Y = 2 T + z0 + 0.5 z1 + noise`: the true average effect is 2.
fn draw(n: usize, p: usize, seed: u64, regime: Regime) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE2);
    let mut z = vec![vec![0.0; n]; p];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        for column in &mut z {
            column[i] = standard_normal(&mut rng);
        }
        t[i] = match regime {
            Regime::Separated => f64::from(u8::from(z[0][i] > 0.0)),
            Regime::Moderate | Regime::Duplicated => {
                let eta = 0.8 * z[0][i] - 0.5 * z[1][i];
                f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())))
            }
        };
        y[i] = 2.0 * t[i] + z[0][i] + 0.5 * z[1][i] + 0.5 * standard_normal(&mut rng);
    }
    if matches!(regime, Regime::Duplicated) {
        let duplicate = z[1].clone();
        z[p - 1].clone_from(&duplicate);
    }
    Raw { t, y, z }
}

fn build(raw: &Raw) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = raw.t.len();
    let p = raw.z.len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..p + 2 {
        let (name, hint) = match index {
            0 => ("t".to_string(), RoleHint::TreatmentCandidate),
            1 => ("y".to_string(), RoleHint::OutcomeCandidate),
            j => (format!("z{}", j - 2), RoleHint::Context),
        };
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut values = vec![raw.t.clone(), raw.y.clone()];
    values.extend(raw.z.iter().cloned());
    let columns = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..p).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

fn ridge(grid: &[f64], inner_folds: usize) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        propensity: PropensityNuisance::ridge_logistic(
            RidgeTuning::new(grid, inner_folds).unwrap(),
        ),
        ..AipwAte::new()
    }
}

fn plain() -> AipwAte {
    AipwAte { bootstrap_replicates: 0, ..AipwAte::new() }
}

fn run(est: &AipwAte, raw: &Raw, seed: u64) -> Result<EffectEstimate, EstimationError> {
    let (data, estimand, query) = build(raw);
    let problem = est.prepare(&data, &estimand, &query)?;
    est.fit(
        &problem,
        &mut AipwWorkspace::default(),
        &ExecutionContext::for_tests(seed),
        AssumptionSet::new(),
    )
}

fn selected_lambdas(estimate: &EffectEstimate) -> Vec<f64> {
    estimate
        .learner_provenance
        .iter()
        .map(|p| {
            let bits = p.spec.strip_prefix("logistic:").expect("a penalized fold's spec");
            f64::from_bits(bits.parse::<u64>().unwrap())
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

// ---- independent score calculation (written here, not taken from the crate) ----------------

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let pivot = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, pivot);
        b.swap(c, pivot);
        let pivot_row = a[c].clone();
        for r in c + 1..n {
            let f = a[r][c] / pivot_row[c];
            for k in c..n {
                a[r][k] -= f * pivot_row[k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s -= a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }
    x
}

/// Newton's method on `-loglik + (lambda / 2) * sum_{j >= 1} beta_j^2`; rows are `[1, z...]`.
fn ridge_logistic(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let p = x[0].len();
    let mut beta = vec![0.0; p];
    for _ in 0..200 {
        let mut g = vec![0.0; p];
        let mut h = vec![vec![0.0; p]; p];
        for (row, &yi) in x.iter().zip(y) {
            let eta: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let mu = expit(eta);
            let w = mu * (1.0 - mu);
            for i in 0..p {
                g[i] += row[i] * (yi - mu);
                for j in 0..p {
                    h[i][j] += w * row[i] * row[j];
                }
            }
        }
        for j in 1..p {
            g[j] -= lambda * beta[j];
            h[j][j] += lambda;
        }
        let step = solve(h, g);
        let mut largest = 0.0_f64;
        for j in 0..p {
            beta[j] += step[j];
            largest = largest.max(step[j].abs());
        }
        if largest < 1e-13 {
            break;
        }
    }
    beta
}

fn ols(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let p = x[0].len();
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..p {
            xty[i] += row[i] * yi;
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    solve(xtx, xty)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Out-of-fold AIPW scores recomputed from the raw data, the table's fold ids and the
/// penalties it selected: standardize on the training rows, fit the ridge logistic and the
/// two arm OLS fits on the training rows, score the held-out rows.
fn independent_scores(
    raw: &Raw,
    fold_ids: &[u32],
    n_folds: usize,
    lambdas: &[f64],
    clip: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = raw.t.len();
    let p = raw.z.len();
    let row = |i: usize| -> Vec<f64> {
        let mut r = vec![1.0];
        r.extend(raw.z.iter().map(|c| c[i]));
        r
    };
    let mut s0 = vec![0.0; n];
    let mut s1 = vec![0.0; n];
    for k in 0..n_folds {
        let train: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize != k).collect();
        let valid: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize == k).collect();
        let mut mean = vec![0.0; p + 1];
        let mut sd = vec![1.0; p + 1];
        for j in 1..=p {
            let col: Vec<f64> = train.iter().map(|&i| raw.z[j - 1][i]).collect();
            let m = col.iter().sum::<f64>() / col.len() as f64;
            let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / col.len() as f64;
            mean[j] = m;
            sd[j] = v.sqrt();
        }
        let standardized = |i: usize| -> Vec<f64> {
            let mut r = row(i);
            for j in 1..=p {
                r[j] = (r[j] - mean[j]) / sd[j];
            }
            r
        };
        let xs: Vec<Vec<f64>> = train.iter().map(|&i| standardized(i)).collect();
        let ts: Vec<f64> = train.iter().map(|&i| raw.t[i]).collect();
        let gamma = ridge_logistic(&xs, &ts, lambdas[k]);
        let arm_fit = |treated: bool| -> Vec<f64> {
            let rows: Vec<usize> =
                train.iter().copied().filter(|&i| (raw.t[i] > 0.5) == treated).collect();
            let x: Vec<Vec<f64>> = rows.iter().map(|&i| row(i)).collect();
            let y: Vec<f64> = rows.iter().map(|&i| raw.y[i]).collect();
            ols(&x, &y)
        };
        let (b0, b1) = (arm_fit(false), arm_fit(true));
        for &i in &valid {
            let e = expit(dot(&standardized(i), &gamma)).clamp(1e-9, 1.0 - 1e-9);
            let e = e.clamp(clip, 1.0 - clip);
            let r = row(i);
            let (m0, m1) = (dot(&r, &b0), dot(&r, &b1));
            let (t, y) = (raw.t[i], raw.y[i]);
            s0[i] = m0 + (1.0 - t) / (1.0 - e) * (y - m0);
            s1[i] = m1 + t / e * (y - m1);
        }
    }
    (s0, s1)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Independent weighted means and covariance of the two arm score columns: the point is
/// `sum w phi / sum w`, the covariance `n / (n - 1) * sum (w_i / sum w)^2 (phi - m)(phi' - m')`.
fn weighted_summary(s0: &[f64], s1: &[f64], w: &[f64]) -> ([f64; 2], [[f64; 2]; 2]) {
    let n = s0.len() as f64;
    let total: f64 = w.iter().sum();
    let m = [
        s0.iter().zip(w).map(|(s, wi)| s * wi).sum::<f64>() / total,
        s1.iter().zip(w).map(|(s, wi)| s * wi).sum::<f64>() / total,
    ];
    let cols = [s0, s1];
    let mut cov = [[0.0; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            let acc: f64 = (0..s0.len())
                .map(|i| (w[i] / total).powi(2) * (cols[a][i] - m[a]) * (cols[b][i] - m[b]))
                .sum();
            cov[a][b] = n / (n - 1.0) * acc;
        }
    }
    (m, cov)
}

fn table(estimate: &EffectEstimate) -> &ScoreTable {
    estimate.score_table.as_ref().expect("the penalized route keeps its score table")
}

// ---- tests ---------------------------------------------------------------------------------

/// Positive known truth under high-dimensional separation: the unpenalized propensity has no
/// MLE (a typed refusal), the ridge propensity does, and its AIPW recovers the effect 2.
#[test]
fn ridge_propensity_recovers_the_effect_where_the_unpenalized_fit_is_refused() {
    let raw = draw(400, 8, 31, Regime::Separated);
    let refused = run(&plain(), &raw, 7).unwrap_err();
    assert!(matches!(refused, EstimationError::Stats(_)), "{refused}");
    let estimate = run(&ridge(&[1.0, 10.0, 100.0, 1000.0], 4), &raw, 7).unwrap();
    assert!((estimate.ate - 2.0).abs() < 0.4, "ate = {}", estimate.ate);
    assert!(selected_lambdas(&estimate).iter().all(|l| [1.0, 10.0, 100.0, 1000.0].contains(l)));
}

/// What the route publishes: a point, the score table and its identity, the overlap report,
/// the fold count and seed, and nothing from which an interval could be built.
#[test]
fn a_penalized_fit_publishes_scores_and_a_point_but_no_interval() {
    let raw = draw(300, 4, 5, Regime::Moderate);
    let est = ridge(&[0.5, 5.0, 50.0], 4);
    let (data, estimand, query) = build(&raw);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let estimate = est
        .fit(
            &problem,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(11),
            AssumptionSet::new(),
        )
        .unwrap();
    assert!(estimate.ate.is_finite());
    assert!(estimate.se_analytic.is_nan());
    assert!(estimate.se_bootstrap.is_none());
    assert!(estimate.joint_covariance.is_none());
    assert!(estimate.score_inference.is_none());
    assert!(estimate.influence.is_none());
    assert!(estimate.overlap_report.is_some());
    assert_eq!(estimate.crossfit_folds, Some(5));
    assert_eq!(estimate.crossfit_seed, Some(11));
    assert_eq!(estimate.learner_provenance.len(), 5);
    let t = table(&estimate);
    assert_eq!(t.row_index.as_ref(), problem.row_index.as_ref());
    assert_eq!(t.n_rows, 300);
    assert_eq!(t.n_folds, 5);
    assert!(provenance_withholds_interval(&t.nuisance_provenance));
}

/// Seeded replay: the same seed gives the same folds, penalties and scores bit for bit; another
/// seed gives another fold plan.
#[test]
fn selected_penalties_replay_from_the_seed() {
    let raw = draw(300, 4, 9, Regime::Moderate);
    let est = ridge(&[0.5, 5.0, 50.0], 4);
    let a = run(&est, &raw, 21).unwrap();
    let b = run(&est, &raw, 21).unwrap();
    let c = run(&est, &raw, 22).unwrap();
    assert_eq!(a.learner_provenance, b.learner_provenance);
    assert_eq!(bits(&table(&a).scores), bits(&table(&b).scores));
    assert_eq!(table(&a).fold_ids, table(&b).fold_ids);
    assert_eq!(a.ate.to_bits(), b.ate.to_bits());
    assert_ne!(table(&a).fold_ids, table(&c).fold_ids);
}

/// Tuning uses training rows only: with the fold plan held fixed, rewriting every row of fold 0
/// (covariates, treatment and outcome) cannot move the penalty chosen for fold 0, whose
/// training rows are the other folds; the held-out scores do move.
#[test]
fn penalty_is_chosen_on_training_rows_only() {
    let raw = draw(250, 4, 13, Regime::Moderate);
    let n = raw.t.len();
    let ids: Vec<u32> = (0..n).map(|i| u32::try_from(i % 5).unwrap()).collect();
    let mut changed = raw.clone();
    for i in (0..n).filter(|i| i % 5 == 0) {
        changed.t[i] = 1.0 - raw.t[i];
        changed.y[i] += 50.0;
        for column in &mut changed.z {
            column[i] += 3.0;
        }
    }
    let est = ridge(&[0.05, 0.5, 5.0, 50.0], 4);
    let fit_with_fixed_folds = |data: &Raw| {
        let (frame, estimand, query) = build(data);
        let mut problem = est.prepare(&frame, &estimand, &query).unwrap();
        problem.fold_assignment = Some(Arc::from(ids.clone()));
        est.fit(
            &problem,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(3),
            AssumptionSet::new(),
        )
        .unwrap()
    };
    let before = fit_with_fixed_folds(&raw);
    let after = fit_with_fixed_folds(&changed);
    assert_eq!(before.learner_provenance[0], after.learner_provenance[0]);
    assert_ne!(bits(&table(&before).scores), bits(&table(&after).scores));
}

/// The point is the mean of the stored scores, and the scores, the retargeted point and the
/// retargeted covariance equal a calculation written here from the raw data, the table's fold
/// ids and the penalties it selected.
#[test]
fn scores_match_an_independent_calculation_and_retarget_exactly() {
    let raw = draw(400, 5, 17, Regime::Moderate);
    let est = ridge(&[0.5, 5.0, 50.0], 4);
    let estimate = run(&est, &raw, 29).unwrap();
    let t = table(&estimate);
    let (c0, c1) = (t.column(0).unwrap(), t.column(1).unwrap());

    // Score-mean identity.
    let contrast: Vec<f64> = c0.iter().zip(c1).map(|(a, b)| b - a).collect();
    assert!((estimate.ate - mean(&contrast)).abs() < 1e-12);

    // Independent out-of-fold scores.
    let lambdas = selected_lambdas(&estimate);
    let (s0, s1) = independent_scores(&raw, &t.fold_ids, 5, &lambdas, 0.01);
    for i in 0..raw.t.len() {
        assert!((c0[i] - s0[i]).abs() < 1e-5, "arm 0 row {i}: {} vs {}", c0[i], s0[i]);
        assert!((c1[i] - s1[i]).abs() < 1e-5, "arm 1 row {i}: {} vs {}", c1[i], s1[i]);
    }
    let independent_ate = mean(&s1) - mean(&s0);
    assert!((estimate.ate - independent_ate).abs() < 1e-5);

    // Retargeted point and covariance, from the table and from the independent scores.
    let weights: Vec<f64> = raw.z[0].iter().map(|z| (0.3 * z).exp()).collect();
    let summary = t.summarize(Some(&weights)).unwrap();
    let (m, cov) = weighted_summary(c0, c1, &weights);
    let (m_ind, cov_ind) = weighted_summary(&s0, &s1, &weights);
    for a in 0..2 {
        assert!((summary.means[a] - m[a]).abs() < 1e-12);
        assert!((m[a] - m_ind[a]).abs() < 1e-5);
        for b in 0..2 {
            let got = summary.covariance.get(a, b);
            assert!((got - cov[a][b]).abs() <= 1e-12 * (1.0 + got.abs()), "cov {a}{b}");
            assert!((cov[a][b] - cov_ind[a][b]).abs() < 1e-5, "independent cov {a}{b}");
        }
    }

    // The table retargets through the shared retarget routine: constant relative weights
    // leave the target unchanged and reproduce the point and the unweighted covariance.
    let ones = vec![1.0; raw.t.len()];
    let (result, overlap_failed) = retarget(t, &ones, &[], None, None, None).unwrap();
    assert!(!overlap_failed);
    let (m_one, cov_one) = weighted_summary(c0, c1, &ones);
    let value = result.contrast.as_ref().expect("a mean contrast").value;
    assert!((value - estimate.ate).abs() < 1e-12);
    assert!(
        (result.summary.means[1] - result.summary.means[0] - (m_one[1] - m_one[0])).abs() < 1e-12
    );
    assert!((result.covariance.get(0, 1) - cov_one[0][1]).abs() < 1e-12);
}

/// A requested interval is refused with its reason code, never ignored: a bootstrap, a
/// non-default analytic kind, and a direct bootstrap attach on a point.
#[test]
fn an_interval_request_is_refused_with_its_reason_code() {
    let raw = draw(200, 3, 3, Regime::Moderate);
    let (data, estimand, query) = build(&raw);
    for requested in [
        AipwAte { bootstrap_replicates: 50, ..ridge(&[1.0], 3) },
        AipwAte { se_kind: AnalyticSeKind::Hc1, ..ridge(&[1.0], 3) },
    ] {
        let error = requested.prepare(&data, &estimand, &query).unwrap_err();
        assert!(error.to_string().contains("penalized_interval_not_licensed"), "{error}");
    }
    let est = ridge(&[1.0, 10.0], 3);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut workspace = AipwWorkspace::default();
    let point = est.fit(&problem, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
    let boot = AipwAte { bootstrap_replicates: 5, ..est };
    let error = boot.attach_bootstrap(&problem, &mut workspace, &ctx, point).unwrap_err();
    assert!(error.to_string().contains("penalized_interval_not_licensed"), "{error}");
}

/// Outside the untrimmed `AllObserved` mean ATE the penalty has no route.
#[test]
fn a_scope_outside_the_license_is_refused() {
    let raw = draw(200, 3, 4, Regime::Moderate);
    let (data, estimand, query) = build(&raw);
    let trimmed = AipwAte {
        overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) },
        ..ridge(&[1.0], 3)
    };
    let error = trimmed.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("route_not_supported"), "{error}");
}

/// Lasso is closed: refused at preparation, and again by the score-table builder itself.
#[test]
fn lasso_is_closed_with_a_typed_refusal() {
    let raw = draw(200, 3, 6, Regime::Moderate);
    let (data, estimand, query) = build(&raw);
    let lasso = AipwAte { propensity: PropensityNuisance::lasso(), ..plain() };
    let error = lasso.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("selection_inference_not_licensed"), "{error}");

    let mut problem = plain().prepare(&data, &estimand, &query).unwrap();
    problem.propensity = PropensityNuisance::lasso();
    let error = build_binary_scores(
        &problem,
        problem.treatment_id,
        &[None],
        5,
        &GlmOptions::default(),
        FaerBackend,
    )
    .unwrap_err();
    assert!(error.to_string().contains("selection_inference_not_licensed"), "{error}");
}

/// A declared ML fallback is closed: a failed GLM fit is refused with the failure recorded,
/// a successful one is the plain GLM result bit for bit, and nothing is silently replaced.
#[test]
fn a_failed_fit_under_a_declared_fallback_is_recorded_and_never_replaced() {
    let fallback = AipwAte {
        propensity: PropensityNuisance::default().with_fallback(NuisanceFallback::Ml),
        ..plain()
    };
    let separated = draw(300, 3, 8, Regime::Separated);
    assert!(matches!(run(&plain(), &separated, 2).unwrap_err(), EstimationError::Stats(_)));
    let error = run(&fallback, &separated, 2).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("nuisance_fallback_not_licensed"), "{message}");
    assert!(message.contains("primary nuisance fit failed"), "{message}");

    let healthy = draw(300, 3, 8, Regime::Moderate);
    let armed = run(&fallback, &healthy, 2).unwrap();
    let glm = run(&plain(), &healthy, 2).unwrap();
    assert_eq!(armed.ate.to_bits(), glm.ate.to_bits());
    assert_eq!(bits(&table(&armed).scores), bits(&table(&glm).scores));
}

/// An exact duplicate column is a rank deficiency: the unpenalized route and the penalized one
/// (whose propensity copes, but whose arm OLS does not) both refuse it with a typed error.
#[test]
fn a_duplicated_column_is_refused_with_a_typed_rank_error() {
    let raw = draw(300, 4, 12, Regime::Duplicated);
    for est in [plain(), ridge(&[1.0, 10.0], 3)] {
        let error = run(&est, &raw, 5).unwrap_err();
        assert!(
            matches!(error, EstimationError::Stats(StatsError::RankDeficient { .. })),
            "{error}"
        );
    }
}

/// Cancellation during penalty selection is a stop with its reason code, never a verdict.
#[test]
fn a_cancelled_fit_is_a_stop_never_a_verdict() {
    let raw = draw(200, 3, 14, Regime::Moderate);
    let est = ridge(&[1.0, 10.0], 3);
    let (data, estimand, query) = build(&raw);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let error =
        est.fit(&problem, &mut AipwWorkspace::default(), &ctx, AssumptionSet::new()).unwrap_err();
    assert!(error.to_string().contains("cancelled_no_claim"), "{error}");
}

/// The score table is an artifact: its wire form round-trips, and its provenance carries the
/// canonical penalty configuration and the penalties selected per fold, so a different grid
/// never shares a score-reuse identity.
#[test]
fn a_penalized_score_table_round_trips_with_its_penalty_identity() {
    let raw = draw(300, 4, 19, Regime::Moderate);
    let narrow = ridge(&[1.0, 10.0], 3);
    let wide = ridge(&[1.0, 10.0, 100.0], 3);
    let a = run(&narrow, &raw, 8).unwrap();
    let b = run(&wide, &raw, 8).unwrap();
    let t = table(&a);
    let wire = t.to_wire();
    assert_eq!(&ScoreTable::from_wire(wire.clone()).unwrap(), t);
    assert!(wire.nuisance_provenance.contains(&narrow.propensity.canonical_key()));
    let chosen: Vec<String> =
        selected_lambdas(&a).iter().map(|l| format!("{:016x}", l.to_bits())).collect();
    assert!(wire.nuisance_provenance.contains(&format!("selected_lambda={}", chosen.join(","))));
    assert!(provenance_withholds_interval(&wire.nuisance_provenance));
    assert_ne!(t.nuisance_provenance, table(&b).nuisance_provenance);
    // The unpenalized table carries no marker.
    let glm = run(&plain(), &raw, 8).unwrap();
    assert!(!provenance_withholds_interval(&table(&glm).nuisance_provenance));
}
