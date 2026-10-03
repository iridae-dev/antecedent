//! Penalized-propensity cross-fitted AIPW (2.2 E2): the ridge and lasso routes fit and replay,
//! tune their penalty (and a lasso's support) on training rows only, keep the score table, row
//! identity, overlap report and retargeting, publish the cross-fitted influence-function SE and
//! the refit-bootstrap SE (which repeats penalty selection in every replicate), and the
//! declared GLM-to-penalized fallback runs and records the failed fit it replaced. A scope
//! outside the license, a machine-learning fallback and a design the outcome stage cannot fit
//! are refused with their reason codes.
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
    OverlapPolicy, PROPENSITY_FIT_STAGE, PenalizedVarianceReport, PropensityNuisance,
    REFIT_BOOTSTRAP_UNCERTAINTY_KIND, ReplicatePolicy, RidgeTuning, ScoreTable,
    crossfit_influence_se, provenance_marks_penalized, provenance_withholds_interval, retarget,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;
use antecedent_stats::StatsError;

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

fn lasso(grid: &[f64], inner_folds: usize) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        propensity: PropensityNuisance::lasso_with(RidgeTuning::new(grid, inner_folds).unwrap()),
        ..AipwAte::new()
    }
}

fn plain() -> AipwAte {
    AipwAte { bootstrap_replicates: 0, ..AipwAte::new() }
}

fn with_bootstrap(est: AipwAte, replicates: u32) -> AipwAte {
    AipwAte { bootstrap_replicates: replicates, ..est }
}

/// The plain GLM route with a declared fallback destination.
fn glm_with_fallback(fallback: NuisanceFallback) -> AipwAte {
    AipwAte { propensity: PropensityNuisance::default().with_fallback(fallback), ..plain() }
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
            let (_, bits) = p.spec.split_once(':').expect("a penalized fold's spec");
            f64::from_bits(bits.parse::<u64>().unwrap())
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The cross-fitted influence-function SE written from the table's scores: with
/// `d_i = phi1_i - phi0_i`, `sqrt(sum (d_i - mean d)^2 / (n (n - 1)))`.
fn hand_influence_se(table: &ScoreTable) -> f64 {
    let d: Vec<f64> =
        table.column(0).unwrap().iter().zip(table.column(1).unwrap()).map(|(c, t)| t - c).collect();
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    (d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n * (n - 1.0))).sqrt()
}

fn variance_of(estimate: &EffectEstimate) -> &PenalizedVarianceReport {
    estimate
        .penalized
        .as_ref()
        .and_then(|report| report.variance.as_ref())
        .expect("the refit bootstrap's report")
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
/// the fold count and seed, and the cross-fitted influence-function interval ingredients: an SE
/// equal to the hand formula on the table's scores, the joint covariance, the score inference
/// and the influence values. No bootstrap was requested, so no bootstrap SE and no report.
#[test]
fn a_penalized_fit_publishes_scores_a_point_and_the_cross_fitted_influence_interval() {
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
    let t = table(&estimate);
    let hand = hand_influence_se(t);
    assert!(hand.is_finite() && hand > 0.0);
    assert!(
        (estimate.se_analytic - hand).abs() <= 1e-12 * hand,
        "{} vs {hand}",
        estimate.se_analytic
    );
    assert!((crossfit_influence_se(t).unwrap() - hand).abs() <= 1e-12 * hand);
    assert_eq!(estimate.se_kind, Some(AnalyticSeKind::Homoskedastic));
    assert!(estimate.se_bootstrap.is_none());
    assert!(estimate.joint_covariance.is_some());
    assert!(estimate.score_inference.is_some());
    assert_eq!(estimate.influence.as_ref().map(|v| v.len()), Some(300));
    assert!(estimate.overlap_report.is_some());
    assert_eq!(estimate.crossfit_folds, Some(5));
    assert_eq!(estimate.crossfit_seed, Some(11));
    assert_eq!(estimate.learner_provenance.len(), 5);
    assert!(estimate.penalized.is_none(), "a ridge fit with no bootstrap has nothing to report");
    assert_eq!(t.row_index.as_ref(), problem.row_index.as_ref());
    assert_eq!(t.n_rows, 300);
    assert_eq!(t.n_folds, 5);
    assert!(provenance_marks_penalized(&t.nuisance_provenance));
    assert!(!provenance_withholds_interval(&t.nuisance_provenance));
    // A dependence-robust analytic kind is accepted too, computed by the shared SE routine,
    // and the iid-only joint covariance is then not published.
    let hc1 = AipwAte { se_kind: AnalyticSeKind::Hc1, ..ridge(&[0.5, 5.0, 50.0], 4) };
    let robust = run(&hc1, &raw, 11).unwrap();
    assert_eq!(robust.ate.to_bits(), estimate.ate.to_bits());
    assert_eq!(robust.se_kind, Some(AnalyticSeKind::Hc1));
    assert!(robust.se_analytic.is_finite() && robust.se_analytic > 0.9 * hand);
    assert!(robust.joint_covariance.is_none());
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

/// Rewriting every row of fold 0 (covariates, treatment, outcome) with the fold plan held fixed,
/// returns the estimate before and after.
fn rewrite_fold_zero(est: &AipwAte, raw: &Raw) -> (EffectEstimate, EffectEstimate) {
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
    (fit_with_fixed_folds(raw), fit_with_fixed_folds(&changed))
}

/// Tuning uses training rows only: with the fold plan held fixed, rewriting every row of fold 0
/// cannot move the penalty chosen for fold 0, whose training rows are the other folds; the
/// held-out scores do move.
#[test]
fn penalty_is_chosen_on_training_rows_only() {
    let raw = draw(250, 4, 13, Regime::Moderate);
    let (before, after) = rewrite_fold_zero(&ridge(&[0.05, 0.5, 5.0, 50.0], 4), &raw);
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

/// The refit bootstrap publishes its SE as the bootstrap SE, with the replicate accounting, the
/// penalty every successful replicate selected on its own resample (one per fold of that
/// replicate's fold plan, from the declared grid), the influence SE for comparison, and replays
/// bit for bit from the seed.
#[test]
fn the_refit_bootstrap_publishes_its_se_and_the_penalties_it_selected_in_every_replicate() {
    let raw = draw(200, 3, 41, Regime::Moderate);
    let grid = [0.5, 5.0, 50.0];
    let est = with_bootstrap(ridge(&grid, 3), 30);
    let a = run(&est, &raw, 5).unwrap();
    let b = run(&est, &raw, 5).unwrap();
    let se = a.se_bootstrap.expect("a refit bootstrap SE");
    assert!(se.is_finite() && se > 0.0, "{se}");
    let variance = variance_of(&a);
    assert_eq!(variance.uncertainty_kind, REFIT_BOOTSTRAP_UNCERTAINTY_KIND);
    assert_eq!(variance.replicates_requested, 30);
    assert_eq!(variance.replicates_ok + variance.replicates_failed, 30);
    assert_eq!(a.bootstrap_replicates_ok, Some(variance.replicates_ok));
    assert_eq!(a.bootstrap_replicates_failed, Some(variance.replicates_failed));
    assert!(!variance.cancelled);
    assert_eq!(variance.refit_bootstrap_se, Some(se));
    assert_eq!(variance.replicate_penalties.len(), variance.replicates_ok as usize);
    for replicate in &variance.replicate_penalties {
        assert!(replicate.replicate < 30);
        assert_eq!(replicate.lambdas.len(), 5, "one penalty per outer fold");
        assert!(replicate.lambdas.iter().all(|l| grid.contains(l)), "{:?}", replicate.lambdas);
    }
    // The influence SE rides beside it and is both the analytic SE and the hand formula.
    assert!((variance.influence_se - a.se_analytic).abs() <= 1e-12 * a.se_analytic);
    assert!((variance.influence_se - hand_influence_se(table(&a))).abs() < 1e-10);
    // Replay: same seed, same SE bits and penalty record; another seed resamples differently.
    assert_eq!(a.se_bootstrap.map(f64::to_bits), b.se_bootstrap.map(f64::to_bits));
    assert_eq!(a.penalized, b.penalized);
    let c = run(&est, &raw, 6).unwrap();
    assert_ne!(a.se_bootstrap.map(f64::to_bits), c.se_bootstrap.map(f64::to_bits));
}

/// The refit-bootstrap SE is close to an independent bootstrap written here: it resamples the
/// rows with its own generator, builds a data set from each resample and calls the public fit
/// (which re-tunes the penalty on the resample). The two differ only by Monte Carlo error and by
/// the library keeping a copied row's unit identity inside one fold (copies in different folds
/// leak). The tolerance is a ratio in [0.6, 1.6]: with 60 replicates each the SE of an SE is
/// about `1 / sqrt(2 * 60)` = 9%, so the ratio of two independent estimates has sd near 13% and
/// the band is about 3.5 sd either side of 1, with room for the leakage.
#[test]
fn the_refit_bootstrap_is_close_to_an_independent_bootstrap_through_the_public_fit() {
    let raw = draw(250, 3, 47, Regime::Moderate);
    let n = raw.t.len();
    let grid = [0.5, 5.0, 50.0];
    let library = run(&with_bootstrap(ridge(&grid, 3), 60), &raw, 9).unwrap();
    let library_se = library.se_bootstrap.expect("a refit bootstrap SE");

    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut draw_index = |bound: usize| -> usize {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % bound as u64).unwrap()
    };
    let est = ridge(&grid, 3);
    let mut estimates = Vec::new();
    for replicate in 0..60 {
        let idx: Vec<usize> = (0..n).map(|_| draw_index(n)).collect();
        let resampled = Raw {
            t: idx.iter().map(|&i| raw.t[i]).collect(),
            y: idx.iter().map(|&i| raw.y[i]).collect(),
            z: raw.z.iter().map(|c| idx.iter().map(|&i| c[i]).collect()).collect(),
        };
        if let Ok(estimate) = run(&est, &resampled, 100 + replicate) {
            estimates.push(estimate.ate);
        }
    }
    assert!(estimates.len() >= 50, "{} of 60 independent replicates fit", estimates.len());
    let m = mean(&estimates);
    let independent_se = (estimates.iter().map(|a| (a - m) * (a - m)).sum::<f64>()
        / (estimates.len() as f64 - 1.0))
        .sqrt();
    let ratio = library_se / independent_se;
    assert!((0.6..1.6).contains(&ratio), "library {library_se} vs independent {independent_se}");
    // Both estimate the same sampling variability as the influence-function SE does.
    let influence = library.se_analytic;
    assert!((0.6..1.7).contains(&(library_se / influence)), "{library_se} vs IF {influence}");
}

/// A replicate that cannot be fit (a resample that leaves a fold's training rows without two
/// distinct treated units) is counted as failed, never replaced: the successes and failures
/// add up to the replicates attempted, the SE is published exactly when the shared replicate
/// policy allows it, and each success still records its penalties.
#[test]
fn failed_replicates_are_counted_and_the_se_follows_the_failure_policy() {
    let mut raw = draw(40, 2, 53, Regime::Moderate);
    for i in 0..40 {
        raw.t[i] = f64::from(u8::from(i % 7 == 0));
        raw.y[i] = 2.0 * raw.t[i] + raw.z[0][i] + 0.3 * (((i * 7) % 13) as f64 / 13.0 - 0.5);
    }
    let est = with_bootstrap(ridge(&[0.5, 5.0], 2), 100);
    let estimate = run(&est, &raw, 4).unwrap();
    let variance = variance_of(&estimate);
    assert_eq!(variance.replicates_requested, 100);
    assert_eq!(variance.replicates_ok + variance.replicates_failed, 100);
    assert!(variance.replicates_failed > 0, "a 6-treated sample must lose some resamples");
    assert!(variance.replicates_ok > 0);
    assert_eq!(variance.replicate_penalties.len(), variance.replicates_ok as usize);
    let allowed = ReplicatePolicy::BOOTSTRAP
        .decide(100, variance.replicates_ok, variance.replicates_failed)
        .is_ok();
    assert_eq!(estimate.se_bootstrap.is_some(), allowed);
    assert_eq!(variance.refit_bootstrap_se.is_some(), allowed);
    assert_eq!(estimate.bootstrap_replicates_failed, Some(variance.replicates_failed));
}

/// Cancellation of the bootstrap is a stop, never a verdict: the flag is reported, no SE is
/// published from the partial run, and the point estimate is untouched.
#[test]
fn a_cancelled_refit_bootstrap_is_a_stop_with_no_se() {
    let raw = draw(200, 3, 59, Regime::Moderate);
    let est = with_bootstrap(ridge(&[1.0, 10.0], 3), 20);
    let (data, estimand, query) = build(&raw);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(2);
    let mut workspace = AipwWorkspace::default();
    let point =
        ridge(&[1.0, 10.0], 3).fit(&problem, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
    let ate = point.ate;
    let cancelled = ExecutionContext::for_tests(2);
    cancelled.cancellation.cancel();
    let estimate = est.attach_bootstrap(&problem, &mut workspace, &cancelled, point).unwrap();
    assert!(estimate.bootstrap_cancelled);
    assert!(estimate.se_bootstrap.is_none());
    assert_eq!(estimate.ate.to_bits(), ate.to_bits());
    let variance = variance_of(&estimate);
    assert!(variance.cancelled && variance.refit_bootstrap_se.is_none());
    assert_eq!(variance.replicates_ok, 0);
}

/// Outside the untrimmed `AllObserved` mean ATE a penalty has no route: ridge is
/// `route_not_supported`, and a lasso (which would select on the rows it scores outside the
/// cross-fit) is `selection_inference_not_licensed`.
#[test]
fn a_scope_outside_the_license_is_refused() {
    let raw = draw(200, 3, 4, Regime::Moderate);
    let (data, estimand, query) = build(&raw);
    let trim = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) };
    let trimmed_ridge = AipwAte { overlap: trim, ..ridge(&[1.0], 3) };
    let error = trimmed_ridge.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("route_not_supported"), "{error}");
    let trimmed_lasso = AipwAte { overlap: trim, ..lasso(&[1.0], 3) };
    let error = trimmed_lasso.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("selection_inference_not_licensed"), "{error}");
    // A fallback beside a penalized primary could never run, so it is refused, not ignored.
    let incoherent = AipwAte {
        propensity: PropensityNuisance::ridge_logistic(RidgeTuning::default())
            .with_fallback(NuisanceFallback::Ml),
        ..plain()
    };
    let error = incoherent.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("invalid_argument"), "{error}");
}

/// Sparse truth: the treatment depends on `z0` and `z1` among ten covariates. The lasso
/// propensity keeps `z0` on every fold and `z1` on at least four, drops most noise columns,
/// records the support per fold in the report and in the table's provenance, recovers the
/// effect 2, publishes the influence SE and the retargetable table, and replays bit for bit.
#[test]
fn a_lasso_propensity_recovers_the_support_and_the_effect_and_records_it_per_fold() {
    let raw = draw(600, 10, 61, Regime::Moderate);
    let est = lasso(&[3.0, 10.0, 30.0, 100.0], 4);
    let estimate = run(&est, &raw, 13).unwrap();
    assert!((estimate.ate - 2.0).abs() < 0.35, "ate = {}", estimate.ate);
    let report = estimate.penalized.as_ref().expect("a lasso records its selected support");
    assert!(report.fallback.is_none() && report.variance.is_none());
    assert_eq!(report.selected_support.len(), 5);
    let kept = |name: &str| {
        report.selected_support.iter().filter(|s| s.names.iter().any(|n| n == name)).count()
    };
    assert_eq!(kept("V2"), 5, "{:?}", report.selected_support);
    assert!(kept("V3") >= 4, "{:?}", report.selected_support);
    let noise: usize = report
        .selected_support
        .iter()
        .map(|s| s.names.iter().filter(|n| !["V2", "V3"].contains(&n.as_str())).count())
        .sum();
    // A cross-validated lasso over-selects noise (its penalty minimizes prediction loss, not
    // selection error), so the CV-chosen claim is only that the pooled selection is short of all
    // 40 noise slots; sparsity proper is shown below with a declared strong penalty.
    assert!(noise < 30, "{noise} of 40 noise slots selected: {:?}", report.selected_support);
    let strong = run(&lasso(&[200.0], 4), &raw, 13).unwrap();
    let strong_noise: usize = strong
        .penalized
        .as_ref()
        .unwrap()
        .selected_support
        .iter()
        .map(|s| s.names.iter().filter(|n| !["V2", "V3"].contains(&n.as_str())).count())
        .sum();
    assert!(
        strong_noise < noise,
        "a stronger declared penalty is sparser: {strong_noise} vs {noise}"
    );
    for (fold, support) in report.selected_support.iter().enumerate() {
        assert_eq!(support.fold, fold);
        assert_eq!(support.columns.len(), support.names.len());
        assert!(support.columns.iter().all(|c| (1..=10).contains(c)));
    }
    // Score-mean identity, the influence SE, the joint covariance and the provenance.
    let t = table(&estimate);
    let contrast: Vec<f64> =
        t.column(0).unwrap().iter().zip(t.column(1).unwrap()).map(|(a, b)| b - a).collect();
    assert!((estimate.ate - mean(&contrast)).abs() < 1e-12);
    let hand = hand_influence_se(t);
    assert!((estimate.se_analytic - hand).abs() <= 1e-12 * hand);
    assert!(estimate.joint_covariance.is_some() && estimate.score_inference.is_some());
    assert_eq!(estimate.learner_provenance.len(), 5);
    assert!(estimate.learner_provenance.iter().all(|p| p.spec.starts_with("lasso_logistic:")));
    let provenance = t.nuisance_provenance.as_ref();
    assert!(provenance.contains(";propensity=lasso.cv("), "{provenance}");
    let support = provenance.split(";selected_support=").nth(1).expect("per-fold support");
    assert_eq!(support.matches('|').count(), 4, "{support}");
    assert!(support.starts_with("0:"), "{support}");
    assert!(provenance_marks_penalized(provenance) && !provenance_withholds_interval(provenance));
    // The frozen table retargets through the shared routine.
    let ones = vec![1.0; raw.t.len()];
    let (retargeted, overlap_failed) = retarget(t, &ones, &[], None, None, None).unwrap();
    assert!(!overlap_failed);
    assert!((retargeted.contrast.as_ref().unwrap().value - estimate.ate).abs() < 1e-12);
    // Seeded replay is bit for bit, support included.
    let again = run(&est, &raw, 13).unwrap();
    assert_eq!(again.ate.to_bits(), estimate.ate.to_bits());
    assert_eq!(again.penalized, estimate.penalized);
    assert_eq!(bits(&table(&again).scores), bits(t.scores.as_ref()));
}

/// The lasso is cross-fitted on training rows only: rewriting fold 0 cannot move the penalty or
/// the support chosen for fold 0, and the held-out scores do move.
#[test]
fn a_lasso_support_and_penalty_are_chosen_on_training_rows_only() {
    let raw = draw(250, 5, 67, Regime::Moderate);
    let (before, after) = rewrite_fold_zero(&lasso(&[1.0, 5.0, 20.0, 80.0], 4), &raw);
    assert_eq!(before.learner_provenance[0], after.learner_provenance[0]);
    let support = |e: &EffectEstimate| e.penalized.as_ref().unwrap().selected_support[0].clone();
    assert_eq!(support(&before), support(&after));
    assert_ne!(bits(&table(&before).scores), bits(&table(&after).scores));
}

/// Where the unpenalized logistic has no MLE (complete separation) the lasso propensity fits
/// and its AIPW recovers the effect.
#[test]
fn a_lasso_propensity_fits_where_the_plain_logistic_is_refused() {
    let raw = draw(400, 8, 31, Regime::Separated);
    assert!(matches!(run(&plain(), &raw, 7).unwrap_err(), EstimationError::Stats(_)));
    let estimate = run(&lasso(&[5.0, 20.0, 80.0], 4), &raw, 7).unwrap();
    assert!((estimate.ate - 2.0).abs() < 0.5, "ate = {}", estimate.ate);
    assert!(estimate.se_analytic.is_finite() && estimate.se_analytic > 0.0);
}

/// The refit bootstrap repeats the lasso's penalty and support selection on every resample.
#[test]
fn the_lasso_refit_bootstrap_repeats_selection_and_records_each_replicate() {
    let raw = draw(200, 4, 71, Regime::Moderate);
    let grid = [2.0, 10.0, 40.0];
    let estimate = run(&with_bootstrap(lasso(&grid, 3), 20), &raw, 3).unwrap();
    let variance = variance_of(&estimate);
    assert_eq!(variance.replicates_ok + variance.replicates_failed, 20);
    assert!(variance.replicates_ok >= 2);
    assert!(estimate.se_bootstrap.is_some_and(|se| se.is_finite() && se > 0.0));
    for replicate in &variance.replicate_penalties {
        assert_eq!(replicate.lambdas.len(), 5);
        assert!(replicate.lambdas.iter().all(|l| grid.contains(l)));
    }
    assert!(!estimate.penalized.as_ref().unwrap().selected_support.is_empty());
}

/// A declared ML fallback stays closed: a failed GLM fit is refused with the failure recorded,
/// a successful one is the plain GLM result bit for bit, and nothing is silently replaced.
#[test]
fn a_machine_learning_fallback_is_refused_with_the_failed_fit_recorded() {
    let fallback = glm_with_fallback(NuisanceFallback::Ml);
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

/// A declared GLM-to-ridge fallback runs when the GLM propensity fit fails: the result is the
/// destination route's (the same scores and point as declaring it directly) with the failed fit
/// (stage, class, message) and the selected destination recorded on the result and in the
/// table's provenance; a healthy GLM fit is untouched and records nothing.
#[test]
fn a_failed_glm_fit_runs_the_declared_ridge_fallback_and_records_both() {
    let tuning = RidgeTuning::new(&[1.0, 10.0, 100.0], 3).unwrap();
    let fallback = glm_with_fallback(NuisanceFallback::RidgeLogistic(tuning.clone()));
    let separated = draw(300, 3, 8, Regime::Separated);
    let estimate = run(&fallback, &separated, 2).unwrap();
    assert!((estimate.ate - 2.0).abs() < 0.5, "ate = {}", estimate.ate);

    let record = estimate.penalized.as_ref().and_then(|r| r.fallback.as_ref()).expect("a record");
    assert_eq!(record.failed_fit.stage, PROPENSITY_FIT_STAGE);
    assert!(
        ["separated", "non_converged", "boundary_saturated"].contains(&record.failed_fit.reason),
        "{}",
        record.failed_fit.reason
    );
    assert!(record.failed_fit.fold < 5 && !record.failed_fit.message.is_empty());
    let destination = PropensityNuisance::ridge_logistic(tuning.clone());
    assert_eq!(record.destination, destination.canonical_key());

    // The result is the destination's: identical to declaring the ridge route directly.
    let direct = run(&ridge(&[1.0, 10.0, 100.0], 3), &separated, 2).unwrap();
    assert_eq!(estimate.ate.to_bits(), direct.ate.to_bits());
    assert_eq!(bits(&table(&estimate).scores), bits(&table(&direct).scores));
    assert_eq!(estimate.se_analytic.to_bits(), direct.se_analytic.to_bits());
    assert_eq!(estimate.learner_provenance, direct.learner_provenance);
    assert_eq!(estimate.learner_provenance.len(), 5);

    // Both identities are on the table: the destination's configuration, the failure and the
    // declared fallback.
    let provenance = table(&estimate).nuisance_provenance.as_ref();
    assert!(provenance.contains(&destination.canonical_key()), "{provenance}");
    assert!(provenance.contains(";nuisance_fallback=glm_failed:"), "{provenance}");
    assert!(provenance.contains(&fallback.propensity.canonical_key()), "{provenance}");
    assert_ne!(
        table(&estimate).nuisance_provenance,
        table(&direct).nuisance_provenance,
        "a fallback result is never the same artifact as the declared route"
    );
    assert!(!provenance_withholds_interval(provenance));

    // A healthy GLM fit under the same declaration is the plain GLM result and records nothing.
    let healthy = draw(300, 3, 8, Regime::Moderate);
    let armed = run(&fallback, &healthy, 2).unwrap();
    let glm = run(&plain(), &healthy, 2).unwrap();
    assert_eq!(armed.ate.to_bits(), glm.ate.to_bits());
    assert!(armed.penalized.is_none());
    assert!(!table(&armed).nuisance_provenance.as_ref().contains(";nuisance_fallback"));

    // The lasso destination runs the same way and records its support.
    let lasso_fallback = glm_with_fallback(NuisanceFallback::Lasso(
        RidgeTuning::new(&[5.0, 20.0, 80.0], 3).unwrap(),
    ));
    let via_lasso = run(&lasso_fallback, &separated, 2).unwrap();
    let report = via_lasso.penalized.as_ref().unwrap();
    assert!(report.fallback.as_ref().unwrap().destination.starts_with("lasso.cv("));
    assert_eq!(report.selected_support.len(), 5);
}

/// With a fallback that ran, the bootstrap replicates run the destination (the claim is the
/// destination's) and a failed GLM fit is never retried inside them.
#[test]
fn the_bootstrap_of_a_fallback_result_runs_the_destination() {
    let tuning = RidgeTuning::new(&[5.0, 50.0], 3).unwrap();
    let est = with_bootstrap(glm_with_fallback(NuisanceFallback::RidgeLogistic(tuning)), 12);
    let separated = draw(240, 3, 8, Regime::Separated);
    let estimate = run(&est, &separated, 2).unwrap();
    let report = estimate.penalized.as_ref().unwrap();
    assert!(report.fallback.is_some());
    let variance = report.variance.as_ref().expect("the destination's refit bootstrap");
    assert_eq!(variance.replicates_ok + variance.replicates_failed, 12);
    assert!(variance.replicates_ok >= 2);
    assert!(
        variance
            .replicate_penalties
            .iter()
            .all(|r| r.lambdas.iter().all(|l| [5.0, 50.0].contains(l)))
    );
    assert!(estimate.se_bootstrap.is_some());
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
    assert!(provenance_marks_penalized(&wire.nuisance_provenance));
    assert!(!provenance_withholds_interval(&wire.nuisance_provenance));
    assert_ne!(t.nuisance_provenance, table(&b).nuisance_provenance);
    // The unpenalized table carries no marker.
    let glm = run(&plain(), &raw, 8).unwrap();
    assert!(!provenance_marks_penalized(&table(&glm).nuisance_provenance));
    // A lasso table with its selected support round-trips too.
    let l = run(&lasso(&[2.0, 10.0], 3), &raw, 8).unwrap();
    let lasso_wire = table(&l).to_wire();
    assert_eq!(&ScoreTable::from_wire(lasso_wire.clone()).unwrap(), table(&l));
    assert!(lasso_wire.nuisance_provenance.contains(";selected_support=0:"));
}

/// Repeated sampling on known truth (a test with fixed seeds, not a coverage record): over 100
/// simulated data sets with both nuisances correctly specified, the 95% interval from the
/// cross-fitted influence-function SE of the ridge and of the lasso route covers the effect 2.
/// The bound is the nominal 0.95 minus about three Monte Carlo standard errors (0.022) and a
/// finite-sample allowance; the measured coverage lives in the calibration suite.
#[test]
fn the_influence_interval_covers_the_known_effect_in_repeated_sampling() {
    for (name, est) in [("ridge", ridge(&[0.5, 50.0], 3)), ("lasso", lasso(&[2.0, 20.0], 3))] {
        let mut covered = 0u32;
        let reps = 100u32;
        for r in 0..reps {
            let raw = draw(200, 3, 900 + u64::from(r), Regime::Moderate);
            let estimate = run(&est, &raw, u64::from(r)).unwrap();
            if (estimate.ate - 2.0).abs() <= 1.96 * estimate.se_analytic {
                covered += 1;
            }
        }
        let coverage = f64::from(covered) / f64::from(reps);
        assert!((0.87..=0.995).contains(&coverage), "{name}: coverage {coverage}");
    }
}
