//! Declared sampling weights across the observational estimators.
//!
//! Every estimator that reads weights is checked three ways:
//!
//! * against an oracle computed independently of the library (weighted normal equations and a
//!   weighted Newton logistic fit written here, a case-weight perturbation derivative of the
//!   whole estimator for the standard errors, a hand-rolled matched-pair computation);
//! * for the weight contract: rescaling the weights changes nothing, a zero-weight row is the
//!   same as a deleted row and is counted, invalid weights are refused naming the first one;
//! * against the truth of a simulated super-population that is sampled with known, unequal
//!   inclusion probabilities, where the unweighted estimate of the same estimator is biased.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "test fixtures index small literals; the oracles are written as the formulas read"
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
    AipwAte, AipwWorkspace, AnalyticSeKind, ClusterDml, DistanceMatching, DmlAte, DmlScore,
    DrLearner, EffectEstimate, EstimationWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityNuisance, PropensityStratification, PropensityWeighting, RidgeTuning,
    SamplingWeights,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{kish_ess, sigmoid, standard_normal};
use antecedent_learn::{ElasticNetSpec, LearnerSpec};
use antecedent_stats::GlmFamily;

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(19)
}

fn dataset(t: &[f64], y: &[f64], z: &[f64]) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
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
    let columns = [(T, t), (Y, y), (Z, z)]
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(T, Y)
}

/// Rows of a confounded study with unequal sampling weights.
#[derive(Clone)]
struct Study {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
}

impl Study {
    fn len(&self) -> usize {
        self.t.len()
    }

    fn data(&self) -> (TabularData, IdentifiedEstimand) {
        dataset(&self.t, &self.y, &self.z)
    }

    fn weights(&self) -> SamplingWeights {
        SamplingWeights::new(self.w.clone()).unwrap()
    }

    fn rescaled(&self, factor: f64) -> Self {
        Self { w: self.w.iter().map(|w| w * factor).collect(), ..self.clone() }
    }

    /// The study with extra rows of weight zero and arbitrary values mixed in at `positions`
    /// (indices into the returned study, ascending).
    fn with_zero_weight_rows(&self, positions: &[usize]) -> Self {
        let total = self.len() + positions.len();
        let (mut t, mut y, mut z, mut w) = (vec![], vec![], vec![], vec![]);
        let mut next = 0;
        for row in 0..total {
            if positions.contains(&row) {
                t.push(1.0 - f64::from(row as u32 % 2));
                y.push(1.0e6 * (row as f64 + 1.0));
                z.push(-17.0 + row as f64);
                w.push(0.0);
            } else {
                t.push(self.t[next]);
                y.push(self.y[next]);
                z.push(self.z[next]);
                w.push(self.w[next]);
                next += 1;
            }
        }
        Self { t, y, z, w }
    }
}

/// A small study whose weights vary by a factor of about ten.
fn small_study(n: usize, seed: u64) -> Study {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x51);
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(rng.next_f64() < sigmoid(-0.2 + 0.8 * zi));
        z[i] = zi;
        t[i] = ti;
        y[i] = 0.5 * zi + (1.0 + 0.4 * zi) * ti + 0.6 * standard_normal(&mut rng);
        w[i] = 0.4 + 4.0 * rng.next_f64() * rng.next_f64() + if i % 7 == 0 { 3.0 } else { 0.0 };
    }
    Study { t, y, z, w }
}

/// A super-population with `E[Y(1) - Y(0)] = 1` (effect `1 + z/2`, `z ~ N(0, 1)`), sampled by
/// Poisson sampling with inclusion probability `expit(-1.4 + z)`: units with large `z` are
/// over-represented, and the weight of a sampled unit is its inverse inclusion probability.
fn survey(population: usize, seed: u64) -> Study {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x52);
    let (mut t, mut y, mut z, mut w) = (vec![], vec![], vec![], vec![]);
    for _ in 0..population {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * zi));
        let yi = zi + (1.0 + 0.5 * zi) * ti + 0.7 * standard_normal(&mut rng);
        let inclusion = sigmoid(-1.4 + zi);
        if rng.next_f64() < inclusion {
            t.push(ti);
            y.push(yi);
            z.push(zi);
            w.push(1.0 / inclusion);
        }
    }
    Study { t, y, z, w }
}

// ---------------------------------------------------------------------------------------------
// Independent numerics
// ---------------------------------------------------------------------------------------------

/// Solve `a x = b` for a small dense system (row-major `a`) by Gaussian elimination.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        let pivot = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..k {
            let f = a[row][col] / a[col][col];
            for c in col..k {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let tail: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

fn inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a.len();
    let columns: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            let mut e = vec![0.0; k];
            e[c] = 1.0;
            solve(a.to_vec(), e)
        })
        .collect();
    (0..k).map(|r| (0..k).map(|c| columns[c][r]).collect()).collect()
}

fn weighted_least_squares(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Vec<f64> {
    let k = x[0].len();
    let mut xtwx = vec![vec![0.0; k]; k];
    let mut xtwy = vec![0.0; k];
    for ((row, &yi), &wi) in x.iter().zip(y).zip(w) {
        for a in 0..k {
            xtwy[a] += wi * row[a] * yi;
            for b in 0..k {
                xtwx[a][b] += wi * row[a] * row[b];
            }
        }
    }
    solve(xtwx, xtwy)
}

/// Weighted logistic maximum likelihood by Newton's method, coefficients on `[1, z]`.
fn weighted_logit(z: &[f64], t: &[f64], mass: &[f64]) -> [f64; 2] {
    let mut beta = [0.0_f64; 2];
    for _ in 0..200 {
        let mut score = [0.0; 2];
        let mut information = [0.0; 3];
        for i in 0..t.len() {
            let e = sigmoid(beta[0] + beta[1] * z[i]);
            let residual = mass[i] * (t[i] - e);
            let v = mass[i] * e * (1.0 - e);
            score[0] += residual;
            score[1] += residual * z[i];
            information[0] += v;
            information[1] += v * z[i];
            information[2] += v * z[i] * z[i];
        }
        let det = information[0] * information[2] - information[1].powi(2);
        let step = [
            (information[2] * score[0] - information[1] * score[1]) / det,
            (information[0] * score[1] - information[1] * score[0]) / det,
        ];
        beta[0] += step[0];
        beta[1] += step[1];
        if step[0].abs().max(step[1].abs()) < 1e-14 {
            break;
        }
    }
    beta
}

fn normalized(weights: &[f64]) -> Vec<f64> {
    let n = weights.len() as f64;
    let sum: f64 = weights.iter().sum();
    weights.iter().map(|w| w * n / sum).collect()
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{what}: {actual} vs {expected}"
    );
}

// ---------------------------------------------------------------------------------------------
// Estimator runners
// ---------------------------------------------------------------------------------------------

fn linear(
    study: &Study,
    weighted: bool,
    configure: impl Fn(LinearAdjustmentAte) -> LinearAdjustmentAte,
) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = configure(LinearAdjustmentAte::new().with_bootstrap_replicates(0));
    if weighted && matches!(estimator.se_kind, AnalyticSeKind::Homoskedastic) {
        estimator = estimator.with_se_kind(AnalyticSeKind::Hc1);
    }
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut EstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn no_clip() -> OverlapPolicy {
    OverlapPolicy::RequireDiagnostics { clip: None, trim: None }
}

fn weighting(
    study: &Study,
    configure: impl Fn(PropensityWeighting) -> PropensityWeighting,
) -> EffectEstimate {
    let (data, estimand) = study.data();
    let estimator = configure(
        PropensityWeighting::new()
            .with_bootstrap_replicates(0)
            .with_overlap(no_clip())
            .with_sampling_weights(study.weights()),
    );
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn unweighted_weighting(study: &Study) -> EffectEstimate {
    let (data, estimand) = study.data();
    let estimator = PropensityWeighting::new().with_bootstrap_replicates(0).with_overlap(no_clip());
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn stratification(study: &Study, strata: u32, weighted: bool) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = PropensityStratification::new()
        .with_bootstrap_replicates(0)
        .with_overlap(no_clip())
        .with_n_strata(strata);
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn matching(study: &Study, weighted: bool) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = PropensityMatching::new().with_overlap(no_clip());
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn distance_matching(study: &Study, weighted: bool) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = DistanceMatching::new().with_overlap(no_clip());
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn glm(
    study: &Study,
    family: GlmFamily,
    weighted: bool,
    configure: impl Fn(GlmAdjustmentAte) -> GlmAdjustmentAte,
) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = configure(
        GlmAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_family(family)
            .with_se_kind(AnalyticSeKind::Hc1),
    );
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn dml(study: &Study, score: DmlScore, weighted: bool) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = DmlAte::new().with_score(score);
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap()
}

fn dr_learner(study: &Study, weighted: bool) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = DrLearner::new();
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap()
}

/// The same study with every row repeated `w_i` times (integer weights only), unweighted.
fn replicated(study: &Study) -> Study {
    let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
    for i in 0..study.len() {
        for _ in 0..(study.w[i].round() as usize) {
            t.push(study.t[i]);
            y.push(study.y[i]);
            z.push(study.z[i]);
        }
    }
    let w = vec![1.0; t.len()];
    Study { t, y, z, w }
}

/// A small study whose weights are the integers 1, 2 and 3.
fn integer_study(n: usize, seed: u64) -> Study {
    let study = small_study(n, seed);
    let w = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
    Study { w, ..study }
}

/// The study with its outcome dichotomized at `0.5`.
fn binary_outcome(study: &Study) -> Study {
    Study { y: study.y.iter().map(|&y| f64::from(y > 0.5)).collect(), ..study.clone() }
}

fn aipw(study: &Study, configure: impl Fn(AipwAte) -> AipwAte) -> EffectEstimate {
    let (data, estimand) = study.data();
    let estimator = configure(AipwAte::new().with_bootstrap_replicates(0));
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator.fit(&prepared, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Linear adjustment
// ---------------------------------------------------------------------------------------------

/// Weighted-least-squares coefficient of `t` in `y ~ 1 + t + z` and its sandwich variance,
/// with scores `w_i x_i e_i`, optionally summed within `groups`.
fn wls_oracle(study: &Study, groups: Option<&[u32]>) -> (f64, f64) {
    let rows: Vec<usize> = (0..study.len()).filter(|&i| study.w[i] > 0.0).collect();
    let w = normalized(&rows.iter().map(|&i| study.w[i]).collect::<Vec<_>>());
    let x: Vec<Vec<f64>> = rows.iter().map(|&i| vec![1.0, study.t[i], study.z[i]]).collect();
    let y: Vec<f64> = rows.iter().map(|&i| study.y[i]).collect();
    let beta = weighted_least_squares(&x, &y, &w);
    let mut xtwx = vec![vec![0.0; 3]; 3];
    for (row, &wi) in x.iter().zip(&w) {
        for a in 0..3 {
            for b in 0..3 {
                xtwx[a][b] += wi * row[a] * row[b];
            }
        }
    }
    let bread = inverse(&xtwx);
    let n = rows.len() as f64;
    // Score contribution of every row, grouped when clusters are declared.
    let mut cells: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    for (k, row) in x.iter().enumerate() {
        let residual = y[k] - (0..3).map(|c| row[c] * beta[c]).sum::<f64>();
        let key = groups.map_or(k as u32, |g| g[rows[k]]);
        let cell = cells.entry(key).or_insert_with(|| vec![0.0; 3]);
        for c in 0..3 {
            cell[c] += w[k] * row[c] * residual;
        }
    }
    let mut meat = vec![vec![0.0; 3]; 3];
    for cell in cells.values() {
        for a in 0..3 {
            for b in 0..3 {
                meat[a][b] += cell[a] * cell[b];
            }
        }
    }
    let factor = match groups {
        Some(_) => {
            let g = cells.len() as f64;
            g / (g - 1.0) * (n - 1.0) / (n - 3.0)
        }
        None => n / (n - 3.0),
    };
    let variance_tt = (0..3)
        .map(|a| (0..3).map(|b| bread[1][a] * meat[a][b] * bread[b][1]).sum::<f64>())
        .sum::<f64>()
        * factor;
    (beta[1], variance_tt.sqrt())
}

#[test]
fn linear_adjustment_is_the_weighted_least_squares_fit_with_a_sampling_sandwich() {
    let study = small_study(120, 3);
    let (beta, se) = wls_oracle(&study, None);
    let fit = linear(&study, true, |e| e);
    close(fit.ate, beta, 1e-9, "weighted coefficient");
    close(fit.se_analytic(), se, 1e-9, "HC1 sandwich with sampling-weighted scores");
    assert_eq!(fit.uncertainty.analytic.kind, Some(AnalyticSeKind::Hc1));
    let unweighted = linear(&study, false, |e| e);
    assert!((unweighted.ate - fit.ate).abs() > 1e-4, "the weights must move the estimate");
}

#[test]
fn a_classical_standard_error_request_with_weights_is_refused() {
    let study = small_study(80, 4);
    let (data, estimand) = study.data();
    let error = LinearAdjustmentAte::new()
        .with_sampling_weights(study.weights())
        .prepare(&data, &estimand, &query())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("sampling_weights.route_unsupported") && error.contains("hc0..hc3"),
        "{error}"
    );
}

#[test]
fn linear_adjustment_cluster_se_is_the_weighted_cluster_sandwich() {
    let study = small_study(150, 5);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i / 6) as u32).collect();
    let (beta, se) = wls_oracle(&study, Some(&groups));
    let fit = linear(&study, true, |e| {
        e.with_se_kind(AnalyticSeKind::Cluster).with_source_cluster_ids(groups.clone())
    });
    close(fit.ate, beta, 1e-9, "weighted coefficient");
    close(fit.se_analytic(), se, 1e-9, "cluster sandwich over weighted scores");
}

#[test]
fn linear_adjustment_reports_the_weight_record() {
    let study = small_study(100, 6).with_zero_weight_rows(&[3, 40, 77]);
    let fit = linear(&study, true, |e| e);
    let record = fit.diagnostics.sampling_weights.expect("a weighted fit carries its record");
    assert_eq!(record.rows_used, 100);
    assert_eq!(record.zero_weight_rows, 3);
    let positive: Vec<f64> = study.w.iter().copied().filter(|w| *w > 0.0).collect();
    close(record.effective_sample_size, kish_ess(&positive), 1e-12, "Kish effective size");
    close(record.weight_sum, positive.iter().sum::<f64>(), 1e-12, "declared weight sum");
    assert_eq!(fit.provenance.n_obs, Some(100));
    assert!(linear(&study, false, |e| e).diagnostics.sampling_weights.is_none());
}

// ---------------------------------------------------------------------------------------------
// The weight contract, for every estimator
// ---------------------------------------------------------------------------------------------

fn each_estimate(study: &Study) -> Vec<(&'static str, EffectEstimate)> {
    vec![
        ("linear", linear(study, true, |e| e)),
        ("weighting", weighting(study, |e| e)),
        ("stratification", stratification(study, 4, true)),
        ("matching", matching(study, true)),
        ("distance matching", distance_matching(study, true)),
        ("glm", glm(study, GlmFamily::GaussianIdentity, true, |e| e)),
    ]
}

#[test]
fn rescaling_the_weights_changes_neither_the_estimate_nor_the_interval_nor_the_effective_size() {
    let study = small_study(160, 7);
    let base = each_estimate(&study);
    for factor in [0.001, 1000.0] {
        let scaled = each_estimate(&study.rescaled(factor));
        for ((name, a), (_, b)) in base.iter().zip(&scaled) {
            close(b.ate, a.ate, 1e-9, &format!("{name} estimate at scale {factor}"));
            close(b.se_analytic(), a.se_analytic(), 1e-9, &format!("{name} SE at scale {factor}"));
            let (ra, rb) =
                (a.diagnostics.sampling_weights.unwrap(), b.diagnostics.sampling_weights.unwrap());
            close(rb.effective_sample_size, ra.effective_sample_size, 1e-9, name);
            close(rb.weight_sum, ra.weight_sum * factor, 1e-9, name);
        }
    }
}

#[test]
fn a_zero_weight_row_is_a_deleted_row_that_is_counted() {
    let study = small_study(140, 8);
    let padded = study.with_zero_weight_rows(&[0, 5, 6, 90, 143]);
    let bare = each_estimate(&study);
    let with_zeros = each_estimate(&padded);
    for ((name, a), (_, b)) in bare.iter().zip(&with_zeros) {
        close(b.ate, a.ate, 1e-12, &format!("{name} estimate"));
        close(b.se_analytic(), a.se_analytic(), 1e-12, &format!("{name} SE"));
        let record = b.diagnostics.sampling_weights.unwrap();
        assert_eq!((record.rows_used, record.zero_weight_rows), (140, 5), "{name}");
    }
}

#[test]
fn invalid_weights_are_refused_naming_the_first_offender() {
    let mut weights = vec![1.0; 30];
    weights[11] = -2.0;
    weights[20] = f64::NAN;
    let error = SamplingWeights::new(weights).unwrap_err().to_string();
    assert!(error.starts_with("reason=sampling_weights_invalid:"), "{error}");
    assert!(error.contains("source row 11") && error.contains("-2"), "{error}");
    assert!(SamplingWeights::new(vec![0.0; 30]).is_err(), "no positive weight");

    // A declaration of the wrong length is refused at preparation, not misaligned.
    let study = small_study(40, 9);
    let (data, estimand) = study.data();
    let short = SamplingWeights::new(vec![1.0; 39]).unwrap();
    let error = LinearAdjustmentAte::new()
        .with_se_kind(AnalyticSeKind::Hc1)
        .with_sampling_weights(short)
        .prepare(&data, &estimand, &query())
        .unwrap_err()
        .to_string();
    assert!(error.contains("39 sampling weights") && error.contains("40 rows"), "{error}");
}

#[test]
fn unit_weights_reproduce_the_unweighted_estimators() {
    let study = Study { w: vec![1.0; 200], ..small_study(200, 10) };
    close(
        linear(&study, true, |e| e.with_se_kind(AnalyticSeKind::Hc1)).ate,
        linear(&study, false, |e| e).ate,
        1e-10,
        "linear",
    );
    let (weighted, plain) = (weighting(&study, |e| e), unweighted_weighting(&study));
    close(weighted.ate, plain.ate, 1e-9, "weighting estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-9, "weighting SE");
    let (weighted, plain) = (stratification(&study, 4, true), stratification(&study, 4, false));
    close(weighted.ate, plain.ate, 1e-9, "stratification estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-9, "stratification SE");
    let (weighted, plain) = (matching(&study, true), matching(&study, false));
    close(weighted.ate, plain.ate, 1e-9, "matching estimate");
}

// ---------------------------------------------------------------------------------------------
// Inverse-probability weighting
// ---------------------------------------------------------------------------------------------

/// The Hajek effect of the weighted-logistic inverse-probability weights, as a function of
/// the case masses `mass` (the sampling weights, perturbed row by row below).
fn hajek_effect(study: &Study, mass: &[f64]) -> f64 {
    let beta = weighted_logit(&study.z, &study.t, mass);
    let (mut num, mut den) = ([0.0; 2], [0.0; 2]);
    for i in 0..study.len() {
        let e = sigmoid(beta[0] + beta[1] * study.z[i]);
        let arm = usize::from(study.t[i] > 0.5);
        let ipw = if arm == 1 { 1.0 / e } else { 1.0 / (1.0 - e) };
        num[arm] += mass[i] * ipw * study.y[i];
        den[arm] += mass[i] * ipw;
    }
    num[1] / den[1] - num[0] / den[0]
}

/// Weighted score of every row: the derivative of the whole estimator (logistic refit
/// included) with respect to the row's case mass, by central differences, times that mass.
/// With-replacement linearization of an estimator of the weighted measure `Σ m_i δ_i` has
/// per-unit contributions `m_i ∂T/∂m_i`, whose squares sum to the variance.
fn weighted_scores(study: &Study) -> Vec<f64> {
    let mut mass = normalized(&study.w);
    let step = 1e-5;
    (0..study.len())
        .map(|row| {
            let base = mass[row];
            mass[row] = base + step;
            let plus = hajek_effect(study, &mass);
            mass[row] = base - step;
            let minus = hajek_effect(study, &mass);
            mass[row] = base;
            base * (plus - minus) / (2.0 * step)
        })
        .collect()
}

#[test]
fn weighting_is_the_hajek_effect_of_the_weighted_logistic_fit() {
    let study = small_study(70, 11);
    let fit = weighting(&study, |e| e);
    close(fit.ate, hajek_effect(&study, &normalized(&study.w)), 1e-8, "weighted Hajek effect");
    let derivatives = weighted_scores(&study);
    let n = study.len() as f64;
    let iid =
        (derivatives.iter().map(|d| d * d).sum::<f64>() * n / (n - 1.0) * n / (n - 2.0)).sqrt();
    close(fit.se_analytic(), iid, 1e-6, "SE equals the weighted-score linearization oracle");
}

#[test]
fn weighting_cluster_se_is_the_cluster_sandwich_with_the_propensity_correction() {
    let study = small_study(90, 12);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i % 15) as u32).collect();
    let derivatives = weighted_scores(&study);
    let mut totals = std::collections::BTreeMap::<u32, f64>::new();
    for (d, g) in derivatives.iter().zip(&groups) {
        *totals.entry(*g).or_default() += d;
    }
    let g = totals.len() as f64;
    let expected = (g / (g - 1.0) * totals.values().map(|v| v * v).sum::<f64>()).sqrt();
    let fit = weighting(&study, |e| e.with_source_cluster_ids(groups.clone()));
    close(fit.se_analytic(), expected, 1e-6, "cluster sandwich equals the derivative oracle");
    assert_eq!(fit.uncertainty.analytic.kind, Some(AnalyticSeKind::Cluster));
    let independent = weighting(&study, |e| e);
    assert!((fit.se_analytic() - independent.se_analytic()).abs() > 1e-6);
}

// ---------------------------------------------------------------------------------------------
// Stratification
// ---------------------------------------------------------------------------------------------

/// The documented stratification rule: cut at the sample quantiles of the propensity, assign
/// by `score <= cut`; pool within-stratum weighted contrasts by weighted stratum mass.
fn stratification_oracle(study: &Study, k: usize) -> (f64, f64) {
    let mass = normalized(&study.w);
    let beta = weighted_logit(&study.z, &study.t, &mass);
    let n = study.len();
    let scores: Vec<f64> =
        (0..n).map(|i| sigmoid(beta[0] + beta[1] * study.z[i]).clamp(1e-12, 1.0 - 1e-12)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let cuts: Vec<f64> = (1..k).map(|s| scores[order[((s * n) / k).min(n - 1)]]).collect();
    let stratum: Vec<usize> = scores
        .iter()
        .map(|score| cuts.iter().position(|cut| score <= cut).unwrap_or(cuts.len()).min(k - 1))
        .collect();
    let (mut effect, mut variance, mut total) = (0.0, 0.0, 0.0);
    let mut pooled = Vec::new();
    for s in 0..k {
        let arm_rows = |arm: f64| -> Vec<usize> {
            (0..n).filter(|&i| stratum[i] == s && study.t[i] == arm).collect()
        };
        let (treated, control) = (arm_rows(1.0), arm_rows(0.0));
        if treated.is_empty() || control.is_empty() {
            continue;
        }
        let arm = |rows: &[usize]| -> (f64, f64, f64) {
            let w: f64 = rows.iter().map(|&i| mass[i]).sum();
            let m = rows.iter().map(|&i| mass[i] * study.y[i]).sum::<f64>() / w;
            let count = rows.len() as f64;
            let ss: f64 = rows.iter().map(|&i| (mass[i] * (study.y[i] - m)).powi(2)).sum();
            (w, m, ss * count / (count - 1.0) / (w * w))
        };
        let ((w1, m1, v1), (w0, m0, v0)) = (arm(&treated), arm(&control));
        pooled.push((m1 - m0, w1 + w0, v1 + v0));
        total += w1 + w0;
    }
    for (d, share, v) in &pooled {
        effect += d * share / total;
        variance += v * (share / total).powi(2);
    }
    (effect, variance.sqrt())
}

#[test]
fn stratification_pools_weighted_strata_and_matches_the_oracle() {
    let study = small_study(400, 13);
    let (effect, se) = stratification_oracle(&study, 5);
    let fit = stratification(&study, 5, true);
    close(fit.ate, effect, 1e-7, "weighted stratified effect");
    close(fit.se_analytic(), se, 1e-7, "linearized stratum variance");
}

#[test]
fn stratification_cluster_se_sums_the_stratified_influence_within_clusters() {
    let study = small_study(400, 14);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i / 10) as u32).collect();
    let (data, estimand) = study.data();
    let estimator = PropensityStratification::new()
        .with_bootstrap_replicates(0)
        .with_overlap(no_clip())
        .with_n_strata(5)
        .with_sampling_weights(study.weights())
        .with_source_cluster_ids(groups.clone());
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    let fit = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    assert_eq!(fit.uncertainty.analytic.kind, Some(AnalyticSeKind::Cluster));
    let iid = stratification(&study, 5, true);
    close(fit.ate, iid.ate, 1e-12, "clusters do not move the point");
    assert!(fit.se_analytic().is_finite() && fit.se_analytic() > 0.0);
    assert!((fit.se_analytic() - iid.se_analytic()).abs() > 1e-9);
}

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

/// Weighted nearest-neighbour matched effect and Abadie-Imbens variance of the weighted matched
/// mean, from first principles: one donor per query in the opposite arm on the logit of the
/// weighted propensity, the donors' linear bias correction, and `σ² Σ c_i²` with
/// `σ² = ½ Σ w̃_q (τ_q − τ)²`.
fn matching_oracle(study: &Study) -> (f64, f64) {
    let n = study.len();
    let mass = normalized(&study.w);
    let beta = weighted_logit(&study.z, &study.t, &mass);
    let eta: Vec<f64> = (0..n).map(|i| beta[0] + beta[1] * study.z[i]).collect();
    let arm = |value: f64| -> Vec<usize> { (0..n).filter(|&i| study.t[i] == value).collect() };
    let (treated, control) = (arm(1.0), arm(0.0));
    let bias_slope = |donors: &[usize]| -> f64 {
        let m = donors.len() as f64;
        let (mx, my) = (
            donors.iter().map(|&i| eta[i]).sum::<f64>() / m,
            donors.iter().map(|&i| study.y[i]).sum::<f64>() / m,
        );
        let sxy: f64 = donors.iter().map(|&i| (eta[i] - mx) * (study.y[i] - my)).sum();
        let sxx: f64 = donors.iter().map(|&i| (eta[i] - mx).powi(2)).sum();
        sxy / sxx
    };
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new(); // (query, donor, signed effect)
    for (queries, donors, sign) in [(&treated, &control, 1.0), (&control, &treated, -1.0)] {
        let slope = bias_slope(donors);
        for &q in queries {
            let d = *donors
                .iter()
                .min_by(|&&a, &&b| (eta[a] - eta[q]).abs().total_cmp(&(eta[b] - eta[q]).abs()))
                .unwrap();
            let raw = study.y[q] - study.y[d] - slope * (eta[q] - eta[d]);
            pairs.push((q, d, sign * raw));
        }
    }
    let total: f64 = pairs.iter().map(|&(q, ..)| mass[q]).sum();
    let ate = pairs.iter().map(|&(q, _, e)| mass[q] * e).sum::<f64>() / total;
    let mut coefficient = vec![0.0; n];
    for &(q, d, _) in &pairs {
        coefficient[q] += mass[q] / total;
        coefficient[d] += mass[q] / total;
    }
    // Every unit is a query of the pooled ATE; a unit with no matched query would still be
    // credited its own `w / total` (the unweighted Abadie-Imbens sum runs over every unit).
    let queried: std::collections::BTreeSet<usize> = pairs.iter().map(|&(q, ..)| q).collect();
    for unit in (0..n).filter(|unit| !queried.contains(unit)) {
        coefficient[unit] += mass[unit] / total;
    }
    // Reliability-weights correction of the weighted variance: 1 / (1 - sum of squared shares).
    let concentration: f64 = pairs.iter().map(|&(q, ..)| (mass[q] / total).powi(2)).sum();
    let sigma2 = 0.5
        * pairs.iter().map(|&(q, _, e)| mass[q] / total * (e - ate).powi(2)).sum::<f64>()
        / (1.0 - concentration);
    (ate, (sigma2 * coefficient.iter().map(|c| c * c).sum::<f64>()).sqrt())
}

#[test]
fn matching_is_the_weighted_mean_of_matched_effects_with_its_donor_variance() {
    let study = small_study(120, 15);
    let (ate, se) = matching_oracle(&study);
    let fit = matching(&study, true);
    close(fit.ate, ate, 1e-7, "weighted matched effect");
    close(fit.se_analytic(), se, 1e-7, "weighted Abadie-Imbens standard error");
}

// ---------------------------------------------------------------------------------------------
// Augmented inverse-probability weighting
// ---------------------------------------------------------------------------------------------

#[test]
fn aipw_reads_weights_through_weighted_nuisances_and_a_weighted_score() {
    let study = small_study(300, 16);
    let weighted = aipw(&study, |e| e.with_sampling_weights(study.weights()));
    let plain = aipw(&study, |e| e);
    assert!((weighted.ate - plain.ate).abs() > 1e-4, "the weights must move the estimate");
    let record = weighted.diagnostics.sampling_weights.expect("record");
    close(record.effective_sample_size, kish_ess(&study.w), 1e-12, "effective size");

    // Unit weights are the unweighted cross-fit.
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    let unit = aipw(&ones, |e| e.with_sampling_weights(ones.weights()));
    close(unit.ate, plain.ate, 1e-9, "unit-weight AIPW");
    close(unit.se_analytic(), plain.se_analytic(), 1e-9, "unit-weight AIPW SE");

    // Rescaling changes nothing.
    let scaled = aipw(&study, |e| e.with_sampling_weights(study.rescaled(250.0).weights()));
    close(scaled.ate, weighted.ate, 1e-9, "rescaled estimate");
    close(scaled.se_analytic(), weighted.se_analytic(), 1e-9, "rescaled SE");
}

#[test]
fn aipw_cluster_se_is_the_cluster_sandwich_of_the_weighted_influence() {
    let study = small_study(300, 17);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i / 10) as u32).collect();
    let fit = aipw(&study, |e| {
        e.with_sampling_weights(study.weights())
            .with_se_kind(AnalyticSeKind::Cluster)
            .with_source_cluster_ids(groups.clone())
    });
    let psi = fit.scores.influence.as_ref().expect("influence values");
    let n = psi.len() as f64;
    let mean = psi.iter().sum::<f64>() / n;
    let mut totals = std::collections::BTreeMap::<u32, f64>::new();
    for (value, g) in psi.iter().zip(&groups) {
        *totals.entry(*g).or_default() += value - mean;
    }
    let g = totals.len() as f64;
    let expected = (g / (g - 1.0) * totals.values().map(|v| v * v).sum::<f64>() / (n * n)).sqrt();
    close(fit.se_analytic(), expected, 1e-10, "cluster sandwich of the weighted influence");
}

fn aipw_target(
    study: &Study,
    weighted: bool,
    population: antecedent_core::TargetPopulation,
    overlap: OverlapPolicy,
) -> EffectEstimate {
    let (data, estimand) = study.data();
    let mut estimator = AipwAte::new().with_bootstrap_replicates(0).with_overlap(overlap);
    if weighted {
        estimator = estimator.with_sampling_weights(study.weights());
    }
    let prepared =
        estimator.prepare(&data, &estimand, &query().with_target_population(population)).unwrap();
    estimator.fit(&prepared, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap()
}

fn in_sample_constructions() -> Vec<(&'static str, antecedent_core::TargetPopulation, OverlapPolicy)>
{
    use antecedent_core::TargetPopulation::{AllObserved, Treated, Untreated};
    vec![
        ("att", Treated, no_clip()),
        ("atc", Untreated, no_clip()),
        (
            "trimmed ate",
            AllObserved,
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) },
        ),
    ]
}

/// The weighted ATT, ATC and trimmed ATE are the unweighted ones of the replicated rows: the
/// weighted logistic and per-arm least-squares fits solve the replicated normal equations, the
/// arm share is the weighted share, and trimming reads the same fitted scores.
#[test]
fn aipw_in_sample_constructions_equal_the_replicated_rows_for_integer_weights() {
    let study = integer_study(240, 31);
    let rows = replicated(&study);
    for (name, population, overlap) in in_sample_constructions() {
        let weighted = aipw_target(&study, true, population.clone(), overlap);
        let plain = aipw_target(&rows, false, population, overlap);
        close(weighted.ate, plain.ate, 1e-7, &format!("{name} point"));
        let record = weighted.diagnostics.sampling_weights.expect("weight record");
        assert!(record.rows_used > 0 && record.rows_used <= 240, "{name}");
    }
}

#[test]
fn aipw_in_sample_constructions_have_unit_weight_and_scale_invariance() {
    let study = small_study(260, 32);
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    for (name, population, overlap) in in_sample_constructions() {
        let weighted = aipw_target(&ones, true, population.clone(), overlap);
        let plain = aipw_target(&ones, false, population.clone(), overlap);
        close(weighted.ate, plain.ate, 1e-9, &format!("{name} unit-weight point"));
        close(weighted.se_analytic(), plain.se_analytic(), 1e-9, &format!("{name} unit-weight SE"));
        let base = aipw_target(&study, true, population.clone(), overlap);
        let scaled = aipw_target(&study.rescaled(400.0), true, population, overlap);
        close(scaled.ate, base.ate, 1e-9, &format!("{name} rescaled point"));
        close(scaled.se_analytic(), base.se_analytic(), 1e-9, &format!("{name} rescaled SE"));
        assert!(base.se_analytic().is_finite() && base.se_analytic() > 0.0, "{name}");
    }
}

#[test]
fn aipw_cluster_dml_reads_weights_through_the_weighted_cluster_sandwich() {
    let study = small_study(300, 33);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i / 10) as u32).collect();
    let run = |study: &Study, weighted: bool| {
        let (data, estimand) = study.data();
        let mut estimator = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_cluster_dml(ClusterDml::new(20).unwrap())
            .with_source_cluster_ids(groups.clone());
        if weighted {
            estimator = estimator.with_sampling_weights(study.weights());
        }
        let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
        estimator
            .fit(&prepared, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap()
    };
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    let (unit, plain) = (run(&ones, true), run(&ones, false));
    close(unit.ate, plain.ate, 1e-9, "unit-weight cluster-DML point");
    close(unit.se_analytic(), plain.se_analytic(), 1e-9, "unit-weight cluster-DML SE");
    let weighted = run(&study, true);
    assert!((weighted.ate - plain.ate).abs() > 1e-4 || study.w.iter().all(|w| *w == study.w[0]));
    assert!(weighted.se_analytic().is_finite() && weighted.se_analytic() > 0.0);
    let scaled = run(&study.rescaled(60.0), true);
    close(scaled.ate, weighted.ate, 1e-9, "rescaled cluster-DML point");
    close(scaled.se_analytic(), weighted.se_analytic(), 1e-9, "rescaled cluster-DML SE");
}

/// A penalized propensity under weights: the penalty is chosen by the weighted inner
/// cross-validated log loss and every fold fit is the prior-weighted penalized likelihood. Unit
/// weights reproduce the unweighted penalized fit and the scale of the weights changes nothing.
#[test]
fn a_penalized_propensity_reads_weights_through_a_weighted_penalty_selection() {
    let study = small_study(320, 18);
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    let tuning = || RidgeTuning::new(&[0.1, 1.0, 10.0], 3).unwrap();
    for nuisance in
        [PropensityNuisance::ridge_logistic(tuning()), PropensityNuisance::lasso_with(tuning())]
    {
        let run = |s: &Study, weighted: bool| {
            aipw(s, |e| {
                let e = AipwAte { propensity: nuisance.clone(), ..e };
                if weighted { e.with_sampling_weights(s.weights()) } else { e }
            })
        };
        let (unit, plain) = (run(&ones, true), run(&ones, false));
        close(unit.ate, plain.ate, 1e-9, "unit-weight penalized point");
        close(unit.se_analytic(), plain.se_analytic(), 1e-9, "unit-weight penalized SE");
        let (base, scaled) = (run(&study, true), run(&study.rescaled(80.0), true));
        close(scaled.ate, base.ate, 1e-9, "rescaled penalized point");
        close(scaled.se_analytic(), base.se_analytic(), 1e-9, "rescaled penalized SE");
        assert!((base.ate - run(&study, false).ate).abs() > 1e-6, "the weights must move it");
        assert!(base.diagnostics.sampling_weights.is_some());
    }
}

#[test]
fn aipw_refuses_a_non_mean_functional_under_weights() {
    let study = small_study(120, 19);
    let (data, estimand) = study.data();
    let mut non_mean = query();
    non_mean.outcome_functional = antecedent_core::OutcomeFunctional::exceedance(0.0);
    let error = AipwAte::new()
        .with_sampling_weights(study.weights())
        .prepare(&data, &estimand, &non_mean)
        .unwrap_err()
        .to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// GLM adjustment
// ---------------------------------------------------------------------------------------------

#[test]
fn a_gaussian_glm_with_weights_is_the_weighted_linear_adjustment() {
    let study = small_study(200, 34);
    let glm_fit = glm(&study, GlmFamily::GaussianIdentity, true, |e| e);
    let linear_fit = linear(&study, true, |e| e.with_se_kind(AnalyticSeKind::Hc1));
    close(glm_fit.ate, linear_fit.ate, 1e-9, "weighted Gaussian GLM point");
    close(glm_fit.se_analytic(), linear_fit.se_analytic(), 1e-7, "weighted Gaussian GLM SE");
}

#[test]
fn a_logit_glm_with_integer_weights_equals_the_replicated_rows() {
    let study = binary_outcome(&integer_study(260, 35));
    let weighted = glm(&study, GlmFamily::BinomialLogit, true, |e| e);
    let plain = glm(&replicated(&study), GlmFamily::BinomialLogit, false, |e| e);
    close(weighted.ate, plain.ate, 1e-6, "weighted logit g-computation point");
    assert!(weighted.se_analytic().is_finite() && weighted.se_analytic() > 0.0);
    let record = weighted.diagnostics.sampling_weights.expect("weight record");
    assert_eq!(record.rows_used, 260);
}

#[test]
fn the_glm_poisson_and_probit_families_read_integer_weights_like_replicated_rows() {
    let base = integer_study(260, 36);
    let counts =
        Study { y: base.y.iter().map(|y| (y.abs() * 2.0).round()).collect(), ..base.clone() };
    let weighted = glm(&counts, GlmFamily::PoissonLog, true, |e| e);
    let plain = glm(&replicated(&counts), GlmFamily::PoissonLog, false, |e| e);
    close(weighted.ate, plain.ate, 1e-6, "weighted Poisson g-computation point");
    let binary = binary_outcome(&base);
    let weighted = glm(&binary, GlmFamily::BinomialProbit, true, |e| e);
    let plain = glm(&replicated(&binary), GlmFamily::BinomialProbit, false, |e| e);
    close(weighted.ate, plain.ate, 1e-6, "weighted probit g-computation point");
}

#[test]
fn glm_refuses_the_classical_se_under_weights() {
    let study = binary_outcome(&small_study(120, 37));
    let (data, estimand) = study.data();
    let classical = GlmAdjustmentAte::new()
        .with_family(GlmFamily::BinomialLogit)
        .with_sampling_weights(study.weights());
    let error = classical.prepare(&data, &estimand, &query()).unwrap_err().to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
    assert!(error.contains("precision"), "{error}");
}

/// The negative-binomial family reads weights through a weighted dispersion update: the fit
/// is finite under unequal weights and reduces to the unweighted fit at unit weights.
#[test]
fn the_negative_binomial_glm_reads_weights_through_a_weighted_dispersion() {
    let base = integer_study(300, 44);
    let counts = Study {
        y: base.y.iter().zip(&base.z).map(|(y, z)| (y.abs() * 3.0 + z.abs()).round()).collect(),
        ..base.clone()
    };
    let weighted = glm(&counts, GlmFamily::NegativeBinomial, true, |e| e);
    assert!(weighted.ate.is_finite() && weighted.se_analytic() > 0.0);
    let unit = Study { w: vec![1.0; counts.len()], ..counts.clone() };
    let (a, b) = (
        glm(&unit, GlmFamily::NegativeBinomial, true, |e| e),
        glm(&unit, GlmFamily::NegativeBinomial, false, |e| e),
    );
    close(a.ate, b.ate, 1e-9, "unit-weight negative-binomial point");
    close(a.se_analytic(), b.se_analytic(), 1e-9, "unit-weight negative-binomial SE");
}

// ---------------------------------------------------------------------------------------------
// Distance matching
// ---------------------------------------------------------------------------------------------

#[test]
fn constant_non_unit_weights_reproduce_the_unweighted_matching_standard_errors() {
    let study = Study { w: vec![3.7; 200], ..small_study(200, 43) };
    let (weighted, plain) = (distance_matching(&study, true), distance_matching(&study, false));
    close(weighted.ate, plain.ate, 1e-12, "distance matching estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-12, "distance matching SE");
    let (weighted, plain) = (matching(&study, true), matching(&study, false));
    close(weighted.ate, plain.ate, 1e-12, "propensity matching estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-12, "propensity matching SE");
}

#[test]
fn distance_matching_with_unit_weights_is_the_unweighted_fit_and_reports_the_record() {
    let study = Study { w: vec![1.0; 200], ..small_study(200, 38) };
    let (weighted, plain) = (distance_matching(&study, true), distance_matching(&study, false));
    close(weighted.ate, plain.ate, 1e-9, "distance matching estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-9, "distance matching SE");
    let unequal = small_study(200, 38);
    let moved = distance_matching(&unequal, true);
    assert!((moved.ate - plain.ate).abs() > 1e-6, "the weights must move the estimate");
    let record = moved.diagnostics.sampling_weights.expect("weight record");
    close(record.effective_sample_size, kish_ess(&unequal.w), 1e-12, "effective size");
}

// ---------------------------------------------------------------------------------------------
// Cross-fitted DML and the DR-learner
// ---------------------------------------------------------------------------------------------

#[test]
fn dml_and_the_dr_learner_with_unit_weights_are_the_unweighted_fits() {
    let study = Study { w: vec![1.0; 240], ..small_study(240, 39) };
    for score in [DmlScore::Aipw, DmlScore::PartiallyLinear] {
        let (weighted, plain) = (dml(&study, score, true), dml(&study, score, false));
        close(weighted.ate, plain.ate, 1e-9, &format!("dml {score:?} estimate"));
        close(weighted.se_analytic(), plain.se_analytic(), 1e-9, &format!("dml {score:?} SE"));
    }
    let (weighted, plain) = (dr_learner(&study, true), dr_learner(&study, false));
    close(weighted.ate, plain.ate, 1e-9, "dr-learner estimate");
    close(weighted.se_analytic(), plain.se_analytic(), 1e-9, "dr-learner SE");
}

#[test]
fn dml_and_the_dr_learner_are_invariant_to_the_scale_of_the_weights_and_move_with_them() {
    let study = small_study(240, 40);
    for score in [DmlScore::Aipw, DmlScore::PartiallyLinear] {
        let base = dml(&study, score, true);
        let scaled = dml(&study.rescaled(0.01), score, true);
        close(scaled.ate, base.ate, 1e-8, &format!("dml {score:?} rescaled estimate"));
        close(
            scaled.se_analytic(),
            base.se_analytic(),
            1e-8,
            &format!("dml {score:?} rescaled SE"),
        );
        let plain = dml(&study, score, false);
        assert!((base.ate - plain.ate).abs() > 1e-6, "the weights must move dml {score:?}");
        let record = base.diagnostics.sampling_weights.expect("weight record");
        close(record.effective_sample_size, kish_ess(&study.w), 1e-12, "effective size");
        assert!(base.scores.table.is_none(), "a weighted fit keeps no retargetable table");
    }
    let base = dr_learner(&study, true);
    let scaled = dr_learner(&study.rescaled(0.01), true);
    close(scaled.ate, base.ate, 1e-8, "dr-learner rescaled estimate");
    close(scaled.se_analytic(), base.se_analytic(), 1e-8, "dr-learner rescaled SE");
    assert!(base.heterogeneity.cate.is_some());
}

/// The linear final stage's pointwise standard errors are the HC0 sandwich of the weighted least
/// squares: unit weights reproduce the unweighted ones and the scale of the weights changes
/// nothing.
#[test]
fn the_dr_learner_publishes_weighted_pointwise_standard_errors() {
    let study = small_study(260, 45);
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    let se = |fit: &EffectEstimate| {
        fit.heterogeneity
            .cate_se
            .as_ref()
            .expect("weighted linear final stage keeps cate_se")
            .to_vec()
    };
    let (unit, plain) = (se(&dr_learner(&ones, true)), se(&dr_learner(&ones, false)));
    for (a, b) in unit.iter().zip(&plain) {
        close(*a, *b, 1e-9, "unit-weight pointwise SE");
    }
    let base = se(&dr_learner(&study, true));
    let scaled = se(&dr_learner(&study.rescaled(0.02), true));
    for (a, b) in scaled.iter().zip(&base) {
        close(*a, *b, 1e-8, "rescaled pointwise SE");
    }
    assert!(base.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn a_learner_without_prior_weights_is_refused_naming_it() {
    let study = small_study(160, 41);
    let (data, estimand) = study.data();
    let estimator = DmlAte::new()
        .with_outcome(LearnerSpec::ElasticNet(ElasticNetSpec { lambda: 0.1, l1_ratio: 0.5 }))
        .with_sampling_weights(study.weights());
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    let error = estimator.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap_err().to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
    assert!(error.contains("elastic_net"), "{error}");
}

#[test]
fn dml_and_the_dr_learner_publish_the_weighted_cluster_sandwich() {
    let study = small_study(300, 42);
    let groups: Vec<u32> = (0..study.len()).map(|i| (i / 10) as u32).collect();
    let ones = Study { w: vec![1.0; study.len()], ..study.clone() };
    let run_dml = |s: &Study, weighted: bool| {
        let (data, estimand) = s.data();
        let mut estimator = DmlAte::new()
            .with_cluster_dml(ClusterDml::new(20).unwrap())
            .with_source_cluster_ids(groups.clone());
        if weighted {
            estimator = estimator.with_sampling_weights(s.weights());
        }
        let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
        estimator.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap()
    };
    let run_dr = |s: &Study, weighted: bool| {
        let (data, estimand) = s.data();
        let mut estimator = DrLearner::new()
            .with_cluster_dml(ClusterDml::new(20).unwrap())
            .with_source_cluster_ids(groups.clone());
        if weighted {
            estimator = estimator.with_sampling_weights(s.weights());
        }
        let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
        estimator.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap()
    };
    // Unit weights: the weighted cluster sandwich of the weighted influence is the unweighted
    // cluster-DML sandwich of the score contrast.
    close(run_dml(&ones, true).se_analytic(), run_dml(&ones, false).se_analytic(), 1e-9, "dml");
    close(run_dr(&ones, true).se_analytic(), run_dr(&ones, false).se_analytic(), 1e-9, "dr");
    // The scale of the weights changes nothing; unequal weights move the point.
    let (base, scaled) = (run_dml(&study, true), run_dml(&study.rescaled(30.0), true));
    close(scaled.ate, base.ate, 1e-8, "rescaled dml point");
    close(scaled.se_analytic(), base.se_analytic(), 1e-8, "rescaled dml SE");
    assert!(base.se_analytic().is_finite() && base.se_analytic() > 0.0);
    assert!((base.ate - run_dml(&study, false).ate).abs() > 1e-6);
}

// ---------------------------------------------------------------------------------------------
// Known truth: a sampled super-population
// ---------------------------------------------------------------------------------------------

/// The population effect is exactly `1`. Units with larger `z` are over-sampled and have larger
/// effects, so the unweighted estimators speak for the sampled units (about `1.3`) while the
/// weighted ones speak for the population; each weighted estimate lands within four of its
/// own standard errors of the truth.
#[test]
fn weighted_estimators_recover_the_population_effect_that_unweighted_ones_miss() {
    let study = survey(30_000, 2024);
    assert!(study.len() > 5_000, "sample of {}", study.len());
    let truth = 1.0;

    let estimates = [
        ("ipw", weighting(&study, |e| e)),
        ("stratification", stratification(&study, 10, true)),
        ("matching", matching(&study, true)),
        ("aipw", aipw(&study, |e| e.with_sampling_weights(study.weights()))),
        ("dml", dml(&study, DmlScore::Aipw, true)),
        ("dr-learner", dr_learner(&study, true)),
        ("distance matching", distance_matching(&study, true)),
    ];
    for (name, estimate) in &estimates {
        let se = estimate.se_analytic();
        assert!(se.is_finite() && se > 0.0 && se < 0.15, "{name} SE {se}");
        // Stratification has a residual within-stratum bias; allow it its own scale.
        let slack =
            if matches!(*name, "stratification" | "distance matching") { 0.06 } else { 0.0 };
        assert!(
            (estimate.ate - truth).abs() < 4.0 * se + slack,
            "{name}: {} vs the population effect {truth} (se {se})",
            estimate.ate
        );
    }
    let biased = [
        ("ipw", unweighted_weighting(&study).ate),
        ("matching", matching(&study, false).ate),
        ("aipw", aipw(&study, |e| e).ate),
        ("dml", dml(&study, DmlScore::Aipw, false).ate),
    ];
    for (name, ate) in biased {
        assert!(ate > truth + 0.15, "unweighted {name} should speak for the sample: {ate}");
    }
    let record = estimates[0].1.diagnostics.sampling_weights.unwrap();
    assert!(record.effective_sample_size < 0.95 * record.rows_used as f64);
}

// ---------------------------------------------------------------------------------------------
// Clusters: bootstrap of whole clusters, and the causal forest
// ---------------------------------------------------------------------------------------------

/// A study of `groups` clusters of `size` rows whose outcomes share a cluster effect.
fn clustered_study(groups: usize, size: usize, seed: u64) -> (Study, Vec<u32>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x53);
    let (mut t, mut y, mut z, mut w, mut labels) = (vec![], vec![], vec![], vec![], vec![]);
    for g in 0..groups {
        let shared = 0.8 * standard_normal(&mut rng);
        for _ in 0..size {
            let zi = standard_normal(&mut rng) + 0.5 * shared;
            let ti = f64::from(rng.next_f64() < sigmoid(-0.2 + 0.7 * zi));
            t.push(ti);
            z.push(zi);
            y.push(0.5 * zi + (1.0 + 0.3 * zi) * ti + shared + 0.5 * standard_normal(&mut rng));
            w.push(0.5 + 2.0 * rng.next_f64());
            labels.push(g as u32);
        }
    }
    (Study { t, y, z, w }, labels)
}

#[test]
fn the_weighting_bootstrap_resamples_whole_clusters_and_tracks_the_cluster_sandwich() {
    let (study, labels) = clustered_study(30, 10, 31);
    let (data, estimand) = study.data();
    let estimator = PropensityWeighting::new()
        .with_bootstrap_replicates(150)
        .with_overlap(no_clip())
        .with_sampling_weights(study.weights())
        .with_source_cluster_ids(labels);
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    let fit = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    let bootstrap = fit.se_bootstrap().expect("the cluster bootstrap reports a standard error");
    let analytic = fit.se_analytic();
    assert!(fit.uncertainty.bootstrap.replicates_ok.unwrap() >= 140);
    assert!(
        bootstrap > 0.5 * analytic && bootstrap < 2.0 * analytic,
        "bootstrap {bootstrap} vs cluster sandwich {analytic}"
    );
    // Declaring the clusters, not just the weights, is what widens the interval over rows.
    let flat = weighting(&study, |e| e);
    assert!(
        analytic > 1.2 * flat.se_analytic(),
        "cluster {analytic} vs rows {}",
        flat.se_analytic()
    );
}

#[test]
fn the_stratification_bootstrap_resamples_whole_clusters() {
    let (study, labels) = clustered_study(30, 10, 32);
    let (data, estimand) = study.data();
    let estimator = PropensityStratification::new()
        .with_bootstrap_replicates(120)
        .with_overlap(no_clip())
        .with_n_strata(4)
        .with_sampling_weights(study.weights())
        .with_source_cluster_ids(labels);
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    let fit = estimator
        .fit(&prepared, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    let bootstrap = fit.se_bootstrap().expect("bootstrap SE");
    assert!(bootstrap.is_finite() && bootstrap > 0.0);
    assert!(fit.uncertainty.bootstrap.replicates_ok.unwrap() >= 100);
}

#[test]
fn a_clustered_causal_forest_subsamples_whole_clusters_and_reports_a_cluster_se() {
    use antecedent_estimate::{CausalForest, ClusterDml};
    let (study, labels) = clustered_study(40, 12, 33);
    let (data, estimand) = study.data();
    let forest = CausalForest::new()
        .with_n_trees(16)
        .with_min_leaf(6)
        .with_cluster_dml(ClusterDml::new(20).unwrap())
        .with_source_cluster_ids(labels.clone());
    let prepared = forest.prepare(&data, &estimand, &query()).unwrap();
    let fit = forest.fit(&prepared, &ctx(), AssumptionSet::new()).unwrap();
    assert_eq!(fit.uncertainty.analytic.kind, Some(AnalyticSeKind::Cluster));
    assert!(fit.se_analytic().is_finite() && fit.se_analytic() > 0.0);
    assert!(fit.heterogeneity.cate.is_some());

    // Fewer clusters than the declared minimum cannot support a cluster sandwich.
    let few: Vec<u32> = labels.iter().map(|l| l % 8).collect();
    let coarse = CausalForest::new()
        .with_n_trees(8)
        .with_cluster_dml(ClusterDml::new(20).unwrap())
        .with_source_cluster_ids(few);
    let error = coarse.prepare(&data, &estimand, &query()).unwrap_err().to_string();
    assert!(error.contains("too_few_clusters"), "{error}");

    // Labels without a declared unit (or the reverse) are not a declaration.
    let half = CausalForest::new().with_source_cluster_ids(labels);
    assert!(half.prepare(&data, &estimand, &query()).is_err());
}
