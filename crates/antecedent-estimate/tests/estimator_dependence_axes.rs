//! Declared dependence on the observational contrast estimators, against closed forms.
//!
//! On a small deterministic design (clusters of four rows, a cluster-level shock, a
//! serially smooth disturbance) the cluster-robust and Newey-West standard errors of the
//! linear and Gaussian-GLM adjustment estimators are compared with a sandwich computed here
//! from the definition (ordinary least squares by Gaussian elimination, scores `x_i e_i`
//! summed within a cluster, or Bartlett-weighted autocovariances of the scores). The same
//! rows relabelled and reordered must give the same standard errors, and dependence that the
//! estimator cannot read must be refused rather than ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "test fixtures index small literals; the oracles are written as the formulas read"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AnalyticSeKind, EffectEstimate, EstimationWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_stats::GlmFamily;

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(41)
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

/// Deterministic rows: `groups` clusters of `size` rows. The cluster shock enters the
/// outcome, so rows of one cluster are positively dependent; the disturbance also drifts
/// smoothly with the row index, so neighbouring rows are serially dependent.
struct Rows {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    cluster: Vec<u32>,
}

fn rows(groups: usize, size: usize) -> Rows {
    let n = groups * size;
    let (mut t, mut y, mut z, mut cluster) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let g = i / size;
        let shock = 1.5 * ((g as f64) * 2.399).sin();
        let drift = 0.8 * ((i as f64) * 0.35).sin();
        let zi = ((i as f64) * 0.71).cos() + 0.3 * ((i * i % 7) as f64);
        let ti = f64::from(((i * 5 + g * 3) % 7) < 3);
        t.push(ti);
        z.push(zi);
        y.push(1.0 + 2.0 * ti + 0.5 * zi + shock + drift + 0.2 * ((i as f64) * 1.7).sin());
        cluster.push(g as u32);
    }
    Rows { t, y, z, cluster }
}

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

/// Ordinary least squares of `y` on `[1, t, z]`: the coefficients, the inverse Gram matrix
/// and the row scores `x_i e_i`.
fn ols(r: &Rows) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = r.t.len();
    let x: Vec<[f64; 3]> = (0..n).map(|i| [1.0, r.t[i], r.z[i]]).collect();
    let mut xtx = vec![vec![0.0; 3]; 3];
    let mut xty = vec![0.0; 3];
    for i in 0..n {
        for a in 0..3 {
            xty[a] += x[i][a] * r.y[i];
            for b in 0..3 {
                xtx[a][b] += x[i][a] * x[i][b];
            }
        }
    }
    let beta = solve(xtx.clone(), xty);
    let scores = (0..n)
        .map(|i| {
            let e = r.y[i] - (0..3).map(|c| x[i][c] * beta[c]).sum::<f64>();
            (0..3).map(|c| x[i][c] * e).collect()
        })
        .collect();
    (beta, inverse(&xtx), scores)
}

/// `B Ω B` read at the treatment coefficient.
fn sandwich_tt(bread: &[Vec<f64>], omega: &[Vec<f64>]) -> f64 {
    (0..3).map(|a| (0..3).map(|b| bread[1][a] * omega[a][b] * bread[b][1]).sum::<f64>()).sum()
}

/// Treatment coefficient and its CR1 cluster-robust standard error, from the definition.
fn cluster_oracle(r: &Rows) -> (f64, f64) {
    let n = r.t.len() as f64;
    let (beta, bread, scores) = ols(r);
    let mut cells: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    for (i, s) in scores.iter().enumerate() {
        let cell = cells.entry(r.cluster[i]).or_insert_with(|| vec![0.0; 3]);
        for c in 0..3 {
            cell[c] += s[c];
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
    let g = cells.len() as f64;
    let factor = g / (g - 1.0) * (n - 1.0) / (n - 3.0);
    (beta[1], (sandwich_tt(&bread, &meat) * factor).sqrt())
}

/// Treatment coefficient and the Newey-West standard error with Bartlett weights
/// `1 - l / (lag + 1)` and no small-sample factor (the lag-zero sandwich is HC0), from the
/// definition.
fn newey_west_oracle(r: &Rows, lag: usize) -> (f64, f64) {
    let n = r.t.len();
    let (beta, bread, scores) = ols(r);
    let mut omega = vec![vec![0.0; 3]; 3];
    for l in 0..=lag {
        let weight = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
        for i in l..n {
            for a in 0..3 {
                for b in 0..3 {
                    let term = weight * scores[i][a] * scores[i - l][b];
                    omega[a][b] += term;
                    if l > 0 {
                        omega[b][a] += term;
                    }
                }
            }
        }
    }
    (beta[1], sandwich_tt(&bread, &omega).sqrt())
}

fn linear(r: &Rows, configure: impl Fn(LinearAdjustmentAte) -> LinearAdjustmentAte) -> EffectEstimate {
    let (data, estimand) = dataset(&r.t, &r.y, &r.z);
    let estimator = configure(LinearAdjustmentAte::new().with_bootstrap_replicates(0));
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut EstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn glm(r: &Rows, configure: impl Fn(GlmAdjustmentAte) -> GlmAdjustmentAte) -> EffectEstimate {
    let (data, estimand) = dataset(&r.t, &r.y, &r.z);
    let estimator = configure(
        GlmAdjustmentAte::new().with_bootstrap_replicates(0).with_family(GlmFamily::GaussianIdentity),
    );
    let prepared = estimator.prepare(&data, &estimand, &query()).unwrap();
    estimator
        .fit(&prepared, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn linear_and_gaussian_glm_cluster_se_are_the_closed_form_cluster_sandwich() {
    let r = rows(12, 4);
    let (beta, se) = cluster_oracle(&r);
    let independent = linear(&r, |e| e.with_se_kind(AnalyticSeKind::Hc1));
    assert!(
        (independent.se_analytic() - se).abs() > 1e-3,
        "the design must make clustering matter: {} vs {se}",
        independent.se_analytic()
    );
    let by_linear = linear(&r, |e| {
        e.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(r.cluster.clone())
    });
    close(by_linear.ate, beta, 1e-9, "linear coefficient");
    close(by_linear.se_analytic(), se, 1e-9, "linear cluster sandwich");
    assert_eq!(by_linear.uncertainty.analytic.kind, Some(AnalyticSeKind::Cluster));
    let by_glm = glm(&r, |e| {
        e.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(r.cluster.clone())
    });
    close(by_glm.ate, beta, 1e-9, "gaussian glm coefficient");
    close(by_glm.se_analytic(), se, 1e-9, "gaussian glm cluster sandwich");
}

#[test]
fn cluster_se_is_invariant_to_cluster_relabelling_and_to_the_order_of_whole_clusters() {
    let r = rows(12, 4);
    let base = linear(&r, |e| {
        e.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(r.cluster.clone())
    });
    // Reverse the order of whole clusters (each keeps its internal row order) and relabel.
    let mut order: Vec<usize> = vec![];
    for g in (0..12).rev() {
        order.extend(g * 4..g * 4 + 4);
    }
    let moved = Rows {
        t: order.iter().map(|&i| r.t[i]).collect(),
        y: order.iter().map(|&i| r.y[i]).collect(),
        z: order.iter().map(|&i| r.z[i]).collect(),
        cluster: order.iter().map(|&i| 1000 + 7 * r.cluster[i]).collect(),
    };
    for (name, fit) in [
        (
            "linear",
            linear(&moved, |e| {
                e.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(moved.cluster.clone())
            }),
        ),
        (
            "glm",
            glm(&moved, |e| {
                e.with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(moved.cluster.clone())
            }),
        ),
    ] {
        close(fit.ate, base.ate, 1e-9, name);
        close(fit.se_analytic(), base.se_analytic(), 1e-9, name);
    }
}

#[test]
fn newey_west_se_of_linear_and_gaussian_glm_adjustment_is_the_closed_form_hac() {
    let r = rows(12, 4);
    for lag in [1usize, 3] {
        let (beta, se) = newey_west_oracle(&r, lag);
        let by_linear = linear(&r, |e| e.with_se_kind(AnalyticSeKind::NeweyWest { lag }));
        close(by_linear.ate, beta, 1e-9, "linear coefficient");
        close(by_linear.se_analytic(), se, 1e-9, &format!("linear newey-west lag {lag}"));
        let by_glm = glm(&r, |e| e.with_se_kind(AnalyticSeKind::NeweyWest { lag }));
        close(by_glm.se_analytic(), se, 1e-9, &format!("glm newey-west lag {lag}"));
    }
    // The lag-zero Newey-West sandwich is the HC0 sandwich.
    let zero = linear(&r, |e| e.with_se_kind(AnalyticSeKind::NeweyWest { lag: 0 }));
    let hc0 = linear(&r, |e| e.with_se_kind(AnalyticSeKind::Hc0));
    close(zero.se_analytic(), hc0.se_analytic(), 1e-9, "lag zero is HC0");
}

#[test]
fn a_cluster_standard_error_without_cluster_labels_or_with_one_cluster_is_refused() {
    let r = rows(12, 4);
    let (data, estimand) = dataset(&r.t, &r.y, &r.z);
    let linear_fit = |ids: Option<Vec<u32>>| {
        let mut estimator = LinearAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_se_kind(AnalyticSeKind::Cluster);
        if let Some(ids) = ids {
            estimator = estimator.with_cluster_ids(ids);
        }
        estimator.prepare(&data, &estimand, &query()).and_then(|prepared| {
            estimator.fit(
                &prepared,
                &mut EstimationWorkspace::default(),
                &ctx(),
                AssumptionSet::new(),
            )
        })
    };
    let glm_fit = |ids: Option<Vec<u32>>| {
        let mut estimator = GlmAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_family(GlmFamily::GaussianIdentity)
            .with_se_kind(AnalyticSeKind::Cluster);
        if let Some(ids) = ids {
            estimator = estimator.with_cluster_ids(ids);
        }
        estimator.prepare(&data, &estimand, &query()).and_then(|prepared| {
            estimator.fit(
                &prepared,
                &mut GlmAdjustmentWorkspace::default(),
                &ctx(),
                AssumptionSet::new(),
            )
        })
    };
    for (name, error) in [
        ("linear without labels", linear_fit(None).map(|fit| fit.ate).unwrap_err()),
        ("glm without labels", glm_fit(None).map(|fit| fit.ate).unwrap_err()),
    ] {
        assert!(error.to_string().to_lowercase().contains("cluster"), "{name}: {error}");
    }
    let one = vec![0_u32; r.t.len()];
    for (name, error) in [
        ("linear", linear_fit(Some(one.clone())).map(|fit| fit.ate).unwrap_err()),
        ("glm", glm_fit(Some(one)).map(|fit| fit.ate).unwrap_err()),
    ] {
        assert!(error.to_string().contains("at least 2 clusters"), "{name}: {error}");
    }
}
