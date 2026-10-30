//! The Bayesian static mediation posterior against its exact conjugate closed form.
//!
//! The mediator mechanism `M ~ 1 + T` and the outcome mechanism `Y ~ 1 + T + M` carry the
//! Normal-Inverse-Gamma conjugate posterior `b | s2 ~ N(0, s2 c0^2 I)`, `s2 ~ Inv-Gamma(1e-3,
//! 1e-3)`, so each coefficient vector has the mean `(X'X + I/c0^2)^-1 X'y` and the Student-t
//! covariance `bn / (an - 1) (X'X + I/c0^2)^-1`. The two mechanisms are independent, so with
//! `delta` the treatment gap the mediated quantity `delta a b` has the mean `delta E[a] E[b]` and
//! the variance `delta^2 (Va Vb + Va E[b]^2 + Vb E[a]^2)`, and the direct quantity `delta c` has
//! `delta E[c]` and `delta^2 Vc`. Those are evaluated here in closed form, independently of the
//! product, and the route must reproduce them to the Monte Carlo error of its draws.
//!
//! At a small sample the posterior mean is pulled toward the prior mean zero, so it differs from
//! the least-squares (frequentist) estimate, and that difference is an exact consequence of the
//! prior, not a defect: it is bounded by the prior's pull on each coefficient and it vanishes as
//! the sample grows. Three sample sizes of one law show the gap shrinking at the closed-form
//! rate. A tolerance fitted to the observed gap would hide this; the bound here is derived.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, MediationContrast, MediationQuery, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const DRAWS: usize = 200_000;
const PRIOR_SCALE: f64 = 2.0;
const TREATED: f64 = 1.0;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Deterministic law: `M = 0.3 + 0.9 T + e`, `Y = 1 + 0.7 T + 1.4 M + u`.
fn law(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..n).map(|i| if (i * 5 + 2) % 7 < 3 { 1.0 } else { 0.0 }).collect();
    let m: Vec<f64> =
        (0..n).map(|i| 0.3 + 0.9 * t[i] + 0.5 * (((i * 7 + 1) % 13) as f64 - 6.0) / 6.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.7 * t[i] + 1.4 * m[i] + 0.4 * (((i * 11 + 3) % 17) as f64 - 8.0) / 8.0)
        .collect();
    (t, m, y)
}

/// Solve the symmetric system `a x = b` by Gaussian elimination with pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        for r in 0..n {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..n {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    (0..n).map(|i| b[i] / a[i][i]).collect()
}

fn inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let cols: Vec<Vec<f64>> = (0..n)
        .map(|j| solve(a.to_vec(), (0..n).map(|i| f64::from(u8::from(i == j))).collect()))
        .collect();
    (0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect()
}

/// Posterior mean and covariance of one mechanism's coefficients, and the least-squares fit.
struct Mechanism {
    mean: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    least_squares: Vec<f64>,
}

fn conjugate(rows: &[Vec<f64>], y: &[f64], scale: f64) -> Mechanism {
    let (n, p) = (y.len(), rows[0].len());
    let gram: Vec<Vec<f64>> = (0..p)
        .map(|a| (0..p).map(|b| rows.iter().map(|r| r[a] * r[b]).sum::<f64>()).collect())
        .collect();
    let xty: Vec<f64> =
        (0..p).map(|a| rows.iter().zip(y).map(|(r, y)| r[a] * y).sum::<f64>()).collect();
    let mut precision = gram.clone();
    for (i, row) in precision.iter_mut().enumerate() {
        row[i] += 1.0 / (scale * scale);
    }
    let variance = inverse(&precision);
    let mean: Vec<f64> =
        (0..p).map(|a| (0..p).map(|b| variance[a][b] * xty[b]).sum::<f64>()).collect();
    let yy = y.iter().map(|v| v * v).sum::<f64>();
    let fitted = (0..p).map(|a| mean[a] * (0..p).map(|b| precision[a][b] * mean[b]).sum::<f64>());
    let shape = 1e-3 + n as f64 / 2.0;
    let rate = 1e-3 + (yy - fitted.sum::<f64>()) / 2.0;
    let factor = rate / (shape - 1.0);
    Mechanism {
        mean,
        covariance: variance.iter().map(|r| r.iter().map(|v| v * factor).collect()).collect(),
        least_squares: solve(gram, xty),
    }
}

struct Closed {
    mediated_mean: f64,
    mediated_sd: f64,
    direct_mean: f64,
    direct_sd: f64,
    /// Least-squares mediated and direct estimates.
    mediated_ols: f64,
    direct_ols: f64,
    /// Bound on |posterior mean - least squares| of the mediated quantity from the prior pull.
    mediated_pull: f64,
}

fn closed_form(n: usize, delta: f64) -> Closed {
    let (t, m, y) = law(n);
    let mediator = conjugate(&t.iter().map(|&t| vec![1.0, t]).collect::<Vec<_>>(), &m, PRIOR_SCALE);
    let outcome =
        conjugate(&(0..n).map(|i| vec![1.0, t[i], m[i]]).collect::<Vec<_>>(), &y, PRIOR_SCALE);
    let (ea, va) = (mediator.mean[1], mediator.covariance[1][1]);
    let (ec, vc) = (outcome.mean[1], outcome.covariance[1][1]);
    let (eb, vb) = (outcome.mean[2], outcome.covariance[2][2]);
    let var_ab = va * vb + va * eb * eb + vb * ea * ea;
    // The prior pull on a coefficient vector is (X'X + I/c0^2)^-1 (I/c0^2) beta_ols, exact; the
    // product a b moves by at most |da||b_ols| + |a_ols||db| + |da||db|.
    let pull = |mechanism: &Mechanism, index: usize| {
        (mechanism.mean[index] - mechanism.least_squares[index]).abs()
    };
    let (da, db) = (pull(&mediator, 1), pull(&outcome, 2));
    let (a_ols, b_ols) = (mediator.least_squares[1], outcome.least_squares[2]);
    Closed {
        mediated_mean: delta * ea * eb,
        mediated_sd: delta * var_ab.sqrt(),
        direct_mean: delta * ec,
        direct_sd: delta * vc.sqrt(),
        mediated_ols: delta * a_ols * b_ols,
        direct_ols: delta * outcome.least_squares[1],
        mediated_pull: delta * (da * b_ols.abs() + a_ols.abs() * db + da * db),
    }
}

fn graph() -> Dag {
    let mut dag = Dag::with_variables(3);
    for (s, d) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(d)).unwrap();
    }
    dag
}

struct Fitted {
    effect: f64,
    sd: f64,
}

fn route(n: usize, contrast: MediationContrast, inference: InferenceMode) -> Fitted {
    let (t, m, y) = law(n);
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
    query.control = Intervention::set(v(0), Value::f64(0.0));
    query.active = Intervention::set(v(0), Value::f64(TREATED));
    let ctx = ExecutionContext::for_tests(41);
    let study = Study::tabular(data.clone())
        .graph(graph())
        .query(CausalQuery::Mediation(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    let sd = result.posterior.as_ref().map_or(f64::NAN, |post| {
        let column = post.effect_column().expect("effect column");
        post.summaries.sd[column]
    });
    Fitted { effect: result.effect(), sd }
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(PRIOR_SCALE))
}

#[test]
fn the_bayesian_mediation_posterior_equals_the_closed_form_prior_shrunk_value_and_sd() {
    let root = (DRAWS as f64).sqrt();
    for n in [12_usize, 48, 192] {
        let closed = closed_form(n, TREATED);
        for (contrast, mean, sd) in [
            (MediationContrast::NaturalIndirect, closed.mediated_mean, closed.mediated_sd),
            (MediationContrast::NaturalDirect, closed.direct_mean, closed.direct_sd),
        ] {
            let fitted = route(n, contrast, bayesian());
            assert!(
                (fitted.effect - mean).abs() < 5.0 * sd / root,
                "n={n} {contrast:?}: posterior mean {} vs closed form {mean} (mc sd {})",
                fitted.effect,
                sd / root
            );
            assert!(
                (fitted.sd / sd - 1.0).abs() < 8.0 / root,
                "n={n} {contrast:?}: posterior sd {} vs closed form {sd}",
                fitted.sd
            );
        }
    }
}

#[test]
fn the_bayesian_and_frequentist_mediation_differ_by_at_most_the_prior_pull_and_converge_with_n() {
    let root = (DRAWS as f64).sqrt();
    let mut gaps = Vec::new();
    for n in [12_usize, 48, 192] {
        let closed = closed_form(n, TREATED);
        let bayes = route(n, MediationContrast::NaturalIndirect, bayesian());
        let freq = route(n, MediationContrast::NaturalIndirect, InferenceMode::Frequentist);
        // The frequentist route is the least-squares product of the two mechanisms.
        assert!(
            (freq.effect - closed.mediated_ols).abs() < 1e-9,
            "n={n}: frequentist {} vs least squares {}",
            freq.effect,
            closed.mediated_ols
        );
        let gap = (bayes.effect - freq.effect).abs();
        let monte_carlo = 5.0 * closed.mediated_sd / root;
        assert!(
            gap <= closed.mediated_pull + monte_carlo,
            "n={n}: gap {gap} exceeds the prior pull {} (+ mc {monte_carlo})",
            closed.mediated_pull
        );
        // The prior pulls toward zero: the posterior magnitude is not larger.
        assert!(bayes.effect.abs() <= freq.effect.abs() + monte_carlo, "n={n}");
        let direct_bayes = route(n, MediationContrast::NaturalDirect, bayesian());
        let direct_freq = route(n, MediationContrast::NaturalDirect, InferenceMode::Frequentist);
        assert!((direct_freq.effect - closed.direct_ols).abs() < 1e-9, "n={n}");
        gaps.push((gap, (direct_bayes.effect - direct_freq.effect).abs(), closed.mediated_pull));
    }
    // The closed-form pull shrinks with the sample, and the observed gaps follow it down.
    assert!(gaps[0].2 > gaps[1].2 && gaps[1].2 > gaps[2].2, "{gaps:?}");
    assert!(gaps[2].2 < 0.25 * gaps[0].2, "{gaps:?}");
    assert!(gaps[2].0 < gaps[0].0 && gaps[2].1 < gaps[0].1, "{gaps:?}");
}
