//! The Bayesian augmented two-period panel contrast through the public study facade.
//!
//! The posterior is compared with the arm-specific least-squares reference written out here
//! (each arm's change score regressed on a constant and the supplied propensity and
//! untreated-change columns; the contrast is the treated mean change minus the comparison
//! regression evaluated at the treated covariate means), against the known effect of the
//! simulated panel, and at its boundaries: a thin arm, declared sampling weights and a
//! frequentist-only declaration are refused with the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

const EFFECT: f64 = 1.5;

struct Panel {
    pre: Vec<f64>,
    post: Vec<f64>,
    propensity: Vec<f64>,
    prediction: Vec<f64>,
    treated: Vec<bool>,
}

fn panel(units: usize, treated_every: usize) -> Panel {
    let mut p = Panel {
        pre: vec![],
        post: vec![],
        propensity: vec![],
        prediction: vec![],
        treated: vec![],
    };
    for i in 0..units {
        let x = ((i * 37 % 101) as f64) / 100.0;
        let m = ((i * 53 % 89) as f64) / 89.0;
        let d = i % treated_every == 0;
        let base = ((i as f64) * 0.77).sin();
        let noise = 0.3 * ((i as f64) * 2.31).cos();
        p.pre.push(base);
        p.post.push(base + 0.5 + 1.2 * m - 0.8 * x + if d { EFFECT } else { 0.0 } + noise);
        p.propensity.push(x);
        p.prediction.push(m);
        p.treated.push(d);
    }
    p
}

fn table(p: &Panel) -> TabularData {
    TabularData::from_f64_columns([
        ("pre", p.pre.as_slice()),
        ("post", p.post.as_slice()),
        ("p", p.propensity.as_slice()),
        ("m0", p.prediction.as_slice()),
    ])
    .unwrap()
}

fn query(p: &Panel) -> PanelDidQuery {
    let ids: Vec<Arc<str>> = (0..p.pre.len()).map(|i| Arc::<str>::from(format!("s{i}"))).collect();
    PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        p.treated.clone(),
        ids.clone(),
        ids,
        true,
    )
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(100.0).n_draws(20_000))
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

/// Treated mean change minus the comparison-arm least-squares fit at the treated covariate means.
fn reference(p: &Panel) -> f64 {
    let rows = |arm: bool| -> Vec<usize> { (0..p.pre.len()).filter(|&i| p.treated[i] == arm).collect() };
    let design = |i: usize| [1.0, p.propensity[i], p.prediction[i]];
    let comparison = rows(false);
    let mut xtx = vec![vec![0.0; 3]; 3];
    let mut xty = vec![0.0; 3];
    for &i in &comparison {
        let x = design(i);
        let change = p.post[i] - p.pre[i];
        for a in 0..3 {
            xty[a] += x[a] * change;
            for b in 0..3 {
                xtx[a][b] += x[a] * x[b];
            }
        }
    }
    let beta = solve(xtx, xty);
    let treated = rows(true);
    let count = treated.len() as f64;
    let mean_change = treated.iter().map(|&i| p.post[i] - p.pre[i]).sum::<f64>() / count;
    let mean_x = [
        1.0,
        treated.iter().map(|&i| p.propensity[i]).sum::<f64>() / count,
        treated.iter().map(|&i| p.prediction[i]).sum::<f64>() / count,
    ];
    mean_change - (0..3).map(|c| beta[c] * mean_x[c]).sum::<f64>()
}

#[test]
fn the_bayesian_augmented_panel_posterior_matches_the_arm_regression_reference_and_the_truth() {
    let p = panel(240, 3);
    let data = table(&p);
    let ctx = ExecutionContext::for_tests(31);
    let result = Study::tabular(data.clone())
        .query(query(&p))
        .inference(bayes())
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.panel_did"));
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    let expected = reference(&p);
    assert!((mean - expected).abs() < 0.02, "posterior mean {mean} vs reference {expected}");
    assert!((mean - EFFECT).abs() < 0.15, "posterior mean {mean} vs truth {EFFECT}");
    assert!(sd > 0.0 && sd < 0.2, "posterior sd {sd}");
    assert!(lower < EFFECT && EFFECT < upper, "[{lower}, {upper}]");
    assert_eq!(result.support_status(), None, "the Bayesian design route is unlicensed");
}

#[test]
fn a_thin_arm_and_declared_weights_are_refused_with_their_next_attempts() {
    let ctx = ExecutionContext::for_tests(32);
    // Two treated units cannot support a constant and two covariates.
    let thin = panel(60, 30);
    let error = Study::tabular(table(&thin))
        .query(query(&thin))
        .inference(bayes())
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err()
        .to_string();
    assert!(error.contains("bayesian_did_arm_support"), "{error}");
    assert!(error.contains("Next attempt"), "{error}");

    let p = panel(120, 3);
    let weights: Vec<f64> = (0..p.pre.len()).map(|i| 1.0 + (i % 3) as f64).collect();
    let data = table(&p).with_weights(Arc::from(weights)).unwrap();
    let error = Study::tabular(data)
        .query(query(&p))
        .inference(bayes())
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err()
        .to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
    assert!(error.contains("Next attempt"), "{error}");
}
