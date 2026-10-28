//! The conjugate normal EVSI against 60-digit quadrature of the preposterior value.
//!
//! The reference (`oracle_data/evsi_conjugate_reference.py`, frozen in the JSON beside it)
//! integrates the best posterior expected utility against the marginal density of the sample
//! mean, split at every break-even point, rather than using the preposterior-variance closed
//! form of the product. It covers one to six affine actions, a dominated action, equal slopes,
//! utility offsets of 1e9, information ratios from 1e-12 to 1e12, and a prior at, far from and
//! tightly concentrated around the break-even state.

use antecedent_design::evsi::{
    AffineStateUtility, FutureStudyProgram, NormalMeanFutureData, NormalMeanKnownVarianceUpdate,
    NormalStateBelief, StudyAction,
};
use serde_json::Value;

fn reference() -> Vec<Value> {
    let text = include_str!("oracle_data/evsi_conjugate_reference.json");
    serde_json::from_str::<Value>(text).expect("frozen reference parses").as_array().cloned().expect("array")
}

fn f(value: &Value, key: &str) -> f64 {
    value[key].as_f64().expect("number")
}

fn floats(value: &Value, key: &str) -> Vec<f64> {
    value[key].as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn program(case: &Value) -> FutureStudyProgram {
    let intercept = floats(case, "intercept");
    let slope = floats(case, "slope");
    let actions = (0..intercept.len())
        .map(|i| StudyAction::try_new(&format!("a{i}")).expect("action"))
        .collect();
    FutureStudyProgram::try_new(
        NormalStateBelief::try_new(f(case, "prior_mean"), f(case, "prior_variance")).expect("belief"),
        NormalMeanFutureData::try_new(case["sample_size"].as_u64().expect("n"), f(case, "noise_variance"))
            .expect("data"),
        NormalMeanKnownVarianceUpdate,
        actions,
        AffineStateUtility::try_new(intercept, slope).expect("utility"),
        None,
    )
    .expect("program")
}

#[test]
fn gross_evsi_matches_high_precision_quadrature() {
    for case in reference() {
        let name = case["name"].as_str().expect("name");
        let value = program(&case).evaluate_future_study().expect("evsi");
        let want = f(&case, "gross_evsi");
        // Relative 1e-9 of the value, with an absolute floor tied to the utility scale.
        let scale = floats(&case, "intercept").iter().fold(1.0_f64, |m, a| m.max(a.abs()));
        let tol = 1e-9 * want.abs() + 1e-15 * scale + 1e-300;
        assert!(
            (value.gross_evsi - want).abs() <= tol,
            "{name}: evsi {} vs reference {want} (tol {tol:e})",
            value.gross_evsi
        );
        assert!(value.net_value.is_none(), "{name}: no cost declared");
        let pre = value.preposterior_expected_utility;
        let want_pre = f(&case, "preposterior_expected_utility");
        assert!((pre - want_pre).abs() <= 1e-9 * want_pre.abs().max(1.0), "{name}: {pre} vs {want_pre}");
        assert!(
            (value.prior_expected_utility - f(&case, "prior_expected_utility")).abs()
                <= 1e-12 * scale,
            "{name}"
        );
        // EVSI never exceeds perfect information on the same belief.
        assert!(value.gross_evsi <= f(&case, "evpi") * (1.0 + 1e-9) + 1e-15 * scale, "{name}");
    }
}

#[test]
fn updated_decision_matches_the_exact_conjugate_posterior() {
    for case in reference() {
        let name = case["name"].as_str().expect("name");
        let prog = program(&case);
        let (mu0, tau2) = (f(&case, "prior_mean"), f(&case, "prior_variance"));
        let (n, sigma2) = (case["sample_size"].as_u64().expect("n") as f64, f(&case, "noise_variance"));
        for ybar in [-3.0_f64, -0.25, 0.0, 0.5, 4.0] {
            let prec = 1.0 / tau2 + n / sigma2;
            let mean = (mu0 / tau2 + n * ybar / sigma2) / prec;
            let got = prog.updated_decision(ybar).expect("update");
            assert!((got.posterior_mean - mean).abs() <= 1e-12 * mean.abs().max(1.0), "{name}");
            assert!((got.posterior_variance - 1.0 / prec).abs() <= 1e-12 / prec, "{name}");
            let alpha = floats(&case, "intercept");
            let beta = floats(&case, "slope");
            let best = alpha
                .iter()
                .zip(&beta)
                .map(|(a, b)| a + b * mean)
                .fold(f64::NEG_INFINITY, f64::max);
            let scale = alpha.iter().fold(1.0_f64, |m, a| m.max(a.abs()));
            assert!((got.expected_utility - best).abs() <= 1e-9 * scale, "{name}");
        }
    }
}

#[test]
fn net_value_subtracts_the_declared_cost_only() {
    let case = &reference()[0];
    let base = program(case);
    let with_cost = FutureStudyProgram::try_new(
        NormalStateBelief::try_new(f(case, "prior_mean"), f(case, "prior_variance")).expect("belief"),
        NormalMeanFutureData::try_new(4, 12.0).expect("data"),
        NormalMeanKnownVarianceUpdate,
        vec![StudyAction::try_new("a0").expect("a"), StudyAction::try_new("a1").expect("a")],
        AffineStateUtility::try_new(vec![0.0, 0.0], vec![0.0, 1.0]).expect("u"),
        Some(0.01),
    )
    .expect("program");
    let gross = base.evaluate_future_study().expect("gross").gross_evsi;
    let net = with_cost.evaluate_future_study().expect("net");
    assert!((net.gross_evsi - gross).abs() < 1e-15);
    assert!((net.net_value.expect("net") - (gross - 0.01)).abs() < 1e-15);
}
