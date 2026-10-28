//! The scalar state-space helpers against the joint Gaussian law in 80-digit arithmetic.
//!
//! `oracle_data/lgssm_joint_gaussian_reference.py` (frozen in the JSON beside it) builds the
//! covariance of the states `x_0, ..., x_{n-1}` and of the observations directly, and obtains
//! every filtered, predicted and smoothed moment, the lag-one smoothed covariance and the
//! per-row predictive log density by Gaussian conditioning. It runs no recursion. The
//! cases cover a stable, a unit-root, an explosive and a negative coefficient, an observation
//! noise 10^7 times the process noise and one 10^-7 times it, and scales from 1e-8 to 1e8.

use antecedent_model::batch::ParentBatch;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::lgssm::{kalman_filter, rts_smooth};
use antecedent_model::mechanism::log_prob_column;
use serde_json::Value;

fn column(case: &Value, name: &str) -> Vec<f64> {
    case[name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn agree(label: &str, case: &str, actual: &[f64], expected: &[f64], floor: f64, relative: f64) {
    assert_eq!(actual.len(), expected.len(), "{case} {label}: length");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= relative * e.abs().max(floor),
            "{case} {label}[{t}]: {a:.17e} vs {e:.17e}"
        );
    }
}

#[test]
fn filter_smoother_and_density_match_the_joint_gaussian() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/lgssm_joint_gaussian_reference.json"))
            .unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let a = case["a"].as_f64().unwrap();
        let q = case["process_std"].as_f64().unwrap();
        let r = case["obs_std"].as_f64().unwrap();
        let m0 = case["initial_mean"].as_f64().unwrap();
        let y = column(case, "observations");
        let n = y.len();
        let ymax = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let variance_floor =
            column(case, "predicted_variance").iter().fold(0.0_f64, |m, v| m.max(*v));

        let (fm, fv, pm, pv) = kalman_filter(&y, a, q * q, r * r, m0, q * q);
        let (sm, sv, lag) = rts_smooth(a, &fm, &fv, &pm, &pv);
        let mean_floor = 1e-6 * ymax;
        agree("filtered_mean", name, &fm, &column(case, "filtered_mean"), mean_floor, 1e-9);
        agree(
            "filtered_variance",
            name,
            &fv,
            &column(case, "filtered_variance"),
            variance_floor,
            1e-9,
        );
        agree("predicted_mean", name, &pm, &column(case, "predicted_mean"), mean_floor, 1e-9);
        agree(
            "predicted_variance",
            name,
            &pv,
            &column(case, "predicted_variance"),
            variance_floor,
            1e-9,
        );
        agree("smoothed_mean", name, &sm, &column(case, "smoothed_mean"), mean_floor, 1e-9);
        agree(
            "smoothed_variance",
            name,
            &sv,
            &column(case, "smoothed_variance"),
            variance_floor,
            1e-9,
        );
        agree(
            "lag_one_covariance",
            name,
            &lag,
            &column(case, "lag_one_covariance"),
            variance_floor,
            1e-9,
        );

        let slot = MechanismSlot::LinearGaussianStateSpace {
            a,
            process_std: q,
            obs_std: r,
            initial_mean: m0,
        };
        let mut logp = vec![0.0; n];
        log_prob_column(&slot, &y, ParentBatch { values: &[], n_rows: n, n_parents: 0 }, &mut logp)
            .unwrap();
        agree("row_log_density", name, &logp, &column(case, "row_log_density"), 1.0, 1e-9);
    }
}
