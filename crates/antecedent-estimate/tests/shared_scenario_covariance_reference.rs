//! Shared-row scenario covariance against exact multinomial enumeration in rationals.
//!
//! `oracle_data/shared_scenario_covariance_reference.py` (frozen in the JSON beside it) lists every
//! count vector of n draws from n rows with its exact probability n!/(prod c_i!)/n^n, drops a
//! vector on which any estimator is undefined for all estimators, renormalizes, and writes the exact
//! mean, covariance and plug-in entry sampling variance. Designs cover ties, two rows, an exactly
//! collinear pair with an extreme score, and unbalanced strata where most vectors fail. This test
//! rebuilds the estimators from the same integer data and requires the exact-enumeration routine
//! to reproduce every figure and the dropped mass. It also checks the weighted moment routine on
//! fixed replicate matrices (including the R/(R-1) correction and a zero weight), and the bootstrap
//! path: the count vectors it hands to the estimators are recorded and the covariance it reports is
//! recomputed from them in exact integer arithmetic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "tiny integer fixtures"
)]

use std::sync::{Arc, Mutex};

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::scenario_covariance::{
    ExactEnumerationOptions, RowDependence, ScenarioRowEstimator, SharedRowBootstrapOptions,
    covariance_estimator_moments, exact_enumeration_covariance, shared_row_bootstrap_covariance,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/shared_scenario_covariance_reference.json"))
        .unwrap()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn near(got: f64, want: f64, scale: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-11 * (want.abs() + scale) + 1e-300,
        "{what}: {got} vs {want} (scale {scale})"
    );
}

fn units(n: usize) -> Vec<Arc<str>> {
    (0..n)
        .map(|i| {
            let id: Arc<str> = Arc::from(format!("u{i}"));
            id
        })
        .collect()
}

fn estimator(idx: usize, data: &Value, spec: &Value, n: usize) -> ScenarioRowEstimator {
    let kind = spec["kind"].as_str().unwrap().to_owned();
    let key = spec["key"].as_str().map(str::to_owned);
    let get = |name: &str| ints(&data[name]);
    let (a, x, z, s, t, y) = match kind.as_str() {
        "linear" => (get(key.as_deref().unwrap()), vec![], vec![], vec![], vec![], vec![]),
        "ratio" => (vec![], get("x"), get("z"), vec![], vec![], vec![]),
        _ => (vec![], vec![], vec![], get("s"), get("t"), get("y")),
    };
    ScenarioRowEstimator::new(format!("s{idx}"), "snap", units(n), move |c: &[u32]| {
        let total: i64 = c.iter().map(|&v| i64::from(v)).sum();
        match kind.as_str() {
            "linear" => {
                let sum: i64 = c.iter().zip(&a).map(|(&v, &w)| i64::from(v) * w).sum();
                Ok(sum as f64 / total as f64)
            }
            "ratio" => {
                let num: i64 = c.iter().zip(&x).map(|(&v, &w)| i64::from(v) * w).sum();
                let den: i64 = c.iter().zip(&z).map(|(&v, &w)| i64::from(v) * w).sum();
                if den == 0 {
                    Err(EstimationError::data_msg("undefined ratio"))
                } else {
                    Ok(num as f64 / den as f64)
                }
            }
            _ => {
                let mut acc = 0.0;
                let mut strata: Vec<i64> = s.clone();
                strata.sort_unstable();
                strata.dedup();
                for stratum in strata {
                    let n_s: i64 =
                        (0..c.len()).filter(|&i| s[i] == stratum).map(|i| i64::from(c[i])).sum();
                    if n_s == 0 {
                        continue;
                    }
                    let arm = |want: i64| {
                        let (mut sum, mut cnt) = (0_i64, 0_i64);
                        for i in 0..c.len() {
                            if s[i] == stratum && t[i] == want {
                                sum += i64::from(c[i]) * y[i];
                                cnt += i64::from(c[i]);
                            }
                        }
                        (cnt > 0).then(|| sum as f64 / cnt as f64)
                    };
                    let (Some(m1), Some(m0)) = (arm(1), arm(0)) else {
                        return Err(EstimationError::data_msg("stratum lacks an arm"));
                    };
                    acc += n_s as f64 / total as f64 * (m1 - m0);
                }
                Ok(acc)
            }
        }
    })
}

#[test]
fn exact_enumeration_matches_rational_reference_with_joint_dropping() {
    let reference = reference();
    let designs = reference["designs"].as_array().unwrap();
    assert_eq!(designs.len(), 4);
    for design in designs {
        let name = design["name"].as_str().unwrap();
        let n = design["n"].as_u64().unwrap() as usize;
        let scenarios: Vec<ScenarioRowEstimator> = design["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| estimator(i, &design["data"], s, n))
            .collect();
        let k = scenarios.len();
        let options = ExactEnumerationOptions { max_compositions: 100_000, max_failure_mass: 1.0 };
        let got =
            exact_enumeration_covariance(&scenarios, &options, &ExecutionContext::for_tests(1))
                .unwrap();
        assert_eq!(got.replicates_total, design["count_vectors"].as_u64().unwrap(), "{name}");
        assert_eq!(got.failed_replicates, design["failed_vectors"].as_u64().unwrap(), "{name}");
        near(got.failed_mass, design["failed_mass"]["value"].as_f64().unwrap(), 0.0, name);
        let scale = (0..k)
            .map(|i| design["covariance"][i][i]["value"].as_f64().unwrap().abs())
            .fold(0.0, f64::max);
        for i in 0..k {
            near(got.means[i], design["means"][i]["value"].as_f64().unwrap(), 1.0, name);
            for j in 0..k {
                near(
                    got.entry(i, j),
                    design["covariance"][i][j]["value"].as_f64().unwrap(),
                    1e-3 * scale,
                    &format!("{name} cov[{i}][{j}]"),
                );
                near(
                    got.entry_sampling_variance[i * k + j],
                    design["entry_sampling_variance"][i][j]["value"].as_f64().unwrap(),
                    1e-3 * scale * scale,
                    &format!("{name} var[{i}][{j}]"),
                );
            }
        }
    }
}

#[test]
fn moment_routine_matches_rational_reference() {
    let reference = reference();
    for case in reference["moment_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = case["replicates"].as_array().unwrap();
        let k = rows[0].as_array().unwrap().len();
        let flat: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let weights: Vec<f64> =
            case["weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let (cov, var) = covariance_estimator_moments(
            &flat,
            &weights,
            k,
            case["correction"].as_f64().unwrap(),
            case["n_rows"].as_u64().unwrap() as usize,
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                near(
                    cov[i * k + j],
                    case["covariance"][i][j]["value"].as_f64().unwrap(),
                    1e-6,
                    name,
                );
                near(
                    var[i * k + j],
                    case["entry_sampling_variance"][i][j]["value"].as_f64().unwrap(),
                    1e-6,
                    name,
                );
            }
        }
    }
}

#[test]
fn bootstrap_covariance_equals_the_corrected_covariance_of_the_recorded_replicates() {
    // Integer-valued estimators, so the recomputation below is exact in i128.
    let a = [3_i64, 0, 5, 5, 1, 9];
    let b = [2_i64, 7, 0, 1, 1, 4];
    let n = a.len();
    let seen: Arc<Mutex<Vec<Vec<u32>>>> = Arc::default();
    let record = Arc::clone(&seen);
    let first = ScenarioRowEstimator::new("a", "snap", units(n), move |c: &[u32]| {
        record.lock().unwrap().push(c.to_vec());
        Ok(c.iter().zip(&a).map(|(&v, &w)| f64::from(v) * w as f64).sum())
    });
    let second = ScenarioRowEstimator::new("b", "snap", units(n), move |c: &[u32]| {
        Ok(c.iter().zip(&b).map(|(&v, &w)| f64::from(v) * w as f64).sum())
    })
    .with_dependence(RowDependence::SharedRows);
    let r = 500_usize;
    let options = SharedRowBootstrapOptions { replicates: r, seed: 17, max_failure_fraction: 0.0 };
    let got = shared_row_bootstrap_covariance(
        &[first, second],
        &options,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let counts = seen.lock().unwrap().clone();
    assert_eq!(counts.len(), r);
    let (mut sa, mut sb, mut saa, mut sab, mut sbb) = (0_i128, 0_i128, 0_i128, 0_i128, 0_i128);
    let mut per_row = vec![0_u64; n];
    for c in &counts {
        assert_eq!(c.iter().map(|&v| u64::from(v)).sum::<u64>(), n as u64);
        for (acc, &v) in per_row.iter_mut().zip(c) {
            *acc += u64::from(v);
        }
        let x: i128 = c.iter().zip(&a).map(|(&v, &w)| i128::from(v) * i128::from(w)).sum();
        let y: i128 = c.iter().zip(&b).map(|(&v, &w)| i128::from(v) * i128::from(w)).sum();
        sa += x;
        sb += y;
        saa += x * x;
        sab += x * y;
        sbb += y * y;
    }
    // Sum of (x - mean)(y - mean) = sxy - sx sy / R; covariance = that / (R - 1).
    let rr = r as i128;
    let cov = |sxy: i128, sx: i128, sy: i128| (rr * sxy - sx * sy) as f64 / (rr * (rr - 1)) as f64;
    near(got.entry(0, 0), cov(saa, sa, sa), 1.0, "var a");
    near(got.entry(0, 1), cov(sab, sa, sb), 1.0, "cov ab");
    near(got.entry(1, 1), cov(sbb, sb, sb), 1.0, "var b");
    near(got.means[0], sa as f64 / r as f64, 1.0, "mean a");
    // Each row is drawn with expected count one: 500 replicates give a standard deviation of
    // about 0.04 for the mean count; six of them is a deterministic sanity bound.
    for (i, total) in per_row.iter().enumerate() {
        let mean = *total as f64 / r as f64;
        assert!((mean - 1.0).abs() < 0.25, "row {i} mean count {mean}");
    }
}
