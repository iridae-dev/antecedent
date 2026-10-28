//! Whole-program randomization, finite-carryover history means and the Gaussian AR(1) history
//! posterior against exact-rational references.
//!
//! `oracle_data/joint_design_reference.py` (frozen in the JSON beside it) computes, with every
//! binary64 input as an exact fraction, the Horvitz-Thompson program means with the
//! estimating-equation sandwich of independent units; the history-inclusion probabilities by
//! enumerating the candidate schedules, the history means and the sequence-cluster sandwich; and
//! the Gaussian posterior of the history means under proper independent priors and a known
//! stationary AR(1) residual, by building and inverting each sequence's full error covariance
//! rather than the tridiagonal precision the product assembles. Posterior draws are compared by
//! their sample moments within five Monte Carlo standard errors. Designs cover unequal known
//! probabilities and weights, an outcome offset of 1e6, the smallest admissible sample, unequal
//! sequence lengths and a carryover of two. This reviews the randomized and switchback owners
//! only; the longitudinal, instrumental, rollout and synthetic owners of the module are not
//! covered here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_design::randomized::{
    JointRandomizedDesign, JointRandomizedUnit, randomized_joint_means,
};
use antecedent_estimate::joint_design::switchback::{
    JointSwitchbackDesign, SwitchbackGaussianPrior, switchback_joint_means,
    switchback_joint_posterior,
};
use antecedent_estimate::switchback::HistoryAssignment;
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/joint_design_reference.json")).unwrap()
        ["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, scale: f64, what: &str) {
    assert!((got - want).abs() <= rel * (want.abs() + scale), "{what}: got {got} want {want}");
}

fn flat(m: &Value) -> Vec<f64> {
    m.as_array().unwrap().iter().flat_map(floats).collect()
}

fn design(k: usize) -> JointRandomizedDesign {
    JointRandomizedDesign {
        worlds: (0..k).map(|j| format!("program{j}")).collect(),
        sampling_unit: "geo".into(),
        population: "fixed census".into(),
        consistency: "complete program including history".into(),
        between_unit_isolation: "no spillovers across geos".into(),
    }
}

#[test]
fn randomized_means_and_sandwich_match_the_exact_reference() {
    let mut seen = 0;
    for case in cases() {
        if case["kind"] != "randomized" {
            continue;
        }
        seen += 1;
        let name = case["name"].as_str().unwrap();
        let k = case["k"].as_u64().unwrap() as usize;
        let units: Vec<JointRandomizedUnit> = case["units"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| JointRandomizedUnit {
                id: u["id"].as_str().unwrap().into(),
                arm: u["arm"].as_u64().unwrap() as usize,
                probabilities: floats(&u["probabilities"]),
                outcome: u["outcome"].as_f64().unwrap(),
                population_weight: u["population_weight"].as_f64().unwrap(),
            })
            .collect();
        let got = randomized_joint_means(&design(k), &units).unwrap();
        let means = floats(&case["means"]);
        let covariance = flat(&case["covariance"]);
        let scale = if name.contains("offset") { 1e6 * 1e6 * 1e-9 } else { 0.0 };
        for j in 0..k {
            close(got.means[j], means[j], 1e-12, 0.0, &format!("{name} mean {j}"));
            for l in 0..k {
                close(
                    got.covariance[j * k + l],
                    covariance[j * k + l],
                    1e-9,
                    scale,
                    &format!("{name} cov {j},{l}"),
                );
            }
        }
        // a prespecified contrast assembled from the joint covariance, not marginal intervals
        let mut c = vec![0.0; k];
        c[0] = 1.0;
        c[k - 1] = -1.0;
        let (mean, variance) = got.contrast(&c).unwrap();
        close(mean, means[0] - means[k - 1], 1e-9, 1e-6, name);
        let want = covariance[0] - 2.0 * covariance[k - 1] + covariance[(k - 1) * k + k - 1];
        close(variance, want, 1e-8, scale, name);
    }
    assert_eq!(seen, 3);
}

#[test]
fn unsupported_programs_and_bad_probabilities_are_refused() {
    let unit = |id: &str, arm: usize, p: [f64; 2]| JointRandomizedUnit {
        id: id.into(),
        arm,
        probabilities: p.to_vec(),
        outcome: 1.0,
        population_weight: 1.0,
    };
    let ok = [unit("a", 0, [0.5, 0.5]), unit("b", 1, [0.5, 0.5])];
    assert!(randomized_joint_means(&design(2), &ok).is_ok());
    let one_arm = [unit("a", 0, [0.5, 0.5]), unit("b", 0, [0.5, 0.5])];
    assert!(randomized_joint_means(&design(2), &one_arm).is_err(), "a program with no unit");
    let bad_sum = [unit("a", 0, [0.5, 0.4]), unit("b", 1, [0.5, 0.5])];
    assert!(randomized_joint_means(&design(2), &bad_sum).is_err());
    let duplicate = [unit("a", 0, [0.5, 0.5]), unit("a", 1, [0.5, 0.5])];
    assert!(randomized_joint_means(&design(2), &duplicate).is_err());
    let mut weightless = ok.clone();
    weightless[0].population_weight = 0.0;
    assert!(randomized_joint_means(&design(2), &weightless).is_err());
    let mut unstated = design(2);
    unstated.consistency = " ".into();
    assert!(randomized_joint_means(&unstated, &ok).is_err(), "premises must be stated");
}

fn switchback_design(case: &Value) -> JointSwitchbackDesign {
    let width = case["width"].as_u64().unwrap() as usize;
    JointSwitchbackDesign {
        worlds: case["histories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| {
                let h: Vec<usize> =
                    h.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
                (h.iter().map(ToString::to_string).collect::<String>(), h)
            })
            .collect(),
        carryover: width - 1,
        no_anticipation: "outcomes do not depend on future arms".into(),
        finite_carryover: "washout after the declared order".into(),
        sequence_isolation: "independent sequences".into(),
        population: "eligible periods".into(),
    }
}

fn sequences(case: &Value) -> Vec<HistoryAssignment> {
    case["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| HistoryAssignment {
            schedules: s["schedules"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
                })
                .collect(),
            probabilities: floats(&s["probabilities"]),
            realized: s["realized"].as_u64().unwrap() as usize,
            outcomes: floats(&s["outcomes"]),
        })
        .collect()
}

#[test]
fn history_means_and_sandwich_match_the_enumerated_inclusion_reference() {
    let mut seen = 0;
    for case in cases() {
        if case["kind"] != "histories" {
            continue;
        }
        seen += 1;
        let name = case["name"].as_str().unwrap();
        let got = switchback_joint_means(&switchback_design(&case), &sequences(&case)).unwrap();
        let k = got.means.len();
        let means = floats(&case["means"]);
        let covariance = flat(&case["covariance"]);
        let scale = if name.contains("offset") { 1e5 * 1e5 * 1e-9 } else { 0.0 };
        for j in 0..k {
            close(got.means[j], means[j], 1e-12, 0.0, &format!("{name} mean {j}"));
            for l in 0..k {
                close(
                    got.covariance[j * k + l],
                    covariance[j * k + l],
                    1e-9,
                    scale,
                    &format!("{name} cov {j},{l}"),
                );
            }
        }
    }
    assert_eq!(seen, 2);
}

#[test]
fn the_ar1_history_posterior_has_the_exact_gaussian_moments() {
    let draws = 400_000;
    for case in cases() {
        if case["kind"] != "histories" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let prior = SwitchbackGaussianPrior {
            mean: floats(&case["prior_mean"]),
            variance: floats(&case["prior_variance"]),
            sequence_variance: floats(&case["sequence_variance"]),
            sequence_rho: floats(&case["sequence_rho"]),
            draws,
            seed: 11,
        };
        let posterior = switchback_joint_posterior(
            &switchback_design(&case),
            &sequences(&case),
            &prior,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let k = prior.mean.len();
        let (mean, covariance) =
            (floats(&case["posterior_mean"]), flat(&case["posterior_covariance"]));
        assert_eq!(posterior.draws.len(), draws);
        for j in 0..k {
            let sd = covariance[j * k + j].sqrt();
            let se = sd / (draws as f64).sqrt();
            assert!(
                (posterior.estimate.means[j] - mean[j]).abs() <= 5.0 * se,
                "{name} posterior mean {j}: {} vs {} (5 se = {})",
                posterior.estimate.means[j],
                mean[j],
                5.0 * se
            );
            for l in 0..k {
                let want = covariance[j * k + l];
                let se = ((covariance[j * k + j] * covariance[l * k + l] + want * want)
                    / draws as f64)
                    .sqrt();
                assert!(
                    (posterior.estimate.covariance[j * k + l] - want).abs() <= 5.0 * se,
                    "{name} posterior cov {j},{l}: {} vs {want}",
                    posterior.estimate.covariance[j * k + l]
                );
            }
        }
        // one row is one joint mechanism realisation: every world shares each draw
        assert!(posterior.draws.iter().all(|row| row.len() == k));
    }
}

#[test]
fn a_history_without_a_declared_posterior_coordinate_or_a_singular_prior_is_refused() {
    let case = cases().into_iter().find(|c| c["kind"] == "histories").unwrap();
    let design = switchback_design(&case);
    let sequences = sequences(&case);
    let prior = |scale: f64| SwitchbackGaussianPrior {
        mean: floats(&case["prior_mean"]),
        variance: floats(&case["prior_variance"]).iter().map(|v| v * scale).collect(),
        sequence_variance: floats(&case["sequence_variance"]),
        sequence_rho: floats(&case["sequence_rho"]),
        draws: 8,
        seed: 3,
    };
    let ctx = ExecutionContext::for_tests(1);
    assert!(switchback_joint_posterior(&design, &sequences, &prior(1.0), &ctx).is_ok());
    assert!(switchback_joint_posterior(&design, &sequences, &prior(0.0), &ctx).is_err());
    let mut wrong_rho = prior(1.0);
    wrong_rho.sequence_rho[0] = 1.0;
    assert!(switchback_joint_posterior(&design, &sequences, &wrong_rho, &ctx).is_err());
    let mut few = prior(1.0);
    few.draws = 1;
    assert!(switchback_joint_posterior(&design, &sequences, &few, &ctx).is_err());
}
