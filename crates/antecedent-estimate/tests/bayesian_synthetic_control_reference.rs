//! The Bayesian synthetic-control posterior against simplex quadrature.
//!
//! `oracle_data/bayesian_synthetic_control_reference.py` (frozen in the JSON beside it) writes
//! the three-donor weight posterior with the residual variance integrated out analytically,
//! `prod_j w_j^(alpha - 1) (b + RSS(w) / 2)^-(a + n / 2)`, integrates it over the simplex by
//! adaptive quadrature, and derives the posterior mean and sd of every weight, of the realised
//! average post-period effect (weight-driven gap plus the independent counterfactual errors) and
//! the pre-fit RMSE at the posterior-mean weights. The Metropolis sampler must reproduce them
//! within a tolerance set from its own effective sample size, for a uniform and a concentrated
//! Dirichlet prior. Independent errors only: the AR(1) continuation is not covered here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_synthetic_control::{
    BayesianSyntheticControlSpec, fit_bayesian_synthetic_control,
};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const UNITS: [&str; 4] = ["d1", "d2", "d3", "t"];

fn donors_at(t: i64) -> [f64; 3] {
    let tf = t as f64;
    [
        1.0 + 0.5 * tf + 0.3 * (((t * 5) % 7) as f64 / 3.0 - 1.0),
        3.0 - 0.2 * tf + 0.3 * (((t * 3) % 8) as f64 / 3.5 - 1.0),
        2.0 + 0.4 * ((t * 3) % 5) as f64,
    ]
}

fn panel() -> (Vec<f64>, Vec<String>, Vec<i64>) {
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=11_i64 {
        let d = donors_at(t);
        let treated = 0.6 * d[0]
            + 0.4 * d[1]
            + 0.15 * (((t * 11) % 13) as f64 / 6.0 - 1.0)
            + if t >= 9 { 2.0 } else { 0.0 };
        for (name, value) in UNITS.iter().zip([d[0], d[1], d[2], treated]) {
            y.push(value);
            units.push((*name).to_string());
            periods.push(t);
        }
    }
    (y, units, periods)
}

#[test]
fn weights_effect_and_pre_fit_match_the_simplex_quadrature() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/bayesian_synthetic_control_reference.json"))
            .unwrap();
    let (y, units, periods) = panel();
    for case in cases {
        let alpha = case["alpha"].as_f64().unwrap();
        let spec = BayesianSyntheticControlSpec {
            dirichlet_concentration: alpha,
            ..BayesianSyntheticControlSpec::new("t", 9)
        };
        let fit =
            fit_bayesian_synthetic_control(&y, &units, &periods, &spec, DrawSpec::new(40_000, 21))
                .unwrap();
        let ess = fit.min_bulk_ess;
        assert!(ess >= 100.0 && fit.max_split_rhat < 1.05, "alpha {alpha}: gate");
        let tolerance = |sd: f64| 6.0 * sd / ess.sqrt() + 1e-3;
        let (mean, sd) =
            (case["effect_mean"].as_f64().unwrap(), case["effect_sd"].as_f64().unwrap());
        assert!(
            (fit.mean - mean).abs() < tolerance(sd),
            "alpha {alpha}: effect {} vs {mean}",
            fit.mean
        );
        assert!((fit.sd / sd - 1.0).abs() < 0.05, "alpha {alpha}: effect sd {} vs {sd}", fit.sd);
        for j in 0..3 {
            let (m, s) = (
                case["weight_means"][j].as_f64().unwrap(),
                case["weight_sds"][j].as_f64().unwrap(),
            );
            assert!(
                (fit.donor_weight_means[j].1 - m).abs() < tolerance(s),
                "alpha {alpha}: weight {j} mean {} vs {m}",
                fit.donor_weight_means[j].1
            );
            assert!(
                (fit.donor_weight_sds[j] / s - 1.0).abs() < 0.08,
                "alpha {alpha}: weight {j} sd {} vs {s}",
                fit.donor_weight_sds[j]
            );
        }
        let rmse = case["pre_fit_rmse"].as_f64().unwrap();
        assert!((fit.pre_fit_rmse - rmse).abs() < 2e-3 + 0.02 * rmse, "alpha {alpha}: rmse");
    }
}
