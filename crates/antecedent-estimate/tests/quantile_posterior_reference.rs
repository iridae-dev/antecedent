//! The binned-Dirichlet quantile posterior against its exact distribution.
//!
//! `oracle_data/quantile_posterior_reference.py` (frozen in the JSON beside it) writes the
//! posterior of each arm's quantile as a one-dimensional integral of Beta tails over the
//! bracket in which the piecewise-linear distribution function crosses the level, including
//! the density guard that leaves a draw unresolved. From it come the exact probability that a
//! draw is resolved, the distribution function of the quantile given resolution, its mean and
//! its equal-tailed 95% interval. The product's sampler is compared with that at 200,000 draws,
//! with tolerances of a few times the observed Monte Carlo deviation (about 3 standard errors on the distribution function, 0.012 on the interval ends), under moderate, vague (0.01) and
//! strong (20) bin concentrations and three levels.

use antecedent_estimate::quantile_posterior::randomized_quantile_posterior;
use serde_json::Value;

fn f(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or_else(|| panic!("{key}"))
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn empirical(draws: &[f64], x: f64) -> f64 {
    draws.iter().filter(|d| **d <= x).count() as f64 / draws.len() as f64
}

#[test]
fn arm_quantile_posteriors_match_the_exact_beta_integral() {
    let data: Value = serde_json::from_str(include_str!("oracle_data/quantile_posterior_reference.json"))
        .expect("frozen reference parses");
    for case in data.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let grid = floats(&case["grid"]);
        let mut outcome: Vec<f64> = Vec::new();
        let control = floats(&case["control_data"]);
        let active = floats(&case["active_data"]);
        outcome.extend(&control);
        outcome.extend(&active);
        let mut treated = vec![false; control.len()];
        treated.extend(vec![true; active.len()]);
        let draws = 200_000;
        let post = randomized_quantile_posterior(
            &outcome,
            &treated,
            &grid,
            f(case, "concentration"),
            f(case, "tau"),
            draws,
            20_261_028,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        for (label, arm_draws, cdf_mean) in [
            ("control", &post.control_draws, &post.control_cdf_mean),
            ("treated", &post.treated_draws, &post.treated_cdf_mean),
        ] {
            let want = &case[label];
            for (got, w) in cdf_mean.iter().zip(floats(&want["cdf_mean"])) {
                assert!((got - w).abs() < 1e-12, "{name}/{label}: cdf mean {got} vs {w}");
            }
            for point in want["cdf_at"].as_array().unwrap() {
                let (x, p) = (point[0].as_f64().unwrap(), point[1].as_f64().unwrap());
                let got = empirical(arm_draws, x);
                assert!((got - p).abs() < 0.003, "{name}/{label}: P(Q <= {x}) {got} vs {p}");
            }
            let mean = arm_draws.iter().sum::<f64>() / arm_draws.len() as f64;
            assert!((mean - f(want, "mean")).abs() < 0.003, "{name}/{label}: mean {mean}");
            let mut sorted = arm_draws.clone();
            sorted.sort_by(f64::total_cmp);
            let at = |p: f64| sorted[((sorted.len() as f64) * p) as usize];
            assert!((at(0.025) - f(want, "q025")).abs() < 0.012, "{name}/{label}: q025 {}", at(0.025));
            assert!((at(0.975) - f(want, "q975")).abs() < 0.012, "{name}/{label}: q975 {}", at(0.975));
        }
        // A draw is kept only when both independent arms resolve.
        let both = f(&case["control"], "resolved") * f(&case["treated"], "resolved");
        let share = post.draws_resolved as f64 / draws as f64;
        assert!((share - both).abs() < 0.0004, "{name}: resolved share {share} vs {both}");
    }
}
