//! Bayesian Gaussian response levels against the exact Normal-Inverse-Gamma posterior.
//!
//! `oracle_data/estimate_response_bayesian_reference.py` (frozen in the JSON beside it) forms, in
//! 40-digit arithmetic, the posterior of a linear functional `w'b` of an adjusted main-effects
//! outcome model: Student-t with `2 a_n` degrees of freedom, location `w'm_n` and scale
//! `(b_n / a_n) w'V_n w`, under an isotropic `N(0, s2 c0^2)` coefficient prior and the weakly
//! informative Inverse-Gamma(1e-3, 1e-3) residual prior. A response level is that functional with
//! every column but the treatment at its sample mean, so a mean-curve coordinate, a set, shift,
//! Bernoulli, Gaussian (its mean) or categorical (its mean index) policy and a joint policy on two
//! treatments each have a closed-form posterior. As in the neighbouring Gaussian g-computation
//! review the draws are Monte Carlo: the mean is held to five Monte Carlo standard errors, the
//! standard deviation to 8 / sqrt(draws) relative and the interval ends to 3 percent of the
//! posterior standard deviation. The fixed-grid credible band the estimator builds from its retained
//! draws is recomputed from those draws (maximum standardized deviation, the `ceil(level (B + 1))`-th
//! smallest, floored at the pointwise interval).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ContinuousDomain, ExecutionContext, GridSpec, IdentificationStatus,
    Intervention, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty,
    ResponseValue, StochasticPolicy, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::BayesianGComputationAte;
use antecedent_estimate::response::ContinuousResponseEstimator;
use serde_json::Value as Json;

const DRAWS: usize = 200_000;

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn intervention(variable: VariableId, policy: &Json) -> Intervention {
    let f = |key: &str| policy[key].as_f64().unwrap();
    match policy["kind"].as_str().unwrap() {
        "set" => Intervention::set(variable, Value::f64(f("value"))),
        "shift" => Intervention::Shift { variable, delta: Value::f64(f("delta")) },
        "bernoulli" => {
            Intervention::Stochastic { variable, policy: StochasticPolicy::Bernoulli { p: f("p") } }
        }
        "categorical" => Intervention::Stochastic {
            variable,
            policy: StochasticPolicy::Categorical {
                probs: policy["probs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            },
        },
        _ => Intervention::Stochastic {
            variable,
            policy: StochasticPolicy::Gaussian { mean: f("mean"), variance: f("var") },
        },
    }
}

fn check(name: &str, mean: f64, sd: f64, lower: f64, upper: f64, want: &Json) {
    let (r_mean, r_sd) = (want["mean"].as_f64().unwrap(), want["sd"].as_f64().unwrap());
    let root = (DRAWS as f64).sqrt();
    assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "{name}: mean {mean} vs {r_mean}");
    assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "{name}: sd {sd} vs {r_sd}");
    assert!((lower - want["lower"].as_f64().unwrap()).abs() < 0.03 * r_sd, "{name}: lower {lower}");
    assert!((upper - want["upper"].as_f64().unwrap()).abs() < 0.03 * r_sd, "{name}: upper {upper}");
}

#[test]
fn response_levels_and_credible_intervals_match_the_exact_posterior() {
    let frozen: Json =
        serde_json::from_str(include_str!("oracle_data/estimate_response_bayesian_reference.json"))
            .unwrap();
    for case in frozen.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let two = case["two_treatments"].as_bool().unwrap();
        let (a, b, z1, z2, y) = (
            floats(&case["a"]),
            floats(&case["b"]),
            floats(&case["z1"]),
            floats(&case["z2"]),
            floats(&case["y"]),
        );
        let mut columns: Vec<(&str, &[f64])> = vec![("a", a.as_slice())];
        if two {
            columns.push(("b", b.as_slice()));
        }
        columns.push(("z1", z1.as_slice()));
        columns.push(("z2", z2.as_slice()));
        columns.push(("y", y.as_slice()));
        let data = TabularData::from_f64_columns(columns.iter().copied()).unwrap();
        let t = if two { 2 } else { 1 };
        let id = |i: usize| VariableId::from_raw(u32::try_from(i).unwrap());
        let adjusters = [id(t), id(t + 1)];
        let outcome = id(t + 2);
        let mut estimator = ContinuousResponseEstimator::new(adjusters);
        estimator.options.bandwidth = Some(0.5);
        let bayes = BayesianGComputationAte::conjugate()
            .with_n_draws(DRAWS)
            .with_seed(17)
            .with_prior_scale(case["prior_scale"].as_f64().unwrap());
        let ctx = ExecutionContext::for_tests(3);
        let run = |estimator: &ContinuousResponseEstimator, functional: ResponseFunctional| {
            estimator
                .estimate_bayesian(
                    &data,
                    &ResponseQuery::new(functional),
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &bayes,
                    &ctx,
                )
                .unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let results = case["results"].as_array().unwrap();
        // Mean-curve coordinates are one query.
        let curve: Vec<&Json> = results.iter().filter(|r| r["query"]["kind"] == "curve").collect();
        if !curve.is_empty() {
            let grid: Vec<f64> =
                curve.iter().map(|r| r["query"]["grid_point"].as_f64().unwrap()).collect();
            let functional = ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from(grid.clone()))),
            };
            let response = run(&estimator, functional.clone());
            let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
                &response.estimate
            else {
                panic!("{name}: a mean curve is a surface");
            };
            let ResponseUncertainty::PointwiseBand { lower, upper, draws, .. } =
                &response.uncertainty
            else {
                panic!("{name}: a pointwise band");
            };
            let draws = draws.as_ref().expect("retained draws");
            for (g, want) in curve.iter().enumerate() {
                let column = draws.column(g).unwrap();
                let m = column.iter().sum::<f64>() / column.len() as f64;
                let sd = (column.iter().map(|v| (v - m).powi(2)).sum::<f64>()
                    / (column.len() - 1) as f64)
                    .sqrt();
                check(&format!("{name} curve {}", grid[g]), mean[g], sd, lower[g], upper[g], want);
            }
            if name == "curve_and_policies" {
                // The simultaneous band is the maximum standardized deviation of the retained draws.
                let mut banded = estimator.clone();
                banded.options.simultaneous_replicates = Some(200);
                let joint = run(&banded, functional);
                let ResponseUncertainty::SimultaneousBand {
                    lower: lo, upper: hi, replicates, ..
                } = &joint.uncertainty
                else {
                    panic!("{name}: a simultaneous band");
                };
                assert_eq!(*replicates as usize, DRAWS);
                let sds: Vec<f64> = (0..grid.len())
                    .map(|g| {
                        let c = draws.column(g).unwrap();
                        let m = c.iter().sum::<f64>() / c.len() as f64;
                        (c.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (c.len() - 1) as f64)
                            .sqrt()
                    })
                    .collect();
                let mut maxima: Vec<f64> = (0..DRAWS)
                    .map(|d| {
                        (0..grid.len())
                            .map(|g| (draws.column(g).unwrap()[d] - mean[g]).abs() / sds[g])
                            .fold(0.0, f64::max)
                    })
                    .collect();
                maxima.sort_by(f64::total_cmp);
                let rank = ((0.95 * (DRAWS as f64 + 1.0) - 1e-9).ceil().max(1.0)) as usize;
                let mut critical = maxima[rank - 1];
                for g in 0..grid.len() {
                    critical = critical
                        .max((mean[g] - lower[g]) / sds[g])
                        .max((upper[g] - mean[g]) / sds[g]);
                }
                for g in 0..grid.len() {
                    // The product's standard deviations are its own; agree to the draw resolution.
                    assert!(
                        (lo[g] - (mean[g] - critical * sds[g])).abs() < 1e-3 * sds[g],
                        "{name} lo {g}"
                    );
                    assert!(
                        (hi[g] - (mean[g] + critical * sds[g])).abs() < 1e-3 * sds[g],
                        "{name} hi {g}"
                    );
                    assert!(
                        lo[g] <= lower[g] && hi[g] >= upper[g],
                        "{name}: nested in the pointwise band"
                    );
                }
            }
        }
        for want in results.iter().filter(|r| r["query"]["kind"] != "curve") {
            let policies = want["query"]["policy"].as_array().unwrap();
            let interventions: Vec<Intervention> = policies
                .iter()
                .enumerate()
                .map(|(i, policy)| intervention(id(i), policy))
                .collect();
            let kind = want["query"]["kind"].as_str().unwrap();
            let response = run(
                &estimator,
                ResponseFunctional::InterventionResponse {
                    outcome,
                    interventions: Arc::from(interventions),
                },
            );
            let ResponseIdentification::PointIdentified(ResponseValue::Scalar(mean)) =
                &response.estimate
            else {
                panic!("{name} {kind}: a scalar");
            };
            let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } =
                &response.uncertainty
            else {
                panic!("{name} {kind}: a scalar interval");
            };
            check(&format!("{name} {kind}"), *mean, *standard_error, *lower, *upper, want);
        }
    }
}
