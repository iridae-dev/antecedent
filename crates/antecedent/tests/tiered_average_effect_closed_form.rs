//! Closed-form checks for `AverageEffect x {CoDetermined, Unknown} x explicit` (both
//! inferences) and `InterventionResponse x CoDetermined x explicit x Frequentist`.
//!
//! The two average-effect cells run on exactly balanced finite designs, so the estimator is
//! exact and the expected value is a closed form computed by hand in the test, never by calling
//! the production estimator:
//!
//! * `CoDetermined`: a noise-free outcome `y = 2 t + z - 0.5 u` on a cell-count design in which
//!   the treatment share depends on `z` (the naive contrast is off by 1/3, so adjustment is
//!   what recovers the closure functional). AIPW's per-arm outcome regressions are exact, so
//!   every residual term vanishes and the estimate is the empirical-law average of the
//!   contrast `mu(1, z, u) - mu(0, z, u)`, which the test enumerates over the rows.
//! * `Unknown`: binary `t`; the four `(era, t)` cells (era = +-1) are weighted 3:1:1:3 and
//!   each is crossed fully with `(e_m, e_y)` in `{-1, 1}^2`. The noise columns are therefore
//!   exactly orthogonal to the constant, `t`, `era` (functions of the cell) and, for `e_y`, to
//!   `m` as well, so the linear adjustments recover the total effect
//!   `beta_t + beta_m gamma_mt = 1` and the direct effect `-1` to machine precision.
//!
//! The frequentist estimates are checked to 1e-9. The Bayesian cells report a mean of
//! posterior draws; those draws are Monte Carlo around an analytically known posterior whose
//! mean equals the closed form up to the prior shrinkage (`1/prior_scale^2` relative to the
//! Gram diagonal, below 1e-6), so the tolerance is the Monte Carlo error of the mean of the
//! reported draws (six standard errors of a posterior whose own spread is also asserted) plus
//! that shrinkage bound. Truths come from `conformance/estimate/tiered_average_effect_truth`
//! and `conformance/estimate/codetermined_joint_level` (the additive law on an exact
//! cell-count design).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{TieredBackground, WithinTier};

fn truths() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/tiered_average_effect_truth/expected.json"
    ))
    .unwrap()
}

/// Posterior draws requested from the Bayesian cells.
const DRAWS: usize = 20_000;

/// Tolerance of a mean of posterior draws about an analytically known posterior mean: six
/// standard errors of the Monte Carlo mean (using a conservative 2 000 retained draws) plus
/// the prior-shrinkage bound.
fn draw_mean_tolerance(posterior_sd: f64) -> f64 {
    6.0 * posterior_sd / 2_000f64.sqrt() + 1e-6
}

const REPLICATES: usize = 10;

/// The noise-free law `y = 2 t + z - 0.5 u`.
fn codetermined_law(t: f64, z: f64, u: f64) -> f64 {
    2.0 * t + z - 0.5 * u
}

/// Rows `(z, u, t)`: `z` and `u` range over `{-1, 0, 1}`; every cell holds `2 R` controls and
/// `(z + 2) R` treated rows, so `P(t = 1 | z) = (z + 2) / (z + 4)` = 1/3, 1/2, 3/5.
/// The rows are then permuted by a stride coprime to their count (360), so no cross-fit fold
/// is a block of one cell.
fn codetermined_rows() -> Vec<(f64, f64, f64)> {
    let mut rows = Vec::new();
    for u in [-1.0, 0.0, 1.0] {
        for (index, z) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
            for _ in 0..2 * REPLICATES {
                rows.push((z, u, 0.0));
            }
            for _ in 0..(index + 1) * REPLICATES {
                rows.push((z, u, 1.0));
            }
        }
    }
    let n = rows.len();
    assert_eq!(n, 360);
    (0..n).map(|i| rows[(i * 7) % n]).collect()
}

fn codetermined_data() -> TabularData {
    let rows = codetermined_rows();
    let z: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let u: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let t: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let y: Vec<f64> = rows.iter().map(|r| codetermined_law(r.2, r.0, r.1)).collect();
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// The closure functional on the empirical law of `(z, u)`: the row average of
/// `mu(1, z, u) - mu(0, z, u)`, enumerated by hand.
fn codetermined_closure_functional() -> f64 {
    let rows = codetermined_rows();
    rows.iter()
        .map(|&(z, u, _)| codetermined_law(1.0, z, u) - codetermined_law(0.0, z, u))
        .sum::<f64>()
        / rows.len() as f64
}

/// The unadjusted contrast `E[y | t = 1] - E[y | t = 0]` of the same rows, by hand.
fn codetermined_naive_contrast() -> f64 {
    let (mut sum1, mut n1, mut sum0, mut n0) = (0.0, 0.0, 0.0, 0.0);
    for (z, u, t) in codetermined_rows() {
        let y = codetermined_law(t, z, u);
        if t > 0.5 {
            sum1 += y;
            n1 += 1.0;
        } else {
            sum0 += y;
            n0 += 1.0;
        }
    }
    sum1 / n1 - sum0 / n0
}

/// Amplitude of the `e_m` column in `m = t + 0.2 era + 0.5 e_m` (coefficient `0.5`, +-1 levels).
const EM: f64 = 0.5;

/// `{era} | {t, m} | {y}` with binary `t` (levels 0 and 1 both observed): cells `(era, t)`
/// weighted 3:1:1:3 (so `t` depends on `era`), each crossed with `(e_m, e_y)` in `{-1, 1}^2`,
/// replicated four times (128 rows); `m = t + 0.2 era + 0.5 e_m`,
/// `y = -t + 2 m + 0.2 era + e_y`. OLS recovers each stated coefficient exactly.
fn unknown_data() -> TabularData {
    let (mut era, mut t, mut m, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..4 {
        for a in [-1.0, 1.0] {
            for t_i in [0.0, 1.0] {
                // Cell weights 3:1 / 1:3 make the treatment depend on era.
                let weight = if (a < 0.0) == (t_i < 0.5) { 3 } else { 1 };
                for _ in 0..weight {
                    for e_m in [-1.0, 1.0] {
                        for e_y in [-1.0, 1.0] {
                            let m_i = t_i + 0.2 * a + EM * e_m;
                            era.push(a);
                            t.push(t_i);
                            m.push(m_i);
                            y.push(-t_i + 2.0 * m_i + 0.2 * a + e_y);
                        }
                    }
                }
            }
        }
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn background(data: &TabularData, tiers: &[Vec<&str>], within: WithinTier) -> TieredBackground {
    TieredBackground::from_named(data.schema(), tiers, within).unwrap()
}

fn query(data: &TabularData) -> AverageEffectQuery {
    let schema = data.schema();
    AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap())
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(1_000.0))
}

fn codetermined_background(data: &TabularData) -> TieredBackground {
    background(data, &[vec!["z", "u"], vec!["t"], vec!["y"]], WithinTier::CoDetermined)
}

fn unknown_background(data: &TabularData) -> TieredBackground {
    background(data, &[vec!["era"], vec!["t", "m"], vec!["y"]], WithinTier::Unknown)
}

/// `AverageEffect x CoDetermined x explicit x {Frequentist (aipw), Bayesian}`: the closure
/// functional of the stated law is the coefficient 2, and on this exact design AIPW and the
/// Bayesian g-computation both recover it.
#[test]
fn codetermined_average_effect_matches_the_closure_functional() {
    // Hand derivation: the law is additive in t, so the contrast is 2 in every cell and its
    // row average is 2 -- the fixture's closure effect.
    let truth = truths()["codetermined"]["closure_effect"].as_f64().unwrap();
    let enumerated = codetermined_closure_functional();
    assert!((enumerated - truth).abs() < 1e-12, "enumerated {enumerated} vs fixture {truth}");
    // The design discriminates: treated rows have mean z = 1/3 above the controls' 0 (cell
    // counts 1:2:3 on z = -1, 0, 1 against 1:1:1), so the unadjusted contrast is 2 + 1/3.
    let naive = codetermined_naive_contrast();
    assert!((naive - (2.0 + 1.0 / 3.0)).abs() < 1e-9, "naive contrast {naive}");

    let data = codetermined_data();
    let ctx = ExecutionContext::for_tests(19);
    let frequentist = Study::tabular(data.clone())
        .tiered_background(codetermined_background(&data))
        .unwrap()
        .query(query(&data))
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = frequentist.estimate().as_effect().unwrap();
    assert!(
        (effect.ate - truth).abs() < 1e-9,
        "aipw {} vs closure functional {truth} (exact design)",
        effect.ate
    );
    let bayesian = Study::tabular(data.clone())
        .tiered_background(codetermined_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let posterior = bayesian.posterior.as_ref().expect("a posterior of the closure functional");
    let column = posterior.effect_column().unwrap();
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    // The posterior concentrates (the weakly informative residual prior floors sigma at about
    // 2e-3), so a draw-mean error far above this tolerance is a wrong posterior, not noise.
    assert!(sd < 1e-2, "posterior sd {sd}");
    assert!(
        (mean - truth).abs() <= draw_mean_tolerance(sd),
        "posterior mean {mean} sd {sd} vs {truth}"
    );
}

/// `AverageEffect x Unknown x explicit x {Frequentist, Bayesian}`: one value per tier-order
/// scenario, the pretreatment total effect 1 and the closure direct effect -1; no scalar.
#[test]
fn unknown_tier_order_average_effects_match_the_scenario_functionals() {
    // Hand derivation. y = -t + 2 m + 0.2 era + e_y with m = t + 0.2 era + 0.5 e_m, so
    // y = t + 0.6 era + e_m + e_y: adjusting {era} the t-coefficient is the total effect
    // -1 + 2 * 1 = 1. Adjusting {era, m} the t-coefficient is the structural -1. On the
    // balanced cells the residual columns are exactly orthogonal to every regressor, so both are
    // exact; the fixture holds the same two numbers.
    let truth_doc = truths();
    let truth: Vec<f64> = truth_doc["unknown"]["scenario_effects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(truth, vec![-1.0 + 2.0 * 1.0, -1.0]);
    let data = unknown_data();
    let ctx = ExecutionContext::for_tests(23);
    let frequentist = Study::tabular(data.clone())
        .tiered_background(unknown_background(&data))
        .unwrap()
        .query(query(&data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = frequentist.estimate().as_effect().unwrap();
    assert!(effect.ate.is_nan(), "Unknown scenarios never collapse to one ATE");
    let values = effect.scenarios.effects.as_ref().expect("scenario effects");
    let bands = effect.scenarios.intervals.as_ref().expect("joint band");
    assert_eq!(values.len(), truth.len());
    for (j, want) in truth.iter().enumerate() {
        assert!((values[j] - want).abs() < 1e-9, "scenario {j}: {} vs {want}", values[j]);
        let (lo, hi) = bands[j];
        assert!(
            lo - 1e-9 <= *want && *want <= hi + 1e-9,
            "scenario {j} joint band [{lo}, {hi}] vs {want}"
        );
    }
    let bayesian = Study::tabular(data.clone())
        .tiered_background(unknown_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = bayesian.estimate().as_effect().unwrap();
    assert!(effect.ate.is_nan(), "no scenario is the effect");
    let values = effect.scenarios.effects.as_ref().expect("scenario posterior means");
    let intervals = effect.scenarios.intervals.as_ref().expect("scenario credible intervals");
    assert_eq!(values.len(), truth.len());
    for ((value, (lo, hi)), want) in values.iter().zip(intervals.iter()).zip(&truth) {
        // A credible interval of level at least 0.9 spans at least 3.2 posterior sd, so
        // width / 3.2 bounds the posterior sd from above.
        let sd_bound = (hi - lo) / 3.2;
        assert!(sd_bound < 0.5, "scenario posterior is not concentrated: width {}", hi - lo);
        assert!(
            (value - want).abs() <= draw_mean_tolerance(sd_bound),
            "{value} vs {want} (interval [{lo}, {hi}])"
        );
        assert!(*lo <= *want && *want <= *hi, "[{lo}, {hi}] vs {want}");
    }
}

/// `InterventionResponse x CoDetermined x explicit x Frequentist`: cell AIPW of the joint
/// level on the fixture's exact cell-count design against `0.2 + 0.4 + 0.3 + 0.2 mean(z)`.
#[test]
fn codetermined_joint_cell_aipw_matches_the_closed_form_additive_level() {
    let document: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/codetermined_joint_level/expected.json"
    ))
    .unwrap();
    let level = document["level"]["value"].as_f64().unwrap();
    let (mut z, mut t1, mut t2, mut y) = (vec![], vec![], vec![], vec![]);
    for cell in document["design"]["cells"].as_array().unwrap() {
        let at = |key: &str| cell[key].as_f64().unwrap();
        for _ in 0..cell["count"].as_u64().unwrap() {
            let i = z.len() as f64;
            z.push(at("z"));
            t1.push(at("t1"));
            t2.push(at("t2"));
            y.push(0.2 + 0.4 * at("t1") + 0.3 * at("t2") + 0.2 * at("z") + 0.01 * (0.31 * i).sin());
        }
    }
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let joint = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    let tiers =
        background(&data, &[vec!["z"], vec!["t1", "t2"], vec!["y"]], WithinTier::CoDetermined);
    let result = Study::tabular(data)
        .tiered_background(tiers)
        .unwrap()
        .query(CausalQuery::Response(joint))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(17))
        .unwrap();
    assert_eq!(result.support_status().unwrap().as_str(), "supported");
    let got = result.response().map_or(result.estimate().as_effect().unwrap().ate, |r| {
        match &r.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => *v,
            _ => result.estimate().as_effect().unwrap().ate,
        }
    });
    assert!((got - level).abs() < 0.03, "joint cell level {got} vs closed form {level}");
}
