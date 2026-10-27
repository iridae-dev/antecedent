//! Bayesian `AverageEffect` under a tier background, against the analytic truths of
//! linear Gaussian DGPs (derived beside each generator): a posterior for the one
//! `CoDetermined` closure functional, and one separate posterior per Unknown scenario.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{TieredBackground, WithinTier};

use common::calibration::gaussian;

/// `CoDetermined` tiers `{z, u} | {t} | {y}` with binary `t`.
///
/// `z, u` share a latent factor (so `z ↔ u` in the closure ADMG),
/// `P(t = 1) = logistic(0.6 z − 0.4 u)`, `y = 2 t + z − 0.5 u + ε`.
/// The outcome model is linear in `(t, z, u)`, so the closure functional
/// `E[E[y | t=1, z, u] − E[y | t=0, z, u]]` is exactly 2.
fn codetermined_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut uniform_state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        uniform_state = uniform_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (uniform_state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut z, mut u, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let common = g();
        z[i] = 0.7 * common + 0.7 * g();
        u[i] = 0.7 * common + 0.7 * g();
        let p = 1.0 / (1.0 + (-(0.6 * z[i] - 0.4 * u[i])).exp());
        t[i] = f64::from(uniform() < p);
        y[i] = 2.0 * t[i] + z[i] - 0.5 * u[i] + g();
    }
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// Truth of the `CoDetermined` closure functional in [`codetermined_data`].
const CODETERMINED_TRUTH: f64 = 2.0;

/// `Unknown` tiers `{era} | {t, m} | {y}` from the true order `era → t → m → y`.
///
/// `t = 0.8 era + ε_t`, `m = t + 0.2 era + ε_m`, `y = −t + 2 m + 0.2 era + ε_y`.
/// Scenario 0 (treatment precedes its peer) adjusts `{era}`: the coefficient
/// of `t` in the linear projection of `y` on `(t, era)` is the total effect
/// `−1 + 2·1 = 1`. Scenario 1 (treatment follows its peer) adjusts
/// `{era, m}`: `y` is exactly linear in `(t, m, era)`, so the coefficient is
/// the direct effect `−1`.
fn unknown_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = g();
        t[i] = 0.8 * era[i] + g();
        m[i] = t[i] + 0.2 * era[i] + 0.5 * g();
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + g();
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// Scenario truths `[pretreatment, closure]` for [`unknown_data`].
const UNKNOWN_TRUTH: [f64; 2] = [1.0, -1.0];

fn codetermined_background(data: &TabularData) -> TieredBackground {
    TieredBackground::from_named(
        data.schema(),
        &[vec!["z", "u"], vec!["t"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap()
}

fn unknown_background(data: &TabularData) -> TieredBackground {
    TieredBackground::from_named(
        data.schema(),
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
    )
    .unwrap()
}

fn query(data: &TabularData) -> AverageEffectQuery {
    let schema = data.schema();
    AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap())
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2_000).prior_scale(1_000.0))
}

#[test]
fn codetermined_bayesian_posterior_covers_the_closure_functional() {
    let data = codetermined_data(4_000, 19);
    let ctx = ExecutionContext::for_tests(19);
    let result = Study::tabular(data.clone())
        .tiered_background(codetermined_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(result.logical_plan.identifier.as_deref(), Some("generalized.adjustment"));
    let posterior = result.posterior.as_ref().expect("a posterior of the closure functional");
    let column = posterior.effect_column().unwrap();
    let mean = posterior.summaries.mean[column];
    let sd = posterior.summaries.sd[column];
    assert!((mean - CODETERMINED_TRUTH).abs() < 4.0 * sd.max(0.03), "{mean} sd {sd}");
    assert!(sd > 0.0);
}

#[test]
fn unknown_bayesian_keeps_one_posterior_per_scenario_and_no_scalar() {
    let data = unknown_data(4_000, 23);
    let ctx = ExecutionContext::for_tests(23);
    let result = Study::tabular(data.clone())
        .tiered_background(unknown_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = result.estimate().as_effect().unwrap();
    assert!(effect.ate.is_nan(), "no scenario is the effect");
    let values = effect.scenarios.effects.as_ref().expect("scenario posterior means");
    let intervals = effect.scenarios.intervals.as_ref().expect("scenario credible intervals");
    assert_eq!(values.len(), 2);
    for ((value, (lo, hi)), truth) in values.iter().zip(intervals.iter()).zip(UNKNOWN_TRUTH) {
        assert!((value - truth).abs() < 0.15, "{value} vs {truth}");
        assert!(*lo <= truth && truth <= *hi, "[{lo}, {hi}] vs {truth}");
    }
    assert!(result.posterior.is_none(), "no cross-scenario posterior exists");
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "tiered.unknown.bayesian_scenarios")
    );
}

/// Both tier-rule posteriors run from the retained plan with the builder dropped: the plan
/// records the Bayesian g-computation estimator, and the prepared click reproduces the shape of
/// the one-shot result (one posterior for the closure; one credible interval per scenario and
/// no scalar for the unknown tier order).
#[test]
fn tiered_bayesian_average_effects_execute_from_the_retained_plan() {
    let ctx = ExecutionContext::for_tests(31);

    let data = codetermined_data(4_000, 31);
    let builder = Study::tabular(data.clone())
        .tiered_background(codetermined_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    let plan = &prepared.plan().logical.record;
    assert_eq!(plan.identifier.as_deref(), Some("generalized.adjustment"));
    assert_eq!(plan.estimator.as_deref(), Some("bayesian.gcomp"));
    let result = prepared.estimate(&data, &ctx).unwrap();
    let posterior = result.posterior.as_ref().expect("a posterior of the closure functional");
    let mean = posterior.summaries.mean[posterior.effect_column().unwrap()];
    assert!((mean - CODETERMINED_TRUTH).abs() < 0.2, "{mean}");

    let data = unknown_data(4_000, 31);
    let builder = Study::tabular(data.clone())
        .tiered_background(unknown_background(&data))
        .unwrap()
        .query(query(&data))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    let plan = &prepared.plan().logical.record;
    assert_eq!(plan.estimator.as_deref(), Some("bayesian.gcomp"));
    let result = prepared.estimate(&data, &ctx).unwrap();
    let effect = result.estimate().as_effect().unwrap();
    assert!(effect.ate.is_nan(), "no scenario is the effect");
    assert_eq!(effect.scenarios.effects.as_ref().expect("scenario posterior means").len(), 2);
    assert!(result.posterior.is_none(), "no cross-scenario posterior exists");
}
