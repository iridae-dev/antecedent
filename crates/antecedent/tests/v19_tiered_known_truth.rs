//! Tier-background evidence: structure-input conflicts and analytic
//! known-truth pins for `AverageEffect` × `CoDetermined` / `Unknown`.
//!
//! Both DGPs are linear with Gaussian noise, so every scenario's adjustment
//! functional has a closed form (derived next to each generator). The pins
//! assert the reported numbers against that truth, not against another run of
//! the same code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown)]

mod common;

use antecedent::{CausalError, EstimatorId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{DenseNodeId, TieredBackground, WithinTier};

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

fn assert_graph_conflict(result: Result<impl std::fmt::Debug, CausalError>) {
    match result {
        Err(CausalError::Conflict { what, .. }) => assert_eq!(what, "graph"),
        other => {
            panic!("expected CausalError::Conflict for graph + tiered_background, got {other:?}")
        }
    }
}

/// R-7: a caller ADMG and a tier background are two sources of truth. The
/// AverageEffect route keys off the background, so the ADMG (here with a
/// drawn `t ↔ y` edge that defeats adjustment) must be refused, not ignored.
#[test]
fn caller_graph_and_tiered_background_conflict_in_either_order() {
    let data = codetermined_data(200, 1);
    let background = codetermined_background(&data);
    let mut admg = background.to_admg(data.schema()).unwrap();
    let t = DenseNodeId::from_raw(data.schema().id_of("t").unwrap().raw());
    let y = DenseNodeId::from_raw(data.schema().id_of("y").unwrap().raw());
    admg.insert_bidirected(t, y).unwrap();

    // tiered_background, then graph: refused at build.
    let after = Study::tabular(data.clone())
        .tiered_background(background.clone())
        .unwrap()
        .graph(admg.clone())
        .query(query(&data))
        .refute(RefuteSuite::None)
        .build();
    assert_graph_conflict(after);

    // graph, then tiered_background: refused immediately.
    let before = Study::tabular(data.clone()).graph(admg.clone()).tiered_background(background);
    assert_graph_conflict(before);

    // The identify-level check the facade can no longer bypass: the closure
    // is not a back-door set on the caller's ADMG.
    let id = antecedent_identify::identify_tiered_on(
        &codetermined_background(&data),
        &admg,
        &query(&data),
    )
    .unwrap();
    assert_eq!(id.status, antecedent_identify::IdentificationStatus::NotIdentified);
}

#[test]
fn codetermined_average_effect_known_truth() {
    let data = codetermined_data(4_000, 19);
    let ctx = ExecutionContext::for_tests(19);
    for estimator in [EstimatorId::LinearAdjustmentAte, EstimatorId::Aipw] {
        let study = Study::tabular(data.clone())
            .tiered_background(codetermined_background(&data))
            .unwrap()
            .query(query(&data))
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let fresh = study.clone().run(&ctx).unwrap();
        let click = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
        for result in [&fresh, &click] {
            assert_eq!(result.support_status.unwrap().as_str(), "licensed");
            assert_eq!(result.logical_plan.identifier.as_deref(), Some("generalized.adjustment"));
            let est = &result.estimate;
            assert!(est.se_analytic.is_finite() && est.se_analytic > 0.0, "{estimator:?}");
            assert!(
                (est.ate - CODETERMINED_TRUTH).abs() < 4.0 * est.se_analytic
                    && (est.ate - CODETERMINED_TRUTH).abs() < 0.12,
                "{estimator:?} CoDetermined ATE {} (se {}) vs analytic truth {CODETERMINED_TRUTH}",
                est.ate,
                est.se_analytic
            );
        }
        assert!((fresh.estimate.ate - click.estimate.ate).abs() < 1e-12);
    }
}

#[test]
fn unknown_scenarios_known_truth() {
    let data = unknown_data(4_000, 23);
    let ctx = ExecutionContext::for_tests(23);
    let study = Study::tabular(data.clone())
        .tiered_background(unknown_background(&data))
        .unwrap()
        .query(query(&data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    let click = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    for result in [&fresh, &click] {
        assert_eq!(result.support_status.unwrap().as_str(), "licensed");
        assert_eq!(format!("{:?}", result.identification.status), "GraphDependent");
        assert!(result.estimate.ate.is_nan(), "Unknown scenarios never collapse to one ATE");
        let effects = result.estimate.scenario_effects.as_ref().expect("scenario effects");
        let cov = result.estimate.joint_covariance.as_ref().expect("joint IF covariance");
        let bands = result.estimate.scenario_intervals.as_ref().expect("joint band");
        assert_eq!(effects.len(), 2);
        for (j, truth) in UNKNOWN_TRUTH.iter().enumerate() {
            let se = cov.se(j);
            assert!(se.is_finite() && se > 0.0);
            assert!(
                (effects[j] - truth).abs() < 4.0 * se && (effects[j] - truth).abs() < 0.08,
                "scenario {j}: {} (se {se}) vs analytic truth {truth}",
                effects[j]
            );
            let (lo, hi) = bands[j];
            assert!(lo <= *truth && *truth <= hi, "scenario {j} joint band [{lo}, {hi}]");
        }
    }
}
