//! 2.2 E7: tier-aware overlap and E-value diagnostics read off executed tiered results.
//!
//! The data generators match `v19_tiered_known_truth.rs` (linear Gaussian, closed-form
//! truth). The E-value oracle is written here from the raw outcome column and the reported
//! effect, not by calling the code that attaches it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "scenario effects are copied, not recomputed")]

mod common;

use antecedent::{
    CellStatus, EstimatorId, RefuteSuite, Study, TierDesign, TierDiagnostics, tier_diagnostics,
};
use antecedent_core::ExecutionContext;
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::Availability;
use antecedent_graph::{TieredBackground, WithinTier};

use common::calibration::gaussian;
use common::fixtures::confounded_scm;

/// `CoDetermined` tiers `{z, u} | {t} | {y}` with binary `t`, `y = 2 t + z - 0.5 u + e`.
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

/// `Unknown` tiers `{era} | {t, m} | {y}`; scenario truths are 1 (pretreatment) and -1 (closure).
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

fn background(data: &TabularData, tiers: &[Vec<&str>], within: WithinTier) -> TieredBackground {
    TieredBackground::from_named(data.schema(), tiers, within).unwrap()
}

fn query(data: &TabularData) -> antecedent_core::AverageEffectQuery {
    let schema = data.schema();
    antecedent_core::AverageEffectQuery::binary_ate(
        schema.id_of("t").unwrap(),
        schema.id_of("y").unwrap(),
    )
}

fn codetermined_study(data: &TabularData, suite: RefuteSuite) -> antecedent::StudyBuilder {
    Study::tabular(data.clone())
        .tiered_background(background(
            data,
            &[vec!["z", "u"], vec!["t"], vec!["y"]],
            WithinTier::CoDetermined,
        ))
        .unwrap()
        .query(query(data))
        .estimator(EstimatorId::Aipw)
        .refute(suite)
        .bootstrap_replicates(0)
}

/// Independent oracle of the tier cell's point E-value: `RR = exp(0.91 |effect| / sd(y))` with
/// the sample SD of the raw outcome column, then `RR + sqrt(RR (RR - 1))`.
fn oracle_point_evalue(data: &TabularData, effect: f64) -> f64 {
    let y = data.float64_values(data.schema().id_of("y").unwrap()).unwrap();
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let rr = (0.91 * effect.abs() / sd).exp();
    rr + (rr * (rr - 1.0)).sqrt()
}

fn assert_interval_withheld(diagnostics: &TierDiagnostics) {
    let interval = &diagnostics.evalue.interval;
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "tier_diagnostics.evalue_interval_withheld");
}

#[test]
fn codetermined_diagnostics_consume_the_executed_design_overlap_and_point_evalue() {
    let data = codetermined_data(3_000, 41);
    let ctx = ExecutionContext::for_tests(41);
    let result = codetermined_study(&data, RefuteSuite::None).build().unwrap().run(&ctx).unwrap();
    let diagnostics = tier_diagnostics(&result).unwrap();

    let schema = data.schema();
    let TierDesign::CoDeterminedClosure { adjustment, rows, propensity_scored } =
        &diagnostics.design
    else {
        panic!("a CoDetermined study reports the closure design, got {:?}", diagnostics.design);
    };
    assert_eq!(adjustment, &[schema.id_of("z").unwrap(), schema.id_of("u").unwrap()]);
    assert_eq!(*rows, result.estimate.n_obs);
    assert!(*propensity_scored);

    // Overlap is the estimator's own report on that design, never a refit.
    let Availability::Available(report) = &diagnostics.overlap else {
        panic!("AIPW fits a propensity, so the overlap report is carried");
    };
    assert_eq!(Some(report), result.estimate.overlap_report.as_ref());
    assert!(0.0 < report.propensity_min && report.propensity_min < report.propensity_max);
    assert!(report.propensity_max < 1.0);
    // The estimator reports effective sample size per clip threshold, not as one number.
    let sensitivity = report.clip_sensitivity.as_ref().expect("clip sensitivity carries the ESS");
    assert!(sensitivity.ess.iter().all(|&ess| 0.0 < ess && ess <= 3_000.0));

    // Point E-value: independent oracle from the raw outcome column and the reported effect.
    let Availability::Available(point) = &diagnostics.evalue.point else {
        panic!("the tier cell attaches a point E-value");
    };
    let expected = oracle_point_evalue(&data, result.estimate.ate);
    assert!((point.value - expected).abs() <= 1e-9 * expected, "{} vs {expected}", point.value);
    assert!(point.value > 1.0);
    assert_eq!(point.method.as_ref(), "vanderweele_outcome_sd");
    assert_interval_withheld(&diagnostics);
}

#[test]
fn an_evalue_refuter_value_embeds_an_interval_endpoint_and_is_not_a_point_value() {
    let data = codetermined_data(3_000, 41);
    let ctx = ExecutionContext::for_tests(41);
    // A tiered study refuses the Cheap refuter suite (no licensed scalar-refuter state), so a
    // refuter-mirrored E-value is constructed on an executed result the way the mirror sets it.
    assert!(
        codetermined_study(&data, RefuteSuite::Cheap).build().is_err(),
        "the Cheap suite is not licensed on a tiered study"
    );
    let mut result =
        codetermined_study(&data, RefuteSuite::None).build().unwrap().run(&ctx).unwrap();
    let effect = result.estimate.as_effect_mut().expect("an effect estimate");
    effect.evalue_threshold = Some(1.5);
    let diagnostics = tier_diagnostics(&result).unwrap();
    let reason = diagnostics.evalue.point.unavailable().expect("a limit-minimum is not a point");
    assert_eq!(reason.code, "diagnostic_not_available");
    assert_eq!(reason.detail, "tier_diagnostics.evalue_embeds_interval_limit");
    assert_interval_withheld(&diagnostics);
    // The overlap report of the same design is still carried.
    assert!(diagnostics.overlap.available().is_some());
}

#[test]
fn unknown_scenarios_state_the_declared_orientations_and_no_fabricated_diagnostic() {
    let data = unknown_data(2_000, 47);
    let ctx = ExecutionContext::for_tests(47);
    let result = Study::tabular(data.clone())
        .tiered_background(background(
            &data,
            &[vec!["era"], vec!["t", "m"], vec!["y"]],
            WithinTier::Unknown,
        ))
        .unwrap()
        .query(query(&data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let diagnostics = tier_diagnostics(&result).unwrap();
    let schema = data.schema();
    let (era, m) = (schema.id_of("era").unwrap(), schema.id_of("m").unwrap());

    let TierDesign::UnknownScenarios { scenarios } = &diagnostics.design else {
        panic!("an Unknown study reports its scenarios, got {:?}", diagnostics.design);
    };
    assert_eq!(scenarios.len(), 2);
    assert_eq!(scenarios[0].method.as_ref(), "tiered.unknown.pretreatment");
    assert_eq!(scenarios[0].adjustment, vec![era]);
    assert_eq!(scenarios[1].method.as_ref(), "tiered.unknown.closure");
    assert_eq!(scenarios[1].adjustment, vec![era, m]);
    let effects = result.estimate.scenario_effects.as_ref().unwrap();
    assert_eq!(scenarios[0].effect, effects[0]);
    assert_eq!(scenarios[1].effect, effects[1]);

    // No propensity, no single effect: each is a typed absence with its own detail.
    let overlap = diagnostics.overlap.unavailable().expect("no overlap on the Unknown cell");
    assert_eq!(overlap.code, "diagnostic_not_available");
    assert_eq!(overlap.detail, "tier_diagnostics.unknown_scenarios_no_overlap");
    let point = diagnostics.evalue.point.unavailable().expect("no E-value across scenarios");
    assert_eq!(point.code, "diagnostic_not_available");
    assert_eq!(point.detail, "tier_diagnostics.unknown_scenarios_no_evalue");
    assert_interval_withheld(&diagnostics);
}

#[test]
fn diagnostics_are_the_same_off_the_fresh_click_and_refreshed_results_and_the_artifact() {
    let data = codetermined_data(3_000, 43);
    let ctx = ExecutionContext::for_tests(43);
    let builder = codetermined_study(&data, RefuteSuite::None);
    let fresh = builder.clone().build().unwrap().run(&ctx).unwrap();
    let mut prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    let reference = tier_diagnostics(&fresh).unwrap();
    let (ref_report, ref_point) = (
        reference.overlap.available().expect("overlap report"),
        reference.evalue.point.available().expect("point E-value"),
    );
    for result in [&click, &refreshed] {
        assert_eq!(result.support_status, Some(CellStatus::Licensed));
        let again = tier_diagnostics(result).unwrap();
        // The fresh run and the prepared lifecycle agree to the estimator's own tolerance.
        assert_eq!(again.design, reference.design);
        let report = again.overlap.available().expect("overlap report");
        assert!((report.propensity_min - ref_report.propensity_min).abs() < 1e-9);
        assert!((report.propensity_max - ref_report.propensity_max).abs() < 1e-9);
        // The prepared lifecycle attaches the same tier-cell point E-value as the fresh run.
        let point = again.evalue.point.available().expect("point E-value on a prepared result");
        assert!((point.value - ref_point.value).abs() < 1e-9);
        assert_eq!(point.method, ref_point.method);
        assert!(ref_point.value > 1.0);
        assert_eq!(again.evalue.interval, reference.evalue.interval);
    }
    // The diagnostics are a view over the result: the result's own artifact is consumed by
    // the independent verifier before the view is trusted.
    let artifact = prepared.encode_contracted_result(&refreshed, "tier-diagnostics", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{:?}",
        consumed.acceptance.unresolved
    );
}

#[test]
fn a_result_that_is_not_a_tiered_average_effect_is_refused() {
    let ctx = ExecutionContext::for_tests(53);
    let (data, dag, query) = confounded_scm(512, 53);
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let error = tier_diagnostics(&result).unwrap_err();
    assert_eq!(error.code, "invalid_argument");
    assert_eq!(error.detail, "tier_diagnostics.not_a_tiered_average_result");
}
