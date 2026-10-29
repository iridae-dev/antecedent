//! Natural mediation on one series under a supplied `TemporalPag`, run through the compiled
//! route and checked against the effects computed from the structural equations that generated
//! the series.
//!
//! The template is unfolded over the window the pulse and its outcome need, each stationary
//! MAG completion is read through the ADMGs that share its MAG, and the identified functionals
//! are evaluated on the lag-aligned window table with circular blocks of consecutive windows.
//! The expected values below come from the generating equations, not from the engine.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::TemporalPag;

const C: u32 = 0;
const X: u32 = 1;
const W: u32 = 2;
const M: u32 = 3;
const Y: u32 = 4;

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bern(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

// The generating equations. `x` is the treatment of the period it is read in.
fn p_w(w: usize) -> f64 {
    bern(0.35, w)
}
fn p_m(m: usize, x: usize, w: usize) -> f64 {
    bern(0.15 + 0.4 * x as f64 + 0.3 * w as f64, m)
}
fn p_y(y: usize, x: usize, m: usize) -> f64 {
    bern(0.1 + 0.2 * x as f64 + 0.3 * m as f64 + 0.2 * (x * m) as f64, y)
}
fn p_m_given(m: usize, x: usize) -> f64 {
    (0..2).map(|w| p_w(w) * p_m(m, x, w)).sum()
}

/// `E[Y(1, M(0))] - E[Y(0, M(0))]`.
fn structural_direct() -> f64 {
    (0..2).map(|m| p_m_given(m, 0) * (p_y(1, 1, m) - p_y(1, 0, m))).sum()
}

/// `E[Y(0, M(1))] - E[Y(0, M(0))]` (the pure natural indirect effect).
fn structural_indirect() -> f64 {
    (0..2).map(|m| p_y(1, 0, m) * (p_m_given(m, 1) - p_m_given(m, 0))).sum()
}

/// `E[Y(1)] - E[Y(0)]`.
fn structural_total() -> f64 {
    (0..2).map(|m| p_y(1, 1, m) * p_m_given(m, 1) - p_y(1, 0, m) * p_m_given(m, 0)).sum()
}

fn structural(contrast: MediationContrast) -> f64 {
    match contrast {
        MediationContrast::Total => structural_total(),
        MediationContrast::NaturalDirect => structural_direct(),
        _ => structural_indirect(),
    }
}

const CONTRASTS: [MediationContrast; 3] = [
    MediationContrast::Total,
    MediationContrast::NaturalDirect,
    MediationContrast::NaturalIndirect,
];

/// `c(t-1) -> x(t)`, `x -> m`, `w -> m`, `m -> y`, `x -> y` within a period: every edge into
/// `m` and `y` and out of `x` is visible because `c` points into `x` and `w` into `m`.
fn lagged_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(var(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The same template with `c(t)` instead of `c(t-1)`: no lag, a window of one period.
fn contemporaneous_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c0 = graph.add_lagged(var(C), Lag::CONTEMPORANEOUS).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c0, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// Deterministic uniform draws (splitmix64).
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn flip(&mut self, p: f64) -> f64 {
        f64::from(self.unit() < p)
    }
}

fn frame(columns: [&[f64]; 5]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [
            ("c", columns[0]),
            ("x", columns[1]),
            ("w", columns[2]),
            ("m", columns[3]),
            ("y", columns[4]),
        ],
        1,
    )
    .unwrap()
}

/// A series whose treatment follows the previous period's `c` and whose other variables follow
/// the generating equations within the period.
fn lagged_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        y[t] = draws.flip(0.1 + 0.2 * x[t] + 0.3 * m[t] + 0.2 * x[t] * m[t]);
    }
    frame([&c, &x, &w, &m, &y])
}

/// The cells of the within-period joint of `(c, x, w, m, y)` as `rows` rows, in cell order.
fn contemporaneous_series(rows: f64) -> TimeSeriesData {
    let p_c = |c: usize| bern(0.45, c);
    let p_x = |x: usize, c: usize| bern(0.25 + 0.5 * c as f64, x);
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 5];
    for cell in 0..32_usize {
        let level = |position: usize| (cell >> (4 - position)) & 1;
        let (c, x, w, m, y) = (level(0), level(1), level(2), level(3), level(4));
        let probability = p_c(c) * p_x(x, c) * p_w(w) * p_m(m, x, w) * p_y(y, x, m);
        let count = (rows * probability).round() as usize;
        for (column, value) in columns.iter_mut().zip([c, x, w, m, y]) {
            column.extend(std::iter::repeat_n(value as f64, count));
        }
    }
    frame([&columns[0], &columns[1], &columns[2], &columns[3], &columns[4]])
}

fn query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(var(X), var(Y), [var(M)], contrast)
}

fn study(
    series: TimeSeriesData,
    graph: TemporalPag,
    query: MediationQuery,
    inference: InferenceMode,
    replicates: u32,
) -> Study {
    Study::series(series)
        .graph(graph)
        .query(CausalQuery::Mediation(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn diagnostic<'a>(
    result: &'a antecedent::StudyResult,
    code: &str,
) -> Option<&'a antecedent_core::Diagnostic> {
    result.diagnostics.iter().find(|d| d.code.as_ref() == code)
}

fn field(diagnostic: &antecedent_core::Diagnostic, key: &str) -> f64 {
    diagnostic
        .fields
        .iter()
        .find(|(k, _)| k.as_ref() == key)
        .unwrap_or_else(|| panic!("field {key}"))
        .1
        .parse()
        .unwrap()
}

#[test]
fn the_reference_effects_are_material() {
    let (total, direct, indirect) =
        (structural_total(), structural_direct(), structural_indirect());
    assert!(direct > 0.1 && indirect > 0.08 && total > 0.15, "{total} {direct} {indirect}");
}

#[test]
fn a_window_of_one_period_reproduces_the_structural_decomposition() {
    let ctx = ExecutionContext::for_tests(71);
    let series = contemporaneous_series(100_000.0);
    for contrast in CONTRASTS {
        let built = study(
            series.clone(),
            contemporaneous_template(),
            query(contrast),
            InferenceMode::Frequentist,
            0,
        );
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        let expected = structural(contrast);
        assert!((result.effect() - expected).abs() < 1e-3, "{contrast:?}: {}", result.effect());
        let components = result.payload().mediation().expect("the decomposition");
        assert!((components.total.unwrap() - structural_total()).abs() < 1e-3);
        assert!((components.direct.unwrap() - structural_direct()).abs() < 1e-3);
        assert!((components.mediated.unwrap() - structural_indirect()).abs() < 1e-3);
        let window =
            diagnostic(&result, "estimate.temporal_pag.window").expect("window disclosure");
        assert_eq!(field(window, "periods"), 1.0);
        assert_eq!(field(window, "history"), 0.0);
    }
}

#[test]
fn a_lagged_window_matches_the_structural_decomposition_of_the_series() {
    let ctx = ExecutionContext::for_tests(72);
    let series = lagged_series(60_000, 17);
    for contrast in CONTRASTS {
        let result = study(
            series.clone(),
            lagged_template(),
            query(contrast),
            InferenceMode::Frequentist,
            0,
        )
        .run(&ctx)
        .unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        let expected = structural(contrast);
        // Measured deviations on this 60 000-row series are at most 0.0054.
        assert!((result.effect() - expected).abs() < 0.01, "{contrast:?}: {}", result.effect());
        let components = result.payload().mediation().expect("the decomposition");
        assert!((components.total.unwrap() - structural_total()).abs() < 0.01);
        assert!((components.direct.unwrap() - structural_direct()).abs() < 0.01);
        assert!((components.mediated.unwrap() - structural_indirect()).abs() < 0.01);
        let window =
            diagnostic(&result, "estimate.temporal_pag.window").expect("window disclosure");
        // One template lag of history and the pulse period.
        assert_eq!(field(window, "periods"), 2.0);
        assert_eq!(field(window, "history"), 1.0);
        assert_eq!(field(window, "windows"), 59_999.0);
        assert!(diagnostic(&result, "identify.pag.envelope").is_some());
        let certificate = result.certificate.as_ref().expect("the identification certificate");
        assert!(matches!(
            certificate.identification,
            antecedent::Identification::TemporalEnvelope { .. }
        ));
    }
}

#[test]
fn uncertainty_resamples_blocks_never_shorter_than_the_window_into_an_identified_set_interval() {
    let ctx = ExecutionContext::for_tests(73);
    let series = lagged_series(6_000, 23);
    let result = study(
        series,
        lagged_template(),
        query(MediationContrast::NaturalIndirect),
        InferenceMode::Frequentist,
        60,
    )
    .run(&ctx)
    .unwrap();
    let window = diagnostic(&result, "estimate.temporal_pag.window").expect("window disclosure");
    assert!(field(window, "block_length") >= 2.0, "{}", field(window, "block_length"));
    // The window MAG has two candidate ADMGs whose functionals differ, so the indirect effect
    // is an identified set: the interval comes from the block-bootstrap refits of every
    // candidate on the same resampled windows.
    let mixture = result.structural_response.as_ref().expect("the identified-set mixture");
    let interval = mixture.identified_set_interval.as_ref().expect("an identified-set interval");
    assert!(interval.lower <= interval.bound_lower && interval.bound_upper <= interval.upper);
    assert!(
        interval.lower - 0.02 <= structural_indirect()
            && structural_indirect() <= interval.upper + 0.02,
        "[{}, {}] against {}",
        interval.lower,
        interval.upper,
        structural_indirect()
    );
}

#[test]
fn the_bayesian_bootstrap_shares_one_weight_per_block_into_an_identified_set_interval() {
    let ctx = ExecutionContext::for_tests(74);
    let series = lagged_series(20_000, 29);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
    let result =
        study(series, lagged_template(), query(MediationContrast::NaturalDirect), inference, 0)
            .run(&ctx)
            .unwrap();
    assert!((result.effect() - structural_direct()).abs() < 0.01, "{}", result.effect());
    // Two candidate ADMGs identify the direct effect by different functionals, so the result is
    // an identified set whose interval comes from the Dirichlet draws of every candidate,
    // each draw shared across a block of consecutive windows.
    let mixture = result.structural_response.as_ref().expect("the identified-set mixture");
    let interval = mixture.identified_set_interval.as_ref().expect("an identified-set interval");
    assert!(interval.lower <= interval.bound_lower && interval.bound_upper <= interval.upper);
    assert!(
        interval.lower - 0.03 <= structural_direct()
            && structural_direct() <= interval.upper + 0.03,
        "[{}, {}] against {}",
        interval.lower,
        interval.upper,
        structural_direct()
    );
}

#[test]
fn a_treatment_with_only_invisible_edges_is_not_identified_and_the_refusal_names_the_search() {
    // x -> m -> y and x -> y with nothing pointing into x: every edge is invisible.
    let mut graph = TemporalPag::empty();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(x0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    let ctx = ExecutionContext::for_tests(75);
    let error = Study::series(lagged_series(4_000, 31))
        .graph(graph)
        .query(CausalQuery::Mediation(query(MediationContrast::NaturalIndirect)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|built| built.run(&ctx))
        .unwrap_err();
    assert_eq!(error.reason_code(), Some("effect_not_identified"), "{error}");
    assert!(error.to_string().contains("temporal PAG mediation not identified"), "{error}");
}

#[test]
fn a_request_for_two_horizons_is_refused_with_the_next_attempt() {
    let ctx = ExecutionContext::for_tests(76);
    let two_horizons = query(MediationContrast::Total).with_horizons(vec![1, 2]).unwrap();
    let error = Study::series(lagged_series(4_000, 37))
        .graph(lagged_template())
        .query(CausalQuery::Mediation(two_horizons))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|built| built.run(&ctx))
        .unwrap_err();
    assert!(error.to_string().contains("one outcome horizon"), "{error}");
}

/// The temporal PAG decomposition publishes a contracted result that an independent consumer
/// reads back, on a one-period window and on a two-period (lagged) window, for both inferences.
#[test]
fn a_temporal_pag_decomposition_publishes_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(76);
    let cases = [
        ("one-period", contemporaneous_series(20_000.0), contemporaneous_template()),
        ("two-period", lagged_series(6_000, 29), lagged_template()),
    ];
    for (window, series, template) in cases {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
        ] {
            let built = study(
                series.clone(),
                template.clone(),
                query(MediationContrast::NaturalDirect),
                inference,
                0,
            );
            let prepared = built.prepare(&ctx).unwrap();
            let result = prepared.estimate_retained(&ctx).unwrap();
            let artifact = prepared
                .encode_contracted_result(&result, "temporal-pag-mediation", &ctx)
                .unwrap_or_else(|e| panic!("{window}: {e}"));
            let consumed = antecedent_io::consume_analysis_result(&artifact)
                .unwrap_or_else(|e| panic!("{window}: {e}"));
            assert_eq!(consumed.body.estimate, Some(result.effect()), "{window}");
        }
    }
}

/// The same lagged law read as a `TemporalAdmg` (a two-period window): the decomposition
/// publishes a contracted result that an independent consumer reads back.
#[test]
fn a_two_period_temporal_admg_decomposition_publishes_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(77);
    let mut graph = antecedent_graph::TemporalAdmg::empty();
    let c1 = graph.add_lagged(var(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    let series = lagged_series(6_000, 31);
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
    ] {
        let built = Study::series(series.clone())
            .graph(graph.clone())
            .query(CausalQuery::Mediation(query(MediationContrast::NaturalDirect)))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let prepared = built.prepare(&ctx).unwrap();
        let result = prepared.estimate_retained(&ctx).unwrap();
        assert!((result.effect() - structural_direct()).abs() < 0.05, "{}", result.effect());
        let artifact =
            prepared.encode_contracted_result(&result, "temporal-admg-mediation", &ctx).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(result.effect()));
    }
}

#[test]
fn retained_pag_mediation_uses_its_callers_context_for_window_identification() {
    let data = contemporaneous_series(4_000.0);
    let built = study(
        data.clone(),
        contemporaneous_template(),
        query(MediationContrast::NaturalDirect),
        InferenceMode::Frequentist,
        0,
    );
    let ctx = ExecutionContext::for_tests(76);
    let prepared = built.prepare(&ctx).unwrap();
    let expected = prepared.estimate_series(&data, &ctx).unwrap();
    let limited = ExecutionContext::for_tests(76).with_memory_limit(Some(1));
    assert!(matches!(built.prepare(&limited), Err(antecedent::CausalError::Resource { .. })));
    let cancelled = ExecutionContext::for_tests(76);
    cancelled.cancellation.cancel();
    assert!(matches!(built.prepare(&cancelled), Err(antecedent::CausalError::Cancelled { .. })));
    let mut disabled = ExecutionContext::for_tests(76);
    disabled.cache_policy = antecedent_core::CachePolicy::disabled();
    let rechecked = built.prepare(&disabled).unwrap();
    assert_eq!(prepared.contract().unwrap().identities, rechecked.contract().unwrap().identities);
    assert_eq!(
        expected.effect().to_bits(),
        rechecked.estimate_series(&data, &disabled).unwrap().effect().to_bits()
    );
}
