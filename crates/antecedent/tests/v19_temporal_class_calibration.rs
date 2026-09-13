//! 1.9 coverage of temporal class envelopes not covered by `v19_calibration`
//! (WP-H): Frequentist multi-step Sustained on `TemporalCpdag` / `TemporalPag`,
//! `TemporalPag` Sustained in both inference modes, AR(1) variants of the
//! `TemporalPag` cells, and the identified-set interval (C-3 / K-2).
//!
//! The `TemporalPag` fixture is [`fixtures::chain_pag`] (the structure of
//! `conformance/estimate/temporal_class_envelope/identified_pag.json`): six
//! identified completions at two different effects plus one unidentified.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]

mod common;

use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, ResponseValue, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use common::calibration::{CoverageTally, Z90, n_sim, normal_interval, quantile_interval};
use common::fixtures::{
    self, B1, chain_pag, chain_pag_series, confounded_cpdag, confounded_series,
};

const LEVEL: f64 = 0.9;
const N: usize = 160;
const BOOT: u32 = 199;
const DRAWS: usize = 1000;
const SHORT_SERIES: &str = "estimate.temporal.circular_block_se.short_series";

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(8.0))
}

fn run(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: &TemporalEffectQuery,
    inference: InferenceMode,
    prior: Option<ClassPrior>,
    boot: u32,
    ctx_seed: u64,
) -> StudyResult {
    let mut builder = Study::series(data)
        .graph(graph.into())
        .query(CausalQuery::TemporalEffect(query.clone()))
        .inference(inference);
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    builder
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(ctx_seed))
        .unwrap()
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

fn atom_values(result: &StudyResult) -> Vec<Option<f64>> {
    result
        .structural_response
        .as_ref()
        .expect("structural atoms")
        .atoms
        .iter()
        .map(|atom| match atom.value {
            Some(ResponseValue::Scalar(v)) => Some(v),
            _ => None,
        })
        .collect()
}

fn close(label: &str, got: f64, want: f64, tol: f64) {
    assert!((got - want).abs() <= tol, "{label}: got {got}, want {want} ± {tol}");
}

/// Frequentist coverage of `ate ± z·se_bootstrap`, with bias and SE/SD for diagnosis.
struct FreqTally {
    name: String,
    tally: CoverageTally,
    estimates: Vec<f64>,
    ses: Vec<f64>,
    truth: f64,
    warned: u32,
}

impl FreqTally {
    fn new(name: &str, truth: f64) -> Self {
        Self {
            name: name.to_owned(),
            tally: CoverageTally::new(name, LEVEL),
            estimates: Vec::new(),
            ses: Vec::new(),
            truth,
            warned: 0,
        }
    }

    fn record(&mut self, result: &StudyResult) {
        let est = &result.estimate;
        self.tally.record(normal_interval(est.ate, est.se_bootstrap, Z90), self.truth);
        if est.ate.is_finite() {
            self.estimates.push(est.ate);
        }
        if let Some(se) = est.se_bootstrap.filter(|se| se.is_finite()) {
            self.ses.push(se);
        }
        self.warned += u32::from(has_diagnostic(result, SHORT_SERIES));
    }

    fn report(&self) {
        let n = self.estimates.len().max(2) as f64;
        let mean = self.estimates.iter().sum::<f64>() / n;
        let mc_sd =
            (self.estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let mean_se = self.ses.iter().sum::<f64>() / self.ses.len().max(1) as f64;
        eprintln!(
            "info {}: bias={:+.4} mc_sd={mc_sd:.4} mean_se={mean_se:.4} se/sd={:.3} \
             short_series warnings {}/{}",
            self.name,
            mean - self.truth,
            mean_se / mc_sd,
            self.warned,
            self.estimates.len()
        );
    }

    fn assert(&self) {
        self.report();
        self.tally.assert();
    }
}

// ---------------------------------------------------------------------------
// Fixture truths
// ---------------------------------------------------------------------------

/// Identified completions of `chain_pag` in envelope order, `true` = adjusts `z`.
/// Pinned by `chain_pag_fixture_identifies_six_completions_at_two_effects`.
const CHAIN_ADJUSTS: [Option<bool>; 7] =
    [Some(true), Some(true), Some(false), None, Some(false), Some(true), Some(true)];

/// Multi-step Sustained over `-2..=-1` identifies its own (lag-2) window: four
/// directed completions evaluate (two per effect); the unidentified completion
/// and the bidirected MAG completions (unevaluable by sequential g-computation)
/// carry no value. The composed contrast equals the lag-1 effect of each
/// completion because nothing at lag 2 reaches `y`.
const CHAIN_MULTI_ADJUSTS: [Option<bool>; 7] =
    [Some(true), Some(true), Some(false), Some(false), None, None, None];

fn is_multi_step(query: &TemporalEffectQuery) -> bool {
    matches!(query.policy, TemporalPolicy::Sustained { from, until } if from != until)
}

fn chain_truths(query: &TemporalEffectQuery) -> Vec<Option<f64>> {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    let order = if is_multi_step(query) { &CHAIN_MULTI_ADJUSTS } else { &CHAIN_ADJUSTS };
    order.iter().map(|a| a.map(|adj| if adj { adjusted } else { unadjusted })).collect()
}

/// Frozen-weight mixture over identified, evaluable `chain_pag` completions
/// (equal enumeration weights).
fn chain_mixture_truth(query: &TemporalEffectQuery) -> f64 {
    let truths: Vec<f64> = chain_truths(query).into_iter().flatten().collect();
    truths.iter().sum::<f64>() / truths.len() as f64
}

fn cpdag_mixture_truth(rho: f64) -> f64 {
    fixtures::cpdag_completion_truths(rho).iter().sum::<f64>() / 2.0
}

// ---------------------------------------------------------------------------
// Frequentist frozen-weight mixtures (shared circular block over aligned rows)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Design {
    label: &'static str,
    rho: f64,
    n: usize,
}

const IID: Design = Design { label: "iid n=160", rho: 0.0, n: N };
const AR05_160: Design = Design { label: "AR(1) rho=0.5 n=160", rho: 0.5, n: 160 };
const AR09_400: Design = Design { label: "AR(1) rho=0.9 n=400", rho: 0.9, n: 400 };
const AR05_60: Design = Design { label: "AR(1) rho=0.5 n=60", rho: 0.5, n: 60 };

fn frequentist_chain_pag_case(what: &str, query: &TemporalEffectQuery, design: Design, seed: u64) {
    chain_pag_tally(what, query, design, seed).assert();
}

/// Boundary record, as in `v19_temporal_frequentist`: the design sits below the
/// effective-row floor, so the short-series warning must fire on at least three
/// quarters of replicates; coverage is measured and printed, not gated.
fn frequentist_chain_pag_boundary(
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    let tally = chain_pag_tally(what, query, design, seed);
    tally.report();
    let (lo, hi) = common::calibration::coverage_band(n_sim(), LEVEL);
    eprintln!(
        "calibration-boundary {}: coverage={:.3} band=[{lo:.3}, {hi:.3}] mean_length={:.4} \
         (not gated; short_series warnings {}/{})",
        tally.name,
        tally.tally.rate(),
        tally.tally.mean_length(),
        tally.warned,
        n_sim()
    );
    assert!(
        tally.warned * 4 >= n_sim() * 3,
        "boundary design must carry the short-series warning: {}/{}",
        tally.warned,
        n_sim()
    );
}

fn chain_pag_tally(
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) -> FreqTally {
    let mut tally = FreqTally::new(
        &format!("frequentist TemporalPag {what} [{}]", design.label),
        chain_mixture_truth(query),
    );
    for s in 0..n_sim() {
        let result = run(
            chain_pag_series(design.n, design.rho, seed + u64::from(s)),
            chain_pag(),
            query,
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        assert!(
            has_diagnostic(&result, "estimate.temporal_class.frequentist.shared_block"),
            "class envelope must publish the shared-block SE"
        );
        tally.record(&result);
    }
    tally
}

fn frequentist_cpdag_multistep_case(design: Design, seed: u64) {
    let mut tally = FreqTally::new(
        &format!("frequentist TemporalCpdag multi-step Sustained [{}]", design.label),
        cpdag_mixture_truth(design.rho),
    );
    for s in 0..n_sim() {
        let result = run(
            confounded_series(design.n, 0.0, design.rho, seed + u64::from(s)),
            confounded_cpdag(),
            &multi_sustained(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        tally.record(&result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_multistep_sustained_nominal_90_coverage() {
    frequentist_cpdag_multistep_case(IID, 60_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_multistep_sustained_ar1_rho05_n160_nominal_90_coverage() {
    frequentist_cpdag_multistep_case(AR05_160, 61_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_multistep_sustained_nominal_90_coverage() {
    frequentist_chain_pag_case("multi-step Sustained", &multi_sustained(), IID, 62_000);
}

macro_rules! frequentist_pag_gate {
    ($name:ident, $what:literal, $query:expr, $design:expr, $seed:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            frequentist_chain_pag_case($what, &$query, $design, $seed);
        }
    };
}

frequentist_pag_gate!(
    frequentist_temporal_pag_multi_completion_pulse_nominal_90_coverage,
    "Pulse",
    pulse_query(),
    IID,
    63_000
);
frequentist_pag_gate!(
    frequentist_temporal_pag_sustained_nominal_90_coverage,
    "single-step Sustained",
    single_sustained(),
    IID,
    64_000
);
frequentist_pag_gate!(
    frequentist_temporal_pag_pulse_ar1_rho05_n160_nominal_90_coverage,
    "Pulse",
    pulse_query(),
    AR05_160,
    65_000
);
/// Below the effective-row floor on every replicate (the six-completion mixture
/// score at ρ = 0.9, n = 400): measured, warned, not gated.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_pulse_ar1_rho09_n400_short_series_boundary() {
    frequentist_chain_pag_boundary("Pulse", &pulse_query(), AR09_400, 66_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_sustained_ar1_rho09_n400_short_series_boundary() {
    frequentist_chain_pag_boundary("single-step Sustained", &single_sustained(), AR09_400, 69_000);
}
frequentist_pag_gate!(
    frequentist_temporal_pag_pulse_ar1_rho05_n60_nominal_90_coverage,
    "Pulse",
    pulse_query(),
    AR05_60,
    67_000
);
frequentist_pag_gate!(
    frequentist_temporal_pag_sustained_ar1_rho05_n160_nominal_90_coverage,
    "single-step Sustained",
    single_sustained(),
    AR05_160,
    68_000
);
frequentist_pag_gate!(
    frequentist_temporal_pag_sustained_ar1_rho05_n60_nominal_90_coverage,
    "single-step Sustained",
    single_sustained(),
    AR05_60,
    70_000
);

// ---------------------------------------------------------------------------
// Bayesian TemporalPag Sustained / Pulse with a class prior (BMA posterior)
// ---------------------------------------------------------------------------

fn posterior_interval(post: Option<&antecedent::CausalPosterior>) -> Option<(f64, f64)> {
    let post = post?;
    let col = post.effect_column()?;
    quantile_interval(post.draws.column(col).ok()?, LEVEL)
}

/// `circle_pag` with class prior `[0.4, 0.6]`: the draw-level BMA over the one
/// identified completion is that completion's posterior (`θ = B1`), and the
/// unidentified completion's 0.4 stays a separate axis. `chain_pag` cannot carry
/// a class-prior mixture: its global m-separation audit is capped, so the class
/// scope is not full and mixing is refused (its no-prior identified-set interval
/// is calibrated below instead).
fn bayesian_circle_pag_class_prior_case(
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    let prior = ClassPrior::from_ordered([0.4, 0.6]).unwrap();
    let mut tally = CoverageTally::new(
        format!("Bayesian TemporalPag {what} class-prior [{}]", design.label),
        LEVEL,
    );
    for s in 0..n_sim() {
        let result = run(
            fixtures::pag_series_ar1(design.n, design.rho, seed + u64::from(s)),
            fixtures::circle_pag(),
            query,
            bayes(),
            Some(prior.clone()),
            0,
            u64::from(s),
        );
        let post = result.posterior.as_ref().expect("PAG class-prior posterior");
        assert!(
            (post.unidentified_mass - 0.4).abs() < 1e-12,
            "unidentified mass must stay a separate axis, got {}",
            post.unidentified_mass
        );
        tally.record(posterior_interval(Some(post)), B1);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_class_prior_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case("single-step Sustained", &single_sustained(), IID, 71_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_pulse_class_prior_ar1_rho05_n160_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case("Pulse", &pulse_query(), AR05_160, 72_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_class_prior_ar1_rho05_n160_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case(
        "single-step Sustained",
        &single_sustained(),
        AR05_160,
        73_000,
    );
}

/// Frequentist `circle_pag` (one identified completion) under AR(1): the
/// reported aggregate is that completion's shared-block interval.
fn frequentist_circle_pag_case(what: &str, query: &TemporalEffectQuery, design: Design, seed: u64) {
    let mut tally = FreqTally::new(
        &format!("frequentist TemporalPag one-completion {what} [{}]", design.label),
        B1,
    );
    for s in 0..n_sim() {
        let result = run(
            fixtures::pag_series_ar1(design.n, design.rho, seed + u64::from(s)),
            fixtures::circle_pag(),
            query,
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        tally.record(&result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_one_completion_sustained_ar1_rho05_n160_nominal_90_coverage() {
    frequentist_circle_pag_case("single-step Sustained", &single_sustained(), AR05_160, 80_000);
}

// ---------------------------------------------------------------------------
// Identified-set intervals (C-3 Frequentist, K-2 Bayesian no-ClassPrior)
// ---------------------------------------------------------------------------

/// Scores `structural_response.identified_set_interval` against the causal effect
/// of the generating DAG, which is one completion of the class.
///
/// Assertion: the two-sided `level ± 3·MCSE` band. Imbens–Manski intervals are
/// conservative (coverage above nominal) only while the set's width is small
/// relative to sampling noise. In every fixture below the identified effects are
/// 4–10 bound SDs apart at the design's `n`, so the width is retained, the
/// critical value is ≈ one-sided, and the true effect sits at an endpoint of the
/// set: coverage is then nominal (a miss happens only in the tail on the side of
/// the true completion). A one-completion set is a point and the interval is
/// the two-sided atom interval, also nominal. Over-coverage in these designs
/// would mean the construction is wider than IM requires, which the band is
/// meant to catch. Coverage of the *other* endpoint (the non-causal completion's
/// estimand) is printed for the record.
struct SetTally {
    name: String,
    truth: (CoverageTally, f64),
    other: Option<(CoverageTally, f64)>,
    width_retained: u32,
    replicates: u32,
    /// `(bound_lower, lower_se, critical_value)` per replicate.
    lower: Vec<(f64, f64, f64)>,
}

impl SetTally {
    fn new(name: &str, truth: f64, other: Option<f64>) -> Self {
        Self {
            name: name.to_owned(),
            truth: (
                CoverageTally::new(format!("{name} identified-set interval [true effect]"), LEVEL),
                truth,
            ),
            other: other.map(|value| {
                (
                    CoverageTally::new(
                        format!("{name} identified-set interval [other endpoint, info]"),
                        LEVEL,
                    ),
                    value,
                )
            }),
            width_retained: 0,
            replicates: 0,
            lower: Vec::new(),
        }
    }

    fn record(&mut self, result: &StudyResult) {
        let set = result
            .structural_response
            .as_ref()
            .expect("class mixture")
            .identified_set_interval
            .as_ref();
        let interval = set.map(|s| (s.lower, s.upper));
        if let Some(s) = set {
            self.lower.push((s.bound_lower, s.lower_se, s.critical_value));
        }
        self.truth.0.record(interval, self.truth.1);
        if let Some((tally, value)) = self.other.as_mut() {
            tally.record(interval, *value);
        }
        self.width_retained += u32::from(set.is_some_and(|s| s.width_retained));
        self.replicates += 1;
    }

    fn assert(&self) {
        let n = self.lower.len().max(2) as f64;
        let mean = self.lower.iter().map(|l| l.0).sum::<f64>() / n;
        let sd = (self.lower.iter().map(|l| (l.0 - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let se = self.lower.iter().map(|l| l.1).sum::<f64>() / n;
        let c = self.lower.iter().map(|l| l.2).sum::<f64>() / n;
        eprintln!(
            "info {}: lower bound bias={:+.4} mc_sd={sd:.4} mean_se={se:.4} se/sd={:.3} mean_c={c:.3}",
            self.name,
            mean - self.truth.1,
            se / sd
        );
        eprintln!(
            "info {}: set width retained in {}/{} replicates",
            self.name, self.width_retained, self.replicates
        );
        if let Some((tally, value)) = &self.other {
            eprintln!(
                "info {}: coverage of the other endpoint {value:.4} = {:.3} (not asserted)",
                self.name,
                tally.rate()
            );
        }
        self.truth.0.assert();
    }
}

fn identified_set_case<F>(name: &str, truth: f64, other: Option<f64>, mut result_for: F)
where
    F: FnMut(u32) -> StudyResult,
{
    let mut tally = SetTally::new(name, truth, other);
    for s in 0..n_sim() {
        let result = result_for(s);
        tally.record(&result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_identified_set_interval_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    identified_set_case("frequentist TemporalPag Pulse", adjusted, Some(unadjusted), |s| {
        run(
            chain_pag_series(N, 0.0, 74_000 + u64::from(s)),
            chain_pag(),
            &pulse_query(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        )
    });
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_multistep_identified_set_interval_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    identified_set_case(
        "frequentist TemporalPag multi-step Sustained",
        adjusted,
        Some(unadjusted),
        |s| {
            run(
                chain_pag_series(N, 0.0, 75_000 + u64::from(s)),
                chain_pag(),
                &multi_sustained(),
                InferenceMode::Frequentist,
                None,
                BOOT,
                u64::from(s),
            )
        },
    );
}

/// `confounded_series` is generated by the `z -> t` completion (adjust `z`), `θ = B1`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_identified_set_interval_nominal_90_coverage() {
    let [unadjusted, adjusted] = fixtures::cpdag_completion_truths(0.0);
    identified_set_case("frequentist TemporalCpdag Pulse", adjusted, Some(unadjusted), |s| {
        run(
            confounded_series(N, 0.0, 0.0, 76_000 + u64::from(s)),
            confounded_cpdag(),
            &pulse_query(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        )
    });
}

/// `circle_pag` identifies one completion: the set is a point and the interval
/// is the two-sided interval of that completion.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_point_identified_set_interval_nominal_90_coverage() {
    identified_set_case("frequentist TemporalPag one-completion Pulse", B1, None, |s| {
        run(
            fixtures::pag_series(N, 77_000 + u64::from(s)),
            fixtures::circle_pag(),
            &pulse_query(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        )
    });
}

/// K-2: without a `ClassPrior` the Bayesian class path publishes per-completion
/// posteriors only (no blended posterior). The identified-set credible interval
/// from those draws is what carries a coverage claim.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_no_class_prior_identified_set_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    identified_set_case(
        "Bayesian TemporalPag Pulse (no ClassPrior)",
        adjusted,
        Some(unadjusted),
        |s| {
            let result = run(
                chain_pag_series(N, 0.0, 78_000 + u64::from(s)),
                chain_pag(),
                &pulse_query(),
                bayes(),
                None,
                0,
                u64::from(s),
            );
            assert!(result.estimate.ate.is_nan(), "no blended posterior without a class prior");
            result
        },
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_no_class_prior_identified_set_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    identified_set_case(
        "Bayesian TemporalPag single-step Sustained (no ClassPrior)",
        adjusted,
        Some(unadjusted),
        |s| {
            run(
                chain_pag_series(N, 0.0, 79_000 + u64::from(s)),
                chain_pag(),
                &single_sustained(),
                bayes(),
                None,
                0,
                u64::from(s),
            )
        },
    );
}

const SANITY_N: usize = 20_000;

#[test]
fn chain_pag_fixture_identifies_six_completions_at_two_effects() {
    for rho in [0.0, 0.5] {
        let data = chain_pag_series(SANITY_N, rho, 3);
        for query in [pulse_query(), single_sustained(), multi_sustained()] {
            let result =
                run(data.clone(), chain_pag(), &query, InferenceMode::Frequentist, None, 0, 41);
            let values = atom_values(&result);
            eprintln!("chain_pag rho={rho} {:?}: {values:?}", query.policy);
            let truths = chain_truths(&query);
            assert_eq!(values.len(), truths.len(), "seven completions");
            for (value, truth) in values.iter().zip(&truths) {
                match truth {
                    Some(truth) => {
                        close("chain completion plim", value.expect("identified"), *truth, 0.03);
                    }
                    None => assert!(value.is_none(), "completion must carry no value"),
                }
            }
            close("chain mixture", result.estimate.ate, chain_mixture_truth(&query), 0.03);
        }
    }
}

#[test]
fn confounded_cpdag_multistep_composes_each_completion() {
    let data = confounded_series(SANITY_N, 0.0, 0.5, 5);
    let result =
        run(data, confounded_cpdag(), &multi_sustained(), InferenceMode::Frequentist, None, 0, 41);
    let values = atom_values(&result);
    eprintln!("confounded_cpdag multi-step: {values:?}");
    for (value, truth) in values.iter().zip(fixtures::cpdag_completion_truths(0.5)) {
        close("cpdag multi-step completion", value.expect("identified"), truth, 0.03);
    }
    let _ = B1;
    let _ = quantile_interval(&[0.0, 1.0], LEVEL);
}
