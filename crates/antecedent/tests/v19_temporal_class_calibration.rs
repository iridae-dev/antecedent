//! Coverage of temporal class envelopes not covered by `v19_calibration`
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

#![allow(clippy::cast_possible_truncation, clippy::float_cmp, clippy::too_many_lines)]

mod common;

use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, ResponseValue, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use common::calibration::{
    BASE_GRID_POINT, CoverageTally, REPORTED_LEVEL, RecordKey, Z90, Z95, grid_n, grid_point, n_sim,
    normal_interval, quantile_interval, smoke,
};
use common::calibration_bind::{bind, bind_all};
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
    run_study(data, graph, query, inference, prior, boot, ctx_seed).1
}

/// [`run`], returning the study as well (coverage records bind to it).
fn run_study(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: &TemporalEffectQuery,
    inference: InferenceMode,
    prior: Option<ClassPrior>,
    boot: u32,
    ctx_seed: u64,
) -> (Study, StudyResult) {
    let mut builder = Study::series(data)
        .graph(graph.into())
        .query(CausalQuery::TemporalEffect(query.clone()))
        .inference(inference);
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    let study = builder.refute(RefuteSuite::None).bootstrap_replicates(boot).build().unwrap();
    let result = study.run(&ExecutionContext::for_tests(ctx_seed)).unwrap();
    (study, result)
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
///
/// That interval is the runtime's primary interval on the class path: the
/// frozen-weight mixture's point with its shared circular-block SE, keyed
/// `circular_block_se` under the mixture block family. `tally` is gated at
/// [`LEVEL`] and `reported` scores the reported 95% interval on the same
/// replicates (recorded, not gated).
struct FreqTally {
    name: String,
    tally: CoverageTally,
    reported: CoverageTally,
    estimates: Vec<f64>,
    ses: Vec<f64>,
    truth: f64,
    warned: u32,
}

impl FreqTally {
    fn new(test: &'static str, dgp: &'static str, name: &str, truth: f64) -> Self {
        let key = RecordKey { test, dgp, interval: "circular_block_se" };
        Self {
            name: name.to_owned(),
            tally: CoverageTally::for_record(key, LEVEL),
            reported: CoverageTally::for_record(key, REPORTED_LEVEL).unasserted(),
            estimates: Vec::new(),
            ses: Vec::new(),
            truth,
            warned: 0,
        }
    }

    fn record(&mut self, study: &Study, result: &StudyResult) {
        let est = &result.estimate;
        let interval = normal_interval(est.ate, est.se_bootstrap, Z90);
        if interval.is_some() {
            bind_all(&mut [&mut self.tally, &mut self.reported], study, result);
        }
        self.tally.record(interval, self.truth);
        self.reported.record(normal_interval(est.ate, est.se_bootstrap, Z95), self.truth);
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
        self.reported.emit();
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

fn frequentist_chain_pag_case(
    test: &'static str,
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    chain_pag_tally(test, what, query, design, seed).assert();
}

/// Boundary record, as in `v19_temporal_frequentist`: the six-completion
/// mixture's score effective rows sit below the mixture threshold.
fn frequentist_chain_pag_boundary(
    test: &'static str,
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    boundary(&chain_pag_tally(test, what, query, design, seed));
}

/// Boundary record: coverage is measured and printed, not gated, and the
/// short-series warning must fire on at least 95% of replicates (the statistic
/// is estimated per series, so a few draws land above the mixture threshold).
fn boundary(tally: &FreqTally) {
    tally.report();
    let (lo, hi) = common::calibration::coverage_band(n_sim(), LEVEL);
    eprintln!(
        "info {}: nominal band=[{lo:.3}, {hi:.3}] (not gated; short_series warnings {}/{})",
        tally.name,
        tally.warned,
        n_sim()
    );
    tally.tally.emit_named_boundary();
    tally.reported.emit();
    // The warning is a property of the base sample size the design names; at
    // the other grid points its rate is reported and coverage recorded.
    if grid_point() == BASE_GRID_POINT && !smoke() {
        assert!(
            tally.warned * 20 >= n_sim() * 19,
            "boundary design must carry the short-series warning: {}/{}",
            tally.warned,
            n_sim()
        );
    }
}

fn chain_pag_tally(
    test: &'static str,
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) -> FreqTally {
    let mut tally = FreqTally::new(
        test,
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
        &format!("frequentist TemporalPag {what} [{}]", design.label),
        chain_mixture_truth(query),
    );
    for s in 0..n_sim() {
        let (study, result) = run_study(
            chain_pag_series(grid_n(design.n), design.rho, seed + u64::from(s)),
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
        tally.record(&study, &result);
    }
    tally
}

fn frequentist_cpdag_multistep_case(test: &'static str, design: Design, seed: u64) {
    let mut tally = FreqTally::new(
        test,
        "crates/antecedent/tests/common/fixtures.rs::confounded_series",
        &format!("frequentist TemporalCpdag multi-step Sustained [{}]", design.label),
        cpdag_mixture_truth(design.rho),
    );
    for s in 0..n_sim() {
        let (study, result) = run_study(
            confounded_series(grid_n(design.n), 0.0, design.rho, seed + u64::from(s)),
            confounded_cpdag(),
            &multi_sustained(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        tally.record(&study, &result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_multistep_sustained_nominal_90_coverage() {
    frequentist_cpdag_multistep_case(
        "frequentist_temporal_cpdag_multistep_sustained_nominal_90_coverage",
        IID,
        60_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_multistep_sustained_ar1_rho05_n160_nominal_90_coverage() {
    frequentist_cpdag_multistep_case(
        "frequentist_temporal_cpdag_multistep_sustained_ar1_rho05_n160_nominal_90_coverage",
        AR05_160,
        61_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_multistep_sustained_nominal_90_coverage() {
    frequentist_chain_pag_case(
        "frequentist_temporal_pag_multistep_sustained_nominal_90_coverage",
        "multi-step Sustained",
        &multi_sustained(),
        IID,
        62_000,
    );
}

macro_rules! frequentist_pag_gate {
    ($name:ident, $what:literal, $query:expr, $design:expr, $seed:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            frequentist_chain_pag_case(stringify!($name), $what, &$query, $design, $seed);
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
/// Below the mixture threshold on (nearly) every replicate: the six-completion
/// mixture at ρ = 0.9, n = 400 (every atom's score is close to AR(1)(0.81);
/// score effective rows ≈ 45 against 155). Coverage sits at the band's edge
/// (0.888 at 2000 replicates in `v19_short_series_measurement`): measured,
/// warned, not gated.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_pulse_ar1_rho09_n400_short_series_boundary() {
    frequentist_chain_pag_boundary(
        "frequentist_temporal_pag_pulse_ar1_rho09_n400_short_series_boundary",
        "Pulse",
        &pulse_query(),
        AR09_400,
        66_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_sustained_ar1_rho09_n400_short_series_boundary() {
    frequentist_chain_pag_boundary(
        "frequentist_temporal_pag_sustained_ar1_rho09_n400_short_series_boundary",
        "single-step Sustained",
        &single_sustained(),
        AR09_400,
        69_000,
    );
}

/// The two-completion `TemporalCpdag` Pulse at ρ = 0.95, n = 160: the non-causal
/// completion omits the persistent confounder `z[t-1]`, and its finite-sample
/// bias (−0.4 SD) — not the SE — drives coverage to 0.81 (2000 replicates). No
/// block bootstrap removes a bias; the runtime warns (that completion's score
/// has a weak, slowly decaying component the block-length reading sees), and
/// coverage is recorded.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_ar1_rho095_n160_short_series_boundary() {
    const RHO: f64 = 0.95;
    let mut tally = FreqTally::new(
        "frequentist_temporal_cpdag_pulse_ar1_rho095_n160_short_series_boundary",
        "crates/antecedent/tests/common/fixtures.rs::confounded_series",
        "frequentist TemporalCpdag Pulse [AR(1) rho=0.95 n=160 (boundary)]",
        cpdag_mixture_truth(RHO),
    );
    for s in 0..n_sim() {
        let (study, result) = run_study(
            confounded_series(grid_n(160), 0.0, RHO, 90_000 + u64::from(s)),
            confounded_cpdag(),
            &pulse_query(),
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        tally.record(&study, &result);
    }
    boundary(&tally);
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
    posterior_interval_at(post, LEVEL)
}

fn posterior_interval_at(
    post: Option<&antecedent::CausalPosterior>,
    level: f64,
) -> Option<(f64, f64)> {
    let post = post?;
    let col = post.effect_column()?;
    quantile_interval(post.draws.column(col).ok()?, level)
}

/// `circle_pag` with class prior `[0.4, 0.6]`: the draw-level BMA over the one
/// identified completion is that completion's posterior (`θ = B1`), and the
/// unidentified completion's 0.4 stays a separate axis. `chain_pag` cannot carry
/// a class-prior mixture: its global m-separation audit is capped, so the class
/// scope is not full and mixing is refused (its no-prior identified-set interval
/// is calibrated below instead).
///
/// Gated at [`LEVEL`]; the runtime's reported 95% posterior interval is scored
/// on the same replicates and recorded.
fn bayesian_circle_pag_class_prior_case(
    test: &'static str,
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    let prior = ClassPrior::from_ordered([0.4, 0.6]).unwrap();
    eprintln!("calibration {test}: Bayesian TemporalPag {what} class-prior [{}]", design.label);
    let key = RecordKey {
        test,
        dgp: "crates/antecedent/tests/common/fixtures.rs::pag_series_ar1",
        interval: "posterior_quantile",
    };
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    for s in 0..n_sim() {
        let (study, result) = run_study(
            fixtures::pag_series_ar1(grid_n(design.n), design.rho, seed + u64::from(s)),
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
        let interval = posterior_interval(Some(post));
        if interval.is_some() {
            bind_all(&mut [&mut tally, &mut reported], &study, &result);
        }
        tally.record(interval, B1);
        reported.record(posterior_interval_at(Some(post), REPORTED_LEVEL), B1);
    }
    tally.assert();
    reported.emit();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_class_prior_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case(
        "bayesian_temporal_pag_sustained_class_prior_nominal_90_coverage",
        "single-step Sustained",
        &single_sustained(),
        IID,
        71_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_pulse_class_prior_ar1_rho05_n160_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case(
        "bayesian_temporal_pag_pulse_class_prior_ar1_rho05_n160_nominal_90_coverage",
        "Pulse",
        &pulse_query(),
        AR05_160,
        72_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_class_prior_ar1_rho05_n160_nominal_90_coverage() {
    bayesian_circle_pag_class_prior_case(
        "bayesian_temporal_pag_sustained_class_prior_ar1_rho05_n160_nominal_90_coverage",
        "single-step Sustained",
        &single_sustained(),
        AR05_160,
        73_000,
    );
}

/// Frequentist `circle_pag` (one identified completion) under AR(1): the
/// reported aggregate is that completion's shared-block interval.
fn frequentist_circle_pag_case(
    test: &'static str,
    what: &str,
    query: &TemporalEffectQuery,
    design: Design,
    seed: u64,
) {
    let mut tally = FreqTally::new(
        test,
        "crates/antecedent/tests/common/fixtures.rs::pag_series_ar1",
        &format!("frequentist TemporalPag one-completion {what} [{}]", design.label),
        B1,
    );
    for s in 0..n_sim() {
        let (study, result) = run_study(
            fixtures::pag_series_ar1(grid_n(design.n), design.rho, seed + u64::from(s)),
            fixtures::circle_pag(),
            query,
            InferenceMode::Frequentist,
            None,
            BOOT,
            u64::from(s),
        );
        tally.record(&study, &result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_one_completion_sustained_ar1_rho05_n160_nominal_90_coverage() {
    frequentist_circle_pag_case(
        "frequentist_temporal_pag_one_completion_sustained_ar1_rho05_n160_nominal_90_coverage",
        "single-step Sustained",
        &single_sustained(),
        AR05_160,
        80_000,
    );
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
/// 4–10 completion SDs apart at the design's `n`, so the width is retained, the
/// critical value is ≈ one-sided, and the true effect sits at an endpoint of the
/// set: coverage is then nominal (a miss happens only in the tail on the side of
/// the true completion). A one-completion set is a point and the interval is
/// the two-sided atom interval, also nominal. Over-coverage in these designs
/// would mean the construction is wider than IM requires, which the band is
/// meant to catch. Coverage of the *other* endpoint (the non-causal completion's
/// estimand) is printed for the record. The heterogeneous-SD designs further
/// below sit about one noisy-completion SD apart, where the construction is
/// conservative by design, and are gated one-sided.
///
/// `chain_pag`'s equivalence audit is capped, so its intervals are published
/// flagged `truncated`: they span the retained completions, and the generating
/// graph is one of them.
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
    /// `key` makes the true-effect tally back a coverage record of the reported
    /// `identified_set` interval (at its reported level, [`LEVEL`]).
    fn new(name: &str, key: Option<RecordKey>, truth: f64, other: Option<f64>) -> Self {
        let truth_tally = match key {
            Some(key) => CoverageTally::for_record(key, LEVEL),
            None => {
                CoverageTally::new(format!("{name} identified-set interval [true effect]"), LEVEL)
            }
        };
        Self {
            name: name.to_owned(),
            truth: (truth_tally, truth),
            other: other.map(|value| {
                (
                    // No record: coverage of the non-causal completion's estimand is info only.
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

    /// Score one replicate; `study` binds it to the record when the tally has one.
    fn record(&mut self, study: Option<&Study>, result: &StudyResult) {
        let set = result
            .structural_response
            .as_ref()
            .expect("class mixture")
            .identified_set_interval
            .as_ref();
        let interval = set.map(|s| (s.lower, s.upper));
        if let Some(s) = set {
            self.lower.push((s.bound_lower, s.lower_se, s.critical_value));
            if let Some(study) = study {
                bind(&mut self.truth.0, study, result);
            }
        }
        self.truth.0.record(interval, self.truth.1);
        if let Some((tally, value)) = self.other.as_mut() {
            tally.record(interval, *value);
        }
        self.width_retained += u32::from(set.is_some_and(|s| s.width_retained));
        self.replicates += 1;
    }

    fn assert(&self) {
        self.report();
        self.truth.0.assert();
    }

    fn report(&self) {
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
    }
}

fn identified_set_case<F>(
    key: RecordKey,
    name: &str,
    truth: f64,
    other: Option<f64>,
    mut run_for: F,
) where
    F: FnMut(u32) -> (Study, StudyResult),
{
    let mut tally = SetTally::new(name, Some(key), truth, other);
    for s in 0..n_sim() {
        let (study, result) = run_for(s);
        tally.record(Some(&study), &result);
    }
    tally.assert();
}

/// Record key of an identified-set coverage test on a `fixtures` DGP.
fn set_key(test: &'static str, dgp: &'static str) -> RecordKey {
    RecordKey { test, dgp, interval: "identified_set" }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_identified_set_interval_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    let key = set_key(
        "frequentist_temporal_pag_identified_set_interval_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
    );
    identified_set_case(key, "frequentist TemporalPag Pulse", adjusted, Some(unadjusted), |s| {
        run_study(
            chain_pag_series(grid_n(N), 0.0, 74_000 + u64::from(s)),
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
    let key = set_key(
        "frequentist_temporal_pag_multistep_identified_set_interval_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
    );
    identified_set_case(
        key,
        "frequentist TemporalPag multi-step Sustained",
        adjusted,
        Some(unadjusted),
        |s| {
            run_study(
                chain_pag_series(grid_n(N), 0.0, 75_000 + u64::from(s)),
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
    let key = set_key(
        "frequentist_temporal_cpdag_identified_set_interval_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::confounded_series",
    );
    identified_set_case(key, "frequentist TemporalCpdag Pulse", adjusted, Some(unadjusted), |s| {
        run_study(
            confounded_series(grid_n(N), 0.0, 0.0, 76_000 + u64::from(s)),
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
    let key = set_key(
        "frequentist_temporal_pag_point_identified_set_interval_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::pag_series",
    );
    identified_set_case(key, "frequentist TemporalPag one-completion Pulse", B1, None, |s| {
        run_study(
            fixtures::pag_series(grid_n(N), 77_000 + u64::from(s)),
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
    let key = set_key(
        "bayesian_temporal_pag_no_class_prior_identified_set_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
    );
    identified_set_case(
        key,
        "Bayesian TemporalPag Pulse (no ClassPrior)",
        adjusted,
        Some(unadjusted),
        |s| {
            let (study, result) = run_study(
                chain_pag_series(grid_n(N), 0.0, 78_000 + u64::from(s)),
                chain_pag(),
                &pulse_query(),
                bayes(),
                None,
                0,
                u64::from(s),
            );
            assert!(result.estimate.ate.is_nan(), "no blended posterior without a class prior");
            (study, result)
        },
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_sustained_no_class_prior_identified_set_nominal_90_coverage() {
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    let key = set_key(
        "bayesian_temporal_pag_sustained_no_class_prior_identified_set_nominal_90_coverage",
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
    );
    identified_set_case(
        key,
        "Bayesian TemporalPag single-step Sustained (no ClassPrior)",
        adjusted,
        Some(unadjusted),
        |s| {
            run_study(
                chain_pag_series(grid_n(N), 0.0, 79_000 + u64::from(s)),
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

// ---------------------------------------------------------------------------
// Identified-set intervals with heterogeneous completion SDs
// ---------------------------------------------------------------------------

/// `z → t` loading of the heterogeneous-SD design.
const HET_A: f64 = 1.0;
/// Treatment innovation SD: small, so `t` is nearly collinear with `z`.
const HET_S: f64 = 0.2;
/// Effect of `z[t-1]` on `y[t]`.
const HET_C: f64 = 0.25;
/// Outcome noise SD.
const HET_SD_E: f64 = 0.5;

/// `z` iid N(0, 1), `t = HET_A z + HET_S e`, `y[t] = B1 t[t-1] + HET_C z[t-1] +
/// HET_SD_E u[t]` on the [`confounded_cpdag`] structure. Columns `t, y, z`.
///
/// `t` is nearly collinear with `z`, so the adjusted (true, `z -> t`)
/// completion is noisy — SD ≈ `HET_SD_E / (HET_S √n)` ≈ 0.20 at n = 160 —
/// while the unadjusted completion regresses on all of `t`'s variance and is
/// about five times more precise. The unadjusted plim sits
/// `HET_C·HET_A / (HET_A² + HET_S²)` ≈ 0.24 (≈ 1.2 noisy-completion SDs) above
/// `B1`.
fn heterogeneous_se_series(n: usize, seed: u64) -> TimeSeriesData {
    let total = n + 1;
    let mut draw = common::calibration::gaussian(seed);
    let z: Vec<f64> = (0..total).map(|_| draw()).collect();
    let t: Vec<f64> = z.iter().map(|z| HET_A * z + HET_S * draw()).collect();
    let mut y = vec![0.0; total];
    for i in 1..total {
        y[i] = B1 * t[i - 1] + HET_C * z[i - 1] + HET_SD_E * draw();
    }
    TimeSeriesData::from_f64_columns([("t", &t[1..]), ("y", &y[1..]), ("z", &z[1..])], 1).unwrap()
}

/// Unadjusted completion's plim: `B1 + HET_C·cov(t, z) / var(t)`.
fn heterogeneous_se_unadjusted_plim() -> f64 {
    B1 + HET_C * HET_A / (HET_A * HET_A + HET_S * HET_S)
}

/// One-sided record: the true (noisy) completion must not be under-covered.
///
/// A noisy completion next to a precise one is the case where an interval built
/// on the SD of the min / max of completion estimates under-covers: the min is
/// capped by the precise estimate, so its SD falls well below the noisy
/// completion's own. In an idealised simulation (two jointly normal completion
/// estimates with SDs 1 and 0.2, 1.2 SDs apart, replicates drawn around the
/// estimates, n = 400 selection threshold) that construction covers the noisy
/// completion 0.85 of the time at nominal 0.90. The per-completion
/// construction keeps the noisy completion's own interval inside the published
/// one and, at a width this close to the noise, is conservative (0.95 in the
/// same simulation; 0.955 / 0.950 Frequentist / Bayesian here at 400
/// replicates), so the assertion is the lower edge of the `level ± 3·MCSE` band.
fn heterogeneous_se_case<F>(name: &str, mut result_for: F)
where
    F: FnMut(u32) -> StudyResult,
{
    // No record: this cell is gated one-sided (at least the band's lower edge), which
    // the record harness has no role for; the construction is conservative here by design.
    let mut tally = SetTally::new(name, None, B1, Some(heterogeneous_se_unadjusted_plim()));
    for s in 0..n_sim() {
        tally.record(None, &result_for(s));
    }
    tally.report();
    let (lo, hi) = common::calibration::coverage_band(n_sim(), LEVEL);
    let rate = tally.truth.0.rate();
    eprintln!(
        "calibration {name} identified-set interval [true noisy completion]: coverage={rate:.3} \
         band=[{lo:.3}, {hi:.3}] (one-sided: at least {lo:.3}) mean_length={:.4}",
        tally.truth.0.mean_length()
    );
    assert!(rate >= lo, "{name}: true completion under-covered, {rate:.3} < {lo:.3}");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_heterogeneous_se_identified_set_interval_covers_noisy_completion() {
    heterogeneous_se_case("frequentist TemporalCpdag Pulse heterogeneous SDs", |s| {
        run(
            heterogeneous_se_series(grid_n(N), 81_000 + u64::from(s)),
            confounded_cpdag(),
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
fn bayesian_temporal_cpdag_heterogeneous_se_identified_set_covers_noisy_completion() {
    heterogeneous_se_case("Bayesian TemporalCpdag Pulse heterogeneous SDs (no ClassPrior)", |s| {
        run(
            heterogeneous_se_series(grid_n(N), 82_000 + u64::from(s)),
            confounded_cpdag(),
            &pulse_query(),
            bayes(),
            None,
            0,
            u64::from(s),
        )
    });
}

/// The heterogeneous-SD design delivers what the calibration above relies on:
/// completion plims at `B1` and the unadjusted plim, with the adjusted
/// completion several times noisier than the unadjusted one.
#[test]
fn heterogeneous_se_fixture_has_a_noisy_true_completion() {
    let result = run(
        heterogeneous_se_series(SANITY_N, 9),
        confounded_cpdag(),
        &pulse_query(),
        InferenceMode::Frequentist,
        None,
        BOOT,
        9,
    );
    let values = atom_values(&result);
    close("unadjusted completion", values[0].unwrap(), heterogeneous_se_unadjusted_plim(), 0.02);
    close("adjusted completion", values[1].unwrap(), B1, 0.03);
    let set = result.structural_response.as_ref().unwrap().identified_set_interval.unwrap();
    assert!(!set.truncated, "a CPDAG enumeration is complete");
    // The true completion is the lower one; its own SD sets the lower endpoint.
    assert!(set.lower_se > 3.0 * set.upper_se, "noisy lower completion: {set:?}");
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

/// `chain_pag`'s capped equivalence audit leaves the class scope partial: both
/// inference modes still publish the identified-set interval, flagged
/// `truncated` and paired with a warning.
#[test]
fn chain_pag_identified_set_interval_is_flagged_truncated() {
    const TRUNCATED: &str = "estimate.temporal_class.identified_set_interval_truncated";
    for (inference, boot) in [(InferenceMode::Frequentist, 32), (bayes(), 0)] {
        let result = run(
            chain_pag_series(grid_n(N), 0.0, 11),
            chain_pag(),
            &pulse_query(),
            inference,
            None,
            boot,
            11,
        );
        let structural = result.structural_response.as_ref().expect("class mixture");
        assert!(!structural.full_mass_scope, "fixture: the audit is capped");
        let set = structural.identified_set_interval.expect("interval published");
        assert!(set.truncated, "{set:?}");
        assert_eq!(set.completions, 6, "every retained identified completion enters");
        let warning = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == TRUNCATED)
            .expect("truncation warning");
        assert_eq!(warning.severity, antecedent_core::DiagnosticSeverity::Warning);
    }
}

/// `t@-1 o-o z@-1`, both into `y`: three MAG completions over a fully audited
/// class, one of them `t <-> z` (identified, but unevaluable by sequential
/// g-computation). Under a class prior the multi-step Sustained mixture draws
/// only from the directed completions; the bidirected completion's mass is
/// unevaluable on both the structural mixture and the mixed posterior, never
/// unidentified.
#[test]
fn unevaluable_mag_completion_is_not_unidentified_on_the_mixed_posterior() {
    use antecedent_core::Lag;
    let mut pag = antecedent_graph::TemporalPag::empty();
    let t1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = pag.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    pag.insert_directed(t1, y0).unwrap();
    pag.insert_directed(z1, y0).unwrap();
    pag.insert_circle_circle_with_middle(t1, z1, antecedent_graph::MiddleMark::Empty).unwrap();
    let prior = ClassPrior::from_ordered([0.2, 0.3, 0.5]).unwrap();
    let result = run(
        confounded_series(400, 0.0, 0.0, 3),
        pag,
        &multi_sustained(),
        bayes(),
        Some(prior),
        0,
        3,
    );
    let structural = result.structural_response.as_ref().expect("class mixture");
    assert!(structural.full_mass_scope, "fixture: the class audit is complete");
    assert!(structural.unevaluable_mass > 0.0, "fixture: one bidirected completion");
    let posterior = result.posterior.as_ref().expect("class-prior mixture");
    assert!(
        (posterior.unidentified_mass - structural.unidentified_mass).abs() < 1e-12,
        "posterior unidentified {} vs structural {} (unevaluable {})",
        posterior.unidentified_mass,
        structural.unidentified_mass,
        structural.unevaluable_mass
    );
    let set = structural.identified_set_interval.expect("identified-set interval");
    assert!(!set.truncated);
    assert_eq!(set.completions, 2, "the two evaluable directed completions");
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

/// The mixture short-series threshold on fixed series: the printed score
/// effective rows decide the warning, and the six-completion `chain_pag`
/// envelope warns at ρ = 0.9 and stays quiet at ρ = 0.5 (n = 400).
#[test]
fn short_series_mixture_threshold() {
    use antecedent_estimate::CircularBlockFamily;
    const KEY: &str = "score effective rows ";
    let threshold = CircularBlockFamily::Mixture.min_effective_rows();
    for (rho, expect_warning) in [(0.9, true), (0.5, false)] {
        let result = run(
            chain_pag_series(400, rho, 23),
            chain_pag(),
            &pulse_query(),
            InferenceMode::Frequentist,
            None,
            8,
            23,
        );
        let message = &result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "estimate.temporal_class.frequentist.shared_block")
            .expect("shared-block provenance")
            .message;
        let rest = &message[message.find(KEY).expect("effective rows") + KEY.len()..];
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
        let effective_rows: f64 = rest[..end].parse().unwrap();
        let warned = has_diagnostic(&result, SHORT_SERIES);
        assert_eq!(
            warned,
            effective_rows < threshold,
            "rho={rho}: {effective_rows} vs {threshold}"
        );
        assert_eq!(warned, expect_warning, "rho={rho}: effective rows {effective_rows}");
    }
}
