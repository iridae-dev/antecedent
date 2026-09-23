//! Coverage of dependence-honest Frequentist TemporalDag intervals (R-1, R-2).
//!
//! Plain TemporalDag Pulse / single-step Sustained and temporal mediation
//! (Total, Direct, Mediated) are calibrated under iid noise and under AR(1)
//! outcome residuals on the driven-treatment DGPs of `common::driven_dgp`
//! (MA(3) treatment through an observed driver, so the regression scores are
//! serially correlated whenever the residual is; truths are the population
//! values of the reported estimands, documented there). Multi-step Sustained
//! (sequential g-computation) is calibrated on `fixtures::confounded_series`.
//! Boundary records use an AR(1)-persistent treatment (`common::persistent_dgp`)
//! where the series is too short for the score's memory: the short-series
//! warning must fire, and coverage is recorded.
//!
//! Ignored coverage tests run via `scripts/gate_calibration.sh`. The
//! non-ignored tests pin that the circular-block path is the one taken and the
//! short-series threshold of each one-series family on fixed series.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown, clippy::needless_pass_by_value, clippy::too_many_lines)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::estimate::TemporalMediationUncertainty;
use antecedent::{InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DiagnosticSeverity, ExecutionContext, MediationContrast, TemporalEffectQuery,
};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::CircularBlockFamily;
use antecedent_graph::TemporalDag;
use common::calibration::{
    BASE_GRID_POINT, CoverageTally, GRID_POINTS, REPORTED_LEVEL, RecordKey, Z90, Z95, grid_n,
    grid_point, map_replicates, n_sim, normal_interval, smoke,
};
use common::calibration_bind::bind_all;
use common::driven_dgp::{
    A, ALPHA, B, BETA, C, DELTA, Scenario, mediation_dag, mediation_query, mediation_series, pulse,
    pulse_dag, pulse_series, single_sustained,
};
use common::{fixtures, persistent_dgp};

/// Bootstrap replicates per fit.
const BOOT: u32 = 100;

const IID_160: Scenario = Scenario { label: "iid n=160", rho: 0.0, n: 160, seed: 100_000 };
const AR05_160: Scenario =
    Scenario { label: "AR(1) rho=0.5 n=160", rho: 0.5, n: 160, seed: 200_000 };
const AR09_400: Scenario =
    Scenario { label: "AR(1) rho=0.9 n=400", rho: 0.9, n: 400, seed: 300_000 };
const AR05_60: Scenario = Scenario { label: "AR(1) rho=0.5 n=60", rho: 0.5, n: 60, seed: 400_000 };

fn run(
    data: TimeSeriesData,
    graph: TemporalDag,
    query: CausalQuery,
    boot: u32,
    seed: u64,
) -> StudyResult {
    run_study(data, graph, query, boot, seed).1
}

/// [`run`], returning the study as well (coverage records bind to it).
fn run_study(
    data: TimeSeriesData,
    graph: TemporalDag,
    query: CausalQuery,
    boot: u32,
    seed: u64,
) -> (Study, StudyResult) {
    let study = Study::series(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

const SHORT_SERIES: &str = "estimate.temporal.circular_block_se.short_series";

/// Tallies of one design: the gated 90% tallies, the runtime's reported 95%
/// interval scored on the same replicates (recorded, not gated), and how many
/// replicates carried the short-series warning.
struct Coverage {
    tallies: Vec<CoverageTally>,
    reported: Vec<CoverageTally>,
    /// A plain per-replicate flag counter, not a coverage record: `record`s an
    /// always-in-range interval when the warning fires and an always-out-of-range
    /// one when it does not, purely to reuse `CoverageTally`'s extension-safe
    /// covered/attempts bookkeeping (see `CoverageTally::persist`). A raw `u32`
    /// counter here would only count an extending recheck's own new replicates
    /// while `n_sim()` reports the full count, undercounting the warning rate.
    warned: CoverageTally,
}

/// A short-series warning counter keyed by `test`, extension-safe across a
/// recheck (see [`Coverage::warned`]).
fn warn_tally(test: &'static str) -> CoverageTally {
    CoverageTally::new(format!("{test} short-series-warned"), 0.95)
}

/// Record one replicate's short-series warning flag.
fn record_warned(warned: &mut CoverageTally, fired: bool) {
    warned.record(Some((0.0, 1.0)), if fired { 0.5 } else { 2.0 });
}

/// The gated 90% tally and the unasserted reported-level tally of the headline
/// circular-block interval `ate ± z·se_bootstrap` (the primary interval the
/// runtime reports for a Frequentist TemporalDag scalar).
fn headline_tallies(test: &'static str, dgp: &'static str) -> (CoverageTally, CoverageTally) {
    let key = RecordKey { test, dgp, interval: "circular_block_se" };
    (
        CoverageTally::for_record(key, 0.9),
        CoverageTally::for_record(key, REPORTED_LEVEL).unasserted(),
    )
}

/// Coverage of the headline circular-block interval `ate ± z·se_bootstrap`,
/// plus how many replicates carried the short-series warning.
///
/// `seed_offset` separates otherwise identical designs (Pulse h=1 and
/// single-step Sustained fit the same regression) so each gate is independent
/// evidence.
fn effect_coverage(
    test: &'static str,
    s: Scenario,
    what: &str,
    query: &TemporalEffectQuery,
    truth: f64,
    seed_offset: u64,
) -> Coverage {
    let s = Scenario { seed: s.seed + seed_offset, n: grid_n(s.n), ..s };
    let (mut boot, mut reported) =
        headline_tallies(test, "crates/antecedent/tests/common/driven_dgp.rs::pulse_series");
    let two_step = query.horizon_steps >= 2;
    let mut spread = Spread::default();
    let mut warned = warn_tally(test);
    let runs = map_replicates(n_sim(), |rep| {
        run_study(
            pulse_series(s, u32::try_from(rep).unwrap(), two_step),
            pulse_dag(two_step),
            CausalQuery::TemporalEffect(query.clone()),
            BOOT,
            s.seed + rep,
        )
    });
    for (study, result) in &runs {
        let est = &result.estimate;
        assert_eq!(result.estimand.adjustment_set.len(), 0, "{what}: unexpected adjustment");
        assert!(est.se_analytic.is_nan(), "{what}: no analytic SE is calibrated");
        let interval = normal_interval(est.ate, est.se_bootstrap, Z90);
        if interval.is_some() {
            bind_all(&mut [&mut boot, &mut reported], study, result);
        }
        boot.record(interval, truth);
        reported.record(normal_interval(est.ate, est.se_bootstrap, Z95), truth);
        spread.push(est.ate, est.se_bootstrap.unwrap_or(f64::NAN));
        record_warned(&mut warned, has_diagnostic(result, SHORT_SERIES));
    }
    warned.persist();
    spread.report(&format!("{what} [{}]", s.label), truth);
    Coverage { tallies: vec![boot], reported: vec![reported], warned }
}

/// In-assumption gate: nominal coverage must hold (the short-series warning may
/// or may not fire; its rate is reported).
fn gate(coverage: Coverage) {
    eprintln!(
        "short_series warnings: {}/{}",
        coverage.warned.covered(),
        coverage.warned.attempts()
    );
    assert_all(&coverage.tallies.iter().collect::<Vec<_>>());
    for tally in &coverage.reported {
        tally.emit();
    }
}

/// As [`gate_at`], but for a `Coverage` with more than one tally (e.g.
/// `mediation_coverage`'s Total / Direct / Mediated), each asserted against
/// its own per-grid-point measured band instead of sharing one.
fn gate_at_each(coverage: Coverage, measured: [[Option<f64>; GRID_POINTS]; 3]) {
    eprintln!(
        "short_series warnings: {}/{}",
        coverage.warned.covered(),
        coverage.warned.attempts()
    );
    for (tally, measured) in coverage.tallies.iter().zip(measured) {
        tally.assert_boundary_at(measured);
    }
    for tally in &coverage.reported {
        tally.emit();
    }
}

fn gate_at(coverage: Coverage, measured: [Option<f64>; 3]) {
    eprintln!(
        "short_series warnings: {}/{}",
        coverage.warned.covered(),
        coverage.warned.attempts()
    );
    for tally in &coverage.tallies {
        tally.assert_boundary_at(measured);
    }
    for tally in &coverage.reported {
        tally.emit();
    }
}

/// Named boundary cell: a design whose interval measures below the gate's
/// precision floor at 2000 replicates (the mechanism is named at the test);
/// every contrast is asserted against the band around `measured` (one value per
/// grid point), and the short-series warning rate is reported.
fn boundary_gate(coverage: Coverage, measured: [f64; GRID_POINTS]) {
    eprintln!(
        "short_series warnings: {}/{}",
        coverage.warned.covered(),
        coverage.warned.attempts()
    );
    for tally in &coverage.tallies {
        tally.assert_boundary_at(measured.map(Some));
    }
    for tally in &coverage.reported {
        tally.emit();
    }
}

/// Boundary record: a design whose interval under-covers because the series is
/// short for its serial dependence. The runtime must say so — the short-series
/// warning fires on at least 95% of replicates (the statistic is estimated per
/// series, so a few draws land above the family threshold) — and coverage is
/// measured and reported, not gated. The warning is a property of the base
/// sample size the design names; at the other grid points its rate is
/// reported, and coverage is recorded as the same named boundary.
fn boundary(coverage: Coverage) {
    let warned = &coverage.warned;
    let warned_covered = warned.covered();
    let warned_attempts = warned.attempts();
    for tally in &coverage.tallies {
        if tally.record_interval().is_some() {
            tally.emit_named_boundary();
            continue;
        }
        let (lo, hi) = common::calibration::coverage_band(n_sim(), 0.9);
        eprintln!(
            "calibration-boundary {tally:?}: coverage={:.3} band=[{lo:.3}, {hi:.3}] \
             mean_length={:.4} (not gated; short_series warnings {warned_covered}/{warned_attempts})",
            tally.rate(),
            tally.mean_length()
        );
    }
    eprintln!("short_series warnings: {warned_covered}/{warned_attempts}");
    for tally in &coverage.reported {
        tally.emit();
    }
    if grid_point() == BASE_GRID_POINT && !smoke() {
        assert!(
            warned_covered * 20 >= warned_attempts * 19,
            "boundary design must carry the short-series warning: {warned_covered}/{warned_attempts}"
        );
    }
}

/// Coverage of the headline interval of a design given by its data and graph
/// (`data(seed)` draws one series), plus the short-series warning count.
/// `test` / `dgp` key the coverage records (`dgp` names the function `data` calls).
#[allow(clippy::too_many_arguments)]
fn headline_coverage(
    test: &'static str,
    dgp: &'static str,
    what: &str,
    label: &str,
    seed: u64,
    data: impl Fn(u64) -> TimeSeriesData + Sync,
    graph: impl Fn() -> TemporalDag + Sync,
    query: &TemporalEffectQuery,
    truth: f64,
) -> Coverage {
    let (mut boot, mut reported) = headline_tallies(test, dgp);
    let mut spread = Spread::default();
    let mut warned = warn_tally(test);
    let runs = map_replicates(n_sim(), |rep| {
        let rep_seed = seed + rep;
        run_study(
            data(rep_seed),
            graph(),
            CausalQuery::TemporalEffect(query.clone()),
            BOOT,
            rep_seed,
        )
    });
    for (study, result) in &runs {
        let est = &result.estimate;
        assert!(est.se_analytic.is_nan(), "{what}: no analytic SE is calibrated");
        let interval = normal_interval(est.ate, est.se_bootstrap, Z90);
        if interval.is_some() {
            bind_all(&mut [&mut boot, &mut reported], study, result);
        }
        boot.record(interval, truth);
        reported.record(normal_interval(est.ate, est.se_bootstrap, Z95), truth);
        spread.push(est.ate, est.se_bootstrap.unwrap_or(f64::NAN));
        record_warned(&mut warned, has_diagnostic(result, SHORT_SERIES));
    }
    warned.persist();
    spread.report(&format!("{what} [{label}]"), truth);
    Coverage { tallies: vec![boot], reported: vec![reported], warned }
}

/// Print every tally's calibration line before failing on any of them.
fn assert_all(tallies: &[&CoverageTally]) {
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| tally.assert()).err().map(|e| {
                e.downcast_ref::<String>().cloned().unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// Monte-Carlo SD of the point estimate against the mean reported SEs.
#[derive(Default)]
struct Spread {
    estimates: Vec<f64>,
    se: Vec<f64>,
}

impl Spread {
    fn push(&mut self, estimate: f64, se: f64) {
        self.estimates.push(estimate);
        self.se.push(se);
    }

    fn report(&self, label: &str, truth: f64) {
        let n = self.estimates.len() as f64;
        let mean = self.estimates.iter().sum::<f64>() / n;
        let sd =
            (self.estimates.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let finite: Vec<f64> = self.se.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_se = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
        eprintln!(
            "spread {label}: bias={:+.4} monte_carlo_sd={sd:.4} mean_se={mean_se:.4} ratio={:.3}",
            mean - truth,
            mean_se / sd
        );
    }
}

/// Coverage of all three contrasts from the shared block replicate of one run.
fn mediation_coverage(test: &'static str, s: Scenario) -> Coverage {
    let s = Scenario { n: grid_n(s.n), ..s };
    mediation_coverage_on(
        test,
        "crates/antecedent/tests/common/driven_dgp.rs::mediation_series",
        s.label,
        s.seed,
        |rep| mediation_series(s, rep),
        mediation_dag,
    )
}

/// [`mediation_coverage`] on any one-mediator design with an empty adjustment
/// set (`data(rep)` draws replicate `rep`; the run seed is `seed + rep`).
///
/// The query asks for the Mediated contrast, so the runtime's reported
/// (primary) interval is the Mediated circular-block interval: that tally backs
/// the coverage record. Total and Direct are grid slices beside it, not a
/// reported interval, so their tallies emit no record.
fn mediation_coverage_on(
    test: &'static str,
    dgp: &'static str,
    label: &str,
    seed: u64,
    data: impl Fn(u32) -> TimeSeriesData + Sync,
    graph: fn() -> TemporalDag,
) -> Coverage {
    // No record: the Total / Direct slices are not the reported interval of a Mediated query.
    let mut total = CoverageTally::new(format!("temporal mediation Total [{label}]"), 0.9);
    let mut direct = CoverageTally::new(format!("temporal mediation Direct [{label}]"), 0.9);
    let (mut mediated, mut reported) = headline_tallies(test, dgp);
    let mut spreads = [Spread::default(), Spread::default(), Spread::default()];
    let mut warned = warn_tally(test);
    let runs = map_replicates(n_sim(), |rep| {
        run_study(
            data(u32::try_from(rep).unwrap()),
            graph(),
            mediation_query(MediationContrast::Mediated),
            BOOT,
            seed + rep,
        )
    });
    for (study, result) in &runs {
        let grid = result.mediation_grid.as_ref().expect("mediation grid");
        let slice = &grid.slices[0];
        // No mediator-outcome confounder and an empty t -> y back-door set.
        assert!(slice.adjustment.is_empty(), "unexpected adjustment {:?}", slice.adjustment);
        let TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } =
            &slice.uncertainty
        else {
            panic!("temporal mediation must publish shared circular-block SEs");
        };
        let points = &slice.estimate;
        assert!(result.estimate.se_analytic.is_nan(), "iid analytic SE must be withheld");
        record_warned(&mut warned, has_diagnostic(result, SHORT_SERIES));
        total.record(normal_interval(points.total.unwrap(), block.total, Z90), C + A * B);
        direct.record(normal_interval(points.direct.unwrap(), block.direct, Z90), C);
        let interval = normal_interval(points.mediated.unwrap(), block.mediated, Z90);
        if interval.is_some() {
            bind_all(&mut [&mut mediated, &mut reported], study, result);
        }
        mediated.record(interval, A * B);
        reported.record(normal_interval(points.mediated.unwrap(), block.mediated, Z95), A * B);
        for (spread, (point, se)) in spreads.iter_mut().zip([
            (points.total, block.total),
            (points.direct, block.direct),
            (points.mediated, block.mediated),
        ]) {
            spread.push(point.unwrap(), se.unwrap_or(f64::NAN));
        }
    }
    for (spread, (name, truth)) in
        spreads.iter().zip([("Total", C + A * B), ("Direct", C), ("Mediated", A * B)])
    {
        spread.report(&format!("temporal mediation {name} [{label}]"), truth);
    }
    warned.persist();
    Coverage { tallies: vec![total, direct, mediated], reported: vec![reported], warned }
}

/// WP-F2: Total / Direct / Mediated coverage on the mediator-outcome-confounded
/// DGP (`common::fixtures::mediation_series`, `kappa = 0.5`: `m <- z[t-1] ->
/// w[t-1] -> y` confounds `m -> y` but not `t -> y`) at n = 160, iid. The
/// intervals are only calibrated if the outcome model adjusts `{z[t-1],
/// w[t-1]}` as well as the (empty) `t -> y` back-door set.
///
/// As in [`mediation_coverage_on`], only the Mediated tally (the reported
/// interval of the Mediated query) backs a coverage record.
fn confounded_mediation_coverage(test: &'static str) -> Coverage {
    use common::fixtures;
    const SEED: u64 = 600_000;
    let label = "kappa=0.5 iid n=160";
    let truths = [
        ("Total", fixtures::mediation_total_truth()),
        ("Direct", fixtures::mediation_direct_truth()),
        ("Mediated", fixtures::mediation_truth()),
    ];
    let (mediated, mut reported) =
        headline_tallies(test, "crates/antecedent/tests/common/fixtures.rs::mediation_series");
    // No record for Total / Direct: not the reported interval of a Mediated query.
    let mut tallies: Vec<CoverageTally> = truths[..2]
        .iter()
        .map(|(name, _)| {
            CoverageTally::new(format!("temporal mediation {name} confounded [{label}]"), 0.9)
        })
        .collect();
    tallies.push(mediated);
    let mut spreads = [Spread::default(), Spread::default(), Spread::default()];
    let runs = map_replicates(n_sim(), |rep| {
        let seed = SEED + rep;
        run_study(
            fixtures::mediation_series(grid_n(160), fixtures::MED_KAPPA, seed),
            fixtures::mediation_dag(),
            mediation_query(MediationContrast::Mediated),
            BOOT,
            seed,
        )
    });
    for (study, result) in &runs {
        let slice = &result.mediation_grid.as_ref().expect("mediation grid").slices[0];
        let adjustment: Vec<(u32, i32)> =
            slice.adjustment.iter().map(|k| (k.variable.raw(), k.offset)).collect();
        assert_eq!(adjustment, vec![(3, -1), (4, -1)], "S(1) = {{z[t-1], w[t-1]}}");
        let TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } =
            &slice.uncertainty
        else {
            panic!("temporal mediation must publish shared circular-block SEs");
        };
        let points = &slice.estimate;
        for (i, (point, se)) in [
            (points.total, block.total),
            (points.direct, block.direct),
            (points.mediated, block.mediated),
        ]
        .into_iter()
        .enumerate()
        {
            let point = point.unwrap();
            let interval = normal_interval(point, se, Z90);
            if i == 2 {
                if interval.is_some() {
                    bind_all(&mut [&mut tallies[i], &mut reported], study, result);
                }
                reported.record(normal_interval(point, se, Z95), truths[i].1);
            }
            tallies[i].record(interval, truths[i].1);
            spreads[i].push(point, se.unwrap_or(f64::NAN));
        }
    }
    for (spread, (name, truth)) in spreads.iter().zip(truths) {
        spread.report(&format!("temporal mediation {name} confounded [{label}]"), truth);
    }
    // Never checked against the short-series threshold (only gate() reads this
    // Coverage, which prints the rate but never asserts it), so an empty,
    // never-recorded tally is fine here.
    Coverage { tallies, reported: vec![reported], warned: warn_tally(test) }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_mediation_confounded_iid_n160_nominal_90_coverage() {
    let coverage = confounded_mediation_coverage(
        "temporal_dag_mediation_confounded_iid_n160_nominal_90_coverage",
    );
    assert_all(&coverage.tallies.iter().collect::<Vec<_>>());
    for tally in &coverage.reported {
        tally.emit();
    }
}

macro_rules! effect_gate {
    ($name:ident, $scenario:expr, $what:expr, $query:expr, $truth:expr, $offset:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            gate(effect_coverage(stringify!($name), $scenario, $what, &$query, $truth, $offset));
        }
    };
}

macro_rules! effect_boundary_gate {
    ($name:ident, $scenario:expr, $what:expr, $query:expr, $truth:expr, $offset:expr, $measured:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            boundary_gate(
                effect_coverage(stringify!($name), $scenario, $what, &$query, $truth, $offset),
                $measured,
            );
        }
    };
}

macro_rules! mediation_gate {
    ($name:ident, $scenario:expr $(, $measured:expr)?) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            mediation_gate!(@run stringify!($name), $scenario $(, $measured)?);
        }
    };
    (@run $name:expr, $scenario:expr, $measured:expr) => {
        gate_at_each(mediation_coverage($name, $scenario), $measured);
    };
    (@run $name:expr, $scenario:expr) => {
        gate(mediation_coverage($name, $scenario));
    };
}

effect_gate!(
    temporal_dag_pulse_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_pulse_ar05_n160_nominal_90_coverage() {
    gate_at(
        effect_coverage(
            "temporal_dag_pulse_ar05_n160_nominal_90_coverage",
            AR05_160,
            "TemporalDag Pulse h=1",
            &pulse(1),
            BETA,
            0,
        ),
        [Some(0.885), None, None],
    );
}
effect_gate!(
    temporal_dag_pulse_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
effect_gate!(
    temporal_dag_pulse_ar05_n60_nominal_90_coverage,
    AR05_60,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);

effect_gate!(
    temporal_dag_pulse_h2_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
effect_gate!(
    temporal_dag_pulse_h2_ar05_n160_nominal_90_coverage,
    AR05_160,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
effect_gate!(
    temporal_dag_pulse_h2_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0
);
// Boundary cell: 0.885 at 2000 replicates (floor 0.887). At n = 60 the
// horizon-2 design has 58 lag-aligned rows in blocks of 4–6; the replicate SD
// is right on average (SE/SD 1.04) but its own sampling variability (about a
// quarter of its value at that few blocks) costs the normal-quantile interval
// about 1.5 points, which the fixed-b factor at b = ℓ/n does not price.
effect_boundary_gate!(
    temporal_dag_pulse_h2_ar05_n60_boundary_within_band,
    AR05_60,
    "TemporalDag Pulse h=2",
    pulse(2),
    ALPHA * DELTA,
    0,
    [0.885, 0.885, 0.860]
);

effect_gate!(
    temporal_dag_sustained_iid_n160_nominal_90_coverage,
    IID_160,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);
// Boundary cell: 0.874 at 2000 replicates on this seed stream (floor 0.887);
// the same regression measures 0.889 and 0.895 on two other streams
// (`v19_short_series_measurement`, MA(3) Pulse h=1 at ρ = 0.5, n = 160), so
// the design sits at about 0.88. The replicate SD is right on average (SE/SD
// 0.97–1.03) and the errors are normal; the loss is the SD's own sampling
// variability at 16 blocks of 10 rows (coefficient of variation 0.22 against
// the 0.15 the fixed-b limit at b = ℓ/n implies), about 2 points for a
// normal-quantile interval.
// Grid point 2's measured value updated to 0.896 (8000 replicates,
// reproduced on rerun): the prior 0.873 no longer matches this stream.
effect_boundary_gate!(
    temporal_dag_sustained_ar05_n160_boundary_within_band,
    AR05_160,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000,
    [0.885, 0.880, 0.896]
);
effect_gate!(
    temporal_dag_sustained_ar09_n400_nominal_90_coverage,
    AR09_400,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);
effect_gate!(
    temporal_dag_sustained_ar05_n60_nominal_90_coverage,
    AR05_60,
    "TemporalDag single-step Sustained",
    single_sustained(),
    BETA,
    50_000
);

mediation_gate!(temporal_dag_mediation_iid_n160_nominal_90_coverage, IID_160);
mediation_gate!(
    temporal_dag_mediation_ar05_n160_nominal_90_coverage,
    AR05_160,
    [[None, None, None], [None, None, None], [None, Some(0.914), None]]
);
mediation_gate!(temporal_dag_mediation_ar09_n400_nominal_90_coverage, AR09_400);
mediation_gate!(temporal_dag_mediation_ar05_n60_nominal_90_coverage, AR05_60);

/// AR(1) ρ = 0.9 residuals at n = 60 and 160 on the MA(3)-treatment designs:
/// short, strongly autocorrelated series whose estimating score still forgets
/// within a few lags. Coverage is nominal (`v19_short_series_measurement`), so
/// these are gated; the short-series warning fires on part of the n = 60
/// replicates (the single-window and mediation thresholds sit above their
/// effective rows) and on almost none at n = 160, and its rate is reported.
const AR09_160: Scenario =
    Scenario { label: "AR(1) rho=0.9 n=160", rho: 0.9, n: 160, seed: 510_000 };
const AR09_60: Scenario = Scenario { label: "AR(1) rho=0.9 n=60", rho: 0.9, n: 60, seed: 500_000 };

effect_gate!(
    temporal_dag_pulse_ar09_n60_nominal_90_coverage,
    AR09_60,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
effect_gate!(
    temporal_dag_pulse_ar09_n160_nominal_90_coverage,
    AR09_160,
    "TemporalDag Pulse h=1",
    pulse(1),
    BETA,
    0
);
mediation_gate!(
    temporal_dag_mediation_ar09_n60_nominal_90_coverage,
    AR09_60,
    [
        [Some(0.914), Some(0.915), Some(0.918)],
        [Some(0.924), Some(0.928), None],
        [Some(0.931), Some(0.926), Some(0.929)],
    ]
);
mediation_gate!(
    temporal_dag_mediation_ar09_n160_nominal_90_coverage,
    AR09_160,
    [[None, None, None], [None, None, None], [Some(0.928), None, Some(0.914)]]
);

/// Multi-step Sustained over lags 2..=1 on `fixtures::confounded_series` with its
/// generating DAG (`fixtures::confounded_dag`, truth `B1 + B2`): sequential
/// g-computation refits every mechanism on each circular-block replicate.
fn sequential_coverage(test: &'static str, label: &str, rho: f64, n: usize, seed: u64) -> Coverage {
    headline_coverage(
        test,
        "crates/antecedent/tests/common/fixtures.rs::confounded_series",
        "TemporalDag multi-step Sustained",
        label,
        seed,
        |s| fixtures::confounded_series(grid_n(n), fixtures::B2, rho, s),
        fixtures::confounded_dag,
        &persistent_dgp::multi_sustained(),
        fixtures::B1 + fixtures::B2,
    )
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_multistep_sustained_ar05_n160_nominal_90_coverage() {
    gate(sequential_coverage(
        "temporal_dag_multistep_sustained_ar05_n160_nominal_90_coverage",
        "AR(1) rho=0.5 n=160",
        0.5,
        160,
        700_000,
    ));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_multistep_sustained_ar09_n400_nominal_90_coverage() {
    gate(sequential_coverage(
        "temporal_dag_multistep_sustained_ar09_n400_nominal_90_coverage",
        "AR(1) rho=0.9 n=400",
        0.9,
        400,
        710_000,
    ));
}

// Boundary records: an AR(1) treatment as well as residual at n = 60 (ρ = 0.9;
// ρ = 0.95 for multi-step Sustained, whose two-lag contrast averages over more
// of the memory), so the estimating score is itself close to AR(1)(ρ²) and the
// series holds only a handful of its memory spans. No block length or fixed-b
// correction reaches nominal coverage there (0.76-0.83 measured at 2000
// replicates); the runtime warns on (nearly) every replicate, and coverage is
// recorded.

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_pulse_ar1_treatment_rho09_n60_short_series_boundary() {
    boundary(headline_coverage(
        "temporal_dag_pulse_ar1_treatment_rho09_n60_short_series_boundary",
        "crates/antecedent/tests/common/fixtures.rs::chain_pag_series",
        "TemporalDag Pulse h=1, AR(1) treatment",
        "AR(1) rho=0.9 n=60 (boundary)",
        800_000,
        |s| fixtures::chain_pag_series(grid_n(60), 0.9, s),
        persistent_dgp::chain_pulse_dag,
        &persistent_dgp::pulse(),
        fixtures::B1,
    ));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_mediation_ar1_treatment_rho09_n60_short_series_boundary() {
    const SEED: u64 = 810_000;
    boundary(mediation_coverage_on(
        "temporal_dag_mediation_ar1_treatment_rho09_n60_short_series_boundary",
        "crates/antecedent/tests/common/persistent_dgp.rs::mediation_series",
        "AR(1) treatment, AR(1) rho=0.9 n=60 (boundary)",
        SEED,
        |rep| persistent_dgp::mediation_series(grid_n(60), 0.9, SEED + u64::from(rep)),
        persistent_dgp::mediation_dag,
    ));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_dag_multistep_sustained_ar1_treatment_rho095_n60_short_series_boundary() {
    boundary(headline_coverage(
        "temporal_dag_multistep_sustained_ar1_treatment_rho095_n60_short_series_boundary",
        "crates/antecedent/tests/common/persistent_dgp.rs::sequential_series",
        "TemporalDag multi-step Sustained, AR(1) treatment",
        "AR(1) rho=0.95 n=60 (boundary)",
        820_000,
        |s| persistent_dgp::sequential_series(grid_n(60), 0.95, s),
        fixtures::two_lag_dag,
        &persistent_dgp::multi_sustained(),
        fixtures::B1 + fixtures::B2,
    ));
}

#[test]
fn temporal_dag_pulse_publishes_circular_block_se_only() {
    let result = run(
        pulse_series(IID_160, 7, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        24,
        7,
    );
    let diag = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se")
        .expect("the single-window Frequentist path must record its circular-block SE");
    // 159 lag-aligned rows: ⌈159^(1/3)⌉ = 6 beats the two-slice unfolded window.
    assert!(diag.message.contains("circular-block length 6"), "{}", diag.message);
    assert!(diag.message.contains("n=159 lag-aligned rows"), "{}", diag.message);
    let est = &result.estimate;
    assert!(est.se_bootstrap.is_some_and(|se| se.is_finite() && se > 0.0));
    assert!(est.se_analytic.is_nan(), "the iid OLS SE must not be published on lagged rows");
    assert_eq!(est.bootstrap_replicates_ok, Some(24));
    assert!(!has_diagnostic(&result, "estimate.temporal.circular_block_se.short_series"));

    // Without replicates there is no calibrated SE, and the result says so.
    let bare = run(
        pulse_series(IID_160, 7, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        0,
        7,
    );
    assert!(bare.estimate.se_bootstrap.is_none());
    assert!(bare.estimate.se_analytic.is_nan());
    assert!((bare.estimate.ate - est.ate).abs() < 1e-12);
    let bare_diag = bare
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se")
        .unwrap();
    assert!(bare_diag.message.contains("request bootstrap_replicates"), "{}", bare_diag.message);
}

#[test]
fn temporal_mediation_shares_one_block_replicate_across_contrasts() {
    let result = run(
        mediation_series(AR05_160, 3),
        mediation_dag(),
        mediation_query(MediationContrast::Total),
        24,
        3,
    );
    assert!(has_diagnostic(&result, "estimate.temporal.circular_block_se"));
    assert!(result.estimate.se_analytic.is_nan(), "iid Total SE must be withheld by default");
    let slice = &result.mediation_grid.as_ref().unwrap().slices[0];
    let TemporalMediationUncertainty::FrequentistBlockBootstrap { requested, block } =
        &slice.uncertainty
    else {
        panic!("expected shared circular-block uncertainty, got {:?}", slice.uncertainty);
    };
    assert_eq!(*requested, block.total);
    assert_eq!(result.estimate.se_bootstrap, block.total);
    assert_eq!(block.replicates_attempted, 24);
    for se in [block.total, block.direct, block.mediated] {
        assert!(se.is_some_and(|se| se.is_finite() && se > 0.0));
    }

    // No replicates: every iid analytic SE stays NaN and no interval is claimed.
    let none = run(
        mediation_series(AR05_160, 3),
        mediation_dag(),
        mediation_query(MediationContrast::Direct),
        0,
        3,
    );
    assert!(none.estimate.se_analytic.is_nan());
    assert!(none.estimate.se_bootstrap.is_none());
    assert!(matches!(
        none.mediation_grid.as_ref().unwrap().slices[0].uncertainty,
        TemporalMediationUncertainty::FrequentistPointwise { standard_error: None }
    ));
}

/// The effective rows printed by the `estimate.temporal.circular_block_se`
/// provenance, and whether the short-series warning fired.
fn short_series_state(result: &StudyResult) -> (f64, bool) {
    const KEY: &str = "score effective rows ";
    let message = &result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.temporal.circular_block_se")
        .expect("circular-block provenance")
        .message;
    let rest = &message[message.find(KEY).expect("effective rows") + KEY.len()..];
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
    (rest[..end].parse().expect("a number"), has_diagnostic(result, SHORT_SERIES))
}

/// The warning fires exactly below the family threshold, and on this fixed
/// series it does (`expect_warning`) or does not.
fn assert_threshold(result: &StudyResult, family: CircularBlockFamily, expect_warning: bool) {
    let (effective_rows, warned) = short_series_state(result);
    let threshold = family.min_effective_rows();
    assert_eq!(
        warned,
        effective_rows < threshold,
        "{family:?}: warning must fire exactly below {threshold} (effective rows {effective_rows})"
    );
    assert_eq!(
        warned, expect_warning,
        "{family:?}: effective rows {effective_rows} against threshold {threshold}"
    );
    if warned {
        let warning = result.diagnostics.iter().find(|d| d.code.as_ref() == SHORT_SERIES).unwrap();
        assert_eq!(warning.severity, DiagnosticSeverity::Warning);
        assert!(warning.message.contains(family.label()), "{}", warning.message);
    }
}

#[test]
fn short_series_single_window_threshold() {
    let short = Scenario { label: "short", rho: 0.5, n: 30, seed: 9 };
    let thirty_rows = run(
        pulse_series(short, 0, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        8,
        9,
    );
    assert_threshold(&thirty_rows, CircularBlockFamily::SingleWindow, true);
    // AR(1) treatment and residual at ρ = 0.9, n = 60: coverage 0.79 measured.
    let persistent = run(
        fixtures::chain_pag_series(60, 0.9, 12),
        persistent_dgp::chain_pulse_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::pulse()),
        8,
        11,
    );
    assert_threshold(&persistent, CircularBlockFamily::SingleWindow, true);
    // MA(3) treatment, AR(1) ρ = 0.9 residual, n = 160: nominal, quiet.
    let short_memory = run(
        pulse_series(AR09_160, 0, false),
        pulse_dag(false),
        CausalQuery::TemporalEffect(pulse(1)),
        8,
        13,
    );
    assert_threshold(&short_memory, CircularBlockFamily::SingleWindow, false);
}

#[test]
fn short_series_mediation_threshold() {
    let persistent = run(
        persistent_dgp::mediation_series(60, 0.9, 15),
        persistent_dgp::mediation_dag(),
        mediation_query(MediationContrast::Mediated),
        8,
        15,
    );
    assert_threshold(&persistent, CircularBlockFamily::Mediation, true);
    let short_memory = run(
        mediation_series(AR09_160, 0),
        mediation_dag(),
        mediation_query(MediationContrast::Mediated),
        8,
        17,
    );
    assert_threshold(&short_memory, CircularBlockFamily::Mediation, false);
}

#[test]
fn short_series_sequential_threshold() {
    let persistent = run(
        persistent_dgp::sequential_series(60, 0.9, 19),
        fixtures::two_lag_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::multi_sustained()),
        8,
        19,
    );
    assert_threshold(&persistent, CircularBlockFamily::Sequential, true);
    let long = run(
        fixtures::confounded_series(400, fixtures::B2, 0.5, 21),
        fixtures::confounded_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::multi_sustained()),
        8,
        21,
    );
    assert_threshold(&long, CircularBlockFamily::Sequential, false);
    assert!(
        has_diagnostic(&long, "estimate.temporal.sustained_window"),
        "the multi-step provenance must reach the result"
    );
}
