//! One coverage measurement for a supported coordinate and the sibling cells that
//! report the same interval.
//!
//! A supported coordinate (query, graph class, inference, estimator) is offered under
//! several validation modes and, where the graph is given, as an explicit or an accepted
//! structure. Validation runs after estimation and an accepted graph is estimated as the
//! explicit one, so those siblings report the interval the base execution reports. The
//! driver measures the base execution on every replicate and, on the first
//! [`SIBLING_REPLICATES`] replicates, executes every sibling and requires its interval to
//! be the base interval bit for bit. Each sibling's record is bound from the sibling's
//! own execution (its own runtime match key and validation mode), and scored on the
//! intervals that check proved it reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    reason = "each integration test binary includes this module and uses a subset of it"
)]

use antecedent::{Study, StudyResult};

use crate::common::calibration::{
    Construction, CoverageTally, GRID_POINTS, RecordKey, ScopeFacts, map_replicates, n_sim,
};
use crate::common::calibration_bind::constructions;
use crate::common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate_at};

/// Replicates on which every sibling is executed and compared with the base.
pub const SIBLING_REPLICATES: u64 = 2;

/// `[reported level, gate level]` intervals of one scored target.
pub type IntervalPair = [Option<(f64, f64)>; 2];

/// One scored functional of an execution and the truth its interval must cover.
#[derive(Debug, Clone)]
pub struct Target {
    /// Distinguishes the targets of one execution (a dose, a horizon, a contrast);
    /// `None` for an execution with a single target.
    pub label: Option<String>,
    pub truth: f64,
}

impl Target {
    #[must_use]
    pub fn scalar(truth: f64) -> Vec<Self> {
        vec![Self { label: None, truth }]
    }

    #[must_use]
    pub fn labelled(label: impl Into<String>, truth: f64) -> Self {
        Self { label: Some(label.into()), truth }
    }
}

/// The coordinate a test measures.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    /// The test's own name: the record's identity.
    pub test: &'static str,
    /// The generator's name.
    pub dgp: &'static str,
    /// `IntervalMethod::as_str()` of the scored interval.
    pub interval: &'static str,
    /// The estimator the execution must report.
    pub estimator: &'static str,
    /// How many executions `run` distinguishes: variant 0 is the base measured on every
    /// replicate; variants `1..variants` are its siblings (another validation mode,
    /// the accepted twin of an explicit structure).
    pub variants: usize,
}

type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

fn reported(study: &Study, result: &StudyResult) -> Reported {
    constructions(&study.inspect().expect("inspect"), result)
}

fn same_bits(a: &[IntervalPair], b: &[IntervalPair]) -> bool {
    let bits = |pair: &IntervalPair| pair.map(|i| i.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())));
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| bits(x) == bits(y))
}

/// Measure `cell` over `n_sim()` replicates and gate every record at nominal.
///
/// `run(replicate, variant)` executes one variant on the replicate's dataset (the
/// dataset depends on the replicate only); `score` reads one interval pair per target,
/// in `targets` order.
///
/// # Panics
///
/// When an execution reports another estimator, a sibling's interval differs from the
/// base interval, a sibling's match key is not stable, or a record fails its gate.
#[track_caller]
pub fn measure(
    cell: Cell,
    targets: &[Target],
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    measure_at(cell, targets, run, score, &vec![[[None; GRID_POINTS]; 2]; targets.len()]);
}

/// [`measure`] with named boundaries: `measured[target][level][grid point]` is
/// `Some(m)` where that record is a boundary held to its measured coverage `m`.
#[track_caller]
pub fn measure_at(
    cell: Cell,
    targets: &[Target],
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
    measured: &[[[Option<f64>; GRID_POINTS]; 2]],
) {
    let fixed: Vec<f64> = targets.iter().map(|target| target.truth).collect();
    measure_inner(cell, targets, run, score, measured, |_| fixed.clone(), 0);
}

/// [`measure`] whose base execution is measured but not recorded.
///
/// Some coordinates list no cell for the validation mode a base execution would run cheaply
/// (validation none is another test's cell), and executing a listed validated cell as the
/// base would pay its validation on every replicate. Variant 0 is then a validation-none
/// execution of the same route: it is scored on every replicate and emits no record, and
/// variants `1..variants` are the listed cells. Each listed cell is executed on the first
/// [`SIBLING_REPLICATES`] replicates, must report the base interval bit for bit (validation
/// runs after estimation), and is recorded under its own match key and validation mode, bound
/// from its own executions, and scored on the intervals that check proved it reports.
///
/// # Panics
///
/// As [`measure`], and when `cell.variants < 2` (nothing would be recorded).
#[track_caller]
pub fn measure_from_unrecorded_base(
    cell: Cell,
    targets: &[Target],
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    assert!(cell.variants >= 2, "{}: an unrecorded base needs a recorded sibling", cell.test);
    let fixed: Vec<f64> = targets.iter().map(|target| target.truth).collect();
    let measured = vec![[[None; GRID_POINTS]; 2]; targets.len()];
    measure_inner(cell, targets, run, score, &measured, |_| fixed.clone(), 1);
}

/// [`measure`] for a functional whose true value depends on the replicate's sample (a
/// level averaged over the observed covariates): `truths(replicate)` gives one truth per
/// target, in `targets` order, and each target's own `truth` is not read.
#[track_caller]
pub fn measure_sample_truths(
    cell: Cell,
    targets: &[Target],
    truths: impl Fn(u64) -> Vec<f64> + Sync,
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    let measured = vec![[[None; GRID_POINTS]; 2]; targets.len()];
    measure_inner(cell, targets, run, score, &measured, truths, 0);
}

#[track_caller]
fn measure_inner(
    cell: Cell,
    targets: &[Target],
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
    measured: &[[[Option<f64>; GRID_POINTS]; 2]],
    truths: impl Fn(u64) -> Vec<f64> + Sync,
    first_recorded: usize,
) {
    assert!(cell.variants >= 1, "{}: a cell has its base variant", cell.test);
    assert!(first_recorded < cell.variants, "{}: some variant is recorded", cell.test);
    assert_eq!(targets.len(), measured.len(), "{}: one boundary entry per target", cell.test);
    let key = RecordKey { test: cell.test, dgp: cell.dgp, interval: cell.interval };
    let runs = map_replicates(n_sim(), |rep| {
        let (study, result) = run(rep, 0);
        let intervals = score(&result);
        assert_eq!(intervals.len(), targets.len(), "{}: one interval pair per target", cell.test);
        if rep == 0 {
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(cell.estimator),
                "{}: estimator target",
                cell.test
            );
            assert!(
                intervals.iter().all(|pair| pair[0].is_some()),
                "{}: the base execution must publish every scored interval",
                cell.test
            );
        }
        let base = reported(&study, &result);
        drop((study, result));
        let siblings: Vec<Reported> = if rep < SIBLING_REPLICATES {
            (1..cell.variants)
                .map(|variant| {
                    let (study, result) = run(rep, variant);
                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some(cell.estimator),
                        "{}: variant {variant} estimator target",
                        cell.test
                    );
                    assert!(
                        same_bits(&score(&result), &intervals),
                        "{}: variant {variant} does not report the base interval on replicate {rep}",
                        cell.test
                    );
                    reported(&study, &result)
                })
                .collect()
        } else {
            Vec::new()
        };
        let truth = truths(rep);
        assert_eq!(truth.len(), targets.len(), "{}: one truth per target", cell.test);
        (base, intervals, siblings, truth)
    });

    // Each sibling's match key, from its own executions; stable across them.
    let pick = |reported: &Reported| {
        reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == cell.interval)
            .map(|(construction, _, validation)| (construction.clone(), *validation))
            .unwrap_or_else(|| panic!("{}: no {} interval was reported", cell.test, cell.interval))
    };
    let mut keys: Vec<(Construction, &'static str)> = Vec::new();
    for (_, _, siblings, _) in &runs {
        for (i, sibling) in siblings.iter().enumerate() {
            let found = pick(sibling);
            match keys.get(i) {
                Some(seen) => assert_eq!(seen, &found, "{}: sibling match key", cell.test),
                None => keys.push(found),
            }
        }
    }
    assert_eq!(keys.len(), cell.variants - 1, "{}: every sibling was executed", cell.test);

    let mut tallies: Vec<CoverageTally> = Vec::new();
    let mut boundaries = Vec::new();
    // `tallies` grows by `targets.len() * 2` per recorded variant, in (target, level) order, so
    // the target index below is the same whichever variant recording starts at.
    for variant in first_recorded..cell.variants {
        for (target, measured) in targets.iter().zip(measured) {
            for (level, measured) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().zip(measured) {
                // A sibling's tally keeps its own state and decision under its own name.
                let mut tally = CoverageTally::for_record(key, level);
                if variant > 0 {
                    tally = tally.named(format!("{}#v{variant}", cell.test));
                }
                if let Some(label) = &target.label {
                    tally = tally.labelled(label.clone());
                }
                for (base, intervals, _, truth) in &runs {
                    let (construction, scope, validation) = base
                        .iter()
                        .find(|(construction, _, _)| construction.interval_method == cell.interval)
                        .unwrap_or_else(|| {
                            panic!("{}: no {} interval was reported", cell.test, cell.interval)
                        });
                    match variant {
                        0 => tally.bind_validation(construction, *scope, validation),
                        v => tally.bind_validation(&keys[v - 1].0, *scope, keys[v - 1].1),
                    }
                    let index = tallies.len() % (targets.len() * 2) / 2;
                    let pair = intervals[index];
                    tally.record(
                        if level == REPORTED_LEVEL { pair[0] } else { pair[1] },
                        truth[index],
                    );
                }
                tallies.push(tally);
                boundaries.push(*measured);
            }
        }
    }
    gate_at(&tallies, &boundaries);
}
