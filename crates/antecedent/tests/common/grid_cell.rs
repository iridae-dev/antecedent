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
//! Coordinates of other graph classes and structure routes can report that interval too:
//! one data law, one identifying functional, one estimator. A test measures such
//! coordinates together inside [`sharing_base`]: the first measurement is the base, and
//! every later one is held to the same check (each of its executions, on the first
//! [`SIBLING_REPLICATES`] replicates, must report the base interval bit for bit on the
//! same sample and scope) and recorded the same way.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    reason = "each integration test binary includes this module and uses a subset of it"
)]

use std::cell::RefCell;
use std::rc::Rc;

use antecedent::{Study, StudyResult};

use crate::common::calibration::{
    Construction, CoverageTally, GRID_POINTS, RecordKey, ScopeFacts, map_replicates,
    map_replicates_from, n_sim, replicate_start,
};
use crate::common::calibration_bind::constructions;
use crate::common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate_at};

/// Replicates on which every sibling is executed and compared with the base.
pub const SIBLING_REPLICATES: u32 = 2;

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

/// One replicate of a base execution.
#[derive(Clone)]
struct Replicate {
    /// The base execution's match key, scope and validation mode.
    base: (Construction, ScopeFacts, &'static str),
    intervals: Vec<IntervalPair>,
    truth: Vec<f64>,
}

/// A base execution measured on every replicate.
struct Base {
    test: &'static str,
    labels: Vec<Option<String>>,
    /// The replicates this run measures: all of them, or, in a recheck by extension, the
    /// ones after the first run's.
    replicates: Vec<Replicate>,
    /// Replicates `0..SIBLING_REPLICATES`, on which every sibling is executed; executed
    /// again in a recheck by extension, whose own replicates start later.
    head: Vec<Replicate>,
}

/// The measurements of one [`sharing_base`] scope.
struct Shared {
    base: Option<Rc<Base>>,
    /// Variants measured so far: the next measurement's tallies are numbered from here.
    variants: usize,
}

thread_local! {
    static SHARED: RefCell<Option<Shared>> = const { RefCell::new(None) };
}

/// Measure several coordinates that report one interval on one base execution.
///
/// `coordinates` calls the measuring functions of this module once per coordinate, each
/// with this test's name and the same replicate datasets. The first call measures its
/// base variant on every replicate, as it would alone. Every later call executes each of
/// its recorded variants on the first [`SIBLING_REPLICATES`] replicates only, requires
/// the base interval bit for bit on the base's truths and scope, and records each variant
/// under its own match key and validation mode, scored on the base's intervals.
///
/// # Panics
///
/// When scopes nest or `coordinates` measures nothing, and as [`measure`].
pub fn sharing_base(coordinates: impl FnOnce()) {
    let outer = SHARED.with(|shared| shared.replace(Some(Shared { base: None, variants: 0 })));
    assert!(outer.is_none(), "a shared base does not nest");
    coordinates();
    let shared = SHARED.with(RefCell::take).expect("the scope is open");
    assert!(shared.base.is_some(), "a shared base measures at least one coordinate");
}

fn bits(truth: &[f64]) -> Vec<u64> {
    truth.iter().map(|value| value.to_bits()).collect()
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
    let pick = |reported: &Reported| {
        reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == cell.interval)
            .cloned()
            .unwrap_or_else(|| panic!("{}: no {} interval was reported", cell.test, cell.interval))
    };
    // One execution of a variant that must report `intervals`.
    let sibling = |rep: u64, variant: usize, intervals: &[IntervalPair]| {
        let (study, result) = run(rep, variant);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some(cell.estimator),
            "{}: variant {variant} estimator target",
            cell.test
        );
        assert!(
            same_bits(&score(&result), intervals),
            "{}: variant {variant} does not report the base interval on replicate {rep}",
            cell.test
        );
        pick(&reported(&study, &result))
    };

    let (shared, offset) = SHARED.with(|shared| {
        shared.borrow().as_ref().map_or((None, 0), |shared| (shared.base.clone(), shared.variants))
    });
    // `executed[replicate]`: the match key of each variant from `first_executed` on.
    let (base, first_executed, executed) = match shared {
        None => {
            let one = |rep: u64, with_siblings: bool| {
                let (study, result) = run(rep, 0);
                let intervals = score(&result);
                assert_eq!(
                    intervals.len(),
                    targets.len(),
                    "{}: one interval pair per target",
                    cell.test
                );
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
                let base = pick(&reported(&study, &result));
                drop((study, result));
                let siblings: Vec<_> = if with_siblings {
                    (1..cell.variants).map(|variant| sibling(rep, variant, &intervals)).collect()
                } else {
                    Vec::new()
                };
                let truth = truths(rep);
                assert_eq!(truth.len(), targets.len(), "{}: one truth per target", cell.test);
                (Replicate { base, intervals, truth }, siblings)
            };
            let head_len = n_sim().min(SIBLING_REPLICATES);
            let extending = replicate_start() > 0;
            let runs =
                map_replicates(n_sim(), |rep| one(rep, !extending && rep < u64::from(head_len)));
            let (replicates, mut executed): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
            let head = if extending {
                // The siblings' match keys come from their own executions, which a run that
                // starts after the head replicates has to make itself.
                let (head, keys): (Vec<_>, Vec<_>) =
                    map_replicates_from(0, head_len, |rep| one(rep, true)).into_iter().unzip();
                executed = keys;
                head
            } else {
                replicates[..usize::try_from(head_len).expect("replicate count")].to_vec()
            };
            let base = Rc::new(Base {
                test: cell.test,
                labels: targets.iter().map(|target| target.label.clone()).collect(),
                replicates,
                head,
            });
            (base, 1, executed)
        }
        Some(base) => {
            assert_eq!(cell.test, base.test, "one test records every coordinate of its base");
            assert!(
                targets.iter().map(|target| &target.label).eq(&base.labels),
                "{}: the coordinates of a shared base score the same targets",
                cell.test
            );
            let replicates = base.head.as_slice();
            let executed = map_replicates_from(0, n_sim().min(SIBLING_REPLICATES), |rep| {
                let shared = &replicates[usize::try_from(rep).expect("replicate index")];
                assert_eq!(
                    bits(&truths(rep)),
                    bits(&shared.truth),
                    "{}: a shared base has one truth per target",
                    cell.test
                );
                (first_recorded..cell.variants)
                    .map(|variant| {
                        let found = sibling(rep, variant, &shared.intervals);
                        assert_eq!(
                            found.1, shared.base.1,
                            "{}: variant {variant} does not report the base scope on replicate {rep}",
                            cell.test
                        );
                        found
                    })
                    .collect()
            });
            (base, first_recorded, executed)
        }
    };
    SHARED.with(|shared| {
        if let Some(shared) = shared.borrow_mut().as_mut() {
            shared.base.get_or_insert_with(|| Rc::clone(&base));
            shared.variants += cell.variants;
        }
    });

    // Each executed variant's match key, from its own executions; stable across them.
    let mut keys: Vec<(Construction, &'static str)> = Vec::new();
    for variants in &executed {
        for (i, (construction, _, validation)) in variants.iter().enumerate() {
            let found = (construction.clone(), *validation);
            match keys.get(i) {
                Some(seen) => assert_eq!(seen, &found, "{}: sibling match key", cell.test),
                None => keys.push(found),
            }
        }
    }
    assert_eq!(
        keys.len(),
        cell.variants - first_executed,
        "{}: every sibling was executed",
        cell.test
    );

    let key = RecordKey { test: cell.test, dgp: cell.dgp, interval: cell.interval };
    let mut tallies: Vec<CoverageTally> = Vec::new();
    let mut boundaries = Vec::new();
    for variant in first_recorded..cell.variants {
        for (index, (target, measured)) in targets.iter().zip(measured).enumerate() {
            for (level, measured) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().zip(measured) {
                // Every execution but the base keeps its own state and decision under its own
                // name.
                let mut tally = CoverageTally::for_record(key, level);
                if offset + variant > 0 {
                    tally = tally.named(format!("{}#v{}", cell.test, offset + variant));
                }
                if let Some(label) = &target.label {
                    tally = tally.labelled(label.clone());
                }
                for replicate in &base.replicates {
                    let (construction, scope, validation) = &replicate.base;
                    if variant < first_executed {
                        tally.bind_validation(construction, *scope, validation);
                    } else {
                        let (construction, validation) = &keys[variant - first_executed];
                        tally.bind_validation(construction, *scope, validation);
                    }
                    let pair = replicate.intervals[index];
                    tally.record(
                        if level == REPORTED_LEVEL { pair[0] } else { pair[1] },
                        replicate.truth[index],
                    );
                }
                tallies.push(tally);
                boundaries.push(*measured);
            }
        }
    }
    gate_at(&tallies, &boundaries);
}
