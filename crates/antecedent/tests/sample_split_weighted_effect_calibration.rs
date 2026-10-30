//! Whole-method coverage of the interval for the effect of a treatment in a survey-weighted
//! population, after discovery on a row split.
//!
//! The procedure under test is the facade's weighted construction
//! (`antecedent::analysis::sample_split_analysis`): a seeded split allocating three quarters
//! of participants to discovery,
//! PC with the weighted partial-correlation test (`alpha = 0.01`, conditioning-set size and
//! parent limit four) on the discovery rows under their weights, acceptance when the search
//! finished and treatment incident orientations agree in every conflict-free resolution, and the
//! sampling-weighted least-squares effect of the treatment given its discovered parents on the
//! held-out analysis quarter, with the HC3 leverage-corrected sandwich and Student interval at
//! the effective sample size
//! (`SplitAnalysisEffect::interval`, level 0.95).
//!
//! Three quarters for discovery reduces confounder-selection error under unequal survey
//! weights; analysis remains independent of discovery. Conflict resolution is exhaustive and
//! bounded, and refuses when the treatment neighbourhood is not invariant. Version one
//! (the half split with global conflict refusal) and version two (the three-quarter split with
//! HC1) remain available only for exact legacy replay.
//!
//! # Design
//!
//! Every replicate draws a sparse connected linear-Gaussian model of ten variables (the three
//! topologies of `common::sparse_scm`, rotating by replicate, redrawn until the strong
//! faithfulness margin holds), simulates 3,000 participants and gives them survey expansion
//! weights of coefficient of variation 0.8, log-normal and a function of the design stream
//! alone. Weights independent of every outcome leave the weighted population's total effect
//! equal to the structural total effect, which is the truth: the sum over directed paths of the
//! products of coefficients, computed from the model and never from the rows. A replicate whose
//! discovery is refused (an unfinished search, treatment orientations dependent on conflicts,
//! a parent-limit
//! removal, an undirected edge at the treatment, the outcome oriented into the treatment)
//! reports no interval and is not scored as covered or missed: coverage is the coverage of the
//! intervals the procedure reports, and the rate at which it reports is gated separately. (A
//! refusal makes no claim, so counting it as a miss would hold the reported intervals to a
//! level no procedure that declines some draws can meet; the unweighted parent-set designs
//! score the same way.)
//!
//! The frozen targets of the acceptance manifest are asserted at its 3,000 participants:
//! reporting share at least 0.9 and median width at most 1.5 times the median magnitude of the
//! effect. Once the run has the 2,000 replicates the manifest freezes, coverage of at least
//! 0.93 at the nominal 0.95 is asserted at every size. The other grid points are measured for
//! their coverage.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build). The interval stays
//! labelled unmeasured until this design has been measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use std::sync::Arc;

use antecedent::analysis::sample_split_analysis::{
    AnalysisEffectRequest, SplitRule, discover_on_split,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{OwnedColumnarStorage, TableView, TabularData};
use antecedent_discovery::selective::{
    CiTestSpec, NativeResult, StaticAlgorithm, StaticDiscoveryContract,
};
use antecedent_io::measured_inference::split_parent_set_effect::{
    ConflictEvidence, MAX_CONFLICTING_COLLIDERS, treatment_parent_set,
    treatment_parent_set_under_conflicts,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed,
};
use common::sparse_scm::{Scm, Stream, Topology, Variant};

const BASE_PARTICIPANTS: usize = 3000;
const VARIABLES: usize = 10;
const WEIGHT_CV: f64 = 0.8;
const MAX_DEGREE: usize = 4;
const ALPHA: f64 = 0.01;
const INTERVAL: &str = "student_t_sandwich_effective_n";
const MIN_REPORTED_SHARE: f64 = 0.9;
/// Median width over the median magnitude of the effect, the manifest's precision target.
const MAX_MEDIAN_WIDTH_OVER_EFFECT: f64 = 1.5;
const COVERAGE_FLOOR: f64 = 0.93;
const FROZEN_REPLICATES: u32 = 2000;
const PARENT_LIMIT_DIAGNOSTIC: &str = "parent_limit_removed_adjacencies";

fn v(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).expect("small index"))
}

const ALLOCATION_PROTOCOL: &str = "discovery_fraction_0_75.conflict_resolution_treatment_invariant";
const CURRENT_PROTOCOL: &str =
    "discovery_fraction_0_75.conflict_resolution_treatment_invariant.variance_hc3";

fn protocol_version() -> u32 {
    static VERSION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        let raw =
            std::env::var("ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL").unwrap_or_else(|_| "3".into());
        raw.parse::<u32>()
            .ok()
            .filter(|version| matches!(version, 1..=3))
            .unwrap_or_else(|| panic!("ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL={raw}: want 1, 2 or 3"))
    })
}

fn discovery_fraction(version: u32) -> f64 {
    if version == 1 { 0.5 } else { 0.75 }
}

fn construction_for_protocol(version: u32) -> Construction {
    Construction {
        query: "SampleSplitWeightedEffect".into(),
        graph_class: "Cpdag".into(),
        structure: "discovered_sample_split.treatment_parent_set".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "linear.adjustment.ate.sampling_weighted".into(),
        interval_method: INTERVAL.into(),
        se_kind: if version == 3 { "hc3_sampling_weighted" } else { "hc1_sampling_weighted" }
            .into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: if version == 1 {
            "weighted_population_total_effect".into()
        } else if version == 2 {
            format!("weighted_population_total_effect.{ALLOCATION_PROTOCOL}")
        } else {
            format!("weighted_population_total_effect.{CURRENT_PROTOCOL}")
        },
        identification: "discovered_treatment_parent_set".into(),
        reported_level: REPORTED_LEVEL,
    }
}

/// Survey expansion weights `exp(s z - s^2 / 2)`, `s^2 = ln(1 + cv^2)`: mean one, coefficient
/// of variation `cv`, drawn from the design stream.
fn survey_weights(rows: usize, seed: u64) -> Vec<f64> {
    let sigma2 = (1.0 + WEIGHT_CV * WEIGHT_CV).ln();
    let mut stream = Stream::new(seed);
    (0..rows).map(|_| (sigma2.sqrt() * stream.normal() - sigma2 / 2.0).exp()).collect()
}

fn weighted_table(scm: &Scm, columns: &[Vec<f64>], weights: Vec<f64>) -> TabularData {
    let plain = TabularData::from_f64_columns(
        scm.names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .expect("table");
    let storage = plain.storage();
    TabularData::new(
        OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            None,
            Some(Arc::from(weights)),
        )
        .expect("weighted storage"),
    )
}

/// One replicate: the interval and the truth, or `None` when the procedure reported none.
fn replicate(rep: u64) -> Option<((f64, f64), f64)> {
    let topologies = Topology::for_degree(MAX_DEGREE);
    let topology = topologies[usize::try_from(rep).ok()? % topologies.len()];
    let seed = stream_seed(0x5E7_0057 + VARIABLES as u64, rep);
    let scm =
        Scm::with_degree(topology, VARIABLES, stream_seed(seed, 1), Variant::Licensed, MAX_DEGREE);
    let rows = grid_n(BASE_PARTICIPANTS);
    let columns = scm.simulate(rows, stream_seed(seed, 2));
    let table = weighted_table(&scm, &columns, survey_weights(rows, stream_seed(seed, 3)));

    // The adapter requires the context's master seed to be the contract's frozen seed.
    let discovery_seed = stream_seed(seed, 4);
    let ctx = ExecutionContext::for_tests(discovery_seed);
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..VARIABLES).map(v).collect(),
        CiTestSpec::named("weighted_parcorr"),
        discovery_seed,
    );
    contract.fdr = None;
    contract.constraints.alpha = ALPHA;
    contract.constraints.max_cond_size = MAX_DEGREE;
    contract.constraints.max_parents = Some(MAX_DEGREE);
    let rule = SplitRule::SeededPermutation {
        seed: stream_seed(seed, 5),
        discovery_fraction: discovery_fraction(protocol_version()),
    };
    let split = discover_on_split(&contract, &table, rule, &ctx).ok()?;

    let outcome = split.outcome();
    if outcome.search_incomplete
        || outcome.diagnostics.iter().any(|(code, _)| code.ends_with(PARENT_LIMIT_DIAGNOSTIC))
    {
        return None;
    }
    let treatment = u32::try_from(scm.treatment).ok()?;
    let response = u32::try_from(scm.outcome).ok()?;
    let neighbourhood = match (&outcome.native, outcome.orientation_conflict) {
        (_, false) => treatment_parent_set(&outcome.structure, &scm.names, treatment, response),
        (NativeResult::StaticCpdag(native), true) if protocol_version() >= 2 => {
            treatment_parent_set_under_conflicts(
                &outcome.structure,
                &ConflictEvidence {
                    graph: &native.evidence.graph,
                    sepsets: &native.sepsets,
                    constraints: &contract.constraints,
                    variables: &contract.variables,
                },
                &scm.names,
                treatment,
                response,
                MAX_CONFLICTING_COLLIDERS,
            )
        }
        (_, true) => return None,
    }
    .ok()?;
    let request = AnalysisEffectRequest {
        treatment: v(scm.treatment),
        outcome: v(scm.outcome),
        adjustment: neighbourhood.parents.iter().map(|&p| v(p as usize)).collect(),
        levels: (0.0, 1.0),
        clusters: None,
    };
    let effect = if protocol_version() == 3 {
        split.estimate_analysis_effect(&request, &ctx)
    } else {
        split.estimate_analysis_effect_hc1(&request, &ctx)
    }
    .ok()?;
    let interval = effect.interval(REPORTED_LEVEL).ok()?;
    if !(interval.lower.is_finite() && interval.upper.is_finite()) {
        return None;
    }
    Some(((interval.lower, interval.upper), scm.total_effect()))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_split_effect_interval_calibrated() {
    let rows = grid_n(BASE_PARTICIPANTS);
    eprintln!("calibration-protocol weighted_split_source_version={}", protocol_version());
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "weighted_split_effect_interval_calibrated",
            dgp: "crates/antecedent/tests/common/sparse_scm.rs::simulate",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    if protocol_version() == 2 {
        tally = tally.labelled(ALLOCATION_PROTOCOL);
    } else if protocol_version() == 3 {
        tally = tally.labelled(CURRENT_PROTOCOL);
    }
    let mut widths = Vec::new();
    let mut magnitudes = Vec::new();
    let mut reported = 0_u32;
    let runs = map_replicates(n_sim(), replicate);
    let total = u32::try_from(runs.len()).expect("replicate count fits u32");
    for run in runs {
        tally.bind_validation(
            &construction_for_protocol(protocol_version()),
            ScopeFacts {
                row_count: rows as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        if let Some((interval, truth)) = run {
            reported += 1;
            widths.push(interval.1 - interval.0);
            magnitudes.push(truth.abs());
            tally.record(Some(interval), truth);
        }
    }
    let share = f64::from(reported) / f64::from(total.max(1));
    widths.sort_by(f64::total_cmp);
    magnitudes.sort_by(f64::total_cmp);
    let median = |values: &[f64]| values.get(values.len() / 2).copied().unwrap_or(f64::NAN);
    let (width, effect) = (median(&widths), median(&magnitudes));
    eprintln!(
        "info summary weighted_split_effect_interval_calibrated: replicates={total} \
         reported={reported} reported_share={share:.3} median_width={width:.4} \
         median_effect={effect:.4}"
    );
    tally.assert();
    // The manifest freezes the reporting and precision targets at its 3,000 participants; the
    // other grid points are measured for their coverage.
    if rows == BASE_PARTICIPANTS {
        assert!(
            share >= MIN_REPORTED_SHARE,
            "reporting share {share:.3} below {MIN_REPORTED_SHARE}"
        );
        assert!(
            width <= MAX_MEDIAN_WIDTH_OVER_EFFECT * effect,
            "median width {width:.4} above {MAX_MEDIAN_WIDTH_OVER_EFFECT} times the effect {effect:.4}"
        );
    }
    if n_sim() >= FROZEN_REPLICATES {
        let covered = f64::from(tally.covered()) / f64::from(tally.attempts().max(1));
        assert!(covered >= COVERAGE_FLOOR, "coverage {covered:.3} below the frozen floor");
    }
}

#[test]
fn weighted_split_protocol_preserves_legacy_identity_and_declares_independent_allocation() {
    let legacy = construction_for_protocol(1);
    let intermediate = construction_for_protocol(2);
    let current = construction_for_protocol(3);
    assert_eq!(discovery_fraction(1).to_bits(), 0.5_f64.to_bits());
    assert_eq!(discovery_fraction(2).to_bits(), 0.75_f64.to_bits());
    assert_eq!(discovery_fraction(3).to_bits(), 0.75_f64.to_bits());
    assert_eq!(legacy.functional, "weighted_population_total_effect");
    assert_eq!(current.functional, format!("{}.{}", legacy.functional, CURRENT_PROTOCOL));
    assert_eq!(current.interval_method, legacy.interval_method);
    assert_eq!(intermediate.functional, format!("{}.{}", legacy.functional, ALLOCATION_PROTOCOL));
    assert_eq!(intermediate.se_kind, legacy.se_kind);
    assert_eq!(current.se_kind, "hc3_sampling_weighted");
    assert_eq!(current.estimator, legacy.estimator);
    assert_eq!(current.reported_level.to_bits(), legacy.reported_level.to_bits());
}
