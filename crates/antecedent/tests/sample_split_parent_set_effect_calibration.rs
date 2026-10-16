//! Whole-method coverage of the true effect after discovery on a row split, accepted on the
//! treatment's parent set, for sparse linear and log-linear studies of ten, fifteen and twenty
//! variables.
//!
//! The procedure under test is the facade's construction
//! (`antecedent::analysis::sample_split_parent_set_effect`): the declared transformation, a
//! seeded half split, PC with the partial-correlation test (`alpha = 0.01`, conditioning-set
//! size and parent limit equal to the declared maximum degree of four) on the discovery half,
//! acceptance of the class when every edge incident to the treatment is oriented (nuisance
//! edges may stay undirected), and the least-squares effect of the treatment on the outcome
//! adjusted for the treatment's parents on the analysis half. A replicate scores the 95%
//! interval against the truth computed from the structural coefficients alone: the total
//! effect (linear) or the intervention mean ratio of multiplying the treatment by `e`
//! (log-linear).
//!
//! # Gated designs
//!
//! One test per transformation, variable count and topology
//! (`parent_set_<transformation>_<n>_variables_<topology>_nominal_95_coverage`), each its own
//! record: the calibration key carries the transformation (through the estimator), the variable
//! count, the declared degree and the topology (through the functional), so no record borrows
//! another's evidence and none widens the fully directed five-to-seven-variable record. Total
//! participants are `grid_n(2000)`: 1,000, 2,000 and 4,000, half to each part. The two
//! topologies of `common::sparse_scm` (`hub_confounded_multi_path`,
//! `layered_multi_mediator_nuisance`) have weights redrawn until the strong-faithfulness margin
//! `sparse_scm::MIN_MARGIN` holds. A table does not announce its topology, so the runtime
//! licenses a cell only when the records of both topologies attest it (the conservative
//! intersection, `ParentSetEffect::keyed_bases`).
//!
//! A replicate that reports no interval (an undirected treatment edge, an unfinished search, a
//! parent limit that bound) is not scored as covered or missed: coverage is the coverage *of
//! reported intervals*, and the share of replicates that report one is asserted to be at least
//! 0.9 and printed with the median width and the share whose treatment parents were recovered.
//!
//! # Named boundary designs
//!
//! Each runs the same facade procedure on a family outside the published conditions at fifteen
//! variables (linear), pooled over both topologies, and is keyed under its own label so it can
//! never govern a gated record. A boundary record states where the construction stops holding;
//! it is not a failure.
//!
//! * `weak_signal`: coefficients of magnitude 0.1 to 0.25, below the published margin.
//! * `cancelling_path`: the confounder's coefficient on the outcome cancels the mediated path
//!   exactly, an unfaithful law.
//! * `misspecified_test`: right-skewed noise, for which the partial-correlation test does not
//!   see the dependence among a collider's parents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use antecedent::analysis::sample_split_effect::LEVEL;
use antecedent::analysis::sample_split_parent_set_effect::{
    ParentSetOptions, Transformation, estimate_parent_set_effect,
};
use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, map_replicates, n_sim, stream_seed,
};
use common::sparse_scm::{Scm, Topology, Variant};

const BASE_ROWS: usize = 2000;
const MIN_REPORTED_SHARE: f64 = 0.9;
const INTERVAL: &str = "analytic_se";
const MAX_DEGREE: u32 = 4;
/// The prespecified intervention of a log-linear study: multiply the treatment by `e`.
const LOG_SHIFT: f64 = 1.0;
/// The topologies a boundary design alternates between.
const TOPOLOGY_CYCLE: [Topology; 2] =
    [Topology::HubConfoundedMultiPath, Topology::LayeredMultiMediatorNuisance];

fn bound(basis: &CalibrationBasis) -> (Construction, ScopeFacts) {
    (
        Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    )
}

/// One replicate's report: the interval, the truth, its key and whether the discovered
/// treatment parents are the true ones. `None` when no interval was reported.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: (Construction, ScopeFacts),
    parents_recovered: bool,
}

fn replicate(
    transformation: Transformation,
    variables: usize,
    topology: Topology,
    variant: Variant,
    skewed: bool,
    rep: u64,
) -> Option<Scored> {
    let seed = stream_seed(0x5E7_0017 + variables as u64, rep);
    let scm = Scm::with_variant(topology, variables, stream_seed(seed, 1), variant);
    let rows = grid_n(BASE_ROWS);
    let log = transformation == Transformation::LogLinear;
    let columns = match (log, skewed) {
        (true, _) => scm.simulate_log_linear(rows, stream_seed(seed, 2)),
        (false, true) => scm.simulate_skewed(rows, stream_seed(seed, 2)),
        (false, false) => scm.simulate(rows, stream_seed(seed, 2)),
    };
    let data = TabularData::from_f64_columns(
        scm.names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .expect("table");
    let options = if log {
        ParentSetOptions::log_linear(MAX_DEGREE, LOG_SHIFT)
    } else {
        ParentSetOptions::linear(MAX_DEGREE)
    };
    let ctx = ExecutionContext::for_tests(seed);
    let estimate = estimate_parent_set_effect(
        &data,
        VariableId::from_raw(scm.treatment as u32),
        VariableId::from_raw(scm.outcome as u32),
        options,
        stream_seed(seed, 3),
        &ctx,
    )
    .ok()?;
    let mut true_parents: Vec<u32> = scm.treatment_parents().iter().map(|&p| p as u32).collect();
    true_parents.sort_unstable();
    let reported = estimate.reported();
    Some(Scored {
        interval: (reported.lower, reported.upper),
        truth: if log { scm.mean_ratio(LOG_SHIFT) } else { scm.total_effect() },
        construction: bound(&estimate.keyed_basis_for(topology.name())),
        parents_recovered: estimate.treatment_parents() == true_parents.as_slice(),
    })
}

/// Print the reporting share, coverage, median width and parent recovery of `runs`.
fn print_summary(test: &str, runs: &[Option<Scored>], gated: bool) {
    let mut reported = CoverageTally::new(format!("{test} reported"), LEVEL);
    let mut coverage = CoverageTally::new(format!("{test} covered"), LEVEL);
    let mut recovery = CoverageTally::new(format!("{test} parents"), LEVEL);
    let mut widths = Vec::new();
    for run in runs {
        if let Some(scored) = run {
            reported.record(Some((0.0, 1.0)), 0.5);
            recovery.record(scored.parents_recovered.then_some((0.0, 1.0)), 0.5);
            coverage.record(Some(scored.interval), scored.truth);
            widths.push(scored.interval.1 - scored.interval.0);
        } else {
            reported.record(None, 0.5);
            recovery.record(None, 0.5);
        }
    }
    reported.persist();
    coverage.persist();
    recovery.persist();
    widths.sort_by(f64::total_cmp);
    let median = widths.get(widths.len() / 2).copied().unwrap_or(f64::NAN);
    let share = f64::from(reported.covered()) / f64::from(reported.attempts().max(1));
    eprintln!(
        "info summary {test}: reported_share={share:.3} ({} of {}) coverage={:.3} \
         median_width={median:.3} parents_recovered={:.3}",
        reported.covered(),
        reported.attempts(),
        f64::from(coverage.covered()) / f64::from(coverage.attempts().max(1)),
        f64::from(recovery.covered()) / f64::from(recovery.attempts().max(1)),
    );
    if gated {
        assert!(
            share >= MIN_REPORTED_SHARE,
            "{test}: only {share:.3} of replicates reported an interval (floor \
             {MIN_REPORTED_SHARE})"
        );
    }
}

fn gated(test: &'static str, transformation: Transformation, variables: usize, topology: Topology) {
    let runs = map_replicates(n_sim(), |rep| {
        replicate(transformation, variables, topology, Variant::Licensed, false, rep)
    });
    print_summary(test, &runs, true);
    let dgp = match transformation {
        Transformation::Linear => "crates/antecedent/tests/common/sparse_scm.rs::simulate",
        Transformation::LogLinear => {
            "crates/antecedent/tests/common/sparse_scm.rs::simulate_log_linear"
        }
    };
    let key = RecordKey { test, dgp, interval: INTERVAL };
    let mut tally = CoverageTally::for_record(key, LEVEL);
    for scored in runs.iter().flatten() {
        tally.bind(&scored.construction.0, scored.construction.1);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
}

/// A named boundary at fifteen variables (linear), pooled over both topologies and keyed under
/// its own label.
fn boundary(test: &'static str, label: &str, variant: Variant, skewed: bool) {
    let runs = map_replicates(n_sim(), |rep| {
        let topology = TOPOLOGY_CYCLE[(rep % 2) as usize];
        replicate(Transformation::Linear, 15, topology, variant, skewed, rep).map(|mut scored| {
            scored.construction.0.functional =
                format!("{}.named_boundary:{label}", scored.construction.0.functional);
            scored
        })
    });
    print_summary(test, &runs, false);
    let key = RecordKey {
        test,
        dgp: "crates/antecedent/tests/common/sparse_scm.rs::simulate",
        interval: INTERVAL,
    };
    if runs.iter().flatten().next().is_none() {
        eprintln!("{test}: no replicate reported an interval; no record is emitted");
        return;
    }
    let mut tally = CoverageTally::for_record(key, LEVEL);
    for scored in runs.iter().flatten() {
        tally.bind(&scored.construction.0, scored.construction.1);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.emit_named_boundary();
}

macro_rules! gated_design {
    ($name:ident, $transformation:expr, $variables:expr, $topology:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            gated(stringify!($name), $transformation, $variables, $topology);
        }
    };
}

macro_rules! boundary_design {
    ($name:ident, $label:expr, $variant:expr, $skewed:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            boundary(stringify!($name), $label, $variant, $skewed);
        }
    };
}

gated_design!(
    parent_set_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::Linear,
    10,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::Linear,
    10,
    Topology::LayeredMultiMediatorNuisance
);
gated_design!(
    parent_set_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::Linear,
    15,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::Linear,
    15,
    Topology::LayeredMultiMediatorNuisance
);
gated_design!(
    parent_set_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::Linear,
    20,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::Linear,
    20,
    Topology::LayeredMultiMediatorNuisance
);
gated_design!(
    parent_set_log_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::LogLinear,
    10,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_log_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::LogLinear,
    10,
    Topology::LayeredMultiMediatorNuisance
);
gated_design!(
    parent_set_log_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::LogLinear,
    15,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_log_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::LogLinear,
    15,
    Topology::LayeredMultiMediatorNuisance
);
gated_design!(
    parent_set_log_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage,
    Transformation::LogLinear,
    20,
    Topology::HubConfoundedMultiPath
);
gated_design!(
    parent_set_log_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage,
    Transformation::LogLinear,
    20,
    Topology::LayeredMultiMediatorNuisance
);

boundary_design!(parent_set_weak_signal_boundary_95, "weak_signal", Variant::WeakSignal, false);
boundary_design!(
    parent_set_cancelling_path_boundary_95,
    "cancelling_path",
    Variant::CancellingPath,
    false
);
boundary_design!(
    parent_set_misspecified_test_boundary_95,
    "misspecified_test",
    Variant::Licensed,
    true
);
