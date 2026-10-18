//! Whole-method coverage of the true effect after discovery on a row split, accepted on the
//! treatment's parent set, for sparse linear and log-linear studies of ten, fifteen and twenty
//! variables.
//!
//! The procedure under test is the facade's construction
//! (`antecedent::analysis::sample_split_parent_set_effect`): the declared transformation, a
//! seeded three-to-one split, PC with the partial-correlation test (`alpha = 0.01`, conditioning-set
//! size and parent limit equal to the declared maximum degree of four) on the discovery rows,
//! acceptance of the class when every edge incident to the treatment is oriented (nuisance
//! edges may stay undirected), and the least-squares effect of the treatment on the outcome
//! adjusted for the treatment's parents on the independent analysis quarter. A replicate scores the 95%
//! interval against the truth computed from the structural coefficients alone: the total
//! effect (linear) or the intervention mean ratio of multiplying the treatment by `e`
//! (log-linear).
//!
//! # Gated designs
//!
//! One test per transformation, variable count and topology
//! (`parent_set_<transformation>_<n>_variables_<topology>_nominal_95_coverage`), each its own
//! record: the calibration key carries the transformation (through the estimator), the variable
//! count, the declared degree, any background knowledge and the topology (through the
//! functional), so no record borrows another's evidence and none widens the fully directed
//! five-to-seven-variable record. Total participants are `grid_n(2000)`: 1,000, 2,000 and 4,000,
//! three quarters to discovery and one quarter to analysis. The three topologies of `common::sparse_scm` (`hub_confounded_multi_path`,
//! `layered_multi_mediator_nuisance` and `confounded_direct_and_mediated`, which has a direct
//! edge from the treatment to the outcome plus a mediated path under confounding) have weights
//! redrawn until the strong-faithfulness margin `sparse_scm::MIN_MARGIN` holds. A table does not
//! announce its topology, so the runtime licenses a cell only when the records of every
//! topology of its declared degree attest it (the conservative intersection,
//! `ParentSetEffect::keyed_bases`).
//!
//! Every replicate reruns the whole procedure: the model is drawn, the rows simulated, and
//! discovery, acceptance or refusal and estimation run on them. The truth is computed
//! independently of the estimator and of the path product, by simulating the intervention
//! (`Scm::simulated_total_effect`, `Scm::simulated_mean_ratio`: the treatment's equation
//! replaced by the constant, common noise draws). A replicate that reports no interval is not
//! scored as covered or missed: coverage is the coverage *of reported intervals*. Each design
//! prints, as `info summary`, the reporting denominator and share, every refusal reason with
//! its count, the median interval width, the share whose treatment parents were recovered, and
//! the skeleton and orientation precision and recall of the discovered class against the true
//! graph (orientation recall counts every true edge, including those a CPDAG leaves open).
//!
//! The frozen usefulness targets of the acceptance manifest are asserted: reporting share at
//! least 0.9, median width at most 0.6, 0.42 and 0.3 at 1,000, 2,000 and 4,000 participants,
//! and, once the run has the 2,000 replicates the manifest freezes, coverage of at least 0.93.
//!
//! # Knowledge regimes
//!
//! Each advertised background-knowledge regime is validated end to end at ten variables,
//! linear, with statements that are true of the model:
//! `parent_set_knowledge_<regime>_<topology>_nominal_95_coverage` for `required` (every edge
//! incident to the treatment), `forbidden` (no nuisance node causes a core node) and `tiers`
//! (causal depth), and `parent_limits` (a declared maximum degree of three on a degree-three
//! family). The knowledge enters the calibration key.
//!
//! # Named boundary designs
//!
//! Each runs the same facade procedure on a family outside the published conditions at fifteen
//! variables (linear), pooled over the topologies, and is keyed under its own label so it can
//! never govern a gated record. A boundary record states where the construction stops holding;
//! it is not a failure.
//! These stress records score the whole procedure: a refusal counts as a miss, including when
//! every replicate refuses. Their `whole_procedure` functional and requested-participant scope
//! distinguish them from the conditional coverage and successful analysis-row scope above.
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

use std::collections::{BTreeMap, BTreeSet};

use antecedent::analysis::sample_split_effect::LEVEL;
use antecedent::analysis::sample_split_parent_set_effect::{
    Knowledge, ParentSetOptions, Transformation, estimate_parent_set_effect_with_knowledge,
};
use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_io::IoError;
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, map_replicates, n_sim, stream_seed,
};
use common::sparse_scm::{Scm, Topology, Variant};

const BASE_ROWS: usize = 2000;
const MIN_REPORTED_SHARE: f64 = 0.9;
/// The manifest's frozen coverage floor, applied once a run has its frozen replicate count.
const COVERAGE_FLOOR: f64 = 0.93;
const FROZEN_REPLICATES: u32 = 2000;
const INTERVAL: &str = "analytic_se";
/// The prespecified intervention of a log-linear study: multiply the treatment by `e`.
const LOG_SHIFT: f64 = 1.0;
/// Rows of the intervention simulations that give the truth.
const TRUTH_ROWS: usize = 2_000;

/// Explicit replay of a previously measured version-one source. Ordinary measurements exercise
/// the current facade; legacy replay executes the real original producer and retains its keys.
fn protocol_version() -> u32 {
    static VERSION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        let raw =
            std::env::var("ANTECEDENT_PARENT_PROTOCOL_VERSION").unwrap_or_else(|_| "2".into());
        raw.parse::<u32>()
            .ok()
            .filter(|version| matches!(version, 1 | 2))
            .unwrap_or_else(|| panic!("ANTECEDENT_PARENT_PROTOCOL_VERSION={raw}: want 1 or 2"))
    })
}

/// The frozen median-width target at `participants` total participants.
fn width_target(participants: usize) -> f64 {
    match participants {
        0..=1_500 => 0.6,
        1_501..=3_000 => 0.42,
        _ => 0.3,
    }
}

/// Which background knowledge a design asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regime {
    None,
    Required,
    Forbidden,
    Tiers,
    /// A declared maximum degree of three, on a degree-three family.
    ParentLimits,
}

/// One validated configuration.
#[derive(Clone, Copy)]
struct Design {
    transformation: Transformation,
    variables: usize,
    topology: Topology,
    regime: Regime,
}

impl Design {
    const fn degree(self) -> u32 {
        match self.regime {
            Regime::ParentLimits => 3,
            _ => 4,
        }
    }
}

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

/// One replicate's report: the interval, the truth, its key, and how the discovered class
/// compares with the true graph. `None` when no interval was reported.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: (Construction, ScopeFacts),
    parents_recovered: bool,
    /// `(true positives, reported, true)` of the undirected skeleton.
    skeleton: (u32, u32, u32),
    /// `(correctly directed, reported directed, true edges)`.
    orientation: (u32, u32, u32),
}

/// One replicate: the scored report, or the registered refusal detail that withheld it.
type Replicate = Result<Scored, String>;

fn knowledge_of(scm: &Scm, regime: Regime) -> Option<Knowledge> {
    let columns = |pairs: Vec<(usize, usize)>| -> Vec<(u32, u32)> {
        pairs.into_iter().map(|(a, b)| (a as u32, b as u32)).collect()
    };
    match regime {
        Regime::None | Regime::ParentLimits => None,
        Regime::Required => {
            Some(Knowledge { required: columns(scm.required_links()), ..Knowledge::default() })
        }
        Regime::Forbidden => {
            Some(Knowledge { forbidden: columns(scm.forbidden_links()), ..Knowledge::default() })
        }
        Regime::Tiers => Some(Knowledge {
            tiers: scm
                .tiers()
                .into_iter()
                .map(|tier| tier.into_iter().map(|v| v as u32).collect())
                .collect(),
            ..Knowledge::default()
        }),
    }
}

fn unordered(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

#[allow(
    clippy::too_many_lines,
    reason = "keep one complete discovery, admission, estimation and recovery audit together per replicate"
)]
fn replicate(design: Design, variant: Variant, skewed: bool, rep: u64) -> Replicate {
    let variables = design.variables;
    let seed = stream_seed(0x5E7_0017 + variables as u64, rep);
    let scm = Scm::with_degree(
        design.topology,
        variables,
        stream_seed(seed, 1),
        variant,
        design.degree() as usize,
    );
    let rows = grid_n(BASE_ROWS);
    let log = design.transformation == Transformation::LogLinear;
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
        ParentSetOptions::log_linear(design.degree(), LOG_SHIFT)
    } else {
        ParentSetOptions::linear(design.degree())
    };
    let ctx = ExecutionContext::for_tests(seed);
    let (reported, basis, parents, directed, unresolved) = if protocol_version() == 1 {
        use antecedent_io::measured_inference::split_parent_set_effect::{
            ParentSetEffectSource, run,
        };
        let mut source = ParentSetEffectSource::new(
            scm.names.clone(),
            columns,
            scm.treatment as u32,
            scm.outcome as u32,
            stream_seed(seed, 3),
            ctx.rng.master_seed(),
            options.transformation,
            options.max_degree,
            options.log_shift,
        );
        source.version = 1;
        source.knowledge = knowledge_of(&scm, design.regime);
        run(&source, &ctx).map(|execution| {
            let basis = execution.keyed_basis_for(design.topology.name());
            (
                execution.reported,
                basis,
                execution.neighbourhood.parents,
                execution.neighbourhood.directed,
                execution.neighbourhood.unresolved,
            )
        })
    } else {
        estimate_parent_set_effect_with_knowledge(
            &data,
            VariableId::from_raw(scm.treatment as u32),
            VariableId::from_raw(scm.outcome as u32),
            options,
            knowledge_of(&scm, design.regime),
            stream_seed(seed, 3),
            &ctx,
        )
        .map(|estimate| {
            (
                estimate.reported().clone(),
                estimate.keyed_basis_for(design.topology.name()),
                estimate.treatment_parents().to_vec(),
                estimate.directed_edges().to_vec(),
                estimate.unresolved_edges().to_vec(),
            )
        })
    }
    .map_err(|error| match error {
        IoError::Refused { message, .. } => {
            message.split_once(": ").map_or(message.clone(), |(detail, _)| detail.to_owned())
        }
        other => format!("error: {other}"),
    })?;
    let mut true_parents: Vec<u32> = scm.treatment_parents().iter().map(|&p| p as u32).collect();
    true_parents.sort_unstable();
    // Truth by simulating the intervention, independently of the estimator.
    let truth_seed = stream_seed(seed, 4);
    let truth = if log {
        scm.simulated_mean_ratio(LOG_SHIFT, TRUTH_ROWS, truth_seed)
    } else {
        scm.simulated_total_effect(TRUTH_ROWS, truth_seed)
    };
    let true_edges: BTreeSet<(u32, u32)> =
        scm.edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect();
    let true_skeleton: BTreeSet<(u32, u32)> =
        true_edges.iter().map(|&(a, b)| unordered(a, b)).collect();
    let directed: BTreeSet<(u32, u32)> = directed.into_iter().collect();
    let found_skeleton: BTreeSet<(u32, u32)> = directed
        .iter()
        .map(|&(a, b)| unordered(a, b))
        .chain(unresolved.iter().map(|e| unordered(e.a, e.b)))
        .collect();
    let count = |n: usize| n as u32;
    Ok(Scored {
        interval: (reported.lower, reported.upper),
        truth,
        construction: bound(&basis),
        parents_recovered: parents == true_parents,
        skeleton: (
            count(found_skeleton.intersection(&true_skeleton).count()),
            count(found_skeleton.len()),
            count(true_skeleton.len()),
        ),
        orientation: (
            count(directed.intersection(&true_edges).count()),
            count(directed.len()),
            count(true_edges.len()),
        ),
    })
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    f64::from(numerator) / f64::from(denominator.max(1))
}

/// What a design publishes about its replicates: the reporting denominator, every refusal
/// reason with its count, the median width, and the parent and graph recovery.
struct Summary {
    replicates: u32,
    reported: u32,
    refusals: BTreeMap<String, u32>,
    median_width: f64,
    coverage: f64,
    parents_recovered: f64,
    /// `(precision, recall)` of the discovered skeleton against the true graph.
    skeleton: (f64, f64),
    /// `(precision, recall)` of the discovered orientations against the true edges.
    orientation: (f64, f64),
}

impl Summary {
    fn reported_share(&self) -> f64 {
        ratio(self.reported, self.replicates)
    }
}

/// Tally `runs` into the [`Summary`] a design publishes, and persist its tallies.
fn summarize(test: &str, runs: &[Replicate]) -> Summary {
    let mut reported = CoverageTally::new(format!("{test} reported"), LEVEL);
    let mut coverage = CoverageTally::new(format!("{test} covered"), LEVEL);
    let mut recovery = CoverageTally::new(format!("{test} parents"), LEVEL);
    let mut widths = Vec::new();
    let mut refusals: BTreeMap<String, u32> = BTreeMap::new();
    let (mut skeleton, mut orientation) = ((0, 0, 0), (0, 0, 0));
    for run in runs {
        match run {
            Ok(scored) => {
                reported.record(Some((0.0, 1.0)), 0.5);
                recovery.record(scored.parents_recovered.then_some((0.0, 1.0)), 0.5);
                coverage.record(Some(scored.interval), scored.truth);
                widths.push(scored.interval.1 - scored.interval.0);
                skeleton = (
                    skeleton.0 + scored.skeleton.0,
                    skeleton.1 + scored.skeleton.1,
                    skeleton.2 + scored.skeleton.2,
                );
                orientation = (
                    orientation.0 + scored.orientation.0,
                    orientation.1 + scored.orientation.1,
                    orientation.2 + scored.orientation.2,
                );
            }
            Err(detail) => {
                reported.record(None, 0.5);
                recovery.record(None, 0.5);
                *refusals.entry(detail.clone()).or_default() += 1;
            }
        }
    }
    reported.persist();
    coverage.persist();
    recovery.persist();
    widths.sort_by(f64::total_cmp);
    Summary {
        replicates: reported.attempts(),
        reported: reported.covered(),
        refusals,
        median_width: widths.get(widths.len() / 2).copied().unwrap_or(f64::NAN),
        coverage: ratio(coverage.covered(), coverage.attempts()),
        parents_recovered: ratio(recovery.covered(), recovery.attempts()),
        skeleton: (ratio(skeleton.0, skeleton.1), ratio(skeleton.0, skeleton.2)),
        orientation: (ratio(orientation.0, orientation.1), ratio(orientation.0, orientation.2)),
    }
}

/// Print the [`Summary`] of `runs`; return `(reported share, median width, coverage)`.
fn print_summary(test: &str, runs: &[Replicate]) -> (f64, f64, f64) {
    let summary = summarize(test, runs);
    eprintln!(
        "info summary {test}: replicates={} reported={} reported_share={:.3} \
         refusals={:?} coverage={:.3} median_width={:.3} \
         parents_recovered={:.3} skeleton_precision={:.3} skeleton_recall={:.3} \
         orientation_precision={:.3} orientation_recall={:.3}",
        summary.replicates,
        summary.reported,
        summary.reported_share(),
        summary.refusals,
        summary.coverage,
        summary.median_width,
        summary.parents_recovered,
        summary.skeleton.0,
        summary.skeleton.1,
        summary.orientation.0,
        summary.orientation.1,
    );
    (summary.reported_share(), summary.median_width, summary.coverage)
}

fn gated(test: &'static str, design: Design) {
    eprintln!("calibration-protocol parent_set_source_version={}", protocol_version());
    let runs = map_replicates(n_sim(), |rep| replicate(design, Variant::Licensed, false, rep));
    let (share, median, covered) = print_summary(test, &runs);
    let dgp = match design.transformation {
        Transformation::Linear => "crates/antecedent/tests/common/sparse_scm.rs::simulate",
        Transformation::LogLinear => {
            "crates/antecedent/tests/common/sparse_scm.rs::simulate_log_linear"
        }
    };
    let key = RecordKey { test, dgp, interval: INTERVAL };
    let tally = CoverageTally::for_record(key, LEVEL);
    let mut tally =
        if protocol_version() == 1 { tally } else { tally.labelled("discovery_fraction_0_75") };
    for scored in runs.iter().flatten() {
        tally.bind(&scored.construction.0, scored.construction.1);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
    // The frozen usefulness targets.
    assert!(share >= MIN_REPORTED_SHARE, "{test}: reporting share {share:.3}");
    let target = width_target(grid_n(BASE_ROWS));
    assert!(median <= target, "{test}: median width {median:.3} above the frozen {target}");
    if n_sim() >= FROZEN_REPLICATES {
        assert!(covered >= COVERAGE_FLOOR, "{test}: coverage {covered:.3} below the frozen floor");
    }
}

/// A named boundary at fifteen variables (linear), pooled over the topologies and keyed under
/// its own label.
fn boundary(test: &'static str, label: &str, variant: Variant, skewed: bool) {
    eprintln!("calibration-protocol parent_set_source_version={}", protocol_version());
    let topologies = Topology::for_degree(4);
    let runs = map_replicates(n_sim(), |rep| {
        let design = Design {
            transformation: Transformation::Linear,
            variables: 15,
            topology: topologies[(rep as usize) % topologies.len()],
            regime: Regime::None,
        };
        replicate(design, variant, skewed, rep)
    });
    print_summary(test, &runs);
    let key = RecordKey {
        test,
        dgp: "crates/antecedent/tests/common/sparse_scm.rs::simulate",
        interval: INTERVAL,
    };
    if protocol_version() == 1 {
        // Replay the historical conditional stress record exactly, including its original
        // pooled functional and the absence of a record when every execution refused.
        if runs.iter().flatten().next().is_none() {
            eprintln!("{test}: no replicate reported an interval; no record is emitted");
            return;
        }
        let mut tally = CoverageTally::for_record(key, LEVEL);
        for scored in runs.iter().flatten() {
            let mut construction = scored.construction.0.clone();
            let head = construction.functional.split_once(".topology_").unwrap().0;
            construction.functional = format!("{head}.topology_pooled.named_boundary:{label}");
            tally.bind(&construction, scored.construction.1);
            tally.record(Some(scored.interval), scored.truth);
        }
        tally.emit_named_boundary();
        return;
    }
    // Bind the attempted request, not a successful interval: the whole-procedure stress
    // measure remains defined when discovery refuses every draw. No success-dependent scope
    // facts are supplied, and this functional cannot match the nominal conditional route.
    let construction = Construction {
        query: "AverageEffect".into(),
        graph_class: "Cpdag".into(),
        structure: antecedent_io::measured_inference::split_parent_set_effect::PARENT_SET_STRUCTURE
            .into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: antecedent_io::measured_inference::split_parent_set_effect::LINEAR_ESTIMATOR
            .into(),
        interval_method: INTERVAL.into(),
        se_kind: "homoskedastic".into(),
        dependence: "disjoint_rows".into(),
        posterior: String::new(),
        functional: format!(
            "total_effect.variables_15.max_degree_4.discovery_fraction_0_75.topology_pooled.named_boundary:{label}.whole_procedure"
        ),
        identification: "point".into(),
        reported_level: LEVEL,
    };
    let requested = ScopeFacts {
        row_count: grid_n(BASE_ROWS) as u64,
        replicates_ok: None,
        posterior_draws: None,
        unidentified_mass: 0.0,
    };
    let mut tally =
        CoverageTally::for_record(key, LEVEL).labelled("discovery_fraction_0_75.whole_procedure");
    for run in &runs {
        tally.bind(&construction, requested);
        match run {
            Ok(scored) => tally.record(Some(scored.interval), scored.truth),
            Err(_) => tally.record(None, 0.0),
        }
    }
    tally.emit_named_boundary();
}

const fn design(
    transformation: Transformation,
    variables: usize,
    topology: Topology,
    regime: Regime,
) -> Design {
    Design { transformation, variables, topology, regime }
}

/// A design publishes, for every cell it runs, the denominator of its replicates, the reason of
/// every refusal, the median width of the reported intervals and the parent and graph recovery.
/// A few replicates of one linear and one log-linear cell show each is accounted for and well
/// formed: reported plus refused equals the replicates, every refusal names a registered detail,
/// and the widths and recovery shares are finite and in range.
#[test]
fn discovery_validation_publishes_denominators_refusals_widths_and_recovery() {
    const REPLICATES: u32 = 4;
    for transformation in [Transformation::Linear, Transformation::LogLinear] {
        let cell = design(transformation, 10, Topology::HubConfoundedMultiPath, Regime::None);
        let runs: Vec<Replicate> = (0..u64::from(REPLICATES))
            .map(|rep| replicate(cell, Variant::Licensed, false, rep))
            .collect();
        let summary = summarize("discovery_validation_publication_check", &runs);
        assert_eq!(summary.replicates, REPLICATES, "the denominator is every replicate");
        assert_eq!(
            summary.reported + summary.refusals.values().sum::<u32>(),
            REPLICATES,
            "every replicate is either reported or refused with a stated reason: {:?}",
            summary.refusals
        );
        assert!(summary.refusals.keys().all(|detail| !detail.is_empty()));
        for share in [
            summary.parents_recovered,
            summary.skeleton.0,
            summary.skeleton.1,
            summary.orientation.0,
            summary.orientation.1,
        ] {
            assert!((0.0..=1.0).contains(&share), "recovery share {share} is a proportion");
        }
        if summary.reported > 0 {
            assert!(summary.median_width.is_finite() && summary.median_width > 0.0);
            assert!((0.0..=1.0).contains(&summary.coverage));
        } else {
            assert!(summary.median_width.is_nan(), "no reported interval has no width");
        }
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_10_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_linear_10_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 15, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::Linear, 15, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_15_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_linear_15_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::Linear, 15, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 20, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::Linear, 20, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_linear_20_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_linear_20_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::Linear, 20, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::LogLinear, 10, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_10_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::LogLinear, 10, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_10_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_10_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::LogLinear, 10, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_15_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::LogLinear, 15, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_15_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::LogLinear, 15, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_15_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_15_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::LogLinear, 15, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_20_variables_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::LogLinear, 20, Topology::HubConfoundedMultiPath, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_20_variables_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::LogLinear, 20, Topology::LayeredMultiMediatorNuisance, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_log_linear_20_variables_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_log_linear_20_variables_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::LogLinear, 20, Topology::ConfoundedDirectAndMediated, Regime::None),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_required_edges_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_required_edges_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::HubConfoundedMultiPath, Regime::Required),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_required_edges_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_required_edges_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(
            Transformation::Linear,
            10,
            Topology::LayeredMultiMediatorNuisance,
            Regime::Required,
        ),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_required_edges_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_required_edges_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::ConfoundedDirectAndMediated, Regime::Required),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_forbidden_edges_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_forbidden_edges_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::HubConfoundedMultiPath, Regime::Forbidden),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_forbidden_edges_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_forbidden_edges_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(
            Transformation::Linear,
            10,
            Topology::LayeredMultiMediatorNuisance,
            Regime::Forbidden,
        ),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_forbidden_edges_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_forbidden_edges_confounded_direct_and_mediated_nominal_95_coverage",
        design(
            Transformation::Linear,
            10,
            Topology::ConfoundedDirectAndMediated,
            Regime::Forbidden,
        ),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_tiers_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_tiers_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::HubConfoundedMultiPath, Regime::Tiers),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_tiers_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_tiers_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::LayeredMultiMediatorNuisance, Regime::Tiers),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_tiers_confounded_direct_and_mediated_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_tiers_confounded_direct_and_mediated_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::ConfoundedDirectAndMediated, Regime::Tiers),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_parent_limits_hub_confounded_multi_path_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_parent_limits_hub_confounded_multi_path_nominal_95_coverage",
        design(Transformation::Linear, 10, Topology::HubConfoundedMultiPath, Regime::ParentLimits),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_knowledge_parent_limits_layered_multi_mediator_nuisance_nominal_95_coverage() {
    gated(
        "parent_set_knowledge_parent_limits_layered_multi_mediator_nuisance_nominal_95_coverage",
        design(
            Transformation::Linear,
            10,
            Topology::LayeredMultiMediatorNuisance,
            Regime::ParentLimits,
        ),
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_weak_signal_boundary_95() {
    boundary("parent_set_weak_signal_boundary_95", "weak_signal", Variant::WeakSignal, false);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_cancelling_path_boundary_95() {
    boundary(
        "parent_set_cancelling_path_boundary_95",
        "cancelling_path",
        Variant::CancellingPath,
        false,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn parent_set_misspecified_test_boundary_95() {
    boundary(
        "parent_set_misspecified_test_boundary_95",
        "misspecified_test",
        Variant::Licensed,
        true,
    );
}
