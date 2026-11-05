//! Known-truth checks for the four partial-interference exposure-contrast estimators, each
//! explicitly selected on every supported `InterferenceQuery` coordinate: the cluster Neyman
//! total effect, the exact two-stage saturation contrast, the supplied-propensity observational
//! contrast and the cluster Bayesian bootstrap.
//!
//! The assignment design is a property of the query, not of the coordinate, and the supplied
//! structure only binds the unit schema, so each estimator runs on the one design it models and
//! the test walks the whole coordinate grid: the five static graph classes explicit and accepted,
//! the five posterior atom kinds, and the `none`, `cheap` and `full` validation suites.
//!
//! Every expected value is computed in the test from the definition of the contrast on small
//! hand-counted fixtures, never by calling a production estimator or the library's default
//! estimator:
//!
//! * cluster Neyman: six clusters of 2, 3, 2, 3, 2, 2 units with totals 7, 7, 14, 14, 21, 7;
//!   scaling by `K / n = 3 / 7` gives treated totals 3, 6, 9 and control totals 3, 6, 3, so the
//!   contrast is `6 - 4 = 2`, the variance is `9 / 3 + 3 / 3 = 4` and the Hajek contrast is
//!   `7 - 3.5 = 3.5`;
//! * cluster Bayesian bootstrap: the posterior of the same contrast is the difference of two
//!   independent Dirichlet(1, 1, 1) weighted means of the arm totals, so its mean is 2 and its
//!   variance is `(Sum x^2 / n - mean^2) / (n + 1)` per arm, `1.5 + 0.5 = 2`; the draws are a
//!   Monte Carlo sample of that posterior and are held to six standard errors of the mean;
//! * saturation: outcomes are a deterministic function `1 + 2 own + 3 f + 4 own f` of the
//!   exposure `(own, f)`, so the Hajek contrast is the difference of that function at the two
//!   levels, and the Horvitz-Thompson contrast is rebuilt from the exact level probabilities
//!   averaged over the high and low saturation;
//! * observational: the inverse-probability sums, the Hajek ratio and the cluster-robust
//!   variance of five two-unit clusters are written out by hand.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "test fixtures: small exact indices and one coordinate walk per test"
)]

#[path = "retained_run/mod.rs"]
mod retained;

use retained::run_retained;

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, InterferenceSpec, RefuteSuite,
    Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, IdentificationStatus, InterferenceFunctional, InterferenceQuery,
    ResponseValue, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge};
use antecedent_graph::{Admg, Cpdag, Dag, Pag, TemporalAdmg};
use antecedent_prob::InferenceDiagnostics;
use antecedent_validate::RefutationReport;

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

/// Class and posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("interference-estimators".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

// ---------------------------------------------------------------------------------------------
// Structures: every graph class explicit and accepted, and every posterior atom kind.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Structure {
    Explicit(&'static str),
    Accepted(&'static str),
    Posterior(GraphPosteriorAtomKind),
}

const CLASSES: [&str; 5] = ["dag", "admg", "temporal_admg", "cpdag", "pag"];

fn structures() -> Vec<Structure> {
    let mut all = Vec::new();
    for class in CLASSES {
        all.push(Structure::Explicit(class));
        all.push(Structure::Accepted(class));
    }
    for kind in [
        GraphPosteriorAtomKind::Dag,
        GraphPosteriorAtomKind::Admg,
        GraphPosteriorAtomKind::TemporalAdmg,
        GraphPosteriorAtomKind::Cpdag,
        GraphPosteriorAtomKind::Pag,
    ] {
        all.push(Structure::Posterior(kind));
    }
    all
}

fn bind(builder: StudyBuilder, class: &str, accepted: bool) -> StudyBuilder {
    match (class, accepted) {
        ("dag", false) => builder.graph(Dag::with_variables(1)),
        ("dag", true) => builder.graph(AcceptedGraph::from(Dag::with_variables(1))),
        ("admg", false) => builder.graph(Admg::with_variables(1)),
        ("admg", true) => builder.graph(AcceptedGraph::from(Admg::with_variables(1))),
        ("temporal_admg", false) => builder.graph(temporal_admg()),
        ("temporal_admg", true) => builder.graph(AcceptedGraph::from(temporal_admg())),
        ("cpdag", false) => builder.graph(Cpdag::with_variables(1)),
        ("cpdag", true) => builder.graph(AcceptedGraph::from(Cpdag::with_variables(1))),
        ("pag", false) => builder.graph(Pag::with_variables(1)),
        ("pag", true) => builder.graph(AcceptedGraph::from(Pag::with_variables(1))),
        other => panic!("unknown class {other:?}"),
    }
}

fn temporal_admg() -> TemporalAdmg {
    TemporalAdmg::from_static_contemporaneous(&Admg::with_variables(1)).unwrap()
}

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("interference_estimators_known_truth"),
        rejected_invalid: 0,
    }
}

/// Atoms over `[y, z, w]`: `y -> z`, the empty graph, `z -> y`, and the directed cycle
/// `y -> z -> w -> y`, which is no DAG, ADMG, CPDAG or PAG (weights 0.4, 0.3, 0.1, 0.2).
fn exposure_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let cycle = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 1, 2, true), 3, 2, 0, true);
    GraphPosterior::new(
        3,
        vec![0.4, 0.3, 0.1, 0.2],
        vec![set_edge(0, 3, 0, 1, true), 0, set_edge(0, 3, 1, 0, true), cycle],
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(),
    )
    .unwrap()
    .with_atom_kind(kind)
}

/// The unit table with two covariate columns the posterior's atoms are drawn over.
fn with_covariates(outcomes: &[f64]) -> TabularData {
    let z: Vec<f64> = (0..outcomes.len()).map(|row| (row % 2) as f64).collect();
    let w: Vec<f64> = (0..outcomes.len()).map(|row| ((row / 2) % 2) as f64).collect();
    TabularData::from_f64_columns([("y", outcomes), ("z", z.as_slice()), ("w", w.as_slice())])
        .unwrap()
}

// ---------------------------------------------------------------------------------------------
// Designs and the coordinate walk.
// ---------------------------------------------------------------------------------------------

struct Design {
    outcomes: Vec<f64>,
    data: TabularData,
    spec: InterferenceSpec,
    query: InterferenceQuery,
}

impl Design {
    fn new(
        outcomes: Vec<f64>,
        edges: &[(u64, u64)],
        assignment: Vec<bool>,
        query: InterferenceQuery,
    ) -> Self {
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(
            data.clone(),
            edges.iter().map(|&(from, to)| NetworkEdge::new(from, to, 1.0)).collect::<Vec<_>>(),
        )
        .unwrap();
        Self {
            outcomes,
            data,
            spec: InterferenceSpec { network, assignment: Arc::from(assignment) },
            query,
        }
    }

    fn study(
        &self,
        structure: Structure,
        inference: InferenceMode,
        estimator: EstimatorId,
        suite: RefuteSuite,
    ) -> StudyBuilder {
        let builder = match structure {
            Structure::Explicit(class) => bind(Study::tabular(self.data.clone()), class, false),
            Structure::Accepted(class) => bind(Study::tabular(self.data.clone()), class, true),
            Structure::Posterior(kind) => Study::tabular(with_covariates(&self.outcomes))
                .graph_posterior(exposure_posterior(kind)),
        };
        builder
            .query(CausalQuery::Interference(self.query.clone()))
            .interference(self.spec.clone())
            .inference(inference)
            .estimator(estimator)
            .refute(suite)
    }
}

fn ids(result: &StudyResult) -> Vec<String> {
    let mut found: Vec<String> =
        result.refutations.iter().map(|report| report.refuter.to_string()).collect();
    found.sort();
    found
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == id)
        .unwrap_or_else(|| panic!("no report {id} in {:?}", ids(result)))
}

fn near(got: f64, want: f64, tolerance: f64, label: &str) {
    assert!((got - want).abs() < tolerance, "{label}: {got} vs {want}");
}

/// Run `estimator` explicitly selected on all 45 coordinates of the design (15 structures x 3
/// validation suites). `point` is the value every atom and the published estimate must show,
/// to within `tolerance`.
fn walk(
    design: &Design,
    inference: &InferenceMode,
    estimator: EstimatorId,
    point: f64,
    tolerance: f64,
    seed: u64,
    check: &dyn Fn(&StudyResult, RefuteSuite, &str),
) {
    let ctx = ExecutionContext::for_tests(seed);
    for structure in structures() {
        for suite in SUITES {
            let label = format!("{estimator:?} {structure:?} {suite:?}");
            let result = run_retained(
                design.study(structure, inference.clone(), estimator, suite),
                &label,
                &ctx,
            );
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(estimator.as_str()),
                "{label}"
            );
            near(result.estimate().as_effect().unwrap().ate, point, tolerance, &label);
            check(&result, suite, &label);
            if matches!(structure, Structure::Posterior(_)) {
                // Three atoms are members of the posterior's graph class and carry the one
                // design identification; the directed cycle is no member and is unidentified
                // mass.
                let mixture = result.structural_response.as_ref().expect("a structural mixture");
                near(mixture.identified_mass, 0.8, 1e-9, &label);
                near(mixture.unidentified_mass, 0.2, 1e-9, &label);
                assert!(mixture.unevaluable_mass.abs() < 1e-12, "{label}");
                assert_eq!(mixture.atoms.len(), 4, "{label}");
                assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
                for atom in mixture.atoms.iter().filter(|atom| atom.value.is_some()) {
                    match atom.value {
                        Some(ResponseValue::Scalar(value)) => near(value, point, tolerance, &label),
                        ref other => panic!("{label}: an atom value is a scalar: {other:?}"),
                    }
                }
            }
        }
    }
}

fn contrast_query(
    assignment: AssignmentDesign,
    exposure: ExposureMapping,
    from: (f64, f64),
    to: (f64, f64),
) -> InterferenceQuery {
    InterferenceQuery::new(
        assignment,
        exposure,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: from.0, neighbors: from.1 },
            to: ExposureLevel { own: to.0, neighbors: to.1 },
        },
    )
}

/// Directed edges both ways between every pair of units inside each cluster.
fn within_cluster_edges(clusters: &[u32]) -> Vec<(u64, u64)> {
    let mut edges = Vec::new();
    for (i, a) in clusters.iter().enumerate() {
        for (j, b) in clusters.iter().enumerate() {
            if i != j && a == b {
                edges.push((i as u64, j as u64));
            }
        }
    }
    edges
}

// ---------------------------------------------------------------------------------------------
// The cluster fixture shared by the Neyman and the Bayesian-bootstrap tests.
// ---------------------------------------------------------------------------------------------

const CLUSTER_IDS: [u32; 14] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5];
const CLUSTER_Y: [f64; 14] = [3., 4., 2., 2., 3., 6., 8., 4., 5., 5., 10., 11., 3., 4.];
/// Clusters 0, 2 and 4 are treated.
const CLUSTER_TREATED: [bool; 6] = [true, false, true, false, true, false];

fn cluster_design() -> Design {
    let assignment: Vec<bool> =
        CLUSTER_IDS.iter().map(|&cluster| CLUSTER_TREATED[cluster as usize]).collect();
    Design::new(
        CLUSTER_Y.to_vec(),
        &within_cluster_edges(&CLUSTER_IDS),
        assignment,
        contrast_query(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from(CLUSTER_IDS.to_vec()),
                treated_clusters: 3,
            },
            ExposureMapping::NeighborFraction,
            (0.0, 0.0),
            (1.0, 1.0),
        ),
    )
}

struct ClusterTruth {
    treated: Vec<f64>,
    control: Vec<f64>,
    horvitz_thompson: f64,
    hajek: f64,
    variance: f64,
}

fn sample_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// The cluster-level Neyman quantities from their definitions: scaled cluster totals
/// `total * K / n`, the difference of arm means, the sum of the two `s^2 / m` terms, and the
/// Hajek ratio with first-stage probability `m / K`.
fn cluster_truth() -> ClusterTruth {
    let (k, n) = (6_usize, CLUSTER_IDS.len());
    let mut totals = [0.0_f64; 6];
    let mut sizes = [0_usize; 6];
    for (unit, &cluster) in CLUSTER_IDS.iter().enumerate() {
        totals[cluster as usize] += CLUSTER_Y[unit];
        sizes[cluster as usize] += 1;
    }
    assert_eq!(totals, [7., 7., 14., 14., 21., 7.]);
    assert_eq!(sizes, [2, 3, 2, 3, 2, 2]);
    let scale = k as f64 / n as f64;
    let scaled = |treated: bool| -> Vec<f64> {
        (0..k).filter(|&c| CLUSTER_TREATED[c] == treated).map(|c| totals[c] * scale).collect()
    };
    let (treated, control) = (scaled(true), scaled(false));
    for (got, want) in treated.iter().zip([3.0, 6.0, 9.0]) {
        assert!((got - want).abs() < 1e-12);
    }
    for (got, want) in control.iter().zip([3.0, 6.0, 3.0]) {
        assert!((got - want).abs() < 1e-12);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let horvitz_thompson = mean(&treated) - mean(&control);
    let variance = sample_variance(&treated) / 3.0 + sample_variance(&control) / 3.0;
    let p = 0.5;
    let weighted = |treated_arm: bool, probability: f64| -> (f64, f64) {
        let sum: f64 =
            (0..k).filter(|&c| CLUSTER_TREATED[c] == treated_arm).map(|c| totals[c]).sum();
        let units: usize =
            (0..k).filter(|&c| CLUSTER_TREATED[c] == treated_arm).map(|c| sizes[c]).sum();
        (sum / probability, units as f64 / probability)
    };
    let (treated_sum, treated_weight) = weighted(true, p);
    let (control_sum, control_weight) = weighted(false, 1.0 - p);
    let hajek = treated_sum / treated_weight - control_sum / control_weight;
    // The hand-counted values stated in the module documentation.
    assert!((horvitz_thompson - 2.0).abs() < 1e-12);
    assert!((variance - 4.0).abs() < 1e-12);
    assert!((hajek - 3.5).abs() < 1e-12);
    ClusterTruth { treated, control, horvitz_thompson, hajek, variance }
}

/// Mean and variance, over the 20 equally likely allocations of 3 of 6 clusters, of the
/// Horvitz-Thompson contrast `(1/n) (Sum_treated y / p - Sum_control y / (1 - p))` of the fixed
/// outcomes: the sharp-null re-randomization distribution, enumerated without the engine.
fn cluster_rerandomization_moments() -> (f64, f64) {
    let n = CLUSTER_IDS.len() as f64;
    let mut values = Vec::new();
    for mask in 0_u32..64 {
        if mask.count_ones() != 3 {
            continue;
        }
        let mut total = 0.0;
        for (unit, &cluster) in CLUSTER_IDS.iter().enumerate() {
            if (mask >> cluster) & 1 == 1 {
                total += CLUSTER_Y[unit] / 0.5;
            } else {
                total -= CLUSTER_Y[unit] / 0.5;
            }
        }
        values.push(total / n);
    }
    assert_eq!(values.len(), 20);
    let mean = values.iter().sum::<f64>() / 20.0;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 20.0;
    (mean, variance)
}

/// The realized cluster allocation against the declared design: three whole clusters treated.
fn assert_cluster_calibration(result: &StudyResult, label: &str) {
    let calibration = report(result, "interference.assignment_calibration");
    near(calibration.original_ate, 3.0, 1e-12, label);
    near(calibration.refuted_ate, 3.0, 1e-12, label);
    near(calibration.comparison, 1.0, 1e-12, label);
    assert!(calibration.passed, "{label}: {calibration:?}");
}

#[test]
fn cluster_neyman_matches_the_hand_counted_cluster_totals_on_every_coordinate() {
    on_large_stack(|| {
        let truth = cluster_truth();
        let design = cluster_design();
        let (rerandomized_mean, rerandomized_variance) = cluster_rerandomization_moments();
        assert!(rerandomized_mean.abs() < 1e-12, "{rerandomized_mean}");
        walk(
            &design,
            &InferenceMode::Frequentist,
            EstimatorId::InterferenceClusterNeyman,
            truth.horvitz_thompson,
            1e-9,
            9_101,
            &|result: &StudyResult, suite: RefuteSuite, label: &str| {
                let effect = result.estimate().as_effect().unwrap();
                near(effect.se_analytic(), truth.variance.sqrt(), 1e-9, label);
                let estimated = result.interference().expect("design-based contrast");
                near(estimated.contrast.horvitz_thompson, truth.horvitz_thompson, 1e-9, label);
                near(estimated.contrast.hajek, truth.hajek, 1e-9, label);
                near(estimated.contrast.conservative_variance, truth.variance, 1e-9, label);
                // Three clusters per arm are below the eight the pointwise interval needs.
                let inference = result.interference_inference().expect("cluster inference");
                assert!(inference.interval.is_none(), "{label}");
                assert!(
                    inference.interval_unavailable_reason.unwrap().contains("eight independent"),
                    "{label}"
                );
                let found = ids(result);
                match suite {
                    RefuteSuite::None => assert!(found.is_empty(), "{label}: {found:?}"),
                    RefuteSuite::Cheap => assert_eq!(
                        found,
                        ["interference.assignment_calibration", "interference.exposure_overlap"],
                        "{label}"
                    ),
                    _ => assert_eq!(
                        found,
                        [
                            "interference.assignment_calibration",
                            "interference.exposure_overlap",
                            "interference.randomization_unbiasedness",
                        ],
                        "{label}"
                    ),
                }
                if suite == RefuteSuite::None {
                    return;
                }
                assert_cluster_calibration(result, label);
                // The smallest exposure probability is the share of treated or control clusters.
                let overlap = report(result, "interference.exposure_overlap");
                near(overlap.original_ate, 0.5, 1e-12, label);
                near(overlap.comparison, 0.5, 1e-12, label);
                assert!(overlap.passed, "{label}");
                if suite == RefuteSuite::Full {
                    // The mean of the re-drawn contrasts has the enumerated mean zero and
                    // variance, so its distance from zero is held to four standard errors.
                    let unbiased = report(result, "interference.randomization_unbiasedness");
                    let draws = f64::from(unbiased.replicates);
                    assert!(draws >= 50.0, "{label}: {draws}");
                    assert!(
                        unbiased.original_ate.abs() < 4.0 * (rerandomized_variance / draws).sqrt(),
                        "{label}: {unbiased:?}"
                    );
                    assert!(unbiased.passed, "{label}: {unbiased:?}");
                }
            },
        );
    });
}

#[test]
fn cluster_bayesian_bootstrap_matches_the_dirichlet_posterior_moments_on_every_coordinate() {
    on_large_stack(|| {
        const DRAWS: usize = 20_000;
        let truth = cluster_truth();
        let design = cluster_design();
        // Within an arm the posterior weights are Dirichlet(1, ..., 1), for which
        // Var(Sum v x) = (Sum x^2 / m - mean^2) / (m + 1) = s^2 (m - 1) / (m (m + 1)).
        let arm_variance = |totals: &[f64]| {
            sample_variance(totals) * (totals.len() - 1) as f64
                / (totals.len() * (totals.len() + 1)) as f64
        };
        let posterior_variance = arm_variance(&truth.treated) + arm_variance(&truth.control);
        assert!((posterior_variance - 2.0).abs() < 1e-12);
        let posterior_sd = posterior_variance.sqrt();
        // Six standard errors of the mean of the draws about the exact posterior mean.
        let mean_tolerance = 6.0 * posterior_sd / (DRAWS as f64).sqrt() + 1e-9;
        walk(
            &design,
            &InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)),
            EstimatorId::InterferenceClusterBayesianBootstrap,
            truth.horvitz_thompson,
            mean_tolerance,
            9_102,
            &|result: &StudyResult, suite: RefuteSuite, label: &str| {
                let posterior = result.posterior.as_ref().expect("a contrast posterior");
                assert_eq!(posterior.draws.n_draws, DRAWS, "{label}");
                let column = posterior.effect_column().unwrap();
                near(
                    posterior.summaries.mean[column],
                    truth.horvitz_thompson,
                    mean_tolerance,
                    label,
                );
                // The spread of the draws is the exact posterior sd to within 5%.
                let sd = posterior.summaries.sd[column];
                assert!(
                    (sd / posterior_sd - 1.0).abs() < 0.05,
                    "{label}: sd {sd} vs exact {posterior_sd}"
                );
                let found = ids(result);
                if suite == RefuteSuite::None {
                    assert!(found.is_empty(), "{label}: {found:?}");
                } else {
                    // The model-based route reports no design-based estimate, so only the
                    // realized cluster allocation is checked.
                    assert_eq!(found, ["interference.assignment_calibration"], "{label}");
                    assert_cluster_calibration(result, label);
                }
            },
        );
    });
}

// ---------------------------------------------------------------------------------------------
// Exact two-stage saturation.
// ---------------------------------------------------------------------------------------------

const SATURATION_CLUSTERS: [u32; 12] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
const SATURATION_ASSIGNMENT: [bool; 12] =
    [false, false, false, true, false, false, true, true, false, true, true, true];

/// The outcome as a deterministic function of the exposure `(own, f)`.
fn exposure_outcome(own: f64, fraction: f64) -> f64 {
    1.0 + 2.0 * own + 3.0 * fraction + 4.0 * own * fraction
}

/// `(own, fraction of the two cluster mates treated)` of every unit, counted by hand.
fn saturation_exposures() -> Vec<(f64, f64)> {
    (0..12)
        .map(|unit| {
            let first = unit / 3 * 3;
            let treated_mates = (first..first + 3)
                .filter(|&other| other != unit && SATURATION_ASSIGNMENT[other])
                .count();
            (f64::from(u8::from(SATURATION_ASSIGNMENT[unit])), treated_mates as f64 / 2.0)
        })
        .collect()
}

/// Probability that a unit is at exposure `(own, fraction)`: two of four clusters are high
/// (share one half), a unit is treated with the cluster saturation `q` independently, and each
/// of its two mates likewise.
fn level_probability(own: f64, fraction: f64) -> f64 {
    let treated_mates = (2.0 * fraction).round() as i32;
    let ways = [1.0, 2.0, 1.0][treated_mates as usize];
    0.5 * [0.2_f64, 0.8]
        .iter()
        .map(|&q| {
            (if own > 0.5 { q } else { 1.0 - q })
                * ways
                * q.powi(treated_mates)
                * (1.0 - q).powi(2 - treated_mates)
        })
        .sum::<f64>()
}

fn saturation_design(from: (f64, f64), to: (f64, f64)) -> Design {
    let outcomes: Vec<f64> = saturation_exposures()
        .iter()
        .map(|&(own, fraction)| exposure_outcome(own, fraction))
        .collect();
    Design::new(
        outcomes,
        &within_cluster_edges(&SATURATION_CLUSTERS),
        SATURATION_ASSIGNMENT.to_vec(),
        contrast_query(
            AssignmentDesign::TwoStageSaturation {
                clusters: Arc::from(SATURATION_CLUSTERS.to_vec()),
                low_probability: 0.2,
                high_probability: 0.8,
                high_clusters: 2,
                realized_saturation: Arc::from(
                    [0.2; 6].into_iter().chain([0.8; 6]).collect::<Vec<_>>(),
                ),
            },
            ExposureMapping::NeighborFraction,
            from,
            to,
        ),
    )
}

#[test]
fn saturation_exact_matches_the_exposure_function_and_level_probabilities_on_every_coordinate() {
    on_large_stack(|| {
        // direct (0, .5) -> (1, .5), spillover (0, 0) -> (0, 1), total (0, 0) -> (1, 1).
        for (from, to, hajek_truth) in [
            ((0.0, 0.5), (1.0, 0.5), 4.0),
            ((0.0, 0.0), (0.0, 1.0), 3.0),
            ((0.0, 0.0), (1.0, 1.0), 9.0),
        ] {
            let design = saturation_design(from, to);
            let exposures = saturation_exposures();
            let (p_from, p_to) = (level_probability(from.0, from.1), level_probability(to.0, to.1));
            let at = |level: (f64, f64), unit: usize| {
                (exposures[unit].0 - level.0).abs() < 1e-12
                    && (exposures[unit].1 - level.1).abs() < 1e-12
            };
            // Horvitz-Thompson: (1/n) (Sum_{at to} y / p_to - Sum_{at from} y / p_from).
            let ht_sum = |level: (f64, f64), p: f64| -> f64 {
                (0..12).filter(|&u| at(level, u)).map(|u| design.outcomes[u] / p).sum()
            };
            let horvitz_thompson = (ht_sum(to, p_to) - ht_sum(from, p_from)) / 12.0;
            // Hajek: every unit at a level has the same outcome, so the ratio is the function.
            let hajek = exposure_outcome(to.0, to.1) - exposure_outcome(from.0, from.1);
            assert!((hajek - hajek_truth).abs() < 1e-12);
            let label_contrast = format!("{from:?}->{to:?}");
            walk(
                &design,
                &InferenceMode::Frequentist,
                EstimatorId::InterferenceSaturationExact,
                horvitz_thompson,
                1e-9,
                9_103,
                &|result: &StudyResult, suite: RefuteSuite, label: &str| {
                    let label = format!("{label} {label_contrast}");
                    let estimated = result.interference().expect("saturation contrast");
                    near(estimated.contrast.hajek, hajek, 1e-9, &label);
                    near(estimated.contrast.horvitz_thompson, horvitz_thompson, 1e-9, &label);
                    assert!(estimated.contrast.conservative_variance > 0.0, "{label}");
                    near(estimated.minimum_exposure_probability, p_from.min(p_to), 1e-9, &label);
                    assert!(
                        result.estimate().as_effect().unwrap().se_analytic().is_nan(),
                        "{label}"
                    );
                    // Four clusters are far below the 24 the pointwise interval needs.
                    let inference = result.interference_inference().expect("saturation inference");
                    assert!(inference.interval.is_none(), "{label}");
                    assert!(
                        inference
                            .interval_unavailable_reason
                            .unwrap()
                            .contains("24 independent clusters"),
                        "{label}"
                    );
                    let found = ids(result);
                    if suite == RefuteSuite::None {
                        assert!(found.is_empty(), "{label}: {found:?}");
                        return;
                    }
                    assert_eq!(
                        found,
                        [
                            "interference.assignment_calibration",
                            "interference.exposure_overlap",
                            "interference.saturation_allocation",
                        ],
                        "{label}"
                    );
                    // Six of twelve units treated where the realized saturations imply
                    // 6 * 0.2 + 6 * 0.8 = 6.
                    let calibration = report(result, "interference.assignment_calibration");
                    near(calibration.original_ate, 6.0, 1e-12, &label);
                    near(calibration.refuted_ate, 6.0, 1e-9, &label);
                    near(calibration.comparison, 1.0, 1e-9, &label);
                    assert!(calibration.passed, "{label}");
                    // Clusters 2 and 3 are high where two are declared.
                    let allocation = report(result, "interference.saturation_allocation");
                    near(allocation.original_ate, 2.0, 1e-12, &label);
                    near(allocation.refuted_ate, 2.0, 1e-12, &label);
                    assert!(allocation.passed, "{label}");
                    let overlap = report(result, "interference.exposure_overlap");
                    near(overlap.original_ate, p_from.min(p_to), 1e-9, &label);
                    assert!(overlap.passed, "{label}");
                },
            );
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Supplied-propensity observational exposure.
// ---------------------------------------------------------------------------------------------

const OBSERVED_CLUSTERS: [u32; 10] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
const OBSERVED_Y: [f64; 10] = [1., 3., 5., 9., 7., 8., 2., 2., 4., 6.];
/// Clusters 0 and 3 untreated, 1 and 2 wholly treated, 4 mixed (unit 8 treated, unit 9 not).
const OBSERVED_ASSIGNMENT: [bool; 10] =
    [false, false, true, true, true, true, false, false, true, false];
const P_FROM: [f64; 10] = [0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5];
const P_TO: [f64; 10] = [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25];

fn observed_design() -> Design {
    let edges: Vec<(u64, u64)> =
        (0..5_u64).flat_map(|c| [(2 * c, 2 * c + 1), (2 * c + 1, 2 * c)]).collect();
    Design::new(
        OBSERVED_Y.to_vec(),
        &edges,
        OBSERVED_ASSIGNMENT.to_vec(),
        contrast_query(
            AssignmentDesign::ObservedExposure {
                clusters: Arc::from(OBSERVED_CLUSTERS.to_vec()),
                propensity_from: Arc::from(P_FROM.to_vec()),
                propensity_to: Arc::from(P_TO.to_vec()),
                provenance: ExposurePropensityProvenance::Known,
                assume_network_exchangeability: true,
            },
            ExposureMapping::NeighborCount,
            (0.0, 0.0),
            (1.0, 1.0),
        ),
    )
}

#[test]
fn observational_ipw_matches_the_hand_counted_inverse_probability_sums_on_every_coordinate() {
    on_large_stack(|| {
        // Exposure of unit u is (own, treated mate count); each cluster is a mutual pair.
        let exposure =
            |u: usize| -> (bool, bool) { (OBSERVED_ASSIGNMENT[u], OBSERVED_ASSIGNMENT[u ^ 1]) };
        let at_from: Vec<usize> = (0..10).filter(|&u| exposure(u) == (false, false)).collect();
        let at_to: Vec<usize> = (0..10).filter(|&u| exposure(u) == (true, true)).collect();
        assert_eq!(at_from, [0, 1, 6, 7]);
        assert_eq!(at_to, [2, 3, 4, 5]);
        let from_sum: f64 = at_from.iter().map(|&u| OBSERVED_Y[u] / P_FROM[u]).sum();
        let to_sum: f64 = at_to.iter().map(|&u| OBSERVED_Y[u] / P_TO[u]).sum();
        let from_weight: f64 = at_from.iter().map(|&u| 1.0 / P_FROM[u]).sum();
        let to_weight: f64 = at_to.iter().map(|&u| 1.0 / P_TO[u]).sum();
        assert_eq!((from_sum, to_sum, from_weight, to_weight), (26.0, 90.0, 12.0, 12.0));
        let horvitz_thompson = (to_sum - from_sum) / 10.0;
        let hajek = to_sum / to_weight - from_sum / from_weight;
        // The cluster scores (to - from) / n are -1.4, 4.6, 4.4, -1.2 and 0, with mean 1.28.
        let mut scores = [0.0_f64; 5];
        for u in 0..10 {
            let cluster = OBSERVED_CLUSTERS[u] as usize;
            if at_to.contains(&u) {
                scores[cluster] += OBSERVED_Y[u] / P_TO[u] / 10.0;
            }
            if at_from.contains(&u) {
                scores[cluster] -= OBSERVED_Y[u] / P_FROM[u] / 10.0;
            }
        }
        let mean = scores.iter().sum::<f64>() / 5.0;
        let variance = 5.0 / 4.0 * scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>();
        assert!((horvitz_thompson - 6.4).abs() < 1e-12);
        assert!((hajek - 16.0 / 3.0).abs() < 1e-12);
        assert!((variance - 44.66).abs() < 1e-9);
        // Realized counts against the supplied probabilities, and the clustered variances of
        // the realized-minus-expected totals.
        let expected_from: f64 = P_FROM.iter().sum();
        let expected_to: f64 = P_TO.iter().sum();
        assert_eq!((expected_from, expected_to), (4.5, 3.0));
        let design = observed_design();
        walk(
            &design,
            &InferenceMode::Frequentist,
            EstimatorId::InterferenceObservationalIpw,
            horvitz_thompson,
            1e-9,
            9_104,
            &|result: &StudyResult, suite: RefuteSuite, label: &str| {
                let estimated = result.interference().expect("observational contrast");
                near(estimated.contrast.horvitz_thompson, horvitz_thompson, 1e-9, label);
                near(estimated.contrast.hajek, hajek, 1e-9, label);
                near(estimated.contrast.conservative_variance, variance, 1e-9, label);
                near(estimated.minimum_exposure_probability, 0.25, 1e-12, label);
                // Five clusters are below the 30 the pointwise interval needs.
                assert!(result.estimate().as_effect().unwrap().se_analytic().is_nan(), "{label}");
                let inference = result.interference_inference().expect("observational inference");
                assert!(inference.interval.is_none(), "{label}");
                assert!(
                    inference.interval_unavailable_reason.unwrap().contains("30 independent"),
                    "{label}"
                );
                assert!(
                    result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.interference.observational_ipw"
                            && d.fields.iter().any(|(key, value)| {
                                key.as_ref() == "from_exposed_units" && value.as_ref() == "4"
                            })
                            && d.fields.iter().any(|(key, value)| {
                                key.as_ref() == "to_exposed_units" && value.as_ref() == "4"
                            })
                    }),
                    "{label}"
                );
                let found = ids(result);
                if suite == RefuteSuite::None {
                    assert!(found.is_empty(), "{label}: {found:?}");
                    return;
                }
                assert_eq!(
                    found,
                    [
                        "interference.exposure_calibration_active",
                        "interference.exposure_calibration_baseline",
                        "interference.exposure_overlap",
                    ],
                    "{label}"
                );
                // Four units realized each level; the supplied probabilities imply 4.5 and 3.
                // The clustered realized-minus-expected totals are (1.25, -1, -1, 1.25, -1)
                // for the baseline (z = -0.5 / sqrt(6.125)) and (-.5, 1.25, 1.25, -.5, -.5)
                // for the active level (z = 1 / sqrt(3.875)), both inside the 5% band.
                let baseline = report(result, "interference.exposure_calibration_baseline");
                near(baseline.original_ate, 4.0, 1e-12, label);
                near(baseline.refuted_ate, expected_from, 1e-12, label);
                assert!(baseline.passed, "{label}");
                let active = report(result, "interference.exposure_calibration_active");
                near(active.original_ate, 4.0, 1e-12, label);
                near(active.refuted_ate, expected_to, 1e-12, label);
                assert!(active.passed, "{label}");
                let overlap = report(result, "interference.exposure_overlap");
                near(overlap.original_ate, 0.25, 1e-12, label);
                assert!(overlap.passed, "{label}");
            },
        );
    });
}
