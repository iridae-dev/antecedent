//! Cheap and full validation of transport over a Cpdag or a Pag, and of the exposure designs the
//! interference suite checks against the realized assignment.
//!
//! The references are computed here without the engine: the Poisson-binomial moments of the
//! pinned two-trial-row fixture, the exposures of a four-unit network counted by hand, the
//! saturations of a twelve-unit two-stage design, and the exposure bookkeeping of sixteen
//! two-unit clusters.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    BayesianConfig, CellStatus, InferenceMode, InterferenceSpec, RefuteSuite, Study, StudyResult,
    TransportTrialSpec,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDomain, ExecutionContext, ExposureLevel,
    ExposureMapping, ExposurePropensityProvenance, GridSpec, InterferenceFunctional,
    InterferenceQuery, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg};
use antecedent_validate::RefutationReport;

/// Class estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("class-design-validation".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn ids(result: &StudyResult) -> Vec<String> {
    result.refutations.iter().map(|report| report.refuter.to_string()).collect()
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == id)
        .unwrap_or_else(|| panic!("no report {id} in {:?}", ids(result)))
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

// ---------------------------------------------------------------------------
// Transport over a Cpdag or a Pag.
// ---------------------------------------------------------------------------

const A: u32 = 0;
const Y: u32 = 1;
const TRIAL: u32 = 2;
const SEL: u32 = 3;
const PROP: u32 = 4;
const Z: u32 = 5;
const NODES: u32 = 6;

/// The pinned two-trial-row fixture with a covariate column.
fn transport_data() -> TabularData {
    TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
        ("z", &[0.0, 1.0, 0.0, 1.0][..]),
    ])
    .unwrap()
}

fn transport_query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(A),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(A)])
}

fn trial_spec() -> TransportTrialSpec {
    TransportTrialSpec {
        trial: VariableId::from_raw(TRIAL),
        selection_probability: VariableId::from_raw(SEL),
        treatment_probability: VariableId::from_raw(PROP),
    }
}

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

fn class_study(graph: ClassGraph, inference: InferenceMode, suite: RefuteSuite) -> Study {
    let builder = Study::tabular(transport_data());
    let builder = match graph {
        ClassGraph::Cpdag(graph) => builder.graph(graph),
        ClassGraph::Pag(graph) => builder.graph(graph),
    };
    builder
        .query(CausalQuery::Transport(transport_query()))
        .selection_targets(Arc::<[VariableId]>::from(vec![VariableId::from_raw(Z)]))
        .transport_trial(trial_spec())
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// `a - y - z`: three DAG completions; with the covariate `z` as the selection target two
/// certify the direct formula and one standardizes over `z`.
fn open_chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(NODES);
    graph.insert_undirected(node(A), node(Y)).unwrap();
    graph.insert_undirected(node(Z), node(Y)).unwrap();
    graph
}

fn circle_pair() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_circle_circle(node(A), node(Y)).unwrap();
    pag
}

const TRANSPORT_CHEAP: [&str; 4] = [
    "transport.selection_calibration",
    "transport.arm_calibration",
    "transport.selection_overlap",
    "transport.treatment_overlap",
];

#[test]
fn a_cpdag_and_a_pag_run_the_transport_suite_on_both_inferences() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(9_101);
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(500)),
        ] {
            for name in ["cpdag", "pag"] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    let graph = if name == "cpdag" {
                        ClassGraph::Cpdag(open_chain())
                    } else {
                        ClassGraph::Pag(circle_pair())
                    };
                    let built = class_study(graph, inference.clone(), suite);
                    assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{name}");
                    let result =
                        built.run(&ctx).unwrap_or_else(|e| panic!("{name} {suite:?}: {e}"));
                    let found = ids(&result);
                    match suite {
                        RefuteSuite::None => assert!(found.is_empty(), "{name}: {found:?}"),
                        _ => {
                            for id in TRANSPORT_CHEAP {
                                assert!(found.iter().any(|f| f == id), "{name}: {id} in {found:?}");
                            }
                            assert_eq!(
                                found.iter().any(|f| f.starts_with("transport.bootstrap_range")),
                                suite == RefuteSuite::Full,
                                "{name}: {found:?}"
                            );
                        }
                    }
                    if suite != RefuteSuite::None {
                        // The pinned fixture: two trial rows where the selection
                        // probabilities (one half each, four rows) imply two, and one treated
                        // trial row where the arm probabilities (one half each) imply one.
                        let selection = report(&result, "transport.selection_calibration");
                        assert!((selection.original_ate - 2.0).abs() < 1e-12, "{name}");
                        assert!((selection.refuted_ate - 2.0).abs() < 1e-12, "{name}");
                        assert!((selection.comparison - 1.0).abs() < 1e-12 && selection.passed);
                        let arm = report(&result, "transport.arm_calibration");
                        assert!((arm.original_ate - 1.0).abs() < 1e-12, "{name}");
                        assert!((arm.refuted_ate - 1.0).abs() < 1e-12, "{name}");
                        assert!(arm.passed);
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Exposure designs.
// ---------------------------------------------------------------------------

fn contrast(from: (f64, f64), to: (f64, f64)) -> InterferenceFunctional {
    InterferenceFunctional::ExposureContrast {
        outcome: VariableId::from_raw(0),
        from: ExposureLevel { own: from.0, neighbors: from.1 },
        to: ExposureLevel { own: to.0, neighbors: to.1 },
    }
}

/// Four units in two clusters: the first untreated, the second treated, each pair adjacent.
fn observed_fixture() -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let data = TabularData::from_f64_columns([("y", &[0.0, 0.0, 6.0, 6.0][..])]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ObservedExposure {
            clusters: Arc::from([0, 0, 1, 1]),
            propensity_from: Arc::from([0.5; 4]),
            propensity_to: Arc::from([0.5; 4]),
            provenance: ExposurePropensityProvenance::ExternallyEstimated,
            assume_network_exchangeability: true,
        },
        ExposureMapping::NeighborCount,
        contrast((0.0, 0.0), (1.0, 1.0)),
    );
    let edges = vec![
        NetworkEdge::new(0, 1, 1.0),
        NetworkEdge::new(1, 0, 1.0),
        NetworkEdge::new(2, 3, 1.0),
        NetworkEdge::new(3, 2, 1.0),
    ];
    (data, vec![false, false, true, true], query, edges)
}

/// Twelve units in four clusters of three; two clusters at saturation 0.2 and two at 0.8.
fn saturation_fixture() -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let clusters = [0_u32, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let assignment =
        vec![false, false, false, true, false, false, true, true, false, true, true, true];
    let edges = (0..4)
        .flat_map(|cluster| {
            let first = cluster * 3;
            (first..first + 3).flat_map(move |from| {
                (first..first + 3).filter(move |&to| to != from).map(move |to| {
                    NetworkEdge::new(
                        u64::try_from(from).expect("row id"),
                        u64::try_from(to).expect("row id"),
                        1.0,
                    )
                })
            })
        })
        .collect::<Vec<_>>();
    let outcomes = (0..12)
        .map(|i| {
            let first = i / 3 * 3;
            let neighbors =
                (first..first + 3).filter(|&j| j != i && assignment[j]).count() as f64 / 2.0;
            let own = if assignment[i] { 1.0 } else { 0.0 };
            1.0 + 2.0 * own + 3.0 * neighbors + 4.0 * own * neighbors
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(clusters),
            low_probability: 0.2,
            high_probability: 0.8,
            high_clusters: 2,
            realized_saturation: Arc::from(
                [0.2; 6].into_iter().chain([0.8; 6]).collect::<Vec<_>>(),
            ),
        },
        ExposureMapping::NeighborFraction,
        contrast((0.0, 0.0), (1.0, 1.0)),
    );
    (data, assignment, query, edges)
}

/// Sixteen two-unit clusters, eight treated.
fn cluster_fixture() -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    const K: usize = 16;
    const TREATED: usize = 8;
    let clusters = (0..K).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
    let assignment = (0..K).flat_map(|cluster| [cluster < TREATED; 2]).collect::<Vec<_>>();
    let edges = (0..K)
        .flat_map(|cluster| {
            let first = (cluster * 2) as u32;
            [
                NetworkEdge::new(u64::from(first), u64::from(first + 1), 1.0),
                NetworkEdge::new(u64::from(first + 1), u64::from(first), 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let outcomes = (0..K * 2)
        .map(|unit| {
            let cluster = unit / 2;
            let base = 0.5 * ((cluster * 7) % K) as f64 + 0.1 * (unit % 2) as f64;
            base + if cluster < TREATED { 3.0 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters),
            treated_clusters: TREATED,
        },
        ExposureMapping::NeighborFraction,
        contrast((0.0, 0.0), (1.0, 1.0)),
    );
    (data, assignment, query, edges)
}

fn run_on_class(
    class: &str,
    (data, assignment, query, edges): (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>),
    suite: RefuteSuite,
    ctx: &ExecutionContext,
) -> StudyResult {
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let builder = Study::tabular(data);
    let builder = match class {
        "dag" => builder.graph(Dag::with_variables(1)),
        "admg" => builder.graph(Admg::with_variables(1)),
        "temporal_admg" => builder
            .graph(TemporalAdmg::from_static_contemporaneous(&Admg::with_variables(1)).unwrap()),
        "cpdag" => builder.graph(Cpdag::with_variables(1)),
        "pag" => builder.graph(Pag::with_variables(1)),
        other => panic!("unknown class {other}"),
    };
    builder
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(suite)
        .build()
        .unwrap_or_else(|e| panic!("{class} {suite:?}: {e}"))
        .run(ctx)
        .unwrap_or_else(|e| panic!("{class} {suite:?}: {e}"))
}

const CLASS_NAMES: [&str; 5] = ["dag", "admg", "temporal_admg", "cpdag", "pag"];

#[test]
fn supplied_exposure_probabilities_are_checked_against_the_realized_exposures_on_every_class() {
    let ctx = ExecutionContext::for_tests(9_201);
    for class in CLASS_NAMES {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = run_on_class(class, observed_fixture(), suite, &ctx);
            // Units 0 and 1 have no treated neighbor and are untreated: the baseline level;
            // units 2 and 3 are treated with a treated neighbor: the active level. The supplied
            // probabilities (one half, four units) imply two of each.
            for id in [
                "interference.exposure_calibration_baseline",
                "interference.exposure_calibration_active",
            ] {
                let calibration = report(&result, id);
                assert!((calibration.original_ate - 2.0).abs() < 1e-12, "{class} {id}");
                assert!((calibration.refuted_ate - 2.0).abs() < 1e-12, "{class} {id}");
                // The two clusters' residual totals cancel exactly: z = 0.
                assert!((calibration.comparison - 1.0).abs() < 1e-12, "{class} {id}");
                assert!(calibration.passed);
            }
            // The smallest supplied probability is the overlap statistic.
            let overlap = report(&result, "interference.exposure_overlap");
            assert!((overlap.comparison - 0.5).abs() < 1e-12, "{class}");
            // The supplied-propensity estimator is off the matrix's estimator axis: the design
            // checks run, and the result carries no matrix status of its own.
            assert_eq!(result.support_status(), None, "{class}");
        }
    }
}

#[test]
fn a_two_stage_saturation_is_checked_against_its_realized_saturations() {
    let ctx = ExecutionContext::for_tests(9_202);
    // Six treated of twelve units where six clusters-worth of saturation (0.2 * 6 + 0.8 * 6)
    // imply six: the second stage follows its declaration.
    let treated = saturation_fixture().1.iter().filter(|&&z| z).count();
    assert_eq!(treated, 6);
    for class in CLASS_NAMES {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = run_on_class(class, saturation_fixture(), suite, &ctx);
            let calibration = report(&result, "interference.assignment_calibration");
            assert!((calibration.original_ate - 6.0).abs() < 1e-12, "{class}");
            assert!((calibration.refuted_ate - 6.0).abs() < 1e-12, "{class}");
            assert!((calibration.comparison - 1.0).abs() < 1e-12 && calibration.passed);
            // Two clusters at the high value, as declared, constant within every cluster.
            let allocation = report(&result, "interference.saturation_allocation");
            assert!((allocation.original_ate - 2.0).abs() < 1e-12, "{class}");
            assert!((allocation.refuted_ate - 2.0).abs() < 1e-12, "{class}");
            assert!(allocation.passed, "{class}: {allocation:?}");
        }
    }
}

#[test]
fn a_cluster_randomized_design_is_re_randomized_under_full_validation_on_every_class() {
    let ctx = ExecutionContext::for_tests(9_203);
    for class in CLASS_NAMES {
        let cheap = run_on_class(class, cluster_fixture(), RefuteSuite::Cheap, &ctx);
        assert!(
            cheap
                .refutations
                .iter()
                .all(|r| r.refuter.as_ref() != "interference.randomization_unbiasedness"),
            "{class}"
        );
        let full = run_on_class(class, cluster_fixture(), RefuteSuite::Full, &ctx);
        // Eight whole clusters treated as declared.
        let calibration = report(&full, "interference.assignment_calibration");
        assert!((calibration.original_ate - 8.0).abs() < 1e-12, "{class}");
        assert!(calibration.passed, "{class}");
        // The sharp-null re-randomization ran: it reports the mean of the re-drawn contrasts,
        // and the mean of draws from an unbiased estimator on fixed outcomes is near zero
        // relative to the spread of the outcomes (which is of the order of 4).
        let unbiased = report(&full, "interference.randomization_unbiasedness");
        assert!(unbiased.replicates >= 100, "{class}: {unbiased:?}");
        assert!(unbiased.original_ate.abs() < 1.0, "{class}: {unbiased:?}");
    }
}

/// A transport query over a CPDAG or a PAG publishes a contracted result that an independent
/// consumer reads back, for both inferences.
#[test]
fn a_cpdag_and_a_pag_transport_publish_an_artifact_a_consumer_reads() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(9_102);
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(50)),
        ] {
            for name in ["cpdag", "pag"] {
                let graph = if name == "cpdag" {
                    ClassGraph::Cpdag(open_chain())
                } else {
                    ClassGraph::Pag(circle_pair())
                };
                let built = class_study(graph, inference.clone(), RefuteSuite::None);
                let prepared = built.prepare(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
                let result =
                    prepared.estimate_retained(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
                let artifact = prepared
                    .encode_contracted_result(&result, &format!("class-transport-{name}"), &ctx)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                antecedent_io::consume_analysis_result(&artifact)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    });
}

/// An interference query published under every supplied graph class is a sealed artifact that
/// an independent consumer reads back, for each assignment design.
#[test]
fn every_class_interference_route_publishes_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(9_204);
    for class in CLASS_NAMES {
        for (design, fixture) in [
            ("observed", observed_fixture()),
            ("saturation", saturation_fixture()),
            ("cluster", cluster_fixture()),
        ] {
            let (data, assignment, query, edges) = fixture;
            let network = NetworkData::try_new(data.clone(), edges).unwrap();
            let builder = Study::tabular(data);
            let builder = match class {
                "dag" => builder.graph(Dag::with_variables(1)),
                "admg" => builder.graph(Admg::with_variables(1)),
                "temporal_admg" => builder.graph(
                    TemporalAdmg::from_static_contemporaneous(&Admg::with_variables(1)).unwrap(),
                ),
                "cpdag" => builder.graph(Cpdag::with_variables(1)),
                _ => builder.graph(Pag::with_variables(1)),
            };
            let built = builder
                .query(CausalQuery::Interference(query))
                .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
                .refute(RefuteSuite::None)
                .build()
                .unwrap_or_else(|e| panic!("{class} {design}: {e}"));
            let prepared = built.prepare(&ctx).unwrap_or_else(|e| panic!("{class} {design}: {e}"));
            let result = prepared
                .estimate_retained(&ctx)
                .unwrap_or_else(|e| panic!("{class} {design}: {e}"));
            let artifact = prepared
                .encode_contracted_result(&result, &format!("interference-{class}-{design}"), &ctx)
                .unwrap_or_else(|e| panic!("{class} {design}: {e}"));
            let consumed = antecedent_io::consume_analysis_result(&artifact)
                .unwrap_or_else(|e| panic!("{class} {design}: {e}"));
            assert_eq!(consumed.body.estimate, Some(result.effect()), "{class} {design}");
        }
    }
}
