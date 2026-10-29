//! Known-truth evidence for the transport and interference cells of the support matrix that
//! read a Dag, an accepted structure, a Cpdag, a Pag or a graph posterior, and for their cheap
//! and full validation states.
//!
//! Every expected value is an independent reference computed without the engine:
//!
//! * the closed-form two-row inverse-odds sum pinned in `conformance/estimate/staged_transport`
//!   (`(+2 * 3 - 2 * 1) / 2 = 2`) and its delta-method standard error, `sqrt(36 + 4 + 4 + 4) / 2`
//!   from the per-row influences `+6`, `-2` (trial rows) and `-2`, `-2` (target rows);
//! * the Dirichlet(1, 1) row-weight moments pinned in
//!   `conformance/estimate/bayesian_trial_transport` (the draw is `8 U - 2`);
//! * the Poisson-binomial moments of the pinned fixture, counted by hand: two trial rows where the
//!   supplied selection probabilities (one half, four rows) imply two, one treated trial row where
//!   the arm probabilities (one half, two trial rows) imply one;
//! * the enumerated two-unit Bernoulli design of `conformance/response/randomized_interference`
//!   (Horvitz-Thompson -6, Hajek -3, Young bound 204);
//! * the stated Gaussian exposure model of `conformance/estimate/bayesian_interference`;
//! * the exposures of small networks counted by hand.
//!
//! The query is the same on every structure the cells name, so one value is checked on all of
//! them; what differs is how sID reads the supplied structure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, InterferenceSpec, RefuteSuite, Study,
    StudyBuilder, StudyResult, TransportTrialSpec,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDomain, ExecutionContext, ExposureLevel,
    ExposureMapping, GridSpec, IdentificationStatus, InterferenceFunctional, InterferenceQuery,
    ResponseFunctional, ResponseQuery, ResponseValue, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, adjacency_masks_from_pag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg};
use antecedent_prob::InferenceDiagnostics;
use antecedent_validate::RefutationReport;

/// Class and posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("transport-interference-cells".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn sampling(label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
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

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

// ---------------------------------------------------------------------------
// Transport.
// ---------------------------------------------------------------------------

const A: u32 = 0;
const Y: u32 = 1;
const TRIAL: u32 = 2;
const SEL: u32 = 3;
const PROP: u32 = 4;
const Z: u32 = 5;
const NODES: u32 = 6;

/// The pinned two-trial-row fixture with a covariate column. `selection` is the selection
/// probability the caller supplies on every row.
fn transport_data(selection: f64) -> TabularData {
    TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[selection; 4][..]),
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

fn targets(raw: &[u32]) -> Arc<[VariableId]> {
    Arc::from(raw.iter().map(|id| VariableId::from_raw(*id)).collect::<Vec<_>>())
}

fn transport_inference(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000))
    } else {
        InferenceMode::Frequentist
    }
}

/// The closed-form values the transport cells are checked against.
struct TransportPins {
    ipw: f64,
    ipw_tolerance: f64,
    mean: f64,
    mean_tolerance: f64,
    variance: f64,
    variance_tolerance: f64,
}

fn transport_pins() -> TransportPins {
    let ipw: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_transport/expected.json"
    ))
    .unwrap();
    let posterior: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_trial_transport/expected.json"
    ))
    .unwrap();
    TransportPins {
        ipw: ipw["ipw"].as_f64().unwrap(),
        ipw_tolerance: ipw["tolerance"].as_f64().unwrap(),
        mean: posterior["expected_mean"].as_f64().unwrap(),
        mean_tolerance: posterior["mean_tolerance"].as_f64().unwrap(),
        variance: posterior["expected_variance"].as_f64().unwrap(),
        variance_tolerance: posterior["variance_tolerance"].as_f64().unwrap(),
    }
}

/// `sqrt(sum of squared per-row influences) / target rows` for the pinned fixture: the trial
/// rows contribute `+6` and `-2`, the two target rows `-ipw = -2` each.
fn delta_method_se() -> f64 {
    (36.0_f64 + 4.0 + 4.0 + 4.0).sqrt() / 2.0
}

/// A TemporalAdmg reads the rows as one series: the Bartlett long-run variance of the same
/// influences `(+6, -2, -2, -2)` at the default lag of four rows, two. The autocovariances are
/// `48` at lag zero, `-4` at lag one and `-8` at lag two, so the long-run variance is
/// `48 + 2 * (2/3 * -4 + 1/3 * -8) = 112 / 3`.
fn newey_west_se() -> f64 {
    (112.0_f64 / 3.0).sqrt() / 2.0
}

/// A selection diagram, or a class of them, as the caller supplies it.
enum Graph {
    Dag(Dag),
    Admg(Admg),
    TemporalAdmg(TemporalAdmg),
    Cpdag(Cpdag),
    Pag(Pag),
}

/// How the structure reaches the study: declared outright or accepted from discovery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Supplied {
    Explicit,
    Accepted,
}

fn bind(builder: StudyBuilder, graph: Graph, supplied: Supplied) -> StudyBuilder {
    match (graph, supplied) {
        (Graph::Dag(g), Supplied::Explicit) => builder.graph(g),
        (Graph::Dag(g), Supplied::Accepted) => builder.graph(AcceptedGraph::from(g)),
        (Graph::Admg(g), Supplied::Explicit) => builder.graph(g),
        (Graph::Admg(g), Supplied::Accepted) => builder.graph(AcceptedGraph::from(g)),
        (Graph::TemporalAdmg(g), Supplied::Explicit) => builder.graph(g),
        (Graph::TemporalAdmg(g), Supplied::Accepted) => builder.graph(AcceptedGraph::from(g)),
        (Graph::Cpdag(g), Supplied::Explicit) => builder.graph(g),
        (Graph::Cpdag(g), Supplied::Accepted) => builder.graph(AcceptedGraph::from(g)),
        (Graph::Pag(g), Supplied::Explicit) => builder.graph(g),
        (Graph::Pag(g), Supplied::Accepted) => builder.graph(AcceptedGraph::from(g)),
    }
}

fn finish(
    builder: StudyBuilder,
    selection_targets: &[u32],
    bayesian: bool,
    suite: RefuteSuite,
) -> Study {
    builder
        .query(CausalQuery::Transport(transport_query()))
        .selection_targets(targets(selection_targets))
        .transport_trial(trial_spec())
        .inference(transport_inference(bayesian))
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn transport_study(
    graph: Graph,
    supplied: Supplied,
    selection_targets: &[u32],
    bayesian: bool,
    suite: RefuteSuite,
    selection: f64,
) -> Study {
    finish(
        bind(Study::tabular(transport_data(selection)), graph, supplied),
        selection_targets,
        bayesian,
        suite,
    )
}

fn transport_posterior_study(
    posterior: GraphPosterior,
    selection_targets: &[u32],
    bayesian: bool,
    suite: RefuteSuite,
    selection: f64,
) -> Study {
    finish(
        Study::tabular(transport_data(selection)).graph_posterior(posterior),
        selection_targets,
        bayesian,
        suite,
    )
}

/// `a -> y`: the selection diagram of a model with no latent confounding.
fn direct_dag() -> Dag {
    let mut dag = Dag::with_variables(NODES);
    dag.insert_directed(node(A), node(Y)).unwrap();
    dag
}

fn directed_admg(directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(NODES);
    for &(from, to) in directed {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    for &(left, right) in bidirected {
        graph.insert_bidirected(node(left), node(right)).unwrap();
    }
    graph
}

/// `a - y - z`: no collider, so three DAG completions (`a -> y -> z`, `a <- y -> z` and
/// `a <- y <- z`). With the covariate `z` as the selection target the first two certify the
/// direct formula (`z` is not an ancestor of `y`) and the third standardizes over `z`.
fn open_chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(NODES);
    graph.insert_undirected(node(A), node(Y)).unwrap();
    graph.insert_undirected(node(Z), node(Y)).unwrap();
    graph
}

/// `a -> y <- z` as a Cpdag: the collider is compelled, so the class is the single DAG, and with
/// the covariate `z` as the selection target sID standardizes over it.
fn collider_cpdag() -> Cpdag {
    let mut dag = Dag::with_variables(NODES);
    dag.insert_directed(node(A), node(Y)).unwrap();
    dag.insert_directed(node(Z), node(Y)).unwrap();
    Cpdag::from_dag(&dag)
}

/// `a o-o y`: every completion and every ADMG sharing its m-separation model certifies the
/// direct formula when the selection target is not an ancestor of the outcome.
fn open_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_circle_circle(node(A), node(Y)).unwrap();
    pag
}

/// `a -> y` as a Pag: its MAG completion is itself.
fn direct_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_directed(node(A), node(Y)).unwrap();
    pag
}

/// `a -> z -> y` with `trial <-> s`: the non-singleton c-component leaves sID without a
/// certificate.
fn uncertified_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_directed(node(A), node(Z)).unwrap();
    pag.insert_directed(node(Z), node(Y)).unwrap();
    pag.insert_bidirected(node(TRIAL), node(SEL)).unwrap();
    pag
}

/// The selection diagram as the cell's graph class reads it, with the empty selection target set
/// (so the formula is the direct one on every class and member).
fn point_graph(class: &str) -> Graph {
    match class {
        "dag" => Graph::Dag(direct_dag()),
        "admg" => Graph::Admg(directed_admg(&[(A, Y)], &[])),
        "temporal_admg" => Graph::TemporalAdmg(
            TemporalAdmg::from_static_contemporaneous(&directed_admg(&[(A, Y)], &[])).unwrap(),
        ),
        "cpdag" => Graph::Cpdag(open_chain()),
        "pag" => Graph::Pag(open_pag()),
        other => panic!("unknown class {other}"),
    }
}

/// The weighting values and publishes the pinned contrast on a point-identified cell.
fn assert_transport_point(result: &StudyResult, bayesian: bool, label: &str) {
    let pins = transport_pins();
    let effect = result.estimate().as_effect().unwrap();
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
    assert_eq!(result.logical_plan.identifier.as_deref(), Some("transport.sid"), "{label}");
    if bayesian {
        // Dirichlet(1, 1) over the two trial rows makes the draw 8 U - 2.
        let posterior = result.posterior.as_ref().expect("a transport posterior");
        assert_eq!(posterior.draws.n_draws, 2_000, "{label}");
        assert!((posterior.summaries.mean[0] - pins.mean).abs() < pins.mean_tolerance, "{label}");
        let variance = posterior.summaries.sd[0].powi(2);
        if label.starts_with("temporal_admg") {
            // The rows are one series: the draw weights circular blocks, so only the mean is
            // pinned by the independent row-Dirichlet moments.
            assert!(variance.is_finite() && variance > 0.0, "{label}: {variance}");
        } else {
            assert!((variance - pins.variance).abs() < pins.variance_tolerance, "{label}");
        }
        assert!((effect.ate - posterior.summaries.mean[0]).abs() < 1e-12, "{label}");
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("transport.trial_bayesian_bootstrap"),
            "{label}"
        );
    } else {
        assert!((effect.ate - pins.ipw).abs() <= pins.ipw_tolerance, "{label}: {}", effect.ate);
        assert!((result.transport().unwrap().ipw - pins.ipw).abs() <= pins.ipw_tolerance);
        let want =
            if label.starts_with("temporal_admg") { newey_west_se() } else { delta_method_se() };
        assert!((effect.se_analytic() - want).abs() < 1e-12, "{label}: {}", effect.se_analytic());
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("transport.trial_ipw"),
            "{label}"
        );
    }
}

const TRANSPORT_CHEAP: [&str; 4] = [
    "transport.selection_calibration",
    "transport.arm_calibration",
    "transport.selection_overlap",
    "transport.treatment_overlap",
];

/// The transport-native suite on the pinned fixture, each statistic counted by hand.
fn assert_transport_suite(result: &StudyResult, suite: RefuteSuite, label: &str) {
    let found = ids(result);
    if suite == RefuteSuite::None {
        assert!(found.is_empty(), "{label}: {found:?}");
        return;
    }
    for id in TRANSPORT_CHEAP {
        assert!(found.iter().any(|f| f == id), "{label}: {id} in {found:?}");
    }
    assert_eq!(
        found.iter().any(|f| f.starts_with("transport.bootstrap_range")),
        suite == RefuteSuite::Full,
        "{label}: {found:?}"
    );
    // Two trial rows where the selection probabilities (one half each, four rows) imply two: the
    // z statistic is exactly zero.
    let selection = report(result, "transport.selection_calibration");
    assert!((selection.original_ate - 2.0).abs() < 1e-12, "{label}");
    assert!((selection.refuted_ate - 2.0).abs() < 1e-12, "{label}");
    assert!((selection.comparison - 1.0).abs() < 1e-12 && selection.passed, "{label}");
    // One treated trial row where the arm probabilities (one half each, two trial rows) imply one.
    let arm = report(result, "transport.arm_calibration");
    assert!((arm.original_ate - 1.0).abs() < 1e-12, "{label}");
    assert!((arm.refuted_ate - 1.0).abs() < 1e-12, "{label}");
    assert!((arm.comparison - 1.0).abs() < 1e-12 && arm.passed, "{label}");
    // Equal probabilities give weights of equal size: the effective share is one.
    for id in ["transport.selection_overlap", "transport.treatment_overlap"] {
        let overlap = report(result, id);
        assert!((overlap.comparison - 1.0).abs() < 1e-9 && overlap.passed, "{label}: {id}");
    }
    if suite == RefuteSuite::Full {
        let range = report(result, "transport.bootstrap_range");
        // Resampled tables with no trial row, no target row or a single arm are failed replicates and
        // are not counted.
        assert!(range.replicates >= 20, "{label}: {}", range.replicates);
        assert!(range.passed, "{label}: {range:?}");
    }
}

/// A selection probability the rows contradict fails its calibration: with the supplied
/// probability 0.9 the four rows imply 3.6 trial rows against the two observed,
/// `z = (2 - 3.6) / sqrt(4 * 0.9 * 0.1) = -2.667`, outside the 5% band (two-sided normal
/// p-value 0.00766); the arm probabilities are untouched and still pass. On one dependent series
/// (a TemporalAdmg) the variance is the batch-means one: the residuals `(0.1, 0.1, -0.9, -0.9)`
/// in blocks of two sum to `(0.2, -1.8)`, so the variance is `0.04 + 3.24 = 3.28`,
/// `z = -1.6 / sqrt(3.28) = -0.883` and the p-value is 0.37699: four rows carry too little
/// information for the dependence-aware test to reject.
fn assert_planted_selection_violation(
    study: &Study,
    series: bool,
    label: &str,
    ctx: &ExecutionContext,
) {
    let planted = study.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
    let selection = report(&planted, "transport.selection_calibration");
    assert!((selection.original_ate - 2.0).abs() < 1e-12, "{label}");
    assert!((selection.refuted_ate - 3.6).abs() < 1e-9, "{label}");
    if series {
        assert!(selection.passed, "{label}: {selection:?}");
        assert!((selection.comparison - 0.376_992).abs() < 1e-5, "{label}");
    } else {
        assert!(!selection.passed, "{label}: {selection:?}");
        assert!((selection.comparison - 0.007_660_8).abs() < 1e-4, "{label}");
    }
    assert!(report(&planted, "transport.arm_calibration").passed, "{label}");
}

/// One cell of the point-identified family: the study is licensed before and after running, and
/// publishes the pinned contrast and the hand-counted suite.
fn run_transport_point_cell(
    class: &str,
    supplied: Supplied,
    bayesian: bool,
    suite: RefuteSuite,
    ctx: &ExecutionContext,
) {
    let label = format!("{class} {supplied:?} bayesian={bayesian} {suite:?}");
    let built = transport_study(point_graph(class), supplied, &[], bayesian, suite, 0.5);
    assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{label}");
    let result = built.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_transport_point(&result, bayesian, &label);
    assert_transport_suite(&result, suite, &label);
    assert_eq!(
        result.identification.status,
        IdentificationStatus::NonparametricallyIdentified,
        "{label}"
    );
    if class == "cpdag" || class == "pag" {
        assert!(result.structural_response.is_none(), "{label}: a point, not an identified set");
    }
    assert!(result.estimand.method.as_ref().starts_with("transport.sid"), "{label}");
    if suite != RefuteSuite::None && !bayesian {
        let planted = transport_study(point_graph(class), supplied, &[], false, suite, 0.9);
        assert_planted_selection_violation(&planted, class == "temporal_admg", &label, ctx);
    }
}

#[test]
fn transport_dag_cells_publish_the_pinned_contrast_and_the_hand_counted_suite() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_101);
        for supplied in [Supplied::Explicit, Supplied::Accepted] {
            for bayesian in [false, true] {
                for suite in SUITES {
                    run_transport_point_cell("dag", supplied, bayesian, suite, &ctx);
                }
            }
        }
    });
}

#[test]
fn transport_admg_and_temporal_admg_accepted_cells_publish_the_pinned_contrast() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_102);
        for class in ["admg", "temporal_admg"] {
            for bayesian in [false, true] {
                for suite in SUITES {
                    run_transport_point_cell(class, Supplied::Accepted, bayesian, suite, &ctx);
                }
            }
        }
    });
}

#[test]
fn transport_admg_and_temporal_admg_explicit_cheap_and_full_run_the_hand_counted_suite() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_103);
        for class in ["admg", "temporal_admg"] {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    run_transport_point_cell(class, Supplied::Explicit, bayesian, suite, &ctx);
                }
            }
        }
    });
}

#[test]
fn transport_cpdag_and_pag_cells_publish_the_pinned_contrast_and_the_hand_counted_suite() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_104);
        for class in ["cpdag", "pag"] {
            for supplied in [Supplied::Explicit, Supplied::Accepted] {
                for bayesian in [false, true] {
                    for suite in SUITES {
                        run_transport_point_cell(class, supplied, bayesian, suite, &ctx);
                    }
                }
            }
        }
        assert_cpdag_partial_identification(&ctx);
    });
}

/// A Cpdag whose members certify different formulas leaves the transported quantity partially
/// identified, one case per distinct formula, and the weighting evaluates every case through its
/// certificate: the identified set over the evaluated cases is the one pinned contrast.
fn assert_cpdag_partial_identification(ctx: &ExecutionContext) {
    {
        for supplied in [Supplied::Explicit, Supplied::Accepted] {
            for bayesian in [false, true] {
                for suite in SUITES {
                    let label = format!("cpdag {supplied:?} bayesian={bayesian} {suite:?}");
                    let built = transport_study(
                        Graph::Cpdag(open_chain()),
                        supplied,
                        &[Z],
                        bayesian,
                        suite,
                        0.5,
                    );
                    assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{label}");
                    let result = built.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert_eq!(
                        result.identification.status,
                        IdentificationStatus::PartiallyIdentified,
                        "{label}"
                    );
                    let pins = transport_pins();
                    let effect = result.estimate().as_effect().unwrap();
                    if bayesian {
                        let posterior = result.posterior.as_ref().expect("a transport posterior");
                        assert!(
                            (posterior.summaries.mean[0] - pins.mean).abs() < pins.mean_tolerance,
                            "{label}"
                        );
                    } else {
                        assert!((effect.ate - pins.ipw).abs() <= pins.ipw_tolerance, "{label}");
                        assert!((effect.se_analytic() - delta_method_se()).abs() < 1e-12);
                    }
                    // Three completions, enumerated by hand: a->y->z and a<-y->z keep z out of
                    // the outcome's ancestry (the direct formula), a<-y<-z does not (it
                    // standardizes over z).
                    if !bayesian {
                        let mixture =
                            result.structural_response.as_ref().expect("an identified set");
                        assert_eq!(mixture.atoms.len(), 3, "{label}");
                        assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{label}");
                        assert!(mixture.unidentified_mass.abs() < 1e-12, "{label}");
                        let set = mixture.identified_set.as_ref().expect("an identified set");
                        assert!((set.lower[0] - 2.0).abs() < 1e-12, "{label}");
                        assert!((set.upper[0] - 2.0).abs() < 1e-12, "{label}");
                        let rules = result
                            .diagnostics
                            .iter()
                            .find(|item| item.code.as_ref() == "estimate.cpdag.transport_sid")
                            .expect("the case rules are disclosed");
                        assert!(
                            rules.message.contains("transport.sid.direct=2"),
                            "{}",
                            rules.message
                        );
                        assert!(
                            rules.message.contains("transport.sid.standardize=1"),
                            "{}",
                            rules.message
                        );
                    }
                    assert_transport_suite(&result, suite, &label);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transport over a graph posterior.
// ---------------------------------------------------------------------------

/// The atoms of the Dag, Admg and TemporalAdmg posteriors, with the covariate `z` as the
/// selection target:
///
/// * `direct`: `a -> y`; no population-varying mechanism is an ancestor of `y`.
/// * `standardized`: `a -> y <- z`; `z` is a parentless pre-treatment covariate.
/// * `recursive`: `a -> z -> y`; sID certifies a recursive factorization the weighting cannot
///   evaluate.
/// * `uncertified`: `a -> z -> y` with `trial <-> s`: the non-singleton c-component leaves sID
///   without a certificate (an ADMG only).
fn posterior_atoms() -> [Admg; 4] {
    [
        directed_admg(&[(A, Y)], &[]),
        directed_admg(&[(A, Y), (Z, Y)], &[]),
        directed_admg(&[(A, Z), (Z, Y)], &[]),
        directed_admg(&[(A, Z), (Z, Y)], &[(TRIAL, SEL)]),
    ]
}

fn mask_posterior(
    kind: GraphPosteriorAtomKind,
    weights: &[f64],
    graphs: &[Admg],
    label: &str,
) -> GraphPosterior {
    let n = NODES as usize;
    let masks: Vec<_> = graphs.iter().map(|g| adjacency_mask_from_admg(g).unwrap()).collect();
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling(label),
    )
    .unwrap()
    .with_atom_kind(kind)
}

fn cpdag_posterior(weights: &[f64], graphs: &[Cpdag]) -> GraphPosterior {
    let n = NODES as usize;
    let masks: Vec<_> =
        graphs.iter().map(|graph| adjacency_mask_from_cpdag(graph).unwrap()).collect();
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_transport_cpdag"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn pag_posterior(weights: &[f64], graphs: &[Pag]) -> GraphPosterior {
    let n = NODES as usize;
    let mut masks = Vec::new();
    let mut marks = Vec::new();
    for graph in graphs {
        let (mask, mark) = adjacency_masks_from_pag(graph).unwrap();
        masks.push(mask);
        marks.push(mark);
    }
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_transport_pag"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms(marks)
    .unwrap()
}

/// The posterior of one atom kind and the mass each lost axis holds, decided atom by atom by sID
/// on its own selection diagram.
struct PosteriorCase {
    posterior: GraphPosterior,
    identified: f64,
    unidentified: f64,
    unevaluable: f64,
    status: IdentificationStatus,
}

fn posterior_case(kind: &str) -> PosteriorCase {
    let atoms = posterior_atoms();
    match kind {
        // direct, standardized and recursive.
        "dag" | "temporal_admg" => PosteriorCase {
            posterior: mask_posterior(
                if kind == "dag" {
                    GraphPosteriorAtomKind::Dag
                } else {
                    GraphPosteriorAtomKind::TemporalAdmg
                },
                &[0.5, 0.3, 0.2],
                &atoms[..3],
                "graph_posterior_transport_dag",
            ),
            identified: 0.8,
            unidentified: 0.0,
            unevaluable: 0.2,
            status: IdentificationStatus::GraphDependent,
        },
        // direct, standardized, recursive and uncertified.
        "admg" => PosteriorCase {
            posterior: mask_posterior(
                GraphPosteriorAtomKind::Admg,
                &[0.35, 0.25, 0.2, 0.2],
                &atoms,
                "graph_posterior_transport_admg",
            ),
            identified: 0.6,
            unidentified: 0.2,
            unevaluable: 0.2,
            status: IdentificationStatus::GraphDependent,
        },
        // The open chain (two direct completions, one standardizing) and the compelled collider.
        "cpdag" => PosteriorCase {
            posterior: cpdag_posterior(&[0.7, 0.3], &[open_chain(), collider_cpdag()]),
            identified: 1.0,
            unidentified: 0.0,
            unevaluable: 0.0,
            status: IdentificationStatus::PartiallyIdentified,
        },
        // `a o-o y`, `a -> z -> y` with `trial <-> s`, and `a -> y`.
        "pag" => PosteriorCase {
            posterior: pag_posterior(
                &[0.5, 0.25, 0.25],
                &[open_pag(), uncertified_pag(), direct_pag()],
            ),
            identified: 0.75,
            unidentified: 0.25,
            unevaluable: 0.0,
            status: IdentificationStatus::GraphDependent,
        },
        other => panic!("unknown atom kind {other}"),
    }
}

fn assert_posterior_masses(
    result: &StudyResult,
    case: &PosteriorCase,
    bayesian: bool,
    label: &str,
) {
    let mixture = result.structural_response.as_ref().expect("a structural mixture");
    assert!((mixture.identified_mass - case.identified).abs() < 1e-9, "{label}");
    assert!((mixture.unidentified_mass - case.unidentified).abs() < 1e-9, "{label}");
    assert!((mixture.unevaluable_mass - case.unevaluable).abs() < 1e-9, "{label}");
    for atom in mixture.atoms.iter().filter(|atom| atom.value.is_some()) {
        match atom.value {
            Some(ResponseValue::Scalar(value)) => {
                let pins = transport_pins();
                if bayesian {
                    assert!((value - pins.mean).abs() < pins.mean_tolerance, "{label}: {value}");
                } else {
                    assert!((value - pins.ipw).abs() < 1e-12, "{label}: {value}");
                }
            }
            ref other => panic!("{label}: a transport atom value is a scalar: {other:?}"),
        }
    }
    assert_eq!(result.identification.status, case.status, "{label}");
}

fn run_transport_posterior_cells(kind: &str, ctx: &ExecutionContext) {
    for bayesian in [false, true] {
        for suite in SUITES {
            let case = posterior_case(kind);
            let label = format!("{kind} posterior bayesian={bayesian} {suite:?}");
            let built =
                transport_posterior_study(case.posterior.clone(), &[Z], bayesian, suite, 0.5);
            assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{label}");
            let result = built.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
            let pins = transport_pins();
            let effect = result.estimate().as_effect().unwrap();
            if bayesian {
                let posterior = result.posterior.as_ref().expect("a transport posterior");
                assert_eq!(posterior.draws.n_draws, 2_000, "{label}");
                assert!(
                    (posterior.summaries.mean[0] - pins.mean).abs() < pins.mean_tolerance,
                    "{label}"
                );
                assert!((effect.ate - posterior.summaries.mean[0]).abs() < 1e-12, "{label}");
                assert!((posterior.unidentified_mass - case.unidentified).abs() < 1e-9, "{label}");
                assert!((posterior.unevaluable_mass - case.unevaluable).abs() < 1e-9, "{label}");
            } else {
                assert!((effect.ate - pins.ipw).abs() <= pins.ipw_tolerance, "{label}");
                let want =
                    if kind == "temporal_admg" { newey_west_se() } else { delta_method_se() };
                assert!((effect.se_analytic() - want).abs() < 1e-12, "{label}");
            }
            assert_posterior_masses(&result, &case, bayesian, &label);
            assert_transport_suite(&result, suite, &label);
            if suite != RefuteSuite::None && !bayesian {
                let planted =
                    transport_posterior_study(case.posterior.clone(), &[Z], false, suite, 0.9);
                assert_planted_selection_violation(&planted, kind == "temporal_admg", &label, ctx);
            }
        }
    }
}

#[test]
fn transport_dag_posterior_cells_mix_the_atoms_sid_certifies() {
    on_large_stack(|| run_transport_posterior_cells("dag", &ExecutionContext::for_tests(8_201)));
}

#[test]
fn transport_admg_posterior_cells_state_the_unidentified_and_unevaluable_mass() {
    on_large_stack(|| run_transport_posterior_cells("admg", &ExecutionContext::for_tests(8_202)));
}

#[test]
fn transport_temporal_admg_posterior_cells_mix_the_atoms_sid_certifies() {
    on_large_stack(|| {
        run_transport_posterior_cells("temporal_admg", &ExecutionContext::for_tests(8_203));
    });
}

#[test]
fn transport_cpdag_posterior_cells_run_the_class_envelope_of_sid() {
    on_large_stack(|| run_transport_posterior_cells("cpdag", &ExecutionContext::for_tests(8_204)));
}

#[test]
fn transport_pag_posterior_cells_state_the_mass_sid_does_not_certify() {
    on_large_stack(|| run_transport_posterior_cells("pag", &ExecutionContext::for_tests(8_205)));
}

/// Every transport structure the cells name executes from its retained operation once the
/// builder is gone, and returns the pinned contrast on the data it is handed.
#[test]
fn transport_cells_execute_from_the_retained_operation_after_builder_drop() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_301);
        let pins = transport_pins();
        let data = transport_data(0.5);
        for bayesian in [false, true] {
            let mut cases: Vec<(String, Study)> = Vec::new();
            for class in ["dag", "admg", "temporal_admg", "cpdag", "pag"] {
                for supplied in [Supplied::Explicit, Supplied::Accepted] {
                    cases.push((
                        format!("{class} {supplied:?}"),
                        transport_study(
                            point_graph(class),
                            supplied,
                            &[],
                            bayesian,
                            RefuteSuite::Cheap,
                            0.5,
                        ),
                    ));
                }
            }
            for kind in ["dag", "admg", "temporal_admg", "cpdag", "pag"] {
                cases.push((
                    format!("{kind} posterior"),
                    transport_posterior_study(
                        posterior_case(kind).posterior,
                        &[Z],
                        bayesian,
                        RefuteSuite::Cheap,
                        0.5,
                    ),
                ));
            }
            for (label, builder) in cases {
                let mut prepared =
                    builder.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: prepare {e}"));
                drop(builder);
                assert!(
                    matches!(
                        prepared.retained_operation(),
                        RetainedOperation::TransportTrial(_) | RetainedOperation::EstimatorRoute
                    ),
                    "{label}: the retained operation is a transport plan"
                );
                assert!(prepared.plan().lowered().is_some(), "{label}: a lowered program");
                let result = prepared
                    .estimate(&data, &ctx)
                    .unwrap_or_else(|e| panic!("{label}: estimate {e}"));
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|item| item.code.as_ref() == "exec.identify.cached"),
                    "{label}: the retained proof is reused"
                );
                let ate = result.estimate().as_effect().unwrap().ate;
                if bayesian {
                    assert!((ate - pins.mean).abs() < pins.mean_tolerance, "{label}: {ate}");
                } else {
                    assert!((ate - pins.ipw).abs() <= pins.ipw_tolerance, "{label}: {ate}");
                }
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                assert!(
                    ids(&result).len() == 4,
                    "{label}: the cheap suite ran on the retained plan"
                );
                // A refreshed branch on the same schema keeps the retained operation.
                let again = prepared.estimate(&data, &ctx).unwrap();
                assert!(
                    (again.estimate().as_effect().unwrap().ate - ate).abs() < 1e-12 || bayesian
                );
                let _ = &mut prepared;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Interference.
// ---------------------------------------------------------------------------

fn bernoulli_query() -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 1.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    )
}

fn unit_network(data: &TabularData, edges: &[(u64, u64)]) -> NetworkData {
    NetworkData::try_new(
        data.clone(),
        edges.iter().map(|&(from, to)| NetworkEdge::new(from, to, 1.0)).collect::<Vec<_>>(),
    )
    .unwrap()
}

/// The enumerated two-unit Bernoulli design of `conformance/response/randomized_interference`.
struct BernoulliPin {
    outcomes: Vec<f64>,
    data: TabularData,
    spec: InterferenceSpec,
    horvitz_thompson: f64,
    hajek: f64,
    variance: f64,
    minimum_probability: f64,
    atol: f64,
}

fn bernoulli_pin() -> BernoulliPin {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/randomized_interference/expected.json"
    ))
    .unwrap();
    let outcomes: Vec<f64> =
        pin["outcomes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let assignment: Vec<bool> =
        pin["assignment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let edges: Vec<(u64, u64)> = pin["directed_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| (edge[0].as_u64().unwrap(), edge[1].as_u64().unwrap()))
        .collect();
    let expected = &pin["expected"];
    BernoulliPin {
        spec: InterferenceSpec {
            network: unit_network(&data, &edges),
            assignment: Arc::from(assignment),
        },
        outcomes,
        data,
        horvitz_thompson: expected["horvitz_thompson_contrast"].as_f64().unwrap(),
        hajek: expected["hajek_contrast"].as_f64().unwrap(),
        variance: expected["conservative_variance"].as_f64().unwrap(),
        minimum_probability: expected["minimum_exposure_probability"].as_f64().unwrap(),
        atol: pin["tolerance"]["atol"].as_f64().unwrap(),
    }
}

/// The stated Gaussian exposure model of `conformance/estimate/bayesian_interference`:
/// `y = 2 + 3 own + 4 treated neighbors`, so (own 0, one treated neighbor) -> (own 1, none) is -1.
struct GaussianPin {
    outcomes: Vec<f64>,
    data: TabularData,
    spec: InterferenceSpec,
    contrast: f64,
    tolerance: f64,
}

fn gaussian_pin() -> GaussianPin {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_interference/expected.json"
    ))
    .unwrap();
    let outcomes: Vec<f64> =
        fixture["outcomes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let assignment: Vec<bool> =
        fixture["assignment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let edges: Vec<(u64, u64)> = fixture["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| (edge[0].as_u64().unwrap(), edge[1].as_u64().unwrap()))
        .collect();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    GaussianPin {
        spec: InterferenceSpec {
            network: unit_network(&data, &edges),
            assignment: Arc::from(assignment),
        },
        outcomes,
        data,
        contrast: fixture["expected_contrast"].as_f64().unwrap(),
        tolerance: fixture["tolerance"].as_f64().unwrap(),
    }
}

fn interference_graph(class: &str) -> Graph {
    match class {
        "dag" => Graph::Dag(Dag::with_variables(1)),
        "admg" => Graph::Admg(Admg::with_variables(1)),
        "temporal_admg" => Graph::TemporalAdmg(
            TemporalAdmg::from_static_contemporaneous(&Admg::with_variables(1)).unwrap(),
        ),
        "cpdag" => Graph::Cpdag(Cpdag::with_variables(1)),
        "pag" => Graph::Pag(Pag::with_variables(1)),
        other => panic!("unknown class {other}"),
    }
}

fn interference_inference(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(20_000))
    } else {
        InferenceMode::Frequentist
    }
}

fn interference_study(
    class: &str,
    supplied: Supplied,
    (data, spec): (&TabularData, &InterferenceSpec),
    bayesian: bool,
    suite: RefuteSuite,
) -> Study {
    bind(Study::tabular(data.clone()), interference_graph(class), supplied)
        .query(CausalQuery::Interference(bernoulli_query()))
        .interference(spec.clone())
        .inference(interference_inference(bayesian))
        .refute(suite)
        .build()
        .unwrap()
}

/// Atoms over `[y, z, w]`: `y -> z`, the empty graph, `z -> y`, and the directed cycle
/// `y -> z -> w -> y`, which is no DAG, ADMG, Cpdag or Pag (weights 0.4, 0.3, 0.1, 0.2).
fn exposure_posterior(kind: GraphPosteriorAtomKind, weights: [f64; 4]) -> GraphPosterior {
    let cycle = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 1, 2, true), 3, 2, 0, true);
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![set_edge(0, 3, 0, 1, true), 0, set_edge(0, 3, 1, 0, true), cycle],
        vec![0.0; 9],
        vec![0.0; 9],
        sampling("graph_posterior_interference"),
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

/// The sharp-null re-randomization distribution of the two-unit Bernoulli design, enumerated
/// without the engine: the Horvitz-Thompson contrast of the fixed outcomes under each of the four
/// equally likely assignments. Returns its mean and variance.
///
/// Each unit is in the `from` level (untreated with its neighbor treated) or the `to` level
/// (treated with its neighbor untreated) with probability 1/4 under fair coins on two adjacent
/// units, so the contrast is `(sum_to y / (1/4) - sum_from y / (1/4)) / 2`.
fn rerandomization_moments(outcomes: &[f64]) -> (f64, f64) {
    assert_eq!(outcomes.len(), 2);
    let mut values = Vec::new();
    for bits in 0_u8..4 {
        let z = [bits & 1 == 1, bits & 2 == 2];
        let mut total = 0.0;
        for unit in 0..2 {
            let (own, neighbor) = (z[unit], z[1 - unit]);
            if own && !neighbor {
                total += 4.0 * outcomes[unit];
            }
            if !own && neighbor {
                total -= 4.0 * outcomes[unit];
            }
        }
        values.push(total / 2.0);
    }
    let mean = values.iter().sum::<f64>() / 4.0;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
    (mean, variance)
}

/// The design-native suite on the pinned designs, each statistic counted by hand.
fn assert_interference_suite(
    result: &StudyResult,
    suite: RefuteSuite,
    bayesian: bool,
    label: &str,
) {
    let found = ids(result);
    if suite == RefuteSuite::None {
        assert!(found.is_empty(), "{label}: {found:?}");
        return;
    }
    let calibration = report(result, "interference.assignment_calibration");
    if bayesian {
        // [treated, control, treated, control]: two treated units where fair coins imply two.
        assert!((calibration.original_ate - 2.0).abs() < 1e-12, "{label}");
        assert!((calibration.refuted_ate - 2.0).abs() < 1e-12, "{label}");
        assert_eq!(found.len(), 1, "{label}: {found:?}");
    } else {
        // [treated, control]: one treated unit where fair coins imply one.
        assert!((calibration.original_ate - 1.0).abs() < 1e-12, "{label}");
        assert!((calibration.refuted_ate - 1.0).abs() < 1e-12, "{label}");
        // The smallest exposure probability is the overlap statistic.
        let overlap = report(result, "interference.exposure_overlap");
        assert!((overlap.original_ate - bernoulli_pin().minimum_probability).abs() < 1e-12);
        assert!((overlap.comparison - 0.25).abs() < 1e-12 && overlap.passed, "{label}");
        let randomized = found.iter().any(|id| id == "interference.randomization_unbiasedness");
        assert_eq!(randomized, suite == RefuteSuite::Full, "{label}: {found:?}");
        if suite == RefuteSuite::Full {
            // The mean of the re-drawn contrasts has the enumerated mean zero and variance
            // 18 / draws.
            let unbiased = report(result, "interference.randomization_unbiasedness");
            let (mean, variance) = rerandomization_moments(&bernoulli_pin().outcomes);
            assert!(mean.abs() < 1e-12 && (variance - 18.0).abs() < 1e-12);
            let draws = f64::from(unbiased.replicates);
            assert!(draws >= 50.0, "{label}: {draws}");
            assert!(
                unbiased.original_ate.abs() < 4.0 * (variance / draws).sqrt(),
                "{label}: {unbiased:?}"
            );
            assert!(unbiased.passed, "{label}: {unbiased:?}");
        }
    }
    assert!((calibration.comparison - 1.0).abs() < 1e-12 && calibration.passed, "{label}");
}

/// The pinned contrast on the pinned design.
fn assert_interference_point(result: &StudyResult, bayesian: bool, label: &str) {
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
    let effect = result.estimate().as_effect().unwrap();
    if bayesian {
        let pin = gaussian_pin();
        assert!((effect.ate - pin.contrast).abs() < pin.tolerance, "{label}: {}", effect.ate);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("interference.bayesian_gaussian"),
            "{label}"
        );
        assert!(result.posterior.is_some(), "{label}");
    } else {
        let pin = bernoulli_pin();
        let estimated = result.interference().expect("design-based contrast");
        assert!((estimated.contrast.horvitz_thompson - pin.horvitz_thompson).abs() < pin.atol);
        assert!((estimated.contrast.hajek - pin.hajek).abs() < pin.atol, "{label}");
        assert!((estimated.contrast.conservative_variance - pin.variance).abs() < 1e-9);
        assert!((effect.ate - pin.horvitz_thompson).abs() < pin.atol, "{label}");
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("interference.ht_hajek"),
            "{label}"
        );
    }
}

fn run_interference_cell(
    class: &str,
    supplied: Supplied,
    bayesian: bool,
    suite: RefuteSuite,
    ctx: &ExecutionContext,
) {
    let label = format!("{class} {supplied:?} bayesian={bayesian} {suite:?}");
    let (bernoulli, gaussian) = (bernoulli_pin(), gaussian_pin());
    let design = if bayesian {
        (&gaussian.data, &gaussian.spec)
    } else {
        (&bernoulli.data, &bernoulli.spec)
    };
    let built = interference_study(class, supplied, design, bayesian, suite);
    assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{label}");
    let result = built.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_interference_point(&result, bayesian, &label);
    assert_interference_suite(&result, suite, bayesian, &label);
    if suite != RefuteSuite::None && !bayesian {
        assert_planted_assignment_violation(class, supplied, suite, ctx);
    }
    // The structure binds only the unit schema: nothing in it can move the contrast.
    assert_eq!(result.logical_plan.identifier.as_deref(), Some("interference.design"), "{label}");
}

#[test]
fn interference_accepted_cells_publish_the_enumerated_contrast_and_the_design_suite() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_402);
        for class in ["dag", "admg", "temporal_admg", "cpdag", "pag"] {
            for bayesian in [false, true] {
                for suite in SUITES {
                    run_interference_cell(class, Supplied::Accepted, bayesian, suite, &ctx);
                }
            }
        }
    });
}

#[test]
fn interference_admg_temporal_admg_cpdag_and_pag_explicit_cells_publish_the_enumerated_contrast() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_403);
        for class in ["admg", "temporal_admg", "cpdag", "pag"] {
            for bayesian in [false, true] {
                for suite in SUITES {
                    run_interference_cell(class, Supplied::Explicit, bayesian, suite, &ctx);
                }
            }
        }
    });
}

#[test]
fn interference_dag_explicit_cheap_and_full_run_the_design_suite() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_404);
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                run_interference_cell("dag", Supplied::Explicit, bayesian, suite, &ctx);
            }
        }
    });
}

/// A realized assignment the declared design contradicts fails its calibration, and a design the
/// assignment follows passes: three treated of eight units where a declared probability of 0.9
/// implies 7.2 is outside the 5% band, `z = (3 - 7.2) / sqrt(8 * 0.9 * 0.1) = -4.95`, while the
/// declared 0.4 implies 3.2, `z = -0.2 / sqrt(8 * 0.4 * 0.6) = -0.144`.
fn assert_planted_assignment_violation(
    class: &str,
    supplied: Supplied,
    suite: RefuteSuite,
    ctx: &ExecutionContext,
) {
    let outcomes: Vec<f64> = (0..8).map(f64::from).collect();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let edges: Vec<(u64, u64)> = (0..8_u64).map(|unit| (unit, (unit + 1) % 8)).collect();
    let assignment: Vec<bool> = (0..8).map(|unit| unit % 3 == 0 && unit < 7).collect();
    assert_eq!(assignment.iter().filter(|&&z| z).count(), 3);
    let spec = InterferenceSpec {
        network: unit_network(&data, &edges),
        assignment: Arc::from(assignment),
    };
    for (probability, passes) in [(0.9_f64, false), (0.4_f64, true)] {
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([probability]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let result = bind(Study::tabular(data.clone()), interference_graph(class), supplied)
            .query(CausalQuery::Interference(query))
            .interference(spec.clone())
            .refute(suite)
            .build()
            .unwrap()
            .run(ctx)
            .unwrap_or_else(|e| panic!("{class} {supplied:?} {probability}: {e}"));
        let calibration = report(&result, "interference.assignment_calibration");
        assert!((calibration.original_ate - 3.0).abs() < 1e-12, "{class}");
        assert!((calibration.refuted_ate - 8.0 * probability).abs() < 1e-9, "{class}");
        assert_eq!(calibration.passed, passes, "{class} {probability}: {calibration:?}");
    }
}

fn run_interference_posterior_cells(kind: GraphPosteriorAtomKind, ctx: &ExecutionContext) {
    let (bernoulli, gaussian) = (bernoulli_pin(), gaussian_pin());
    for bayesian in [false, true] {
        for suite in SUITES {
            let label = format!("{kind:?} posterior bayesian={bayesian} {suite:?}");
            let (outcomes, spec) = if bayesian {
                (&gaussian.outcomes, &gaussian.spec)
            } else {
                (&bernoulli.outcomes, &bernoulli.spec)
            };
            let built = Study::tabular(with_covariates(outcomes))
                .graph_posterior(exposure_posterior(kind, [0.4, 0.3, 0.1, 0.2]))
                .query(CausalQuery::Interference(bernoulli_query()))
                .interference(spec.clone())
                .inference(interference_inference(bayesian))
                .refute(suite)
                .build()
                .unwrap();
            assert_eq!(built.support_status(), Some(CellStatus::Licensed), "{label}");
            let result = built.run(ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_interference_point(&result, bayesian, &label);
            assert_interference_suite(&result, suite, bayesian, &label);
            // Three atoms are members of the posterior's graph class and carry the one design
            // identification; the directed cycle is no member and is unidentified mass.
            let mixture = result.structural_response.as_ref().expect("a structural mixture");
            assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{label}");
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9, "{label}");
            assert!(mixture.unevaluable_mass.abs() < 1e-12, "{label}");
            assert_eq!(mixture.atoms.len(), 4, "{label}");
            assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
            for atom in mixture.atoms.iter().filter(|atom| atom.value.is_some()) {
                match atom.value {
                    Some(ResponseValue::Scalar(value)) => {
                        let want =
                            if bayesian { gaussian.contrast } else { bernoulli.horvitz_thompson };
                        let band = if bayesian { gaussian.tolerance } else { bernoulli.atol };
                        assert!((value - want).abs() < band, "{label}: {value}");
                    }
                    ref other => {
                        panic!("{label}: an interference atom value is a scalar: {other:?}")
                    }
                }
            }
            if let Some(posterior) = result.posterior.as_ref() {
                assert!((posterior.unidentified_mass - 0.2).abs() < 1e-9, "{label}");
            }
        }
    }
}

#[test]
fn interference_dag_posterior_cells_carry_the_design_contrast_and_the_atom_mass() {
    on_large_stack(|| {
        run_interference_posterior_cells(
            GraphPosteriorAtomKind::Dag,
            &ExecutionContext::for_tests(8_406),
        );
    });
}

#[test]
fn interference_admg_posterior_cells_carry_the_design_contrast_and_the_atom_mass() {
    on_large_stack(|| {
        run_interference_posterior_cells(
            GraphPosteriorAtomKind::Admg,
            &ExecutionContext::for_tests(8_407),
        );
    });
}

#[test]
fn interference_temporal_admg_posterior_cells_carry_the_design_contrast_and_the_atom_mass() {
    on_large_stack(|| {
        run_interference_posterior_cells(
            GraphPosteriorAtomKind::TemporalAdmg,
            &ExecutionContext::for_tests(8_408),
        );
    });
}

#[test]
fn interference_cpdag_posterior_cells_carry_the_design_contrast_and_the_atom_mass() {
    on_large_stack(|| {
        run_interference_posterior_cells(
            GraphPosteriorAtomKind::Cpdag,
            &ExecutionContext::for_tests(8_409),
        );
    });
}

#[test]
fn interference_pag_posterior_cells_carry_the_design_contrast_and_the_atom_mass() {
    on_large_stack(|| {
        run_interference_posterior_cells(
            GraphPosteriorAtomKind::Pag,
            &ExecutionContext::for_tests(8_410),
        );
    });
}

/// A posterior over other variables than the unit table's is refused, not silently read.
#[test]
fn interference_posterior_over_other_variables_than_the_unit_table_is_refused() {
    on_large_stack(|| {
        let pin = bernoulli_pin();
        let wide = GraphPosterior::new(
            3,
            vec![1.0],
            vec![0],
            vec![0.0; 9],
            vec![0.0; 9],
            sampling("graph_posterior_interference_wide"),
        )
        .unwrap();
        let text = Study::tabular(pin.data.clone())
            .graph_posterior(wide)
            .query(CausalQuery::Interference(bernoulli_query()))
            .interference(pin.spec.clone())
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(8_411)))
            .err()
            .map(|e| e.to_string())
            .expect("the run must be refused");
        assert!(
            text.contains("the posterior's atoms must cover exactly the unit table's variables"),
            "{text}"
        );
    });
}

/// Every interference structure the cells name executes from its retained operation once the
/// builder is gone, and returns the enumerated contrast on the data it is handed.
#[test]
fn interference_cells_execute_from_the_retained_operation_after_builder_drop() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_412);
        let (bernoulli, gaussian) = (bernoulli_pin(), gaussian_pin());
        for bayesian in [false, true] {
            let mut cases: Vec<(String, Study, TabularData)> = Vec::new();
            let (data, spec) = if bayesian {
                (&gaussian.data, &gaussian.spec)
            } else {
                (&bernoulli.data, &bernoulli.spec)
            };
            for class in ["dag", "admg", "temporal_admg", "cpdag", "pag"] {
                for supplied in [Supplied::Explicit, Supplied::Accepted] {
                    cases.push((
                        format!("{class} {supplied:?}"),
                        interference_study(
                            class,
                            supplied,
                            (data, spec),
                            bayesian,
                            RefuteSuite::Cheap,
                        ),
                        data.clone(),
                    ));
                }
            }
            let outcomes = if bayesian { &gaussian.outcomes } else { &bernoulli.outcomes };
            for kind in [
                GraphPosteriorAtomKind::Dag,
                GraphPosteriorAtomKind::Admg,
                GraphPosteriorAtomKind::TemporalAdmg,
                GraphPosteriorAtomKind::Cpdag,
                GraphPosteriorAtomKind::Pag,
            ] {
                let table = with_covariates(outcomes);
                cases.push((
                    format!("{kind:?} posterior"),
                    Study::tabular(table.clone())
                        .graph_posterior(exposure_posterior(kind, [0.4, 0.3, 0.1, 0.2]))
                        .query(CausalQuery::Interference(bernoulli_query()))
                        .interference(spec.clone())
                        .inference(interference_inference(bayesian))
                        .refute(RefuteSuite::Cheap)
                        .build()
                        .unwrap(),
                    table,
                ));
            }
            for (label, builder, table) in cases {
                let prepared =
                    builder.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: prepare {e}"));
                drop(builder);
                assert!(
                    matches!(
                        prepared.retained_operation(),
                        RetainedOperation::Interference(_) | RetainedOperation::EstimatorRoute
                    ),
                    "{label}: the retained operation is the interference plan"
                );
                assert!(prepared.plan().lowered().is_some(), "{label}: a lowered program");
                let result = prepared
                    .estimate(&table, &ctx)
                    .unwrap_or_else(|e| panic!("{label}: estimate {e}"));
                assert_interference_point(&result, bayesian, &label);
                assert!(
                    !ids(&result).is_empty(),
                    "{label}: the cheap suite ran on the retained plan"
                );
            }
        }
    });
}
