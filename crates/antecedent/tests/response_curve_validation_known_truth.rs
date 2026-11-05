//! Known-truth evidence for the cheap and full validation of a static mean response curve.
//!
//! Reference: `conformance/response/curve_refuter_states/expected.json`. The data follow its
//! linear structural law `Y = 1 + 2 T + 0.8 Z + 0.3 e` with confounder `Z` and treatment
//! `T = 0.6 Z + 0.8 cos(1.13 i) + 0.4 e`, where each `e` is the sum of twelve uniforms minus six
//! drawn from one 64-bit linear congruential stream (the fixture's reference regenerates it).
//! The interventional mean at level `x` is `1 + 2 x + 0.8 mean(Z)`, so the published curve is
//! checked against that closed form at every grid point. Tolerance argument: the curve is
//! estimated from 600 rows with residual sd 0.3 and a treatment sd near 0.9, so its pointwise
//! standard error is a few hundredths; the declared band of 0.15 is several standard errors
//! wide and still far narrower than the confounded slope bias (0.8 x 0.6 / 0.9 per unit of
//! `x`), so it separates the adjusted curve from an unadjusted one.
//!
//! One test per distinct estimator and inference path: the adjustment fit under Frequentist
//! inference (`response.kennedy_dr` on a Dag, the first test), the Bayesian posterior-mean fit
//! (`response.bayesian` on a Dag), the per-completion class path (a Cpdag with two completions
//! whose adjustment sets differ, under both inferences) and the ADMG general-ID plug-in
//! (`functional.effect`, under both inferences; its random common cause is not applicable
//! because the plug-in has no adjustment set to extend, and that is asserted).
//!
//! The refuter states are judged by their reports. On this correctly specified design every
//! curve report passes under both structure routes and both suites, and the data-subset report
//! exists exactly under full. A grid beyond the observed treatment range (the planted violation
//! the fixture declares) must fail `curve.support`. The refuters' power against planted breaks
//! of the procedure itself (a curve read from an unpermuted copy of the treatment, a curve that
//! moves when a random covariate is added, a curve that scales with the number of rows) is
//! pinned beside the implementation in `analysis::execute::curve_refutation` on injected refits.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag};
use serde_json::Value;

const ROWS: u32 = 600;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/curve_refuter_states/expected.json"
    ))
    .expect("curve_refuter_states fixture parses")
}

fn number(document: &Value, key: &str) -> f64 {
    document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
}

/// Columns `z`, `t`, `y` in that order (ids 0, 1, 2); returns the table and `mean(z)`.
fn table() -> (TabularData, f64) {
    let mut state = 20_240_611_u64;
    let mut gauss = || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss, reason = "a 53-bit uniform is exact in f64")]
            let uniform = (state >> 11) as f64 / 9_007_199_254_740_992.0;
            sum += uniform;
        }
        sum - 6.0
    };
    let z: Vec<f64> = (0..ROWS).map(|i| (f64::from(i) / 17.0).sin()).collect();
    let t: Vec<f64> = (0..ROWS)
        .map(|i| 0.6 * z[i as usize] + 0.8 * (f64::from(i) * 1.13).cos() + 0.4 * gauss())
        .collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + 0.8 * z + 0.3 * gauss()).collect();
    #[allow(clippy::cast_precision_loss, reason = "the row count is exact in f64")]
    let mean_z = z.iter().sum::<f64>() / f64::from(ROWS);
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    (data, mean_z)
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    graph
}

fn curve(grid: &[f64]) -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(1),
            GridSpec::Values(grid.to_vec().into()),
        ),
    }))
}

fn study(
    data: &TabularData,
    grid: &[f64],
    accepted: bool,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let base = Study::tabular(data.clone());
    let builder = if accepted { base.graph(AcceptedGraph::from(dag())) } else { base.graph(dag()) };
    builder
        .query(curve(grid))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.4), ..Default::default() })
        .refute(suite)
        .bootstrap_replicates(0)
}

fn report<'a>(
    result: &'a antecedent::StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

fn grid(pin: &Value) -> Vec<f64> {
    pin["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

#[test]
fn response_curve_validation_known_truth() {
    let pin = fixture();
    let (data, mean_z) = table();
    let grid = grid(&pin);
    let band = number(&pin, "curve_absolute_tolerance");
    let (intercept, slope, z_coefficient) = (
        number(&pin, "intercept"),
        number(&pin, "treatment_coefficient"),
        number(&pin, "confounder_coefficient"),
    );
    for accepted in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("accepted={accepted} {suite:?}");
            let builder = study(&data, &grid, accepted, suite).build().unwrap();
            let context = ExecutionContext::for_tests(913);
            let prepared = builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
            // A refuted study is not sealed to the curve operation (that one carries no refuter
            // state); it runs its estimator route from the lowered program.
            assert!(
                matches!(prepared.retained_operation(), RetainedOperation::EstimatorRoute),
                "{label}: retained operation {:?}",
                prepared.retained_operation().name()
            );
            drop(builder);
            let result =
                prepared.estimate(&data, &context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.kennedy_dr"));
            assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");

            let response = result.response().expect("response payload");
            let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
                &response.estimate
            else {
                panic!("{label}: unexpected response {:?}", response.estimate);
            };
            assert_eq!(mean.len(), grid.len(), "{label}");
            for (value, x) in mean.iter().zip(&grid) {
                let truth = intercept + slope * x + z_coefficient * mean_z;
                assert!(
                    (value - truth).abs() <= band,
                    "{label}: curve {value} vs closed form {truth} at {x}"
                );
            }

            for name in ["curve.support", "curve.placebo_treatment", "curve.random_common_cause"] {
                let found =
                    report(&result, name).unwrap_or_else(|| panic!("{label}: no `{name}` report"));
                assert!(found.passed, "{label}: {name}: {:?}", found.failure_condition);
            }
            let subset = report(&result, "curve.data_subset");
            assert_eq!(subset.is_some(), suite == RefuteSuite::Full, "{label}");
            if let Some(subset) = subset {
                assert!(subset.passed, "{label}: {:?}", subset.failure_condition);
                assert!(subset.replicates >= 8, "{label}");
            }
            // The scalar contrast refuters do not apply to a curve.
            assert!(report(&result, "placebo.treatment").is_none(), "{label}");
        }
    }
}

#[test]
fn response_curve_validation_detects_a_grid_beyond_the_observed_treatment_range() {
    let pin = fixture();
    let (data, _) = table();
    let mut planted = grid(&pin);
    planted.push(number(&pin, "off_support_at"));
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let clean = study(&data, &grid(&pin), false, suite)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(914))
            .unwrap();
        assert!(report(&clean, "curve.support").is_some_and(|r| r.passed), "{suite:?}");
        let broken = study(&data, &planted, false, suite)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(915))
            .unwrap();
        let support = report(&broken, "curve.support").expect("support report");
        assert!(!support.passed, "{suite:?}: an off-support grid point was not detected");
    }
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
}

fn reports<'a>(
    result: &'a antecedent::StudyResult,
    name: &'a str,
) -> impl Iterator<Item = &'a antecedent_validate::RefutationReport> {
    result.refutations.iter().filter(move |report| report.refuter.as_ref() == name)
}

fn published(result: &antecedent::StudyResult) -> Vec<f64> {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        // An identified set of curves (the completions of an equivalence class) is checked
        // through its bounds: both must lie within the band of the closed form, and the
        // midpoint is returned.
        ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope)) => envelope
            .lower
            .iter()
            .zip(envelope.upper.iter())
            .map(|(lower, upper)| 0.5 * (lower + upper))
            .collect(),
        other => panic!("unexpected response {other:?}"),
    }
}

/// The Bayesian posterior-mean curve is refit under the same posterior path (`response.bayesian`)
/// and judged by the same three perturbations. The conjugate prior shrinks the slope slightly
/// toward zero, a bias of order 1 / n against a slope near 2 at n = 600, inside the same band.
#[test]
fn response_curve_validation_bayesian_dag_known_truth() {
    let pin = fixture();
    let (data, mean_z) = table();
    let grid = grid(&pin);
    let band = number(&pin, "curve_absolute_tolerance");
    for accepted in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("bayesian accepted={accepted} {suite:?}");
            let result = study(&data, &grid, accepted, suite)
                .inference(bayesian())
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(916))
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.bayesian"));
            assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
            for (value, x) in published(&result).iter().zip(&grid) {
                let truth = number(&pin, "intercept")
                    + number(&pin, "treatment_coefficient") * x
                    + number(&pin, "confounder_coefficient") * mean_z;
                assert!((value - truth).abs() <= band, "{label}: {value} vs {truth} at {x}");
            }
            for name in ["curve.support", "curve.placebo_treatment", "curve.random_common_cause"] {
                let found = report(&result, name).unwrap_or_else(|| panic!("{label}: no {name}"));
                assert!(found.passed, "{label}: {name}: {:?}", found.failure_condition);
            }
            assert_eq!(report(&result, "curve.data_subset").is_some(), suite == RefuteSuite::Full);
        }
    }
}

/// `t - z` undirected with `z -> y` and `t -> y`: completions `z -> t` (adjust for `z`) and
/// `t -> z` (adjust for nothing). `Z` is independent of `T` in the data, so both completions
/// publish the same line `1 + 2 x` up to estimation error and each is refuted against its own
/// adjustment estimand.
fn two_completion_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g
}

/// Treatment levels crossed with `z = +-1` and noise `+-0.2`: `Y = 1 + 2 T + 1.5 Z + e`.
fn balanced_table() -> TabularData {
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..31_u32 {
        let level = -1.5 + 3.0 * f64::from(k) / 30.0;
        for (zv, e) in [(1.0, 0.2), (1.0, -0.2), (-1.0, 0.2), (-1.0, -0.2)] {
            z.push(zv);
            t.push(level);
            y.push(1.0 + 2.0 * level + 1.5 * zv + e);
        }
    }
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

#[test]
fn response_curve_validation_class_known_truth() {
    let data = balanced_table();
    let grid = [-0.9, 0.0, 0.9];
    for pag in [false, true] {
        for accepted in [false, true] {
            for bayes in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    class_case(&data, &grid, (pag, accepted, bayes), suite);
                }
            }
        }
    }
}

/// A fully oriented PAG `z -> t`, `z -> y`, `t -> y`: one completion adjusting for `z`.
fn oriented_pag() -> Pag {
    let mut g = Pag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target)).unwrap();
    }
    g
}

fn class_case(
    data: &TabularData,
    grid: &[f64],
    (pag, accepted, bayes): (bool, bool, bool),
    suite: RefuteSuite,
) {
    let label = format!("pag={pag} accepted={accepted} bayesian={bayes} {suite:?}");
    let base = Study::tabular(data.clone());
    let builder = match (pag, accepted) {
        (false, true) => base.graph(AcceptedGraph::from(two_completion_cpdag())),
        (false, false) => base.graph(two_completion_cpdag()),
        (true, true) => base.graph(AcceptedGraph::from(oriented_pag())),
        (true, false) => base.graph(oriented_pag()),
    };
    let builder = builder
        .query(curve(grid))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.6), ..Default::default() })
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = if bayes { builder.inference(bayesian()) } else { builder };
    let result = builder
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(917))
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
    for (value, x) in published(&result).iter().zip(grid) {
        assert!((value - (1.0 + 2.0 * x)).abs() <= 0.25, "{label}: {value} at {x}");
    }
    // Each completion is refuted against its own adjustment estimand: every report
    // of every name passes, and each name is present.
    for name in ["curve.support", "curve.placebo_treatment", "curve.random_common_cause"] {
        let mut found = reports(&result, name).peekable();
        assert!(found.peek().is_some(), "{label}: no {name}");
        for each in found {
            assert!(each.passed, "{label}: {name}: {:?}", each.failure_condition);
        }
    }
    assert_eq!(
        reports(&result, "curve.data_subset").count() > 0,
        suite == RefuteSuite::Full,
        "{label}"
    );
}

/// The ADMG front-door fixture: the curve is the ID-algorithm plug-in, refit whole under a
/// permuted placebo treatment and a row subset; the random common cause is not applicable.
fn admg_fixture() -> (TabularData, Admg, ResponseQuery) {
    let pin: Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|col| *col == name).unwrap()).unwrap(),
        )
    };
    let mut graph = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        graph
            .insert_directed(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        graph
            .insert_bidirected(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
        ),
    });
    (data, graph, query)
}

/// An ADMG with no bidirected edge is identified by back-door adjustment and runs the
/// adjustment-based curve fit: the same three perturbations apply, including the random common cause.
#[test]
fn response_curve_validation_adjustable_admg_known_truth() {
    let (data, _) = table();
    let mut graph = Admg::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    let result = Study::tabular(data.clone())
        .graph(graph)
        .query(curve(&[-0.8, 0.0, 0.8]))
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(919))
        .unwrap();
    assert_eq!(result.support_status(), Some(CellStatus::Supported));
    for name in [
        "curve.support",
        "curve.placebo_treatment",
        "curve.random_common_cause",
        "curve.data_subset",
    ] {
        let found = report(&result, name).unwrap_or_else(|| panic!("no {name} report"));
        assert!(found.passed, "{name}: {:?}", found.failure_condition);
    }
}

#[test]
fn response_curve_validation_admg_known_truth() {
    let (data, graph, query) = admg_fixture();
    for accepted in [false, true] {
        for bayes in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("admg accepted={accepted} bayesian={bayes} {suite:?}");
                let base = Study::tabular(data.clone())
                    .query(CausalQuery::Response(query.clone()))
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalEffect)
                    .refute(suite)
                    .bootstrap_replicates(if bayes { 24 } else { 30 });
                let builder = if accepted {
                    base.graph(AcceptedGraph::from(graph.clone()))
                } else {
                    base.graph(graph.clone())
                };
                let builder = if bayes {
                    builder.inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(256),
                    ))
                } else {
                    builder
                };
                let result = builder
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(918))
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                for name in ["curve.support", "curve.placebo_treatment"] {
                    let found = report(&result, name)
                        .unwrap_or_else(|| panic!("{label}: no {name} report"));
                    assert!(found.passed, "{label}: {name}: {:?}", found.failure_condition);
                }
                assert_eq!(
                    report(&result, "curve.data_subset").is_some(),
                    suite == RefuteSuite::Full,
                    "{label}"
                );
                assert!(report(&result, "curve.random_common_cause").is_none(), "{label}");
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| d.code.as_ref() == "refute.curve.check_not_applicable"
                            && d.message.contains("no adjustment set")),
                    "{label}: the inapplicable random common cause is not stated"
                );
            }
        }
    }
}
