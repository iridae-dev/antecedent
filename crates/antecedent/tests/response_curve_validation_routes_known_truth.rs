//! Known-truth evidence for the cheap and full validation of response curves whose route runs
//! an executor of its own: a temporal series (`TemporalDag`), the per-completion temporal class
//! (`TemporalCpdag`, `TemporalPag`), graph posteriors of temporal atoms (`TemporalDag` and
//! class atoms) and of static `Dag` atoms.
//!
//! Reference: `conformance/response/curve_refuter_route_states/expected.json`. Each law is
//! linear with deterministic near-Gaussian noise (the fixture's reference regenerates the
//! stream and the exact least-squares coefficients), so the interventional mean curve has a
//! closed form: `1.2 + 1.8 x` for the lag-one series law, `1.4 x + 0.6 mean(z)` for the
//! class law, `1 + 2 x` for the static posterior law. Tolerance argument: with residual sd
//! 0.3 and a treatment sd near 0.8 to 1.2 over 600 to 1200 rows the pointwise standard error
//! of each curve is a few hundredths, so the declared band of 0.15 is several standard errors.
//!
//! Every route is validated by re-running its whole executor on perturbed data (a block-shuffled
//! placebo treatment and, under full, a circular 80% window for series, or a permuted placebo
//! and a row subset for tables) and judging the curve it publishes by the sup-norm typicality
//! statistic. On these correctly specified designs every curve report passes; the random common
//! cause is not applicable (the certified graph fixes the covariates) and that is asserted. A
//! grid point beyond the observed treatment range (the planted violation the fixture declares)
//! must fail `curve.support`. The refuters' power against planted breaks of the procedure is
//! pinned on injected refits in `analysis::execute::curve_refutation`.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "retained_run/mod.rs"]
mod retained;

use retained::run_retained;

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TabularData, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, temporal_cpdag_from_dbn_masks,
    temporal_pag_from_dbn_masks,
};
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg, TemporalDag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/curve_refuter_route_states/expected.json"
    ))
    .expect("curve_refuter_route_states fixture parses")
}

fn number(document: &Value, path: &[&str]) -> f64 {
    let mut node = document;
    for key in path {
        node = &node[*key];
    }
    node.as_f64().unwrap_or_else(|| panic!("fixture lacks number `{path:?}`"))
}

fn numbers(document: &Value, key: &str) -> Vec<f64> {
    document[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

/// A 64-bit linear congruential stream: `uniform` is `(state >> 11) / 2^53`, `gauss` is the sum
/// of twelve uniforms minus six.
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss, reason = "a 53-bit uniform is exact in f64")]
        let value = (self.0 >> 11) as f64 / 9_007_199_254_740_992.0;
        value
    }

    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// Law 1: `x_t = 4 u - 2`, `y_t = 1.2 + 1.8 x_{t-1} + 0.3 e`.
fn lag_series(pin: &Value) -> TimeSeriesData {
    let n = pin["lag_series"]["rows"].as_u64().unwrap() as usize;
    let mut stream = Stream(pin["lag_series"]["seed"].as_u64().unwrap());
    let x: Vec<f64> = (0..n).map(|_| 4.0 * stream.uniform() - 2.0).collect();
    let mut y = vec![0.0; n];
    for i in 1..n {
        y[i] = 1.2 + 1.8 * x[i - 1] + 0.3 * stream.gauss();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// Law 2: `z_t = sin(.37 t)`, `x_t = cos(.61 t) + .3 e`, `y_t = 1.4 x_{t-1} + .6 z_{t-1} + .3 e`.
/// Returns the series and `mean(z)` over the lag-aligned rows.
fn class_series(pin: &Value) -> (TimeSeriesData, f64) {
    let n = pin["class_series"]["rows"].as_u64().unwrap() as usize;
    let mut stream = Stream(pin["class_series"]["seed"].as_u64().unwrap());
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.61).cos() + 0.3 * stream.gauss()).collect();
    let mut y = vec![0.0; n];
    for i in 1..n {
        y[i] = 1.4 * x[i - 1] + 0.6 * z[i - 1] + 0.3 * stream.gauss();
    }
    let mean_z = z[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
    let data = TimeSeriesData::from_f64_columns(
        [("t", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    (data, mean_z)
}

/// Law 4: `z = sin(.37 i)`, `t = .5 z + .8 sin(.61 i) + .3 e`, `y = 1 + 2 t + .3 e`.
fn static_table(pin: &Value) -> TabularData {
    let n = pin["static_table"]["rows"].as_u64().unwrap() as usize;
    let mut stream = Stream(pin["static_table"]["seed"].as_u64().unwrap());
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let t: Vec<f64> =
        (0..n).map(|i| 0.5 * z[i] + 0.8 * (i as f64 * 0.61).sin() + 0.3 * stream.gauss()).collect();
    let y: Vec<f64> = t.iter().map(|t| 1.0 + 2.0 * t + 0.3 * stream.gauss()).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn lag_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x, y).unwrap();
    graph
}

fn spec(history: Option<u32>) -> TemporalResponseSpec {
    TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), history).unwrap()
}

fn curve(grid: &[f64], outcome: u32, history: Option<u32>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(outcome),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid.to_vec())),
        ),
    })
    .with_temporal(spec(history))
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
}

fn report<'a>(
    result: &'a StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

fn published(result: &StudyResult) -> Vec<f64> {
    let surface = |value: &ResponseValue| match value {
        ResponseValue::Surface { mean, .. } => Some(mean.to_vec()),
        // An identified set of curves (the completions of an equivalence class): its midpoint.
        ResponseValue::Envelope(envelope) => Some(
            envelope
                .lower
                .iter()
                .zip(envelope.upper.iter())
                .map(|(lower, upper)| 0.5 * (lower + upper))
                .collect(),
        ),
        _ => None,
    };
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => surface(value),
        _ => result
            .structural_response
            .as_ref()
            .and_then(|mixture| mixture.conditional_on_identified.as_ref())
            .and_then(surface),
    }
    .expect("a published curve")
}

/// The reports every route must carry on a correctly specified design, by suite.
fn assert_curve_reports(label: &str, result: &StudyResult, suite: RefuteSuite) {
    for name in ["curve.support", "curve.placebo_treatment"] {
        let found = report(result, name).unwrap_or_else(|| panic!("{label}: no `{name}` report"));
        assert!(found.passed, "{label}: {name}: {:?}", found.failure_condition);
    }
    let subset = report(result, "curve.data_subset");
    assert_eq!(subset.is_some(), suite == RefuteSuite::Full, "{label}");
    if let Some(subset) = subset {
        assert!(subset.passed, "{label}: data_subset: {:?}", subset.failure_condition);
    }
    assert!(report(result, "curve.random_common_cause").is_none(), "{label}");
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.curve.check_not_applicable"
            && d.message.contains("random common cause")),
        "{label}: the inapplicable random common cause is not stated"
    );
}

#[test]
fn response_curve_validation_temporal_dag_known_truth() {
    let pin = fixture();
    let data = lag_series(&pin);
    let grid = numbers(&pin["lag_series"], "grid");
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for accepted in [false, true] {
        for bayes in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("temporal dag accepted={accepted} bayesian={bayes} {suite:?}");
                let base = Study::series(data.clone());
                let builder = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(lag_graph()))
                } else {
                    base.graph(lag_graph())
                };
                let builder = builder
                    .query(CausalQuery::Response(curve(&grid, 1, None)))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let builder = if bayes { builder.inference(bayesian()) } else { builder };
                let result = run_retained(builder, &label, &ExecutionContext::for_tests(931));
                assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                for (value, x) in published(&result).iter().zip(&grid) {
                    assert!((value - (1.2 + 1.8 * x)).abs() <= band, "{label}: {value} at {x}");
                }
                assert_curve_reports(&label, &result, suite);
            }
        }
    }
    // Planted violation: a grid point beyond the observed treatment range.
    let mut planted = grid.clone();
    planted.push(number(&pin["lag_series"], &["off_support_at"]));
    let broken = run_retained(
        Study::series(data.clone())
            .graph(lag_graph())
            .query(CausalQuery::Response(curve(&planted, 1, None)))
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0),
        "retained run",
        &ExecutionContext::for_tests(932),
    );
    assert!(
        report(&broken, "curve.support").is_some_and(|r| !r.passed),
        "an off-support grid point was not detected"
    );
}

fn variables() -> Vec<VariableId> {
    vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)]
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

/// `t@1 -> y`, `z@1 -> y` with no contemporaneous structure.
fn temporal_cpdag() -> antecedent_graph::TemporalCpdag {
    let lag = lag_bit(3, 0, 1) | lag_bit(3, 2, 1);
    temporal_cpdag_from_dbn_masks(0, lag, 3, 1, &variables()).unwrap()
}

/// `z -> t`, `z -> y` contemporaneously and `t@1 -> y`.
fn temporal_pag() -> antecedent_graph::TemporalPag {
    let contemporaneous = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true)
        .to_word()
        .expect("a three-variable graph fits one word");
    temporal_pag_from_dbn_masks(contemporaneous, lag_bit(3, 0, 1), 0, 3, 1, &variables()).unwrap()
}

#[test]
fn response_curve_validation_temporal_class_known_truth() {
    let pin = fixture();
    let (data, mean_z) = class_series(&pin);
    let grid = numbers(&pin["class_series"], "grid");
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for pag in [false, true] {
        for accepted in [false, true] {
            for bayes in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "temporal {} accepted={accepted} bayesian={bayes} {suite:?}",
                        if pag { "pag" } else { "cpdag" }
                    );
                    let base = Study::series(data.clone());
                    let builder = match (pag, accepted) {
                        (false, false) => base.graph(temporal_cpdag()),
                        (false, true) => base.graph(AcceptedGraph::from(temporal_cpdag())),
                        (true, false) => base.graph(temporal_pag()),
                        (true, true) => base.graph(AcceptedGraph::from(temporal_pag())),
                    };
                    let builder = builder
                        .query(CausalQuery::Response(curve(&grid, 1, Some(1))))
                        .refute(suite)
                        .bootstrap_replicates(0);
                    let builder = if bayes { builder.inference(bayesian()) } else { builder };
                    let result = run_retained(builder, &label, &ExecutionContext::for_tests(933));
                    assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                    for (value, x) in published(&result).iter().zip(&grid) {
                        let truth = 1.4 * x + 0.6 * mean_z;
                        assert!((value - truth).abs() <= band, "{label}: {value} vs {truth}");
                    }
                    assert_curve_reports(&label, &result, suite);
                }
            }
        }
    }
}

/// One-atom posterior over the lag-one `TemporalDag` (`x@1 -> y`) with all mass on that atom.
fn dbn_posterior() -> GraphPosterior {
    let lag = lag_bit(2, 0, 1);
    GraphPosterior::from_atoms(
        2,
        vec![1.0],
        vec![0],
        vec![0.0; 4],
        vec![0.0; 4],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("curve_validation_dbn"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap()
}

/// One-atom class posterior (`t@1 -> y`, `z@1 -> y` for a Cpdag; the oriented PAG otherwise).
fn class_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let n = 3;
    let (contemporaneous, lag) = match kind {
        GraphPosteriorAtomKind::Pag => {
            (set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true), lag_bit(3, 0, 1))
        }
        _ => (antecedent_discovery::AdjacencyMask::EMPTY, lag_bit(3, 0, 1) | lag_bit(3, 2, 1)),
    };
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("curve_validation_class"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap()
}

#[test]
fn response_curve_validation_temporal_posterior_known_truth() {
    let pin = fixture();
    let band = number(&pin, &["curve_absolute_tolerance"]);
    let (class_data, mean_z) = class_series(&pin);
    let lag_data = lag_series(&pin);
    let lag_grid = numbers(&pin["lag_series"], "grid");
    let class_grid = numbers(&pin["class_series"], "grid");
    for bayes in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("dbn posterior bayesian={bayes} {suite:?}");
            let builder = Study::series(lag_data.clone())
                .graph_posterior(dbn_posterior())
                .query(CausalQuery::Response(curve(&lag_grid, 1, None)))
                .refute(suite)
                .bootstrap_replicates(0);
            let builder = if bayes { builder.inference(bayesian()) } else { builder };
            let result = run_retained(builder, &label, &ExecutionContext::for_tests(934));
            assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
            for (value, x) in published(&result).iter().zip(&lag_grid) {
                assert!((value - (1.2 + 1.8 * x)).abs() <= band, "{label}: {value} at {x}");
            }
            assert_curve_reports(&label, &result, suite);

            for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
                let label = format!("{kind:?} posterior bayesian={bayes} {suite:?}");
                let builder = Study::series(class_data.clone())
                    .graph_posterior(class_posterior(kind))
                    .query(CausalQuery::Response(curve(&class_grid, 1, Some(1))))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let builder = if bayes { builder.inference(bayesian()) } else { builder };
                let result = run_retained(builder, &label, &ExecutionContext::for_tests(935));
                assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                for (value, x) in published(&result).iter().zip(&class_grid) {
                    assert!((value - (1.4 * x + 0.6 * mean_z)).abs() <= band, "{label}: {value}");
                }
                assert_curve_reports(&label, &result, suite);
            }
        }
    }
}

/// `z -> t -> y` as an adjacency mask over `(t, y, z)`.
fn chain_mask() -> u64 {
    set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true)
        .to_word()
        .expect("a three-variable graph fits one word")
}

/// `y -> t` (reverse causal): no adjustment identifies `do(t)`, so this atom is unidentified mass.
fn reverse_mask() -> u64 {
    set_edge(0, 3, 1, 0, true).to_word().expect("a three-variable graph fits one word")
}

fn dag_posterior(masks: Vec<u64>, weights: Vec<f64>) -> GraphPosterior {
    let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    GraphPosterior::from_atoms(
        3,
        weights,
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess,
            diagnostics: InferenceDiagnostics::analytic("curve_validation_dag"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
    .with_algorithm("known_truth_fixture")
}

fn static_curve(grid: &[f64]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid.to_vec())),
        ),
    })
}

/// A posterior of `Dag` atoms: one identified chain atom, and in the second case an atom that
/// cannot be identified (its mass stays unidentified and the published curve is the
/// conditional-on-identified curve, which the refuter state refits whole).
#[test]
fn response_curve_validation_dag_posterior_known_truth() {
    let pin = fixture();
    let data = static_table(&pin);
    let grid = numbers(&pin["static_table"], "grid");
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for lost_mass in [false, true] {
        for bayes in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label =
                    format!("dag posterior lost_mass={lost_mass} bayesian={bayes} {suite:?}");
                let posterior = if lost_mass {
                    dag_posterior(vec![chain_mask(), reverse_mask()], vec![0.8, 0.2])
                } else {
                    dag_posterior(vec![chain_mask(), chain_mask()], vec![0.6, 0.4])
                };
                let builder = Study::tabular(data.clone())
                    .graph_posterior(posterior)
                    .query(CausalQuery::Response(static_curve(&grid)))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let builder = if bayes { builder.inference(bayesian()) } else { builder };
                let result = run_retained(builder, &label, &ExecutionContext::for_tests(936));
                assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                for (value, x) in published(&result).iter().zip(&grid) {
                    assert!((value - (1.0 + 2.0 * x)).abs() <= band, "{label}: {value} at {x}");
                }
                assert_curve_reports(&label, &result, suite);
            }
        }
    }
}

/// Eight units of law 1, drawn unit after unit from one stream.
fn lag_panel(pin: &Value) -> PanelData {
    let units = pin["panel"]["units"].as_u64().unwrap() as u32;
    let rows = pin["panel"]["rows"].as_u64().unwrap() as usize;
    let mut stream = Stream(pin["panel"]["seed"].as_u64().unwrap());
    let members: Vec<PanelUnit> = (0..units)
        .map(|unit_id| {
            let x: Vec<f64> = (0..rows).map(|_| 4.0 * stream.uniform() - 2.0).collect();
            let mut y = vec![0.0; rows];
            for i in 1..rows {
                y[i] = 1.2 + 1.8 * x[i - 1] + 0.3 * stream.gauss();
            }
            let series =
                TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                    .unwrap();
            PanelUnit { unit_id, series }
        })
        .collect();
    PanelData::try_new(Arc::<[PanelUnit]>::from(members)).unwrap()
}

/// A panel of units is validated by re-running the panel executor on a panel whose units each
/// have their treatment block-shuffled (placebo) and, under full, on a random 80% of whole units.
#[test]
fn response_curve_validation_panel_known_truth() {
    let pin = fixture();
    let panel = lag_panel(&pin);
    let grid = numbers(&pin["panel"], "grid");
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for accepted in [false, true] {
        for bayes in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("panel accepted={accepted} bayesian={bayes} {suite:?}");
                let base = Study::panel(panel.clone());
                let builder = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(lag_graph()))
                } else {
                    base.graph(lag_graph())
                };
                let builder = builder
                    .query(CausalQuery::Response(curve(&grid, 1, None)))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let builder = if bayes { builder.inference(bayesian()) } else { builder };
                let result = run_retained(builder, &label, &ExecutionContext::for_tests(937));
                assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                for (value, x) in published(&result).iter().zip(&grid) {
                    assert!((value - (1.2 + 1.8 * x)).abs() <= band, "{label}: {value} at {x}");
                }
                assert_curve_reports(&label, &result, suite);
            }
        }
    }
}

/// The front-door ADMG fixture read as a `TemporalAdmg` template (`from_static_contemporaneous`),
/// which the executor projects to its contemporaneous ADMG and runs as the general-ID plug-in.
fn temporal_admg_frontdoor() -> (TabularData, TemporalAdmg, ResponseQuery) {
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
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    (data, TemporalAdmg::from_static_contemporaneous(&graph).unwrap(), query)
}

#[test]
fn response_curve_validation_temporal_admg_known_truth() {
    let (data, graph, query) = temporal_admg_frontdoor();
    for accepted in [false, true] {
        for bayes in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("temporal admg accepted={accepted} bayesian={bayes} {suite:?}");
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
                let builder = if bayes { builder.inference(bayesian()) } else { builder };
                let result = run_retained(builder, &label, &ExecutionContext::for_tests(938));
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
            }
        }
    }
}

/// A `TemporalAdmg` template without a bidirected edge is adjustment-identified through the
/// same projection and takes the adjustment fit, including the random common cause.
#[test]
fn response_curve_validation_adjustable_temporal_admg_known_truth() {
    let pin = fixture();
    let data = static_table(&pin);
    let grid = numbers(&pin["static_table"], "grid");
    let mut admg = Admg::with_variables(3);
    // Columns are (t, y, z): z -> t, z -> y, t -> y.
    for (source, target) in [(2, 0), (2, 1), (0, 1)] {
        admg.insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target)).unwrap();
    }
    let graph = TemporalAdmg::from_static_contemporaneous(&admg).unwrap();
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let label = format!("adjustable temporal admg {suite:?}");
        let result = run_retained(
            Study::tabular(data.clone())
                .graph(graph.clone())
                .query(CausalQuery::Response(static_curve(&grid)))
                .refute(suite)
                .bootstrap_replicates(0),
            &label,
            &ExecutionContext::for_tests(939),
        );
        assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
        for (value, x) in published(&result).iter().zip(&grid) {
            assert!((value - (1.0 + 2.0 * x)).abs() <= band, "{label}: {value} at {x}");
        }
        for name in ["curve.support", "curve.placebo_treatment", "curve.random_common_cause"] {
            let found = report(&result, name).unwrap_or_else(|| panic!("{label}: no {name}"));
            assert!(found.passed, "{label}: {name}: {:?}", found.failure_condition);
        }
        assert_eq!(report(&result, "curve.data_subset").is_some(), suite == RefuteSuite::Full);
    }
}

/// Six units of law 2 (the shared confounder `z` is the same wave in every unit), drawn unit
/// after unit from one stream. Returns the panel and `mean(z)` over the lag-aligned rows.
fn class_panel(pin: &Value) -> (PanelData, f64) {
    let units = pin["class_panel"]["units"].as_u64().unwrap() as u32;
    let n = pin["class_panel"]["rows"].as_u64().unwrap() as usize;
    let mut stream = Stream(pin["class_panel"]["seed"].as_u64().unwrap());
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let members: Vec<PanelUnit> = (0..units)
        .map(|unit_id| {
            let x: Vec<f64> =
                (0..n).map(|i| (i as f64 * 0.61).cos() + 0.3 * stream.gauss()).collect();
            let mut y = vec![0.0; n];
            for i in 1..n {
                y[i] = 1.4 * x[i - 1] + 0.6 * z[i - 1] + 0.3 * stream.gauss();
            }
            let series = TimeSeriesData::from_f64_columns(
                [("t", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
                1,
            )
            .unwrap();
            PanelUnit { unit_id, series }
        })
        .collect();
    let mean_z = z[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
    (PanelData::try_new(Arc::<[PanelUnit]>::from(members)).unwrap(), mean_z)
}

/// The class panel executor (`TemporalCpdag` and `TemporalPag` on panel data) is validated by the
/// same wrapper as the series route: each unit's treatment block-shuffled, and under full a random
/// 80% of whole units.
#[test]
fn response_curve_validation_panel_class_known_truth() {
    let pin = fixture();
    let (panel, mean_z) = class_panel(&pin);
    let grid = numbers(&pin["class_panel"], "grid");
    let band = number(&pin, &["curve_absolute_tolerance"]);
    for pag in [false, true] {
        for accepted in [false, true] {
            for bayes in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "panel {} accepted={accepted} bayesian={bayes} {suite:?}",
                        if pag { "pag" } else { "cpdag" }
                    );
                    let base = Study::panel(panel.clone());
                    let builder = match (pag, accepted) {
                        (false, false) => base.graph(temporal_cpdag()),
                        (false, true) => base.graph(AcceptedGraph::from(temporal_cpdag())),
                        (true, false) => base.graph(temporal_pag()),
                        (true, true) => base.graph(AcceptedGraph::from(temporal_pag())),
                    };
                    let builder = builder
                        .query(CausalQuery::Response(curve(&grid, 1, Some(1))))
                        .refute(suite)
                        .bootstrap_replicates(0);
                    let builder = if bayes { builder.inference(bayesian()) } else { builder };
                    let result = run_retained(builder, &label, &ExecutionContext::for_tests(940));
                    assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
                    for (value, x) in published(&result).iter().zip(&grid) {
                        let truth = 1.4 * x + 0.6 * mean_z;
                        assert!((value - truth).abs() <= band, "{label}: {value} vs {truth}");
                    }
                    assert_curve_reports(&label, &result, suite);
                }
            }
        }
    }
}
