//! Cheap and full validation of the temporal cross-world effects (counterfactual, interventional
//! distribution, nested natural direct effect and path-specific effect) on a `TemporalDag`, a
//! `TemporalCpdag` and a `TemporalPag`, supplied, accepted or as a graph posterior, against the
//! closed forms of the one-lag linear mechanism of `conformance/estimate/temporal_cross_world`:
//! `M = 0.5 X(t-1) + e_m`, `Y = 1.5 X(t-1) + 4 M + e_y` give a nested direct effect of 1.5, a
//! mediated path effect of 2, and a total effect and interventional-mean contrast of 3.5.
//!
//! The refuters that read a lead or a pre-period (pre-trend, anticipation, the lagged negative
//! control) are statistical tests of one series, so they are checked across seeded series: the
//! clean mechanism rejects at no more than about its size, and a planted anticipation (the outcome
//! reads the treatment of the same period) is flagged on every seed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test fixtures: small exact indices and one coordinate walk per test"
)]

use std::sync::Arc;

use antecedent::analysis::{RetainedOperation, StudyBuilder};
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, Lag, NestedCounterfactualQuery, NodeRef,
    PathSpecificEffectQuery, Value, VariableId,
};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge as set_mask_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge,
};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_cross_world/expected.json"
    ))
    .unwrap()
}

struct Normals(u64);

impl Normals {
    fn next(&mut self) -> f64 {
        let mut uniform = || {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
            (bits as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let (u, w) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * w).cos()
    }
}

/// Columns `x, m, y`, with seeded Gaussian disturbances; `anticipation` adds `anticipation * x(t)`
/// to the mediator and the outcome, a treatment of the same period that the lag-1 graph does not
/// carry.
fn series_seeded(seed: u64, anticipation: f64) -> TimeSeriesData {
    let n = 400;
    let mut g = Normals(seed);
    let (mut x, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        x[t] = g.next();
        if t > 0 {
            m[t] = 0.5 * x[t - 1] + anticipation * x[t] + 0.3 * g.next();
            y[t] = 1.5 * x[t - 1] + 4.0 * m[t] + anticipation * x[t] + 0.3 * g.next();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn series() -> TimeSeriesData {
    series_seeded(99_991, 0.0)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `X(t-1) -> M`, `X(t-1) -> Y`, `M -> Y`.
fn dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, v(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

/// The DAG with a witness `r(t-1) -> X(t-1)` that makes the treatment edges visible.
fn pag() -> TemporalPag {
    let graph = dag();
    let mut pag = TemporalPag::empty();
    let mut ids = Vec::new();
    for node in graph.nodes() {
        let NodeRef::Lagged { variable, lag } = *node else { panic!("lagged nodes") };
        ids.push(pag.add_lagged(variable, lag).unwrap());
    }
    for edge in graph.edges() {
        pag.insert_directed(ids[edge.a.as_usize()], ids[edge.b.as_usize()]).unwrap();
    }
    let r1 = pag.add_lagged(v(3), Lag::from_raw(1)).unwrap();
    let treatment = graph
        .nodes()
        .iter()
        .position(|node| matches!(node, NodeRef::Lagged { variable, lag } if variable.raw() == 0 && *lag == Lag::from_raw(1)))
        .unwrap();
    pag.insert_directed(r1, ids[treatment]).unwrap();
    pag
}

/// The witness column of the PAG: a series read only through its edge into the treatment.
fn with_witness(data: &TimeSeriesData) -> TimeSeriesData {
    let n = data.row_count();
    let mut columns: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, id) in [("x", 0), ("m", 1), ("y", 2)] {
        let column = data.storage().columns()[id].clone();
        let antecedent_data::OwnedColumn::Float64(column) = column else { panic!("float columns") };
        columns.push((name.to_owned(), column.values.to_vec()));
    }
    columns.push(("r".to_owned(), (0..n).map(|t| ((t as f64) * 0.17).cos()).collect()));
    TimeSeriesData::from_f64_columns(
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())),
        1,
    )
    .unwrap()
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1_u64 << (from * n + to)
}

fn sampling(label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

/// A dynamic (`TemporalDag`) posterior: the correct template sampled at weights 0.7 and 0.3, which
/// are one atom of mass 1.
fn dbn_posterior() -> GraphPosterior {
    let n = 3;
    let correct = |weight: f64| (vec![(1, 2)], vec![(0, 1), (0, 2)], weight);
    let templates = [correct(0.7), correct(0.3)];
    let adjacency: Vec<u64> = templates
        .iter()
        .map(|(c, _, _)| c.iter().fold(0_u64, |mask, &(f, t)| set_edge(mask, n, f, t, true)))
        .collect();
    let lag_masks: Vec<u64> = templates
        .iter()
        .map(|(_, l, _)| l.iter().fold(0_u64, |mask, &(f, t)| mask | lag_bit(n, f, t)))
        .collect();
    let weights: Vec<f64> = templates.iter().map(|(_, _, w)| *w).collect();
    let cell = n * n;
    GraphPosterior::new(
        n,
        weights,
        adjacency,
        vec![0.0; cell],
        vec![0.0; cell],
        sampling("cross_world_temporal_dbn"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(lag_masks)
    .unwrap()
}

/// One class atom (`TemporalCpdag` or, with the witness, `TemporalPag`) of the same template.
fn class_posterior(pag: bool) -> GraphPosterior {
    let base_n = 3;
    let n = if pag { base_n + 1 } else { base_n };
    let mut lag_mask = 0_u64;
    for (from, to) in [(0, 1), (0, 2)] {
        lag_mask |= lag_bit(n, from, to);
    }
    let mut adjacency = AdjacencyMask::EMPTY;
    adjacency = set_mask_edge(adjacency, n, 1, 2, true);
    if pag {
        adjacency = set_mask_edge(adjacency, n, base_n, 0, true);
    }
    let cell = n * n;
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        sampling("cross_world_temporal_class"),
    )
    .unwrap()
    .with_atom_kind(if pag { GraphPosteriorAtomKind::Pag } else { GraphPosteriorAtomKind::Cpdag })
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Dag,
    Cpdag,
    Pag,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Structure {
    Explicit,
    Accepted,
    Posterior,
}

fn queries() -> Vec<(&'static str, CausalQuery)> {
    let (x, m, y) = (v(0), v(1), v(2));
    vec![
        (
            "nested",
            CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap()),
        ),
        (
            "path",
            CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(x, y).with_path_nodes([m])),
        ),
        (
            "counterfactual",
            CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                    .with_control_level(0.0),
            ),
        ),
        (
            "distribution",
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                y,
                [Intervention::set(x, Value::f64(1.0))],
            )),
        ),
    ]
}

fn builder(
    data: TimeSeriesData,
    class: Class,
    structure: Structure,
    query: CausalQuery,
    bayesian: bool,
    suite: RefuteSuite,
) -> StudyBuilder {
    let data = if class == Class::Pag { with_witness(&data) } else { data };
    let base = Study::series(data).query(query);
    let base = match (structure, class) {
        (Structure::Posterior, Class::Dag) => base.graph_posterior(dbn_posterior()),
        (Structure::Posterior, Class::Cpdag) => base.graph_posterior(class_posterior(false)),
        (Structure::Posterior, Class::Pag) => base.graph_posterior(class_posterior(true)),
        (Structure::Explicit, Class::Dag) => base.graph(dag()),
        (Structure::Accepted, Class::Dag) => base.graph(AcceptedGraph::from(dag())),
        (Structure::Explicit, Class::Cpdag) => base.graph(TemporalCpdag::from_temporal_dag(&dag())),
        (Structure::Accepted, Class::Cpdag) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag())))
        }
        (Structure::Explicit, Class::Pag) => base.graph(pag()),
        (Structure::Accepted, Class::Pag) => base.graph(AcceptedGraph::from(pag())),
    };
    let base = base.refute(suite);
    if bayesian {
        base.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(8),
        ))
    } else {
        base.inference(InferenceMode::Frequentist)
    }
}

/// Every coordinate of the three rules: the validations each structure is licensed under.
fn coordinates() -> Vec<(Class, Structure, Vec<RefuteSuite>)> {
    let mut out = Vec::new();
    for class in [Class::Dag, Class::Cpdag, Class::Pag] {
        for structure in [Structure::Explicit, Structure::Accepted] {
            out.push((class, structure, vec![RefuteSuite::Cheap, RefuteSuite::Full]));
        }
        let suites = if class == Class::Dag {
            vec![RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full]
        } else {
            vec![RefuteSuite::Cheap, RefuteSuite::Full]
        };
        out.push((class, Structure::Posterior, suites));
    }
    out
}

fn tolerance(pin: &serde_json::Value, key: &str, bayesian: bool, class: Class) -> f64 {
    let base = pin["tolerance"].as_f64().unwrap();
    // The Bayesian unit counterfactual is the posterior of the slope times the contrast, and a
    // class point is the completion-weighted mean of the linear points.
    if bayesian && (key == "counterfactual" || class != Class::Dag) {
        base.max(0.25)
    } else {
        base.max(0.1)
    }
}

fn effect(result: &StudyResult) -> f64 {
    result.estimate().as_effect().unwrap().ate
}

fn report<'a>(
    result: &'a StudyResult,
    refuter: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == refuter)
}

#[test]
fn temporal_cross_world_cells_match_the_linear_mechanism_under_every_suite() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let pin = pin();
            let ctx = ExecutionContext::for_tests(11);
            for (class, structure, suites) in coordinates() {
                for (key, query) in queries() {
                    let truth = pin[key].as_f64().unwrap();
                    for bayesian in [false, true] {
                        for &suite in &suites {
                            let label = format!(
                                "{class:?} {structure:?} {key} bayesian={bayesian} {suite:?}"
                            );
                            let study =
                                builder(series(), class, structure, query.clone(), bayesian, suite)
                                    .build()
                                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                            assert_eq!(
                                study.support_status(),
                                Some(CellStatus::Licensed),
                                "{label}"
                            );
                            let result = study.run(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                            assert_eq!(
                                result.support_status(),
                                Some(CellStatus::Licensed),
                                "{label}"
                            );
                            let tol = tolerance(&pin, key, bayesian, class);
                            assert!(
                                (effect(&result) - truth).abs() < tol,
                                "{label}: {} vs {truth}",
                                effect(&result)
                            );
                            if structure == Structure::Posterior {
                                let mixture = result.structural_response.as_ref().expect(&label);
                                assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{label}");
                            }
                            if suite == RefuteSuite::None {
                                assert!(result.refutations.is_empty(), "{label}");
                                continue;
                            }
                            // The permuted treatment carries no effect: the refitted value is near zero.
                            let placebo = report(&result, "target.placebo_treatment")
                                .unwrap_or_else(|| panic!("{label}: no placebo"));
                            assert!(placebo.passed, "{label}: {}", placebo.refuted_ate);
                            assert!(
                                placebo.refuted_ate.abs() < 0.6,
                                "{label}: {}",
                                placebo.refuted_ate
                            );
                            if suite == RefuteSuite::Full {
                                let outcome = report(&result, "target.placebo_outcome")
                                    .unwrap_or_else(|| panic!("{label}: no placebo outcome"));
                                assert!(
                                    outcome.passed && outcome.refuted_ate.abs() < 0.6,
                                    "{label}"
                                );
                            }
                            for refuter in ["target.subset_stability", "target.bootstrap_stability"]
                            {
                                if let Some(stability) = report(&result, refuter) {
                                    assert!(stability.passed, "{label}: {refuter}");
                                }
                            }
                        }
                    }
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn temporal_lead_refuters_pass_the_clean_series_at_their_size_and_flag_a_planted_anticipation() {
    let ctx = ExecutionContext::for_tests(11);
    let seeds = 16_u64;
    for (key, query) in queries() {
        for (anticipation, flagged_at_least) in [(0.0, 0_u64), (1.0, 14_u64)] {
            let (mut pre_trend, mut anticipation_fails, mut control) = (0_u64, 0_u64, 0_u64);
            for seed in 1..=seeds {
                let data = series_seeded(1_000 + seed * 7_919, anticipation);
                let result = builder(
                    data,
                    Class::Dag,
                    Structure::Explicit,
                    query.clone(),
                    false,
                    RefuteSuite::Full,
                )
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap();
                let failed = |name: &str| report(&result, name).is_some_and(|r| !r.passed);
                pre_trend += u64::from(failed("target.pre_trend"));
                anticipation_fails += u64::from(failed("target.anticipation"));
                control += u64::from(failed("target.negative_control_outcome"));
            }
            if anticipation == 0.0 {
                // A 5% test with a block-bootstrap spread that is a little narrow: no more than a
                // quarter of seeds reject.
                assert!(
                    pre_trend <= 4 && anticipation_fails <= 5 && control <= 5,
                    "{key}: clean series rejected pre_trend={pre_trend} anticipation={anticipation_fails} control={control} of {seeds}"
                );
            } else {
                assert!(
                    pre_trend >= flagged_at_least,
                    "{key}: planted anticipation flagged on {pre_trend} of {seeds}"
                );
            }
        }
    }
}

#[test]
fn temporal_cross_world_cells_execute_from_the_retained_plan_without_the_builder() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let pin = pin();
            let ctx = ExecutionContext::for_tests(11);
            for (class, structure, suites) in coordinates() {
                for (key, query) in queries() {
                    let truth = pin[key].as_f64().unwrap();
                    for bayesian in [false, true] {
                        for &suite in &suites {
                            let label = format!(
                                "{class:?} {structure:?} {key} bayesian={bayesian} {suite:?}"
                            );
                            let builder =
                                builder(series(), class, structure, query.clone(), bayesian, suite);
                            let study = builder.clone().build().unwrap();
                            let prepared = study.prepare(&ctx).unwrap();
                            drop(builder);
                            drop(study);
                            let operation = format!("{:?}", prepared.retained_operation());
                            assert!(
                                !matches!(
                                    prepared.retained_operation(),
                                    RetainedOperation::FreshRun
                                ),
                                "{label}: {operation}"
                            );
                            let result = prepared
                                .estimate_retained(&ctx)
                                .unwrap_or_else(|e| panic!("{label}: {e}"));
                            assert_eq!(
                                result.support_status(),
                                Some(CellStatus::Licensed),
                                "{label}"
                            );
                            assert!(
                                (effect(&result) - truth).abs()
                                    < tolerance(&pin, key, bayesian, class),
                                "{label}: {}",
                                effect(&result)
                            );
                        }
                    }
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

