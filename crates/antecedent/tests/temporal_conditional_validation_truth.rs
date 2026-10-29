//! Cheap and full validation of a temporal conditional pulse or sustained effect on a
//! `TemporalDag`, `TemporalCpdag` and `TemporalPag` against the marginal conditional effect of a
//! stated law.
//!
//! The fixture `conformance/estimate/temporal_conditional_refuter_states` states
//! `x_t = a r_t + e_t`, `w_t ~ Bernoulli(p)`, `y_t = b x_{t-1} + c w_t + d x_{t-1} w_t +
//! s u_t`: the slope of the pulse at the modifier is `b + d w`, so the effect marginal over the
//! modifier is `b + d p`. The functional refuter state runs the circular-shift placebo
//! treatment, contiguous-window subset stability and, under full, the placebo outcome and the
//! negative-control, pre-trend and anticipation checks (Bayesian: prior sensitivity and the
//! posterior predictive check of the lag-aware reference model). The placebo and control checks
//! have the exact null 0 and the stability checks the original value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/temporal_conditional_refuter_states/expected.json");
const N: usize = 800;

struct Truth {
    effect: f64,
    tolerance: f64,
    null: f64,
    stability: f64,
    law: serde_json::Value,
    document: serde_json::Value,
}

fn truth() -> Truth {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let tolerance = |key: &str| document["tolerance"][key].as_f64().unwrap();
    Truth {
        effect: document["marginal_effect"].as_f64().unwrap(),
        tolerance: tolerance("effect"),
        null: tolerance("null_refuter"),
        stability: tolerance("stability_refuter"),
        law: document["law"].clone(),
        document,
    }
}

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Columns `x, y, w, r`.
fn series(truth: &Truth) -> TimeSeriesData {
    let law = |key: &str| truth.law[key].as_f64().unwrap();
    let mut stream = Stream(77);
    let r: Vec<f64> = (0..N).map(|_| stream.normal()).collect();
    let w: Vec<f64> =
        (0..N).map(|_| f64::from(stream.uniform() < law("modifier_probability"))).collect();
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 0..N {
        x[t] = law("treatment_loading") * r[t] + stream.normal();
    }
    for t in 1..N {
        y[t] = law("slope") * x[t - 1]
            + law("modifier_main") * w[t]
            + law("interaction") * x[t - 1] * w[t]
            + law("noise_sd") * stream.normal();
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice()), ("r", r.as_slice())],
        1,
    )
    .unwrap()
}

fn dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, var(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

/// `r[t-1] -> x[t-1] -> y[t] <- w[t]`: the arrowhead into the lagged treatment makes `x -> y`
/// visible in every completion.
fn pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let r1 = graph.add_lagged(var(3), Lag::from_raw(1)).unwrap();
    let x1 = graph.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(r1, x1).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

#[derive(Clone, Copy, Debug)]
enum Class {
    TemporalDag,
    TemporalCpdag,
    TemporalPag,
}

fn query(sustained: bool) -> CausalQuery {
    let base = TemporalEffectQuery::pulse(var(0), var(1), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
        .with_effect_modifiers([var(2)]);
    CausalQuery::TemporalEffect(if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    })
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64).prior_scale(1_000.0))
}

fn inference_name(inference: &InferenceMode) -> &'static str {
    if matches!(inference, InferenceMode::Frequentist) { "Frequentist" } else { "Bayesian" }
}

fn suite_name(suite: RefuteSuite) -> &'static str {
    match suite {
        RefuteSuite::Cheap => "cheap",
        RefuteSuite::Full => "full",
        _ => "none",
    }
}

fn builder(class: Class, accepted: bool, data: TimeSeriesData) -> antecedent::StudyBuilder {
    let base = Study::series(data);
    match (class, accepted) {
        (Class::TemporalDag, false) => base.graph(dag()),
        (Class::TemporalDag, true) => base.graph(AcceptedGraph::from(dag())),
        (Class::TemporalCpdag, false) => base.graph(TemporalCpdag::from_temporal_dag(&dag())),
        (Class::TemporalCpdag, true) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag())))
        }
        (Class::TemporalPag, false) => base.graph(pag()),
        (Class::TemporalPag, true) => base.graph(AcceptedGraph::temporal_pag(pag())),
    }
}

fn effect_ends(result: &StudyResult) -> (f64, f64) {
    let ate = result.estimate().as_effect().unwrap().ate;
    if ate.is_finite() {
        return (ate, ate);
    }
    let set = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set.as_ref())
        .expect("an identified set");
    (set.lower[0], set.upper[0])
}

fn bare(name: &str) -> &str {
    match name.strip_prefix("completion.") {
        Some(rest) => rest.split_once('.').map_or(rest, |(_, bare)| bare),
        None => name,
    }
}

fn assert_cell(
    result: &StudyResult,
    inference: &InferenceMode,
    suite: RefuteSuite,
    truth: &Truth,
    label: &str,
    status: CellStatus,
) {
    assert_eq!(result.support_status(), Some(status), "{label}");
    let (low, high) = effect_ends(result);
    assert!(
        (low - truth.effect).abs() < truth.tolerance
            && (high - truth.effect).abs() < truth.tolerance,
        "{label}: effect [{low}, {high}] vs marginal conditional effect {}",
        truth.effect
    );
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}");
        return;
    }
    let contract = &truth.document["refuter_contract"];
    let wanted = contract[inference_name(inference)][suite_name(suite)].as_array().unwrap();
    for name in wanted {
        let name = name.as_str().unwrap();
        assert!(
            result.refutations.iter().any(|report| bare(report.refuter.as_ref()) == name),
            "{label}: refuter {name} did not run"
        );
    }
    let listed = |key: &str, name: &str| {
        contract[key].as_array().unwrap().iter().any(|entry| entry.as_str() == Some(name))
    };
    for report in &result.refutations {
        let name = bare(report.refuter.as_ref());
        assert!(
            report.passed,
            "{label}: {name} flags the clean law ({} vs {})",
            report.original_ate, report.refuted_ate
        );
        if listed("null_expected", name) {
            assert!(
                report.refuted_ate.abs() < truth.null,
                "{label}: {name} must sit at the null 0, got {}",
                report.refuted_ate
            );
        }
        if listed("original_expected", name) {
            assert!(
                (report.refuted_ate - report.original_ate).abs() < truth.stability,
                "{label}: {name} moved the effect from {} to {}",
                report.original_ate,
                report.refuted_ate
            );
        }
    }
}

/// `TemporalConditionalPulse` and `TemporalConditionalSustained` x `TemporalDag`,
/// `TemporalCpdag` and `TemporalPag` x explicit and accepted x Frequentist and Bayesian x cheap
/// and full.
#[test]
fn temporal_conditional_cheap_and_full_pass_on_the_clean_law_at_the_marginal_effect() {
    let truth = truth();
    let data = series(&truth);
    let context = ExecutionContext::for_tests(13);
    for class in [Class::TemporalDag, Class::TemporalCpdag, Class::TemporalPag] {
        for accepted in [false, true] {
            for sustained in [false, true] {
                for inference in [InferenceMode::Frequentist, bayesian()] {
                    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                        let label = format!(
                            "{class:?} accepted={accepted} sustained={sustained} {} {}",
                            inference_name(&inference),
                            suite_name(suite)
                        );
                        let result = builder(class, accepted, data.clone())
                            .query(query(sustained))
                            .inference(inference.clone())
                            .refute(suite)
                            .bootstrap_replicates(0)
                            .build()
                            .unwrap_or_else(|error| panic!("{label}: {error}"))
                            .run(&context)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        assert_cell(
                            &result,
                            &inference,
                            suite,
                            &truth,
                            &label,
                            CellStatus::Licensed,
                        );
                    }
                }
            }
        }
    }
}

/// A planted violation is detected: when the outcome responds to the treatment of the next
/// period (an effect of the treatment of the next period), the pre-trend check flags it; the clean law passes it.
#[test]
fn temporal_conditional_pre_trend_detects_planted_anticipation() {
    let truth = truth();
    let clean = series(&truth);
    let context = ExecutionContext::for_tests(14);
    let run = |data: TimeSeriesData| {
        builder(Class::TemporalDag, false, data)
            .query(query(false))
            .refute(RefuteSuite::Full)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&context)
            .unwrap()
    };
    let pre_trend = |result: &StudyResult| {
        result
            .refutations
            .iter()
            .find(|report| report.refuter.as_ref() == "target.pre_trend")
            .map(|report| report.passed)
            .expect("the pre-trend check ran")
    };
    assert!(pre_trend(&run(clean.clone())), "the clean law passes the pre-trend check");
    // y_t also responds to x_t, the treatment of the period the pulse is read from: the
    // outcome moves before the lagged treatment can have acted.
    let storage = clean.storage();
    let column = |id: u32| -> Vec<f64> {
        match &storage.columns()[id as usize] {
            antecedent_data::OwnedColumn::Float64(column) => column.values.to_vec(),
            _ => unreachable!("float columns"),
        }
    };
    let (x, y, w, r) = (column(0), column(1), column(2), column(3));
    let anticipated: Vec<f64> = (0..N).map(|t| y[t] + 1.5 * x[t]).collect();
    let planted = TimeSeriesData::from_f64_columns(
        [
            ("x", x.as_slice()),
            ("y", anticipated.as_slice()),
            ("w", w.as_slice()),
            ("r", r.as_slice()),
        ],
        1,
    )
    .unwrap();
    assert!(!pre_trend(&run(planted)), "the planted anticipation is detected");
}

/// The retained plan runs every class and structure with the builder dropped.
#[test]
fn temporal_conditional_cells_execute_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let data = series(&truth);
    let context = ExecutionContext::for_tests(15);
    for class in [Class::TemporalDag, Class::TemporalCpdag, Class::TemporalPag] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let builder = builder(class, accepted, data.clone())
                    .query(query(false))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let study = builder.clone().build().unwrap();
                let prepared = study.prepare(&context).unwrap();
                drop(builder);
                drop(study);
                assert!(prepared.plan().logical.record.estimator.is_some());
                let result = prepared
                    .estimate_series(&data, &context)
                    .unwrap_or_else(|error| panic!("{class:?} accepted={accepted}: {error}"));
                let (low, high) = effect_ends(&result);
                assert!((low - truth.effect).abs() < truth.tolerance, "{class:?}");
                assert!((high - truth.effect).abs() < truth.tolerance, "{class:?}");
                assert!(!result.refutations.is_empty(), "{class:?} {suite:?}");
            }
        }
    }
}

/// One-atom posterior over `(x, y, w, r)`: `w_t -> y_t`, `r_t -> x_t` and the lag-1 edge
/// `x -> y`; the arrowhead into `x` makes the lagged treatment visible to a PAG atom.
fn posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let n = 4;
    let contemporaneous =
        set_edge(set_edge(0, n, 2, 1, true), n, 3, 0, true).to_word().expect("one word");
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_conditional_validation_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![1_u64 << 1])
    .unwrap()
}

/// `TemporalConditionalPulse` and `TemporalConditionalSustained` x `TemporalDag` (none, cheap and
/// full), `TemporalCpdag` and `TemporalPag` (cheap and full) x graph_posterior x Frequentist and
/// Bayesian: the mixed value is the marginal conditional effect, the refuters refit the whole
/// graph-posterior procedure, and no mass is lost.
#[test]
fn temporal_conditional_graph_posterior_cells_match_the_marginal_effect() {
    let truth = truth();
    let data = series(&truth);
    let context = ExecutionContext::for_tests(16);
    for (kind, suites) in [
        (
            GraphPosteriorAtomKind::Dag,
            vec![RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full],
        ),
        (GraphPosteriorAtomKind::Cpdag, vec![RefuteSuite::Cheap, RefuteSuite::Full]),
        (GraphPosteriorAtomKind::Pag, vec![RefuteSuite::Cheap, RefuteSuite::Full]),
    ] {
        for sustained in [false, true] {
            for inference in [InferenceMode::Frequentist, bayesian()] {
                for &suite in &suites {
                    let class = match kind {
                        GraphPosteriorAtomKind::Dag => "TemporalDag",
                        GraphPosteriorAtomKind::Cpdag => "TemporalCpdag",
                        _ => "TemporalPag",
                    };
                    let label = format!(
                        "{class} graph_posterior sustained={sustained} {} {}",
                        inference_name(&inference),
                        suite_name(suite)
                    );
                    let result = Study::series(data.clone())
                        .graph_posterior(posterior(kind))
                        .query(query(sustained))
                        .inference(inference.clone())
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap_or_else(|error| panic!("{label}: {error}"))
                        .run(&context)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    assert_cell(&result, &inference, suite, &truth, &label, CellStatus::Licensed);
                    let mixture = result.structural_response.as_ref().expect("atom mixture");
                    assert!(mixture.unidentified_mass.abs() < 1e-12, "{label}");
                }
            }
        }
    }
}

/// The retained plan of a graph-posterior conditional study runs with the builder dropped.
#[test]
fn temporal_conditional_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let data = series(&truth);
    let context = ExecutionContext::for_tests(17);
    for kind in
        [GraphPosteriorAtomKind::Dag, GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag]
    {
        let builder = Study::series(data.clone())
            .graph_posterior(posterior(kind))
            .query(query(false))
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0);
        let study = builder.clone().build().unwrap();
        let prepared = study.prepare(&context).unwrap();
        drop(builder);
        drop(study);
        assert!(prepared.plan().logical.record.estimator.is_some(), "{kind:?}");
        let result = prepared
            .estimate_series(&data, &context)
            .unwrap_or_else(|error| panic!("{kind:?}: {error}"));
        let (low, high) = effect_ends(&result);
        assert!((low - truth.effect).abs() < truth.tolerance, "{kind:?}");
        assert!((high - truth.effect).abs() < truth.tolerance, "{kind:?}");
        assert!(!result.refutations.is_empty(), "{kind:?}");
    }
}

/// The result of a graph-posterior temporal conditional study round-trips through the artifact
/// encode and the independent consumer: the identification's node ids sit inside the declared
/// window namespace (variable and offset) the artifact carries.
#[test]
fn temporal_conditional_graph_posterior_results_encode_and_are_consumed() {
    let truth = truth();
    let data = series(&truth);
    let context = ExecutionContext::for_tests(18);
    for kind in
        [GraphPosteriorAtomKind::Dag, GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag]
    {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap] {
            let builder = Study::series(data.clone())
                .graph_posterior(posterior(kind))
                .query(query(false))
                .refute(suite)
                .bootstrap_replicates(0);
            let study = builder.clone().build().unwrap();
            let prepared = study.prepare(&context).unwrap();
            drop(builder);
            drop(study);
            let result = match prepared.estimate_series(&data, &context) {
                Ok(result) => result,
                // The class posteriors refuse the one-shot no-refuter click; they are covered
                // at the other validation depths.
                Err(_)
                    if suite == RefuteSuite::None
                        && !matches!(kind, GraphPosteriorAtomKind::Dag) =>
                {
                    continue;
                }
                Err(error) => panic!("{kind:?} {suite:?}: {error}"),
            };
            let bytes = prepared
                .encode_contracted_result(&result, "temporal-conditional-graph-posterior", &context)
                .unwrap_or_else(|error| panic!("{kind:?} {suite:?}: {error}"));
            let consumed = antecedent_io::consume_analysis_result(&bytes)
                .unwrap_or_else(|error| panic!("{kind:?} {suite:?}: {error}"));
            assert!(consumed.contract.is_some(), "{kind:?} {suite:?}");
        }
    }
}
