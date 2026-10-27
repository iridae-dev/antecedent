//! Path-restricted natural effects and the natural direct effect on a supplied ADMG, run
//! through the compiled route and checked against the effect computed directly from the
//! structural model that generated the table (latent cause included).
//!
//! The reference sums the structural equations over the latent cause with the treatment read
//! at one level along the selected edges and at the control level along the others. The
//! route never sees those equations: it identifies the effect from the graph (the district
//! formula, since the latent cause reaches the outcome's ancestors) and plugs the observed
//! table into the identified functional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use antecedent::analysis::RetainedOperation;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, NestedCounterfactualQuery, PathSpecificEffectQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::TemporalAdmg;
use antecedent_graph::{Admg, Cpdag, DenseNodeId};
use antecedent_io::consume_analysis_result;

const ROWS: f64 = 100_000.0;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn bernoulli(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

/// Columns from the cells of a joint over four binary variables, `ROWS` rows in all.
fn table_from_joint(names: [&str; 4], joint: impl Fn([usize; 4]) -> f64) -> TabularData {
    let mut columns: [Vec<f64>; 4] = Default::default();
    for cell in 0..16_usize {
        let levels = [(cell >> 3) & 1, (cell >> 2) & 1, (cell >> 1) & 1, cell & 1];
        let count = (ROWS * joint(levels)).round() as usize;
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level as f64, count));
        }
    }
    TabularData::from_f64_columns([
        (names[0], columns[0].as_slice()),
        (names[1], columns[1].as_slice()),
        (names[2], columns[2].as_slice()),
        (names[3], columns[3].as_slice()),
    ])
    .unwrap()
}

/// `U -> W`, `U -> Y`; `W -> T`; `W, T -> M`; `W, T, M, U -> Y`.
mod covariate_confounded {
    use super::bernoulli;

    pub fn p_u(u: usize) -> f64 {
        bernoulli(0.4, u)
    }
    pub fn p_w(w: usize, u: usize) -> f64 {
        bernoulli(0.3 + 0.4 * u as f64, w)
    }
    pub fn p_t(t: usize, w: usize) -> f64 {
        bernoulli(0.25 + 0.4 * w as f64, t)
    }
    pub fn p_m(m: usize, w: usize, t: usize) -> f64 {
        bernoulli(0.2 + 0.3 * w as f64 + 0.4 * t as f64, m)
    }
    pub fn p_y(y: usize, w: usize, t: usize, m: usize, u: usize) -> f64 {
        bernoulli(0.05 + 0.15 * w as f64 + 0.15 * t as f64 + 0.25 * m as f64 + 0.2 * u as f64, y)
    }

    /// Observational law of `(w, t, m, y)`.
    pub fn joint(levels: [usize; 4]) -> f64 {
        let [w, t, m, y] = levels;
        (0..2).map(|u| p_u(u) * p_w(w, u) * p_t(t, w) * p_m(m, w, t) * p_y(y, w, t, m, u)).sum()
    }

    /// `E[Y(t = 1 on the direct edge, M(t = 0))] - E[Y(t = 0)]`, from the equations.
    pub fn natural_direct_effect() -> f64 {
        let mut total = 0.0;
        for u in 0..2 {
            for w in 0..2 {
                for m in 0..2 {
                    let weight = p_u(u) * p_w(w, u) * p_m(m, w, 0);
                    total += weight * (p_y(1, w, 1, m, u) - p_y(1, w, 0, m, u));
                }
            }
        }
        total
    }
}

/// `U -> A`, `U -> Y`; `T -> A`, `T -> B`; `A, B -> Y`: columns `t, a, b, y`.
mod routed_through_a {
    use super::bernoulli;

    pub fn p_u(u: usize) -> f64 {
        bernoulli(0.4, u)
    }
    pub fn p_t(t: usize) -> f64 {
        bernoulli(0.5, t)
    }
    pub fn p_a(a: usize, t: usize, u: usize) -> f64 {
        bernoulli(0.15 + 0.4 * t as f64 + 0.3 * u as f64, a)
    }
    pub fn p_b(b: usize, t: usize) -> f64 {
        bernoulli(0.2 + 0.5 * t as f64, b)
    }
    pub fn p_y(y: usize, a: usize, b: usize, u: usize) -> f64 {
        bernoulli(0.05 + 0.3 * a as f64 + 0.2 * b as f64 + 0.25 * u as f64, y)
    }

    /// Observational law of `(t, a, b, y)`.
    pub fn joint(levels: [usize; 4]) -> f64 {
        let [t, a, b, y] = levels;
        p_t(t) * (0..2).map(|u| p_u(u) * p_a(a, t, u) * p_b(b, t) * p_y(y, a, b, u)).sum::<f64>()
    }

    /// `E[Y(t = 1 into A, t = 0 into B)] - E[Y(t = 0)]`, from the equations.
    pub fn effect_through_a() -> f64 {
        let mut total = 0.0;
        for u in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    let routed = p_a(a, 1, u);
                    let reference = p_a(a, 0, u);
                    total += p_u(u) * p_b(b, 0) * (routed - reference) * p_y(1, a, b, u);
                }
            }
        }
        total
    }
}

/// `T -> A`, `T -> B`, `A -> Y`, `B -> Y` without a latent cause: columns `t, a, b, y`.
mod markovian_routed {
    use super::bernoulli;

    pub fn p_t(t: usize) -> f64 {
        bernoulli(0.5, t)
    }
    pub fn p_a(a: usize, t: usize) -> f64 {
        bernoulli(0.15 + 0.4 * t as f64, a)
    }
    pub fn p_b(b: usize, t: usize) -> f64 {
        bernoulli(0.2 + 0.5 * t as f64, b)
    }
    pub fn p_y(y: usize, a: usize, b: usize) -> f64 {
        bernoulli(0.05 + 0.3 * a as f64 + 0.2 * b as f64 + 0.25 * (a * b) as f64, y)
    }

    /// Observational law of `(t, a, b, y)`.
    pub fn joint(levels: [usize; 4]) -> f64 {
        let [t, a, b, y] = levels;
        p_t(t) * p_a(a, t) * p_b(b, t) * p_y(y, a, b)
    }

    /// `E[Y(t = 1 into A, t = 0 into B)] - E[Y(t = 0)]`, from the equations.
    pub fn effect_through_a() -> f64 {
        let mut total = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                total += p_b(b, 0) * (p_a(a, 1) - p_a(a, 0)) * p_y(1, a, b);
            }
        }
        total
    }
}

fn markovian_data() -> TabularData {
    table_from_joint(["t", "a", "b", "y"], markovian_routed::joint)
}

/// Every edge of `T -> A -> Y <- B <- T` compelled: the class has one completion.
fn compelled_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

/// `A -> Y <- B` is a v-structure; `T - A` and `T - B` are undirected: three completions.
fn collider_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(4);
    graph.insert_directed(d(1), d(3)).unwrap();
    graph.insert_directed(d(2), d(3)).unwrap();
    graph.insert_undirected(d(0), d(1)).unwrap();
    graph.insert_undirected(d(0), d(2)).unwrap();
    graph
}

fn covariate_admg() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph.insert_bidirected(d(0), d(3)).unwrap();
    graph
}

fn routed_admg(extra_bidirected: Option<(u32, u32)>) -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph.insert_bidirected(d(1), d(3)).unwrap();
    if let Some((a, b)) = extra_bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn covariate_data() -> TabularData {
    table_from_joint(["w", "t", "m", "y"], covariate_confounded::joint)
}

fn routed_data() -> TabularData {
    table_from_joint(["t", "a", "b", "y"], routed_through_a::joint)
}

fn natural_direct_query() -> CausalQuery {
    CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(v(1), v(2), v(3)).unwrap())
}

fn path_through_a() -> CausalQuery {
    CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]).with_max_len(4),
    )
}

fn base(data: TabularData, graph: Admg, query: CausalQuery) -> antecedent::analysis::StudyBuilder {
    Study::tabular(data).graph(graph).query(query).refute(RefuteSuite::None)
}

fn ate(result: &antecedent::StudyResult) -> f64 {
    result.estimate.as_effect().expect("an effect estimate").ate
}

#[test]
fn natural_direct_effect_with_covariate_confounding_matches_the_structural_effect() {
    let truth = covariate_confounded::natural_direct_effect();
    assert!(truth.abs() > 0.05, "the reference effect must be material: {truth}");
    let study = base(covariate_data(), covariate_admg(), natural_direct_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(11)).unwrap();
    assert!(
        (ate(&result) - truth).abs() < 2e-3,
        "natural direct effect {} against the structural effect {truth}",
        ate(&result)
    );
}

#[test]
fn effect_through_a_confounded_child_uses_the_district_formula() {
    let truth = routed_through_a::effect_through_a();
    assert!(truth.abs() > 0.03, "the reference effect must be material: {truth}");
    let study = base(routed_data(), routed_admg(None), path_through_a())
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(12)).unwrap();
    assert!(
        (ate(&result) - truth).abs() < 2e-3,
        "path-specific effect {} against the structural effect {truth}",
        ate(&result)
    );
}

#[test]
fn a_bootstrap_interval_accompanies_the_frequentist_point() {
    let study = base(routed_data(), routed_admg(None), path_through_a())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(13)).unwrap();
    let effect = result.estimate.as_effect().expect("an effect estimate");
    assert!(
        effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0) > 0,
        "{:?}",
        effect.uncertainty
    );
}

#[test]
fn the_bayesian_posterior_centres_on_the_structural_effect() {
    let truth = routed_through_a::effect_through_a();
    let study = base(routed_data(), routed_admg(None), path_through_a())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(14)).unwrap();
    assert!(
        (ate(&result) - truth).abs() < 6e-3,
        "posterior mean {} against the structural effect {truth}",
        ate(&result)
    );
}

/// Refused at build or at run, never estimated.
fn assert_refused_by_identification(builder: antecedent::analysis::StudyBuilder, seed: u64) {
    let Ok(study) = builder.build() else { return };
    let error = study.run(&ExecutionContext::for_tests(seed)).unwrap_err();
    assert!(
        !error.to_string().contains("unsupported data/graph/query combination"),
        "the refusal must come from identification, got: {error}"
    );
}

#[test]
fn a_recanting_district_is_refused_not_estimated() {
    // `A <-> B` puts the selected child and the complementary child in one district.
    assert_refused_by_identification(
        base(routed_data(), routed_admg(Some((1, 2))), path_through_a()).bootstrap_replicates(0),
        15,
    );
}

#[test]
fn mediator_outcome_confounding_refuses_the_natural_direct_effect() {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph.insert_bidirected(d(2), d(3)).unwrap();
    assert_refused_by_identification(
        base(covariate_data(), graph, natural_direct_query()).bootstrap_replicates(0),
        16,
    );
}

#[test]
fn a_class_with_one_completion_reproduces_the_structural_effect() {
    let truth = markovian_routed::effect_through_a();
    assert!(truth.abs() > 0.03, "the reference effect must be material: {truth}");
    let study = Study::tabular(markovian_data())
        .graph(compelled_cpdag())
        .query(path_through_a())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(17)).unwrap();
    assert!(
        (ate(&result) - truth).abs() < 2e-3,
        "class path-specific effect {} against the structural effect {truth}",
        ate(&result)
    );
    assert!(result.structural_response.is_none(), "one completion is a point, not a set");
}

#[test]
fn completions_that_disagree_publish_the_identified_set() {
    let truth = markovian_routed::effect_through_a();
    let study = Study::tabular(markovian_data())
        .graph(collider_cpdag())
        .query(path_through_a())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(18)).unwrap();
    let mixture = result.structural_response.as_ref().expect("a partially identified class");
    let set = mixture.identified_set.as_ref().expect("an identified set over the completions");
    let (lower, upper) = (set.lower[0], set.upper[0]);
    assert!(
        lower - 2e-3 <= truth && truth <= upper + 2e-3,
        "the generating completion's effect {truth} lies in the identified set [{lower}, {upper}]"
    );
    assert!(upper - lower > 5e-3, "the completions disagree: [{lower}, {upper}]");
    assert!(
        mixture.unidentified_mass > 0.0,
        "the completion with no directed path through A carries unidentified mass"
    );
}

#[test]
fn the_natural_direct_effect_over_a_class_matches_its_one_completion() {
    // `T -> M -> Y` and `T -> Y` with every edge compelled.
    let mut graph = Cpdag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    let p_m = |m: usize, t: usize| bernoulli(0.25 + 0.5 * t as f64, m);
    let p_y = |y: usize, t: usize, m: usize| {
        bernoulli(0.1 + 0.2 * t as f64 + 0.3 * m as f64 + 0.2 * (t * m) as f64, y)
    };
    let mut columns: [Vec<f64>; 3] = Default::default();
    for cell in 0..8_usize {
        let (t, m, y) = ((cell >> 2) & 1, (cell >> 1) & 1, cell & 1);
        let count = (ROWS * 0.5 * p_m(m, t) * p_y(y, t, m)).round() as usize;
        for (column, level) in columns.iter_mut().zip([t, m, y]) {
            column.extend(std::iter::repeat_n(level as f64, count));
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("m", columns[1].as_slice()),
        ("y", columns[2].as_slice()),
    ])
    .unwrap();
    // E[Y(1, M(0))] - E[Y(0, M(0))], from the equations.
    let truth: f64 = (0..2).map(|m| p_m(m, 0) * (p_y(1, 1, m) - p_y(1, 0, m))).sum();
    let query = CausalQuery::NestedCounterfactual(
        NestedCounterfactualQuery::new(v(0), v(1), v(2)).unwrap(),
    );
    let study = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(19)).unwrap();
    assert!(
        (ate(&result) - truth).abs() < 2e-3,
        "natural direct effect {} against the structural effect {truth}",
        ate(&result)
    );
}

/// The identified set over a class's completions is covered by the Imbens-Manski interval.
#[test]
fn the_class_interval_covers_the_identified_set_frequentist() {
    let truth = markovian_routed::effect_through_a();
    let study = Study::tabular(markovian_data())
        .graph(collider_cpdag())
        .query(path_through_a())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(60)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(21)).unwrap();
    let mixture = result.structural_response.as_ref().expect("a partially identified class");
    let set = mixture.identified_set.as_ref().expect("identified set");
    let interval = mixture.identified_set_interval.as_ref().expect("shared-replicate interval");
    assert!(interval.lower <= set.lower[0] + 1e-9 && interval.upper >= set.upper[0] - 1e-9);
    assert!(interval.lower <= truth && truth <= interval.upper, "{interval:?} vs {truth}");
}

#[test]
fn the_class_band_covers_the_identified_set_bayesian() {
    let truth = markovian_routed::effect_through_a();
    let study = Study::tabular(markovian_data())
        .graph(collider_cpdag())
        .query(path_through_a())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(22)).unwrap();
    let mixture = result.structural_response.as_ref().expect("a partially identified class");
    let interval = mixture.identified_set_interval.as_ref().expect("posterior band");
    assert!(interval.lower <= truth && truth <= interval.upper, "{interval:?} vs {truth}");
}

/// One study per route cell: path-specific and natural direct effect on Admg, Cpdag and the
/// projection of a TemporalAdmg.
fn route_cases() -> Vec<(&'static str, antecedent::analysis::StudyBuilder, TabularData)> {
    let nested = || natural_direct_query();
    let mut cases = Vec::new();
    let temporal = |graph: &Admg| TemporalAdmg::from_static_contemporaneous(graph).unwrap();
    cases.push((
        "admg_path",
        base(routed_data(), routed_admg(None), path_through_a()),
        routed_data(),
    ));
    cases.push((
        "admg_nested",
        base(covariate_data(), covariate_admg(), nested()),
        covariate_data(),
    ));
    cases.push((
        "temporal_admg_path",
        Study::tabular(routed_data())
            .graph(temporal(&routed_admg(None)))
            .query(path_through_a())
            .refute(RefuteSuite::None),
        routed_data(),
    ));
    cases.push((
        "temporal_admg_nested",
        Study::tabular(covariate_data())
            .graph(temporal(&covariate_admg()))
            .query(nested())
            .refute(RefuteSuite::None),
        covariate_data(),
    ));
    cases.push((
        "cpdag_path",
        Study::tabular(markovian_data())
            .graph(compelled_cpdag())
            .query(path_through_a())
            .refute(RefuteSuite::None),
        markovian_data(),
    ));
    let mut chain = Cpdag::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        chain.insert_directed(d(from), d(to)).unwrap();
    }
    cases.push((
        "cpdag_nested",
        Study::tabular(covariate_data()).graph(chain).query(nested()).refute(RefuteSuite::None),
        covariate_data(),
    ));
    cases
}

#[test]
fn prepared_handles_run_their_static_cross_world_operation_and_equal_a_fresh_run() {
    for (name, builder, data) in route_cases() {
        let ctx = ExecutionContext::for_tests(31);
        let study =
            builder.bootstrap_replicates(0).build().unwrap_or_else(|e| panic!("{name}: {e}"));
        let fresh = study.clone().run(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
        let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld),
            "{name} retains its static cross-world operation"
        );
        let first = prepared.estimate(&data, &ctx).unwrap();
        let second = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(first.effect().to_bits(), fresh.effect().to_bits(), "{name} prepared vs fresh");
        assert_eq!(first.effect().to_bits(), second.effect().to_bits(), "{name} rerun");
        assert_eq!(
            first.lowered_program_identity(),
            prepared.plan().lowered_program_identity(),
            "{name} lowered identity"
        );
        assert!(prepared.plan().lowered_program_identity().is_some(), "{name}");
    }
}

#[test]
fn contracted_results_round_trip_through_the_consumer() {
    for (name, builder, data) in route_cases() {
        let ctx = ExecutionContext::for_tests(32);
        let study =
            builder.bootstrap_replicates(0).build().unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
        let result = prepared.estimate(&data, &ctx).unwrap();
        let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
        assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{name} refresh");
        let bytes = prepared
            .encode_contracted_result(&refreshed, &format!("path-specific-{name}"), &ctx)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let again = prepared
            .encode_contracted_result(&refreshed, &format!("path-specific-{name}"), &ctx)
            .unwrap();
        assert_eq!(bytes, again, "{name} encoding is deterministic");
        let consumed = consume_analysis_result(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{name} artifact result");
    }
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
