//! Path-restricted natural effects and the natural direct effect on a supplied PAG, run through
//! the compiled route and checked against the effect computed from the structural model that
//! generated the table.
//!
//! A MAG fixes a latent projection only up to its m-separation model, and a path-specific
//! effect is not a function of that model, so the route reads each MAG completion through
//! every ADMG that shares its model. An edge that hides no latent common cause (a visible
//! edge, or a edge whose confounder cannot reach the effect) leaves one functional and a
//! point; an edge that may hide a confounder the effect is sensitive to is refused, since
//! no functional is right for every model the PAG represents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use antecedent::analysis::{RetainedOperation, StudyBuilder};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, NestedCounterfactualQuery, PathSpecificEffectQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Pag};
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

/// Columns from the cells of a joint over `N` binary variables, `ROWS` rows in all.
fn table_from_joint<const N: usize>(
    names: [&str; N],
    joint: impl Fn([usize; N]) -> f64,
) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); N];
    for cell in 0..(1_usize << N) {
        let mut levels = [0_usize; N];
        for (position, level) in levels.iter_mut().enumerate() {
            *level = (cell >> (N - 1 - position)) & 1;
        }
        let count = (ROWS * joint(levels)).round() as usize;
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level as f64, count));
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

fn pag_of(nodes: u32, directed: &[(u32, u32)]) -> Pag {
    let mut pag = Pag::with_variables(nodes);
    for &(from, to) in directed {
        pag.insert_directed(d(from), d(to)).unwrap();
    }
    pag
}

/// `W -> T`; `T -> A`, `T -> B`; `A, B -> Y`: columns `w, t, a, b, y`. Every edge out of `T`
/// and into `Y` is visible because `W` points into `T` and is not adjacent to them.
mod parented_routes {
    use super::bernoulli;

    pub fn p_w(w: usize) -> f64 {
        bernoulli(0.4, w)
    }
    pub fn p_t(t: usize, w: usize) -> f64 {
        bernoulli(0.3 + 0.4 * w as f64, t)
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

    /// Observational law of `(w, t, a, b, y)`.
    pub fn joint(levels: [usize; 5]) -> f64 {
        let [w, t, a, b, y] = levels;
        p_w(w) * p_t(t, w) * p_a(a, t) * p_b(b, t) * p_y(y, a, b)
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

/// `C -> T`; `T -> M`, `T -> Y`; `W -> M`; `M -> Y`: columns `c, w, t, m, y`.
mod mediated_with_parents {
    use super::bernoulli;

    pub fn p_c(c: usize) -> f64 {
        bernoulli(0.45, c)
    }
    pub fn p_w(w: usize) -> f64 {
        bernoulli(0.35, w)
    }
    pub fn p_t(t: usize, c: usize) -> f64 {
        bernoulli(0.25 + 0.5 * c as f64, t)
    }
    pub fn p_m(m: usize, t: usize, w: usize) -> f64 {
        bernoulli(0.15 + 0.4 * t as f64 + 0.3 * w as f64, m)
    }
    pub fn p_y(y: usize, t: usize, m: usize) -> f64 {
        bernoulli(0.1 + 0.2 * t as f64 + 0.3 * m as f64 + 0.2 * (t * m) as f64, y)
    }

    /// Observational law of `(c, w, t, m, y)`.
    pub fn joint(levels: [usize; 5]) -> f64 {
        let [c, w, t, m, y] = levels;
        p_c(c) * p_w(w) * p_t(t, c) * p_m(m, t, w) * p_y(y, t, m)
    }

    /// `E[Y(1, M(0))] - E[Y(0, M(0))]`, from the equations.
    pub fn natural_direct_effect() -> f64 {
        let mut total = 0.0;
        for m in 0..2 {
            let reference: f64 = (0..2).map(|w| p_w(w) * p_m(m, 0, w)).sum();
            total += reference * (p_y(1, 1, m) - p_y(1, 0, m));
        }
        total
    }
}

fn parented_data() -> TabularData {
    table_from_joint(["w", "t", "a", "b", "y"], parented_routes::joint)
}

fn mediated_data() -> TabularData {
    table_from_joint(["c", "w", "t", "m", "y"], mediated_with_parents::joint)
}

/// Every mark decided, every edge into `Y` and out of `T` visible.
fn parented_pag() -> Pag {
    pag_of(5, &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
}

/// `T -> A`, `T -> B`, `A -> Y`, `B -> Y` with nothing pointing into `T`: the treatment edges
/// are invisible, so a latent common cause of `T` and `A` cannot be excluded.
fn unparented_pag() -> Pag {
    pag_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])
}

/// Every mark decided; `W -> M` makes `M -> Y` visible and `C -> T` makes the edges out of
/// `T` visible.
fn mediated_pag() -> Pag {
    pag_of(5, &[(0, 2), (2, 3), (2, 4), (1, 3), (3, 4)])
}

fn path_through_a() -> CausalQuery {
    CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(1), v(4)).with_path_nodes([v(2)]))
}

fn natural_direct_query() -> CausalQuery {
    CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(v(2), v(3), v(4)).unwrap())
}

fn study(data: TabularData, pag: Pag, query: CausalQuery) -> StudyBuilder {
    Study::tabular(data).graph(pag).query(query).refute(RefuteSuite::None)
}

fn ate(result: &antecedent::StudyResult) -> f64 {
    result.estimate.as_effect().expect("an effect estimate").ate
}

#[test]
fn a_pag_with_visible_edges_reproduces_the_structural_effect() {
    let truth = parented_routes::effect_through_a();
    assert!(truth.abs() > 0.03, "the reference effect must be material: {truth}");
    let result = study(parented_data(), parented_pag(), path_through_a())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap();
    assert!(
        (ate(&result) - truth).abs() < 2e-3,
        "PAG path-specific effect {} against the structural effect {truth}",
        ate(&result)
    );
    // `W -> T` may hide a confounder, but it does not reach the effect: one functional.
    assert!(result.structural_response.is_none(), "one functional is a point, not a set");
}

#[test]
fn a_bootstrap_interval_accompanies_the_pag_point() {
    let result = study(parented_data(), parented_pag(), path_through_a())
        .bootstrap_replicates(30)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(42))
        .unwrap();
    let effect = result.estimate.as_effect().expect("an effect estimate");
    assert!(
        effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0) > 0,
        "{:?}",
        effect.uncertainty
    );
}

#[test]
fn the_bayesian_pag_posterior_centres_on_the_structural_effect() {
    let truth = parented_routes::effect_through_a();
    let result = study(parented_data(), parented_pag(), path_through_a())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(43))
        .unwrap();
    assert!(
        (ate(&result) - truth).abs() < 6e-3,
        "posterior mean {} against the structural effect {truth}",
        ate(&result)
    );
}

/// The natural direct effect over a PAG whose candidates are all identified: the district
/// formula of the candidate with `W <-> M` and the edge formula of the others describe the
/// same population effect when the data carry no hidden cause, so the set they span is
/// narrow and contains the structural effect.
#[test]
fn the_natural_direct_effect_over_a_pag_lies_in_its_identified_set() {
    let truth = mediated_with_parents::natural_direct_effect();
    assert!(truth.abs() > 0.03, "the reference effect must be material: {truth}");
    let result = study(mediated_data(), mediated_pag(), natural_direct_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(44))
        .unwrap();
    assert!(
        (ate(&result) - truth).abs() < 3e-3,
        "natural direct effect {} against the structural effect {truth}",
        ate(&result)
    );
    if let Some(mixture) = result.structural_response.as_ref() {
        let set = mixture.identified_set.as_ref().expect("an identified set");
        assert!(
            set.lower[0] - 3e-3 <= truth && truth <= set.upper[0] + 3e-3,
            "the structural effect {truth} lies in the identified set [{}, {}]",
            set.lower[0],
            set.upper[0]
        );
    }
}

fn assert_refused_as_not_identified(builder: StudyBuilder, seed: u64) {
    let Ok(study) = builder.build() else { return };
    let error = study.run(&ExecutionContext::for_tests(seed)).unwrap_err();
    assert_eq!(error.reason_code(), Some("effect_not_identified"), "{error}");
}

/// The same table and effect that a CPDAG identifies: the PAG cannot rule out `T <-> A`, which
/// leaves the kernel of `A` with a hedge, so there is no functional to report.
#[test]
fn an_invisible_treatment_edge_refuses_the_effect() {
    let data = table_from_joint(["t", "a", "b", "y"], |[t, a, b, y]| {
        let p_t = bernoulli(0.5, t);
        let p_a = bernoulli(0.15 + 0.4 * t as f64, a);
        let p_b = bernoulli(0.2 + 0.5 * t as f64, b);
        p_t * p_a * p_b * parented_routes::p_y(y, a, b)
    });
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]),
    );
    assert_refused_as_not_identified(
        study(data, unparented_pag(), query).bootstrap_replicates(0),
        45,
    );
}

/// `M -> Y` is invisible when `T` is adjacent to `Y` and nothing else points into `M`, so a
/// mediator-outcome confounder cannot be excluded and the natural direct effect has a
/// recanting district in one of the ADMGs the PAG stands for.
#[test]
fn an_invisible_mediator_outcome_edge_refuses_the_natural_direct_effect() {
    let pag = pag_of(5, &[(0, 2), (2, 3), (2, 4), (3, 4)]);
    assert_refused_as_not_identified(
        study(mediated_data(), pag, natural_direct_query()).bootstrap_replicates(0),
        46,
    );
}

#[test]
fn a_pag_path_specific_effect_under_cheap_validation_runs_the_unlicensed_refuter_state() {
    let result = Study::tabular(parented_data())
        .graph(parented_pag())
        .query(path_through_a())
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .and_then(|built| built.run(&ExecutionContext::for_tests(47)))
        .unwrap();
    assert_eq!(result.support_status, Some(antecedent::CellStatus::Refused));
    assert!(!result.refutations.is_empty());
    assert!(result.refutations.iter().all(|r| r.refuter.starts_with("target.")));
}

fn route_cases() -> Vec<(&'static str, StudyBuilder, TabularData)> {
    vec![
        ("pag_path", study(parented_data(), parented_pag(), path_through_a()), parented_data()),
        (
            "pag_nested",
            study(mediated_data(), mediated_pag(), natural_direct_query()),
            mediated_data(),
        ),
    ]
}

#[test]
fn prepared_pag_handles_equal_a_fresh_run() {
    for (name, builder, data) in route_cases() {
        let ctx = ExecutionContext::for_tests(51);
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
        assert!(prepared.plan().lowered_program_identity().is_some(), "{name}");
    }
}

#[test]
fn contracted_pag_results_round_trip_through_the_consumer() {
    for (name, builder, data) in route_cases() {
        let ctx = ExecutionContext::for_tests(52);
        let study =
            builder.bootstrap_replicates(0).build().unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{name}: {e}"));
        let result = prepared.estimate(&data, &ctx).unwrap();
        let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
        let label = format!("path-specific-{name}");
        let bytes = prepared
            .encode_contracted_result(&refreshed, &label, &ctx)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let again = prepared.encode_contracted_result(&refreshed, &label, &ctx).unwrap();
        assert_eq!(bytes, again, "{name} encoding is deterministic");
        let consumed = consume_analysis_result(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{name} artifact result");
        assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{name} refresh");
    }
}
