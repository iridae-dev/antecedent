//! Graph posteriors that list the same graph more than once.
//!
//! A supplied [`GraphPosterior`] may repeat a DAG (for example an equal-weight
//! bootstrap or MCMC ensemble passed sample by sample). A repeated graph must
//! contribute its combined sample mass exactly once: listing `A` twice at
//! weight `w` is the same posterior as listing `A` once at `2w`.
//!
//! Static atoms over `[t, y, z]`:
//!
//! | atom | structure | effect |
//! | --- | --- | --- |
//! | A | `T -> Y` (no adjustment) | unadjusted |
//! | B | `Z -> T`, `Z -> Y`, `T -> Y` (adjust `Z`) | adjusted |
//! | C | `Y -> T` | unidentified |
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent::discovery::GraphPosterior;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, ResponseFunctional,
    ResponseQuery, ResponseUncertainty, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::set_edge;
use antecedent_prob::InferenceDiagnostics;

const N: usize = 400;

fn direct() -> u64 {
    set_edge(0, 3, 0, 1, true)
}

fn adjusted() -> u64 {
    set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true)
}

fn unidentified() -> u64 {
    set_edge(0, 3, 1, 0, true)
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `Z ~ Bern(1/2)`, `T | Z ~ Bern(1/4 + Z/2)`, `Y = 2T + 2Z + ε`; the
/// unadjusted and `Z`-adjusted atoms target 3 and 2.
fn binary_data(seed: u64) -> TabularData {
    let mut u = lcg(seed);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..N {
        let zi = f64::from(u8::from(u() < 0.5));
        let ti = f64::from(u8::from(u() < 0.25 + 0.5 * zi));
        y.push(2.0 * ti + 2.0 * zi + (u() - 0.5));
        t.push(ti);
        z.push(zi);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Continuous `T = Z/2 + ν`, `Y = 1 + 2T + 1.5Z + ε`.
fn continuous_data(seed: u64) -> TabularData {
    let mut u = lcg(seed);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..N {
        let zi = 2.0 * u() - 1.0;
        let ti = 0.5 * zi + (u() - 0.5);
        y.push(1.0 + 2.0 * ti + 1.5 * zi + 0.5 * (u() - 0.5));
        t.push(ti);
        z.push(zi);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn posterior(masks: &[u64], weights: &[f64]) -> GraphPosterior {
    GraphPosterior::new(
        3,
        weights.to_vec(),
        masks.to_vec(),
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("repeated_atoms"),
        0,
    )
    .unwrap()
}

/// `[A, A, B, C]` at equal weight: 2/3 of the identified mass on `A`.
fn repeated() -> GraphPosterior {
    posterior(&[direct(), direct(), adjusted(), unidentified()], &[0.25; 4])
}

/// The same posterior with `A` listed once at its combined weight.
fn coalesced() -> GraphPosterior {
    posterior(&[direct(), adjusted(), unidentified()], &[0.5, 0.25, 0.25])
}

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ))
}

fn run(
    data: &TabularData,
    gp: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
) -> StudyResult {
    Study::tabular(data.clone())
        .graph_posterior(gp)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap()
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400).prior_scale(1_000_000.0))
}

/// `(identified, unidentified)` from the Frequentist envelope diagnostic.
fn envelope_masses(result: &StudyResult) -> (f64, f64) {
    let message = &result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.graph_posterior.envelope")
        .expect("graph-posterior envelope diagnostic")
        .message;
    let field = |name: &str| -> f64 {
        let start = message.find(name).unwrap() + name.len();
        message[start..].split([',', ' ', ';']).next().unwrap().parse().unwrap()
    };
    (field("identified_mass="), field("unidentified_mass="))
}

#[test]
fn repeated_static_atom_frequentist_ate_is_the_mass_weighted_mixture() {
    let data = binary_data(0x2A70);
    let only = |mask: u64| {
        run(&data, posterior(&[mask], &[1.0]), ate_query(), InferenceMode::Frequentist).estimate.ate
    };
    let (ate_a, ate_b) = (only(direct()), only(adjusted()));
    assert!((ate_a - ate_b).abs() > 0.5, "atoms must identify different effects");

    let result = run(&data, repeated(), ate_query(), InferenceMode::Frequentist);
    // Direct and adjusted atoms disagree on estimand identity, so the scalar is
    // withheld under GraphDependentAtoms; coalescing repeated listings must not
    // change masses or the identified set.
    assert!(result.estimate.ate.is_nan(), "scalar ate withheld under GraphDependentAtoms");
    assert!(result.estimate.se_analytic.is_nan());
    let structural = result.structural_response.as_ref().unwrap();
    let set = structural.identified_set.as_ref().unwrap();
    assert_eq!(set.lower.len(), 1);
    assert!((set.lower[0] - ate_a.min(ate_b)).abs() < 1e-10);
    assert!((set.upper[0] - ate_a.max(ate_b)).abs() < 1e-10);
    assert!(structural.conditional_on_identified.is_none());
    let (identified, unidentified) = envelope_masses(&result);
    assert!((identified - 0.75).abs() < 1e-12, "identified mass {identified}");
    assert!((unidentified - 0.25).abs() < 1e-12, "unidentified mass {unidentified}");
    assert!((identified + unidentified - 1.0).abs() < 1e-12);

    let reference = run(&data, coalesced(), ate_query(), InferenceMode::Frequentist);
    assert!(reference.estimate.ate.is_nan());
    let ref_set = reference.structural_response.as_ref().unwrap().identified_set.as_ref().unwrap();
    assert!((set.lower[0] - ref_set.lower[0]).abs() < 1e-12);
    assert!((set.upper[0] - ref_set.upper[0]).abs() < 1e-12);
    assert_eq!(structural.atoms.len(), reference.structural_response.as_ref().unwrap().atoms.len());
}

#[test]
fn repeated_static_atom_bayesian_ate_is_the_mass_weighted_mixture() {
    let data = binary_data(0x2A71);
    let result = run(&data, repeated(), ate_query(), bayesian());
    let reference = run(&data, coalesced(), ate_query(), bayesian());
    let post = result.posterior.as_ref().unwrap();
    let ref_post = reference.posterior.as_ref().unwrap();
    assert!((post.summaries.mean[0] - ref_post.summaries.mean[0]).abs() < 1e-12);
    assert!((post.summaries.sd[0] - ref_post.summaries.sd[0]).abs() < 1e-12);
    assert!((post.unidentified_mass - 0.25).abs() < 1e-12);
    let expected = (2.0 * 3.0 + 2.0) / 3.0;
    assert!(
        (post.summaries.mean[0] - expected).abs() < 0.2,
        "Bayesian mixture mean {} vs population mixture {expected}",
        post.summaries.mean[0]
    );
}

fn scalar(value: &ResponseValue) -> f64 {
    let ResponseValue::Scalar(v) = value else { panic!("scalar response expected") };
    *v
}

#[test]
fn repeated_static_atom_response_is_the_mass_weighted_mixture() {
    let data = continuous_data(0x2A72);
    let query =
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.5))]),
        }));
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let only = |mask: u64| {
            let result = run(&data, posterior(&[mask], &[1.0]), query.clone(), inference.clone());
            scalar(result.structural_response.unwrap().conditional_on_identified.as_ref().unwrap())
        };
        let (value_a, value_b) = (only(direct()), only(adjusted()));
        assert!((value_a - value_b).abs() > 0.1, "atoms must identify different responses");
        let expected = (2.0 * value_a + value_b) / 3.0;

        let result = run(&data, repeated(), query.clone(), inference.clone());
        let structural = result.structural_response.as_ref().unwrap();
        let mixed = scalar(structural.conditional_on_identified.as_ref().unwrap());
        assert!(
            (mixed - expected).abs() < 1e-10,
            "repeated-atom response {mixed} != 2/3·{value_a} + 1/3·{value_b} = {expected}"
        );
        assert!((structural.identified_mass - 0.75).abs() < 1e-12);
        assert!((structural.unidentified_mass - 0.25).abs() < 1e-12);
        assert!(structural.unevaluable_mass.abs() < 1e-12);
        let listed: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        assert!((listed - 1.0).abs() < 1e-12);
        // Every listed sample of an evaluable graph carries its value.
        for atom in &structural.atoms {
            assert_eq!(atom.value.is_some(), atom.graph_key != unidentified(), "{atom:?}");
        }

        let reference = run(&data, coalesced(), query.clone(), inference.clone());
        let ref_mixed = scalar(
            reference
                .structural_response
                .as_ref()
                .unwrap()
                .conditional_on_identified
                .as_ref()
                .unwrap(),
        );
        assert!((mixed - ref_mixed).abs() < 1e-12);
        if matches!(inference, InferenceMode::Frequentist) {
            let se = |r: &StudyResult| match r.response.as_ref().unwrap().uncertainty {
                ResponseUncertainty::Scalar { standard_error, .. } => standard_error,
                ref other => panic!("joint-IF scalar interval expected, got {other:?}"),
            };
            let (se, reference_se) = (se(&result), se(&reference));
            assert!(
                se.is_finite() && (se - reference_se).abs() < 1e-12,
                "se={se} reference={reference_se}"
            );
        }
    }
}

#[test]
fn one_graph_key_for_different_masks_is_refused() {
    let mut gp = posterior(&[direct(), adjusted()], &[0.5, 0.5]);
    gp.graph_keys = Arc::from([7_u64, 7]);
    let error = Study::tabular(binary_data(0x2A73))
        .graph_posterior(gp)
        .query(ate_query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap_err();
    assert!(error.to_string().contains("graph key"), "{error}");
}
