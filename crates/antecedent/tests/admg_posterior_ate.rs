//! ADMG graph-posterior `AverageEffect`: functional.effect per atom, posterior mix.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp)]

use antecedent::{
    BayesianConfig, CellStatus, InferenceMode, RefuteSuite, SemanticApplicability,
    StructuralAggregationPolicy, StructuralWeightBasis, Study,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, IdentificationStatus, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, adjacency_mask_from_admg, admg_from_adjacency_mask,
    dag_from_adjacency_mask, mask_is_dag, set_edge,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap()
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(columns: &[&str], name: &str) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap())
}

fn admg_from_pin(pin: &serde_json::Value) -> Admg {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut admg = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        admg.insert_directed(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        admg.insert_bidirected(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    admg
}

fn query_from_pin(pin: &serde_json::Value) -> AverageEffectQuery {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let treatment = node(&columns, pin["query"]["treatment"].as_str().unwrap());
    let outcome = node(&columns, pin["query"]["outcome"].as_str().unwrap());
    AverageEffectQuery::with_levels(
        VariableId::from_raw(treatment.raw()),
        VariableId::from_raw(outcome.raw()),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

/// Directed cycle: not an ADMG, so the atom stays unidentified.
///
/// A 3-node bow `T→Y` plus `T↔Y` cannot be packed (both-bits already mean
/// bidirected). Invalid construction is the same unidentified-mass rule DAG
/// graph-posterior uses when `dag_from_adjacency_mask` fails.
fn cyclic_mask(n: usize) -> u64 {
    let mut mask = 0_u64;
    mask = set_edge(mask, n, 0, 1, true);
    if n >= 3 {
        mask = set_edge(mask, n, 1, 2, true);
        mask = set_edge(mask, n, 2, 0, true);
    } else {
        mask = set_edge(mask, n, 1, 0, true);
    }
    mask
}

fn admg_posterior(n: usize, weights: &[f64], masks: &[u64]) -> GraphPosterior {
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks.to_vec(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("admg_posterior_ate"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

fn run(
    data: TabularData,
    gp: GraphPosterior,
    query: AverageEffectQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
) -> antecedent::StudyResult {
    let ctx = ExecutionContext::for_tests(1);
    Study::tabular(data)
        .graph_posterior(gp)
        .query(query)
        .refute(refute)
        .inference(inference)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
}

#[test]
fn admg_frontdoor_mask_is_not_a_dag() {
    let pin = pin();
    let admg = admg_from_pin(&pin);
    assert!(admg.has_bidirected());
    let mask = adjacency_mask_from_admg(&admg).unwrap();
    assert!(!mask_is_dag(mask, admg.node_count()));
    assert!(
        dag_from_adjacency_mask(mask, admg.node_count()).is_err(),
        "bidirected ADMG packing must not coerce to Dag"
    );
    let back = admg_from_adjacency_mask(mask, admg.node_count()).unwrap();
    assert!(back.has_bidirected());
}

#[test]
fn admg_posterior_average_effect_none_runs() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let query = query_from_pin(&pin);
    let identified = admg_from_pin(&pin);
    let n = identified.node_count();
    let identified_mask = adjacency_mask_from_admg(&identified).unwrap();
    assert!(admg_from_adjacency_mask(cyclic_mask(n), n).is_err());
    let expected = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    let tolerance = pin["frequentist"]["absolute_tolerance"].as_f64().unwrap();
    let weights = [0.8, 0.2];
    let gp = admg_posterior(n, &weights, &[identified_mask, cyclic_mask(n)]);
    assert_eq!(gp.atom_kind, GraphPosteriorAtomKind::Admg);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        let result = run(data.clone(), gp.clone(), query.clone(), inference, RefuteSuite::None);
        assert_eq!(result.logical_plan.identifier.as_deref(), Some("general.id"));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
        assert_eq!(result.support_status.unwrap().as_str(), "licensed");
        let mixture = result.structural_response.as_ref().expect("ADMG posterior mixture");
        assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert!(
            (mixture.unidentified_mass - 0.2).abs() < 1e-12,
            "unidentified posterior mass must be retained: {}",
            mixture.unidentified_mass
        );
        assert_eq!(mixture.atoms.len(), 2);
        let identified_w =
            mixture.atoms.iter().find(|atom| atom.value.is_some()).map(|atom| atom.weight).unwrap();
        let unidentified_w =
            mixture.atoms.iter().find(|atom| atom.value.is_none()).map(|atom| atom.weight).unwrap();
        assert!(
            (identified_w - 0.8).abs() < 1e-12,
            "outer atoms keep posterior weights, not completion mass: {identified_w}"
        );
        assert!((unidentified_w - 0.2).abs() < 1e-12);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert!(
            result.posterior.is_none(),
            "conditional atom posterior must not describe unidentified mass"
        );

        assert!(result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                && d.message.contains("completion enumeration is not posterior probability")
        }));
        assert!(result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.graph_posterior.admg_functional"
                && d.message.contains("functional.effect")
                && d.message.contains("single graphs")
        }));
        assert!(
            has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean)
                || result.estimate.ate.is_finite(),
            "one identified front-door atom should mix as a scalar"
        );
        assert!(
            (result.estimate.ate - expected).abs() < tolerance.max(0.02),
            "identified-atom mixture should recover the front-door pin {expected}, got {}",
            result.estimate.ate
        );
        assert!(
            !result.diagnostics.iter().any(|d| {
                d.code.as_ref() == "estimate.class_graph_posterior"
                    || d.message.contains("bayesian.gcomp")
            }),
            "ADMG atoms must not take the class / gcomp envelope"
        );
    }
}

fn has_policy(result: &antecedent::StudyResult, policy: StructuralAggregationPolicy) -> bool {
    result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
            && d.message.contains(policy.as_str())
    })
}

#[test]
fn admg_graph_posterior_inspect_and_execute_agree_on_support_status() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let query = query_from_pin(&pin);
    let identified = admg_from_pin(&pin);
    let n = identified.node_count();
    let identified_mask = adjacency_mask_from_admg(&identified).unwrap();
    // Inspect encodes every posterior atom; keep only encodable ADMG masks here.
    let gp = admg_posterior(n, &[1.0], &[identified_mask]);
    let ctx = ExecutionContext::for_tests(1);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let builder = Study::tabular(data.clone())
                .graph_posterior(gp.clone())
                .query(query.clone())
                .refute(suite)
                .inference(inference.clone())
                .bootstrap_replicates(0);
            let inspected = builder.clone().inspect().unwrap();
            assert_eq!(
                inspected.support_status,
                Some(CellStatus::Licensed),
                "{inference:?} {suite:?} inspect must publish licensed"
            );
            let preflight = builder.clone().capability().unwrap();
            assert_eq!(
                preflight.applicability,
                SemanticApplicability::Licensed,
                "{inference:?} {suite:?} capability must agree with inspect"
            );
            let built = builder.build().unwrap();
            assert_eq!(
                built.inspect().unwrap().support_status,
                inspected.support_status,
                "built Study::inspect must reuse the same support_status"
            );
            let result = built.run(&ctx).unwrap();
            assert_eq!(
                result.support_status, inspected.support_status,
                "{inference:?} {suite:?} execute must match cheap inspect"
            );
        }
    }
}

#[test]
fn admg_posterior_average_effect_cheap_and_full_run() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let query = query_from_pin(&pin);
    let identified = admg_from_pin(&pin);
    let n = identified.node_count();
    let identified_mask = adjacency_mask_from_admg(&identified).unwrap();
    let weights = [0.8, 0.2];
    let gp = admg_posterior(n, &weights, &[identified_mask, cyclic_mask(n)]);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = run(data.clone(), gp.clone(), query.clone(), inference.clone(), suite);
            assert_eq!(result.support_status.unwrap().as_str(), "licensed");
            assert_eq!(result.logical_plan.identifier.as_deref(), Some("general.id"));
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
            assert!(
                !result.refutations.is_empty(),
                "{inference:?} {suite:?} must run per-atom functional refuters"
            );
            assert!(
                result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "refute.envelope.admg_posterior"
                        && d.message.contains("single graphs")
                        && d.message.contains("posterior")
                }),
                "{inference:?} {suite:?} must name ADMG atom vs posterior mixing"
            );
            assert!(
                !result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "refute.envelope.class_posterior"
                        || d.code.as_ref() == "estimate.class_graph_posterior"
                }),
                "{inference:?} {suite:?} must not route through class envelope refuters"
            );
            assert!(
                has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean),
                "{inference:?} {suite:?} front-door atom should scalar-mix refuters"
            );
            let mixture = result.structural_response.as_ref().expect("ADMG posterior mixture");
            assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);
            assert!(
                result.diagnostics.iter().all(|d| {
                    d.code.as_ref() != "refute.envelope.effect_mixture"
                        || d.message.contains("every contributing atom passes")
                }),
                "{inference:?} {suite:?} outer mix must follow StructuralAggregationPolicy"
            );
        }
    }
}
