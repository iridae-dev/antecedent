//! The identified-atom mixture envelope against exact rational moments.
//!
//! `oracle_data/envelope_reference.py` (frozen in the JSON beside it) pools atoms that share a
//! graph key, drops zero-weight and unidentified atoms from the mixture and computes, with
//! `Fraction` arithmetic on the exact binary draws, the mixture mean and the mixture variance
//! (within-component population variance plus between-component disagreement) of the empirical
//! component laws, and the unidentified mass that is retained rather than redistributed. The
//! published summaries must carry those moments, the status must follow the retained mass, and
//! the input order of the atoms must not matter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::IdentificationStatus;
use antecedent_estimate::{EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope};
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics, WeightedGraphSamples};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/envelope_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn run(case: &Value, order: &[usize]) -> antecedent_estimate::CausalPosterior {
    let atoms = case["atoms"].as_array().unwrap();
    let weights: Vec<f64> = order.iter().map(|&i| atoms[i]["weight"].as_f64().unwrap()).collect();
    let flags: Vec<GraphIdentFlag> = order
        .iter()
        .map(|&i| {
            if atoms[i]["identified"].as_bool().unwrap() {
                GraphIdentFlag::Identified
            } else {
                GraphIdentFlag::Unidentified
            }
        })
        .collect();
    let keys: Vec<u64> = order.iter().map(|&i| atoms[i]["key"].as_u64().unwrap()).collect();
    let graphs = WeightedGraphSamples::new(weights, flags, keys).unwrap();
    let per_graph: Vec<GraphEffectDraws> = case["draws"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, values)| GraphEffectDraws {
            graph_key: key.parse().unwrap(),
            effect_draws: Arc::from(
                values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(),
            ),
        })
        .collect();
    aggregate_effect_envelope(
        &graphs,
        &per_graph,
        InferenceDiagnostics::analytic("reference"),
        EnvelopeOptions::default(),
    )
    .unwrap()
}

#[test]
fn the_published_moments_are_the_exact_mixture_moments() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let order: Vec<usize> = (0..case["atoms"].as_array().unwrap().len()).collect();
        let posterior = run(&case, &order);
        let (mean, sd) = (posterior.summaries.mean[0], posterior.summaries.sd[0]);
        let (r_mean, r_sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        assert!((mean - r_mean).abs() < 1e-12, "{name} mean {mean} vs {r_mean}");
        assert!((sd / r_sd - 1.0).abs() < 1e-9, "{name} sd {sd} vs {r_sd}");
        let unidentified = case["unidentified_mass"].as_f64().unwrap();
        assert!((posterior.unidentified_mass - unidentified).abs() < 1e-12, "{name} mass");
        let expected = if unidentified > 1e-9 {
            IdentificationStatus::GraphDependent
        } else {
            IdentificationStatus::NonparametricallyIdentified
        };
        assert_eq!(posterior.identification, expected, "{name} status");
    }
}

#[test]
fn the_envelope_does_not_depend_on_the_atom_order() {
    for case in cases() {
        let count = case["atoms"].as_array().unwrap().len();
        let forward: Vec<usize> = (0..count).collect();
        let backward: Vec<usize> = (0..count).rev().collect();
        let (a, b) = (run(&case, &forward), run(&case, &backward));
        assert_eq!(a.draws.values, b.draws.values);
        assert_eq!(a.summaries.mean[0], b.summaries.mean[0]);
    }
}

#[test]
fn renormalizing_over_positive_unidentified_mass_is_refused() {
    let case = &cases()[0];
    let atoms = case["atoms"].as_array().unwrap();
    let graphs = WeightedGraphSamples::new(
        atoms.iter().map(|a| a["weight"].as_f64().unwrap()).collect::<Vec<_>>(),
        atoms
            .iter()
            .map(|a| {
                if a["identified"].as_bool().unwrap() {
                    GraphIdentFlag::Identified
                } else {
                    GraphIdentFlag::Unidentified
                }
            })
            .collect::<Vec<_>>(),
        atoms.iter().map(|a| a["key"].as_u64().unwrap()).collect::<Vec<_>>(),
    )
    .unwrap();
    let per_graph: Vec<GraphEffectDraws> = case["draws"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, values)| GraphEffectDraws {
            graph_key: key.parse().unwrap(),
            effect_draws: Arc::from(
                values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(),
            ),
        })
        .collect();
    assert!(
        aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("reference"),
            EnvelopeOptions { renormalize_identified_only: true },
        )
        .is_err()
    );
}
