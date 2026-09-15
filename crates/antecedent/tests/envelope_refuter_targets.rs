//! R-3 (1.9 cell review): multi-atom refuters test each atom against its own estimate.
//!
//! Before 1.9 every envelope atom's refuters compared that atom's perturbation
//! refits with the *pooled* mixture effect, so a perfectly stable atom whose
//! effect differs from the pooled value "failed" `data.subset`, and the
//! unanimous-pass mixture report failed with it. These fixtures have atoms whose
//! effects genuinely differ; the companion unit test
//! `envelope_refuters_compare_each_atom_with_its_own_estimate` (in
//! `crates/antecedent/src/analysis/execute/mod.rs`) shows an atom whose own
//! refits do not reproduce its reported estimate still fails.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

fn cpdag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cpdag_ate_envelope/expected.json"
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

/// Columns are `[t, y, z]`: `z -> y`, `t -> y`, `z — t`.
fn fixture_cpdag() -> Cpdag {
    let (t, y, z) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(z, y).unwrap();
    cpdag.insert_directed(t, y).unwrap();
    cpdag.insert_undirected(z, t).unwrap();
    cpdag
}

fn ate() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == id)
        .unwrap_or_else(|| panic!("{id} report missing; got {:?}", result.refutations))
}

fn assert_atom_targeted_mixture(
    result: &StudyResult,
    refuter: &str,
    expected_original: f64,
    tol: f64,
) {
    let subset = report(result, refuter);
    assert!(
        subset.passed,
        "a stable atom whose effect differs from the pooled value must pass {refuter}: {subset:?}"
    );
    assert!(
        (subset.original_ate - expected_original).abs() < tol,
        "mixed original_ate must be the mass-weighted mean of the per-atom estimates compared \
         ({expected_original}), got {}",
        subset.original_ate
    );
    assert!(
        result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "refute.envelope.effect_mixture"
                && d.message.contains("against that atom's own estimate")
        }),
        "the mixture diagnostic must state what each atom was compared against"
    );
}

#[test]
fn cpdag_heterogeneous_completions_pass_data_subset_against_own_estimates() {
    let pin = cpdag_pin();
    let effects: Vec<f64> = pin["identification"]["completion_effects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!((effects[0] - effects[1]).abs() > 0.1, "fixture atoms must disagree");
    let data = expand_contingency(&pin);
    let freq = Study::tabular(data.clone())
        .graph(fixture_cpdag())
        .query(ate())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let pooled = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    assert!((freq.estimate.ate - pooled).abs() < 1e-12);
    // Equal completion weights: the mixed original is (0.40 + 0.52) / 2.
    assert_atom_targeted_mixture(&freq, "data.subset", 0.5 * (effects[0] + effects[1]), 1e-9);

    let bayes = Study::tabular(data)
        .graph(fixture_cpdag())
        .query(ate())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_scale(10.0),
        ))
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    // Linear-refit stability checks are not applicable to bayesian.gcomp; the
    // E-value is computed from each atom's own posterior mean, and the mixed
    // original is their mass-weighted mean (the envelope posterior mean).
    assert!(bayes.refutations.iter().all(|r| r.refuter.as_ref() != "data.subset"));
    assert_atom_targeted_mixture(&bayes, "sensitivity.evalue", bayes.estimate.ate, 1e-9);
}

/// Known-truth static graph posterior: atom effects 3 (unadjusted) and 2
/// (Z-adjusted), weights 0.5 / 0.3, plus a 0.2 unidentified atom.
fn known_truth_graph_posterior() -> GraphPosterior {
    let weights = [0.5, 0.3, 0.2];
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
}

fn known_truth_data(n: usize) -> TabularData {
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..(n / 16) {
        for (zv, tv, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                t.push(tv);
                z.push(zv);
                y.push(2.0 * tv + 2.0 * zv + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

#[test]
fn graph_posterior_heterogeneous_atoms_pass_data_subset_against_own_estimates() {
    let data = known_truth_data(320);
    let result = Study::tabular(data)
        .graph_posterior(known_truth_graph_posterior())
        .query(ate())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    assert!((result.estimate.ate - 2.625).abs() < 1e-8);
    // Atom effects 3 and 2 at contributing weights 0.5 / 0.3: 2.625 is the
    // mass-weighted mean of what the atoms were compared against, even though
    // neither atom was compared with 2.625 itself.
    assert_atom_targeted_mixture(&result, "data.subset", 2.625, 1e-8);
}
