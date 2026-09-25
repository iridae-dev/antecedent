//! Builder-independent evidence for the frequentist DAG graph-posterior effect route.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_prob::InferenceDiagnostics;

fn data(outcome_shift: f64) -> TabularData {
    let n = 320usize;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(outcome_shift + 2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

fn posterior() -> GraphPosterior {
    // Atom effects are 3 (direct-only), 2 (Z-adjusted), and unidentified.
    // The unidentified reverse-causal atom retains weight 0.2 in the envelope.
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    let weights = Arc::<[f64]>::from([0.5, 0.3, 0.2]);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0] + weights[1];
    marginals[3] = weights[2];
    marginals[6] = weights[1];
    marginals[7] = weights[1];
    GraphPosterior::new(
        3,
        weights.clone(),
        vec![direct, adjusted, unidentified],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_algorithm("known_truth_fixture")
}

#[test]
fn frequentist_dag_graph_posterior_effect_is_sealed_and_preserves_graph_mass() {
    let base = data(0.0);
    let gp = posterior();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(913);

    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        let builder = Study::tabular(base.clone())
            .graph_posterior(gp.clone())
            .query(query.clone())
            .refute(suite)
            .inference(InferenceMode::Frequentist)
            .build()
            .unwrap();
        let one_shot = builder.run(&ctx).unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);

        let plan = prepared
            .checked_graph_posterior_effect_info()
            .expect("prepared graph-posterior operation retained");
        assert_eq!(plan.query, query);
        assert_eq!(plan.estimator.as_str(), "linear.adjustment.ate");
        assert_eq!(plan.validation, suite);
        assert_eq!(plan.weights.as_ref(), &[0.5, 0.3, 0.2]);
        assert_eq!(plan.graph_keys.as_ref(), gp.graph_keys.as_ref());
        assert_eq!(plan.identified.len(), 3);
        assert_eq!(plan.bootstrap_replicates, 199);

        let result = prepared.estimate(&base, &ctx).unwrap();
        assert!(result.estimate.ate.is_nan(), "atom estimands disagree; scalar must be withheld");
        let mixture = result.structural_response.as_ref().expect("structural graph mixture");
        let set = mixture.identified_set.as_ref().expect("identified set retained");
        assert!((set.lower[0] - 2.0).abs() < 0.06);
        assert!((set.upper[0] - 3.0).abs() < 0.06);
        assert!((mixture.identified_mass - 0.8).abs() < 1e-12);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);
        assert!(one_shot.estimate.ate.is_nan());
        assert!(
            one_shot
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code.as_ref() == "exec.identify.cached" }),
            "one-shot run must execute its retained prepared plan"
        );

        let shifted = prepared.refresh(data(4.0), &ctx).unwrap();
        assert!(shifted.estimate.ate.is_nan());
        let refreshed_plan = prepared
            .checked_graph_posterior_effect_info()
            .expect("checked graph-posterior plan remains after refresh");
        assert_eq!(refreshed_plan.graph_keys, plan.graph_keys);
        assert_eq!(refreshed_plan.weights, plan.weights);
        let shifted_mix = shifted.structural_response.as_ref().unwrap();
        let shifted_set = shifted_mix.identified_set.as_ref().unwrap();
        assert!((shifted_set.lower[0] - set.lower[0]).abs() < 1e-12);
        assert!((shifted_set.upper[0] - set.upper[0]).abs() < 1e-12);
        assert!((shifted_mix.identified_mass - mixture.identified_mass).abs() < 1e-12);
        assert!((shifted_mix.unidentified_mass - mixture.unidentified_mass).abs() < 1e-12);

        let artifact = prepared
            .encode_contracted_result(&shifted, "checked-graph-posterior-effect", &ctx)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_graph_posterior_effect_operation"
        }));
        assert!(!consumed.acceptance.accepts_as_verified_program());
    }
}
