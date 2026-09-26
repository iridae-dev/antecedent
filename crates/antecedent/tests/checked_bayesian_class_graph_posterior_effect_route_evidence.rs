//! Builder-independent evidence for Bayesian average effects over CPDAG/PAG
//! graph posteriors.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

use std::sync::Arc;

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SlotAvailability, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_io::consume_analysis_result;
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics};

/// Balanced confounded blocks with `Y = shift + 2T + slope * Z + eps`.
fn blocked_columns(outcome_shift: f64, confounder_slope: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
                outcome.push(outcome_shift + 2.0 * t + confounder_slope * z + epsilon);
            }
        }
    }
    (treatment, outcome, confounder)
}

fn blocked_data(outcome_shift: f64, confounder_slope: f64) -> TabularData {
    let (t, y, z) = blocked_columns(outcome_shift, confounder_slope);
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Same columns in a different order: a different semantic schema.
fn reordered_data() -> TabularData {
    let (t, y, z) = blocked_columns(0.0, 2.0);
    TabularData::from_f64_columns([("z", z.as_slice()), ("y", y.as_slice()), ("t", t.as_slice())])
        .unwrap()
}

/// CPDAG atoms with effects 3 (direct-only, confounded), 2 (`Z`-adjusted),
/// and an unidentified reverse-causal atom carrying weight 0.2.
fn cpdag_posterior() -> GraphPosterior {
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
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
    .with_algorithm("known_truth_fixture")
}

/// `Z -> T -> Y` makes `T -> Y` visible; the reverse atom keeps 0.2 unidentified.
fn visible_pag_posterior() -> GraphPosterior {
    let identified = set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true);
    let reverse = set_edge(0, 3, 1, 0, true);
    let weights = Arc::<[f64]>::from([0.8, 0.2]);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0];
    marginals[3] = weights[1];
    marginals[6] = weights[0];
    GraphPosterior::new(
        3,
        weights.clone(),
        vec![identified, reverse],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        InferenceDiagnostics::analytic("visible_pag_known_truth"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_masks(vec![0, 0])
    .unwrap()
    .with_algorithm("visible_pag_fixture")
}

#[test]
fn bayesian_class_graph_posterior_effect_is_sealed_across_atom_kinds_and_suites() {
    run_on_large_stack(bayesian_class_graph_posterior_effect_body);
}

fn bayesian_class_graph_posterior_effect_body() {
    let ctx = ExecutionContext::for_tests(31_337);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let config = BayesianConfig::conjugate().n_draws(128).prior_scale(100.0);
    let fixtures = [
        ("Cpdag", cpdag_posterior(), blocked_data(0.0, 2.0), blocked_data(4.0, 2.0), 2usize),
        ("Pag", visible_pag_posterior(), blocked_data(0.0, 0.0), blocked_data(4.0, 0.0), 1usize),
    ];
    for (graph_class, posterior, base, shifted, class_atoms) in fixtures {
        for (label, suite) in [
            ("none", RefuteSuite::None),
            ("cheap", RefuteSuite::Cheap),
            ("full", RefuteSuite::Full),
        ] {
            let coordinate =
                format!("AverageEffect:{graph_class}:graph_posterior:Bayesian:{label}");
            let builder = Study::tabular(base.clone())
                .graph_posterior(posterior.clone())
                .query(query.clone())
                .inference(InferenceMode::Bayesian(config.clone()))
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let one_shot = builder.run(&ctx).unwrap();
            let mut prepared = builder.prepare(&ctx).unwrap();
            drop(builder);

            assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
            match &prepared.contract().unwrap().reasoning.support {
                SlotAvailability::Available(slot) => {
                    assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                }
                other => panic!("{coordinate}: support was not available: {other:?}"),
            }
            let plan = prepared
                .checked_class_graph_posterior_effect_info()
                .expect("Bayesian class graph-posterior operation retained");
            assert!(prepared.has_checked_class_graph_posterior_effect_operation());
            assert!(prepared.checked_bayesian_graph_posterior_ate_info().is_none());
            assert_eq!(plan.query, query, "{coordinate}");
            assert!(!plan.conditional, "{coordinate}");
            assert!(plan.inference.starts_with("bayesian:"), "{coordinate}: {}", plan.inference);
            assert_eq!(plan.estimator.as_str(), "bayesian.gcomp", "{coordinate}");
            assert_eq!(plan.validation, suite, "{coordinate}");
            assert_eq!(plan.graph_keys.as_ref(), posterior.graph_keys.as_ref());
            assert_eq!(plan.weights.as_ref(), posterior.weights.as_ref());
            assert_eq!(plan.class_atom_count, class_atoms, "{coordinate}");
            assert_eq!(
                plan.identified.iter().filter(|f| **f == GraphIdentFlag::Unidentified).count(),
                1,
                "{coordinate}: the reverse-causal atom mass stays unidentified"
            );

            let result = prepared.estimate(&base, &ctx).unwrap();
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.gcomp"));
            let mixture = result.structural_response.as_ref().expect("structural graph mixture");
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9, "{coordinate}");
            assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{coordinate}");
            if graph_class == "Pag" {
                assert!(
                    (result.estimate.ate - 2.0).abs() < 0.2,
                    "{coordinate}: {}",
                    result.estimate.ate
                );
                // Unidentified sample mass keeps the aggregate posterior
                // withheld; the retained atom posteriors stay conditional.
                assert!(
                    result.posterior.is_some()
                        || result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "estimate.graph_posterior.posterior_withheld"
                        }),
                    "{coordinate}: withheld aggregate posterior must be disclosed"
                );
            } else {
                assert!(
                    result.estimate.ate.is_nan(),
                    "{coordinate}: atom estimands disagree; scalar must be withheld"
                );
                let set = mixture.identified_set.as_ref().expect("identified set retained");
                assert!((set.lower[0] - 2.0).abs() < 0.2, "{coordinate}: {}", set.lower[0]);
                assert!((set.upper[0] - 3.0).abs() < 0.2, "{coordinate}: {}", set.upper[0]);
            }
            assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None, "{coordinate}");
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                "{coordinate}: the click reuses the retained completion envelopes"
            );
            assert_eq!(one_shot.estimate.ate.is_nan(), result.estimate.ate.is_nan());
            if !one_shot.estimate.ate.is_nan() {
                assert!((one_shot.estimate.ate - result.estimate.ate).abs() < 0.2, "{coordinate}");
            }
            assert!(
                one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                "{coordinate}: one-shot run must execute its retained prepared plan"
            );

            let refreshed = prepared.refresh(shifted.clone(), &ctx).unwrap();
            let retained = prepared.checked_class_graph_posterior_effect_info().unwrap();
            assert_eq!(retained.graph_keys, plan.graph_keys);
            assert_eq!(retained.weights, plan.weights);
            assert_eq!(retained.inference, plan.inference);
            let refreshed_mix = refreshed.structural_response.as_ref().unwrap();
            assert!((refreshed_mix.unidentified_mass - 0.2).abs() < 1e-9, "{coordinate}");
            assert_eq!(refreshed.estimate.ate.is_nan(), result.estimate.ate.is_nan());
            if !refreshed.estimate.ate.is_nan() {
                assert!((refreshed.estimate.ate - 2.0).abs() < 0.2, "{coordinate}");
            }
            assert!(
                prepared.refresh(reordered_data(), &ctx).is_err(),
                "{coordinate}: reordered columns must be refused"
            );

            let artifact = prepared
                .encode_contracted_result(
                    &refreshed,
                    &format!("bayesian-class-graph-posterior-{label}"),
                    &ctx,
                )
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_class_graph_posterior_effect_operation"
                }),
                "{coordinate}: {:?}",
                consumed.acceptance.unresolved
            );
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-bayesian-class-graph-posterior-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
