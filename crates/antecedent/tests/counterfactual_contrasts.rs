//! A counterfactual over several outcomes, a joint baseline of simultaneous interventions, or
//! nested interventions is answered by one `Study`: every outcome is contrasted from the same
//! abduction, with per-outcome uncertainty, support and homogeneity, and the one-outcome,
//! one-`Set` counterfactual keeps its own result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "test scaffolding builds columns from small row indices"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, InterventionSequence,
    SequencedIntervention, TemporalPolicy, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 300;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `x1`, `x2` unrelated causes of two outcomes, each with its own bounded wiggle:
/// `y1 = 1 + 2·x1 + 3·x2 + e1` and `y2 = x1 − 2·x2 + e2`. Variables `0 = x1`, `1 = x2`,
/// `2 = y1`, `3 = y2`. Moving both causes from 0 to 1 moves `y1` by 5 and `y2` by −1.
fn scm() -> (TabularData, Dag) {
    let x1: Vec<f64> = (0..ROWS).map(|i| (i % 17) as f64 * 0.2).collect();
    let x2: Vec<f64> = (0..ROWS).map(|i| ((i * 7) % 11) as f64 * 0.25).collect();
    let y1: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.2 * (i as f64 * 0.37).sin())
        .collect();
    let y2: Vec<f64> =
        (0..ROWS).map(|i| x1[i] - 2.0 * x2[i] + 0.2 * (i as f64 * 0.91).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("x1", x1.as_slice()),
        ("x2", x2.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 2), (1, 2), (0, 3), (1, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn joint_query() -> CounterfactualQuery {
    let mut query = CounterfactualQuery::new(
        v(2),
        vec![Intervention::set(v(0), Value::f64(1.0)), Intervention::set(v(1), Value::f64(1.0))],
    )
    .with_joint_control([
        Intervention::set(v(0), Value::f64(0.0)),
        Intervention::set(v(1), Value::f64(0.0)),
    ]);
    query.outcomes = Arc::from([v(2), v(3)]);
    query
}

fn study(
    data: &TabularData,
    graph: &Dag,
    query: CounterfactualQuery,
    inference: InferenceMode,
    replicates: u32,
) -> antecedent::Study {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Counterfactual(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn estimate(
    data: &TabularData,
    graph: &Dag,
    query: CounterfactualQuery,
    inference: InferenceMode,
    replicates: u32,
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(31);
    study(data, graph, query, inference, replicates)
        .prepare(&ctx)
        .unwrap()
        .estimate(data, &ctx)
        .unwrap()
}

fn diagnostics<'a>(
    result: &'a StudyResult,
    code: &'a str,
) -> impl Iterator<Item = &'a antecedent_core::Diagnostic> {
    result.diagnostics.iter().filter(move |d| d.code.as_ref() == code)
}

/// Two outcomes under two simultaneous interventions against a joint baseline: one result,
/// one contrast per outcome in the query's order, recovering the known effects.
#[test]
fn two_outcomes_under_two_interventions_recover_the_known_effects() {
    let (data, graph) = scm();
    let result = estimate(&data, &graph, joint_query(), InferenceMode::Frequentist, 0);
    assert!(result.counterfactual().is_none(), "a contrast set is not the single form");
    let set = result.counterfactual_contrasts().expect("a contrast set payload");
    assert_eq!(set.treatments.as_ref(), [v(0), v(1)]);
    assert_eq!(set.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [v(2), v(3)]);
    let (first, second) = (&set.contrasts[0], &set.contrasts[1]);
    assert!((first.mean_ite - 5.0).abs() < 0.25, "y1 effect {}", first.mean_ite);
    assert!((second.mean_ite + 1.0).abs() < 0.25, "y2 effect {}", second.mean_ite);
    for contrast in &set.contrasts {
        assert_eq!(contrast.unit_effects.len(), ROWS);
        assert_eq!(contrast.unit_extrapolative.as_ref().map(|flags| flags.len()), Some(ROWS));
        assert!(contrast.standard_error.is_none(), "a point result forms no standard error");
        assert!(contrast.unit_effect_intervals.is_none());
    }
    // The scalar estimate and `effect()` describe the first outcome.
    assert_eq!(result.effect().to_bits(), first.mean_ite.to_bits());
    assert_eq!(result.estimate().as_effect().unwrap().ate.to_bits(), first.mean_ite.to_bits());
    // Support is read per outcome over both intervened variables.
    let support: Vec<_> = diagnostics(&result, "gcm.counterfactual.support").collect();
    assert_eq!(support.len(), 2);
    assert!(support[0].message.contains("outcome=y1") && support[1].message.contains("outcome=y2"));
    assert!(support[0].message.contains("x1=") && support[0].message.contains("x2="));
    let engine = diagnostics(&result, "gcm.counterfactual").next().unwrap();
    assert!(engine.message.contains("control=[x1=0,x2=0]"), "{}", engine.message);
    assert!(engine.message.contains("active=[x1=1,x2=1]"), "{}", engine.message);
}

/// The Bayesian cell resamples the mechanisms once per draw and contrasts every outcome from that
/// draw's abduction: one posterior column and one set of per-unit intervals per outcome.
#[test]
fn bayesian_contrast_set_has_a_posterior_column_and_unit_intervals_per_outcome() {
    let (data, graph) = scm();
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32));
    let result = estimate(&data, &graph, joint_query(), inference, 0);
    let set = result.counterfactual_contrasts().unwrap();
    let posterior = result.posterior.as_ref().expect("the Bayesian cell publishes its posterior");
    assert_eq!(posterior.summaries.mean.len(), 2, "one effect column per outcome");
    for (index, contrast) in set.contrasts.iter().enumerate() {
        let intervals = contrast.unit_effect_intervals.as_ref().expect("per-unit intervals");
        assert_eq!(intervals.lower.len(), ROWS);
        assert_eq!(intervals.method, "unit_posterior_quantile");
        assert!(contrast.standard_error.is_some_and(|se| se >= 0.0 && se.is_finite()));
        assert!((contrast.mean_ite - posterior.summaries.mean[index]).abs() < 1e-12);
    }
    assert!(diagnostics(&result, "gcm.counterfactual.uncertainty_unavailable").next().is_none());
    assert!(diagnostics(&result, "gcm.counterfactual.bayesian").next().is_some());
}

/// The Frequentist row bootstrap refits once per replicate and reports each outcome's own
/// standard error and accounting.
#[test]
fn bootstrapped_contrast_set_reports_each_outcomes_standard_error() {
    let (data, graph) = scm();
    let result = estimate(&data, &graph, joint_query(), InferenceMode::Frequentist, 24);
    let set = result.counterfactual_contrasts().unwrap();
    for contrast in &set.contrasts {
        assert!(contrast.standard_error.is_some_and(|se| se > 0.0 && se.is_finite()));
    }
    assert_eq!(diagnostics(&result, "gcm.frequentist_bootstrap").count(), 2);
    assert!(diagnostics(&result, "gcm.counterfactual.uncertainty_unavailable").next().is_none());
}

/// A nested intervention (a sequence of hard sets active at the static time) is admitted when the
/// query allows nesting, and contrasts exactly as the flat joint intervention does.
#[test]
fn nested_interventions_contrast_like_the_flat_joint_intervention() {
    let (data, graph) = scm();
    let sequence = |level: f64| {
        Intervention::sequence(InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(v(0), Value::f64(level)),
                TemporalPolicy::pulse(0),
            ),
            SequencedIntervention::new(
                Intervention::set(v(1), Value::f64(level)),
                TemporalPolicy::pulse(0),
            ),
        ]))
    };
    let mut nested = CounterfactualQuery::new(v(2), vec![sequence(1.0)])
        .with_control(sequence(0.0))
        .with_nested(true);
    nested.outcomes = Arc::from([v(2), v(3)]);
    let flat = estimate(&data, &graph, joint_query(), InferenceMode::Frequentist, 0);
    let result = estimate(&data, &graph, nested.clone(), InferenceMode::Frequentist, 0);
    let (nested_set, flat_set) =
        (result.counterfactual_contrasts().unwrap(), flat.counterfactual_contrasts().unwrap());
    for (a, b) in nested_set.contrasts.iter().zip(&flat_set.contrasts) {
        assert_eq!(a.outcome, b.outcome);
        for (x, y) in a.unit_effects.iter().zip(b.unit_effects.iter()) {
            assert!((x - y).abs() < 1e-9);
        }
    }
    // Without `allow_nested` the same worlds are refused rather than flattened silently.
    let refused = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Counterfactual(nested.with_nested(false)))
        .build();
    let ctx = ExecutionContext::for_tests(31);
    assert!(refused.is_err() || refused.unwrap().prepare(&ctx).is_err());
}

/// The single-outcome, single-`Set` counterfactual keeps the single result, and the contrast-set
/// route (reached by allowing nesting on the same query) reads the same effects from it.
#[test]
fn single_outcome_results_are_unchanged_and_agree_with_the_contrast_set_route() {
    let (data, graph) = scm();
    let single = CounterfactualQuery::new(v(2), vec![Intervention::set(v(0), Value::f64(1.0))])
        .with_control_level(0.0);
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
    ] {
        let label = format!("{inference:?}");
        let old = estimate(&data, &graph, single.clone(), inference.clone(), 0);
        let cf = old.counterfactual().expect("the single form keeps the single payload");
        assert!(old.counterfactual_contrasts().is_none(), "{label}");
        assert_eq!(cf.unit_effects.len(), ROWS, "{label}");
        // One support diagnostic and the established wording: the single route is untouched.
        let support: Vec<_> = diagnostics(&old, "gcm.counterfactual.support").collect();
        assert_eq!(support.len(), 1, "{label}");
        assert!(support[0].message.starts_with("observed treatment range="), "{label}");

        let via_set = estimate(&data, &graph, single.clone().with_nested(true), inference, 0);
        let set = via_set.counterfactual_contrasts().expect("nesting selects the contrast set");
        assert_eq!(set.contrasts.len(), 1, "{label}");
        let contrast = &set.contrasts[0];
        assert!((contrast.mean_ite - cf.mean_ite).abs() < 1e-9, "{label}");
        for (a, b) in contrast.unit_effects.iter().zip(cf.unit_effects.iter()) {
            assert!((a - b).abs() < 1e-9, "{label}");
        }
    }
}

/// The result artifact carries every contrast: the body holds the section instead of the
/// single-outcome unit effects, and it survives encode, decode and validation.
#[test]
fn the_result_artifact_round_trips_every_contrast() {
    let (data, graph) = scm();
    let ctx = ExecutionContext::for_tests(31);
    let prepared =
        study(&data, &graph, joint_query(), InferenceMode::Frequentist, 0).prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let set = result.counterfactual_contrasts().unwrap();

    let bytes =
        prepared.encode_contracted_result(&result, "counterfactual-contrasts", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.unit_effects.is_none());
    let section = body.counterfactual_contrasts.as_ref().expect("the contrast section");
    assert_eq!(section.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [2, 3]);
    for (wire, contrast) in section.contrasts.iter().zip(&set.contrasts) {
        assert_eq!(wire.mean.to_bits(), contrast.mean_ite.to_bits());
        assert_eq!(wire.unit_effects.effects.as_slice(), contrast.unit_effects.as_ref());
        assert_eq!(wire.unit_effects.extrapolative.as_ref().map(Vec::len), Some(ROWS));
    }
    // A section that names other outcomes than the query, or another unit count, is refused.
    let mut wrong_outcomes = body.clone();
    wrong_outcomes.counterfactual_contrasts.as_mut().unwrap().contrasts.swap(0, 1);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &wrong_outcomes,
            header.variable_names.clone(),
            "wrong-outcomes"
        )
        .is_err()
    );
    let mut ragged = body.clone();
    ragged.counterfactual_contrasts.as_mut().unwrap().contrasts[1].unit_effects.effects.pop();
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &ragged,
            header.variable_names.clone(),
            "ragged"
        )
        .is_err()
    );
}

fn static_round_trip(
    wire: &antecedent_io::StaticResultWire,
    names: &[String],
) -> Result<antecedent_io::StaticResultWire, antecedent_io::IoError> {
    let payload = antecedent_io::CausalPayloadWire::StaticResult(Box::new(wire.clone()));
    let encoded =
        antecedent_io::encode_causal_payload_artifact(&payload, names.to_vec(), "static")?;
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes)?;
    let (_, header, decoded) = antecedent_io::decode_causal_payload_artifact(&bytes)?;
    assert_eq!(header.variable_names, names);
    let antecedent_io::CausalPayloadWire::StaticResult(back) = decoded else {
        panic!("a static result decodes as a static result");
    };
    Ok(*back)
}

/// Whether the CBOR of `wire` has a map key `contrasts` (a nine-byte text string).
fn cbor_names_contrasts(wire: &antecedent_io::StaticResultWire) -> bool {
    let mut cbor = Vec::new();
    ciborium::into_writer(wire, &mut cbor).unwrap();
    let key = b"\x69contrasts";
    cbor.windows(key.len()).any(|window| window == key)
}

/// The portable static result carries every contrast of a result over several outcomes, validates
/// them on encode and decode, and a single-contrast result keeps the bytes it always had.
#[test]
fn the_static_result_wire_carries_every_contrast_and_leaves_one_contrast_unchanged() {
    let (data, graph) = scm();
    let names: Vec<String> = ["x1", "x2", "y1", "y2"].map(String::from).to_vec();
    let query = CausalQuery::Counterfactual(joint_query());
    for replicates in [0, 12] {
        let result = estimate(&data, &graph, joint_query(), InferenceMode::Frequentist, replicates);
        let set = result.counterfactual_contrasts().expect("a contrast set payload");
        let wire = result.static_result_wire(&query, None).expect("a static result of the set");
        assert_eq!(wire.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [2, 3]);
        for (published, contrast) in wire.contrasts.iter().zip(&set.contrasts) {
            assert_eq!(published.mean.to_bits(), contrast.mean_ite.to_bits());
            assert_eq!(published.standard_error, contrast.standard_error);
            assert_eq!(published.unit_effects.effects.as_slice(), contrast.unit_effects.as_ref());
            assert_eq!(
                published.unit_effects.extrapolative.as_ref().map(Vec::len),
                Some(ROWS),
                "the per-unit support flags are carried"
            );
        }
        // The scalar and the single-form fields are the first outcome's.
        let primary = set.primary();
        assert_eq!(wire.estimate.to_bits(), result.estimate().as_effect().unwrap().ate.to_bits());
        assert_eq!(wire.unit_effects.as_deref(), Some(primary.unit_effects.as_ref()));
        assert_eq!((wire.control_level, wire.active_level), (0.0, 1.0));
        assert!(cbor_names_contrasts(&wire), "a contrast set names its contrasts");

        let back = static_round_trip(&wire, &names).expect("a contrast set round trips");
        assert_eq!(back.contrasts, wire.contrasts);
        assert_eq!(back.unit_effects, wire.unit_effects);
        assert_eq!(back.estimate.to_bits(), wire.estimate.to_bits());

        // Contrasts that are not the query's outcomes in order, that cover other units, or whose
        // first is not the result's single form are refused, not repaired.
        let mut swapped = wire.clone();
        swapped.contrasts.swap(0, 1);
        assert!(static_round_trip(&swapped, &names).is_err());
        let mut truncated = wire.clone();
        truncated.contrasts.pop();
        assert!(static_round_trip(&truncated, &names).is_err());
        let mut ragged = wire.clone();
        ragged.contrasts[1].unit_effects.effects.pop();
        assert!(static_round_trip(&ragged, &names).is_err());
        let mut moved = wire.clone();
        moved.unit_effects.as_mut().unwrap()[0] += 1.0;
        assert!(static_round_trip(&moved, &names).is_err());
        let mut non_finite = wire.clone();
        non_finite.contrasts[1].mean = f64::NAN;
        assert!(static_round_trip(&non_finite, &names).is_err());
    }

    // A single-contrast result states no contrasts: its bytes are those it always had, and a
    // body written before the section existed still decodes.
    let single = CounterfactualQuery::new(v(2), vec![Intervention::set(v(0), Value::f64(1.0))])
        .with_control_level(0.0);
    let result = estimate(&data, &graph, single.clone(), InferenceMode::Frequentist, 0);
    let wire = result
        .static_result_wire(&CausalQuery::Counterfactual(single), None)
        .expect("a static result of the single form");
    assert!(wire.contrasts.is_empty());
    assert!(!cbor_names_contrasts(&wire), "a single contrast adds no key to its bytes");
    let json = serde_json::to_value(&wire).unwrap();
    assert!(json.get("contrasts").is_none());
    let back = static_round_trip(&wire, &names).expect("a single contrast round trips");
    assert!(back.contrasts.is_empty());
    assert_eq!(back.unit_effects, wire.unit_effects);
    let parsed: antecedent_io::StaticResultWire = serde_json::from_value(json).unwrap();
    assert!(parsed.contrasts.is_empty());
    // The single form refuses contrasts that are not its one outcome's.
    let mut forged = wire.clone();
    forged.contrasts = vec![antecedent_io::OutcomeContrastWire {
        outcome: 3,
        mean: 0.0,
        standard_error: None,
        unit_effects: antecedent_io::UnitEffectsWire {
            effects: vec![0.0; ROWS],
            homogeneous: false,
            intervals: None,
            extrapolative: None,
        },
    }];
    assert!(static_round_trip(&forged, &names).is_err(), "contrasts must name the query's outcome");
}

/// The query's joint baseline is part of its wire and its canonical identity; a single control
/// is not disturbed.
#[test]
fn the_joint_baseline_survives_the_query_wire() {
    let joint = CausalQuery::Counterfactual(joint_query());
    let wire = antecedent_io::causal_query_to_wire(&joint).unwrap();
    let back = antecedent_io::causal_query_from_wire(&wire).unwrap();
    assert_eq!(back, joint);
    let single = CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(2), vec![Intervention::set(v(0), Value::f64(1.0))])
            .with_control_level(0.0),
    );
    let text =
        serde_json::to_string(&antecedent_io::causal_query_to_wire(&single).unwrap()).unwrap();
    assert!(!text.contains("joint_control"), "a single control adds nothing to its wire: {text}");
    let text = serde_json::to_string(&wire).unwrap();
    assert!(text.contains("joint_control"));
}
