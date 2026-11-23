//! Durable records of accepted preferences and of decisions across a preference set: round trips
//! for every basis and both routes with reals kept bit for bit, loading only under retained
//! identities, replay by recomputation, and every refusal.
//!
//! The standard law has `E[benefit, cost] = (10, 4)` under `A` and `(5.5, 1)` under `B`, and the
//! value is `benefit - w * cost` with `w` in `[0, 3]`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError, PendingReview, PreferenceBasis,
    PreferenceContractParts, estimate_accepted, learn_compatible_set_accepted,
};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::decision_artifact::source_digest;
use antecedent_design::decision_contract::{ActionKind, DecisionCriterion, StructuralPolicy};
use antecedent_design::preference_artifact::{
    AcceptedPreferencesArtifact, PreferenceArtifactError as E, PreferenceDecisionArtifact,
};
use antecedent_design::preference_choice_model::ChoiceModelOptions;
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::PreferenceVerdict;
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationOptions, PreferenceRoute,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as P, PreferenceParameter, PreferenceSet, ValueModel,
};
use antecedent_design::preference_set_learning::SetLearningOptions;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

/// Small fixed-seed xorshift64* generator.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in the open interval `(0, 1)`.
    #[allow(clippy::cast_precision_loss, reason = "53-bit integers are exact in f64")]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn gumbel(&mut self) -> f64 {
        -(-self.uniform().ln()).ln()
    }
}

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.to_owned(), units: units.to_owned() }
}

fn model_with(value: P) -> ValueModel {
    ValueModel {
        attributes: vec![attribute("benefit", "events"), attribute("cost", "usd")],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "w".to_owned(),
                units: "events/usd".to_owned(),
                lower: 0.0,
                upper: 3.0,
                assumption_id: "w-judgment".to_owned(),
            }],
            vec![],
        ),
        value,
        utility_units: "events".to_owned(),
    }
}

/// `benefit - w * cost`.
fn cost_model() -> ValueModel {
    model_with(P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))))
}

/// `benefit - w * w * cost`: the certified route answers it.
fn squared_model() -> ValueModel {
    model_with(P::difference(
        P::Attribute(0),
        P::product(P::product(P::Parameter(0), P::Parameter(0)), P::Attribute(1)),
    ))
}

fn quantity(attribute: &Attribute, action: usize) -> ScientificQuantity {
    ScientificQuantity {
        units: attribute.units.clone(),
        regime_id: format!("do(a={action})"),
        ..common::quantity(&attribute.id)
    }
}

fn parts(model: &ValueModel, policy: PreferencePolicy) -> PreferenceContractParts {
    PreferenceContractParts {
        actions: ["A", "B"]
            .iter()
            .enumerate()
            .map(|(a, id)| PreferenceAction {
                id: (*id).to_owned(),
                kind: ActionKind::Intervention,
                inputs: model
                    .attributes
                    .iter()
                    .map(|attribute| ActionInput::Quantity(quantity(attribute, a)))
                    .collect(),
            })
            .collect(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".to_owned(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: policy,
    }
}

fn contract(model: ValueModel, policy: PreferencePolicy) -> PreferenceContract {
    let parts = parts(&model, policy);
    AcceptedPreferences::declared(model).unwrap().contract(parts)
}

/// `E[benefit, cost]` is `(10, 4)` under `A` and `(5.5, 1)` under `B`; `shift` moves one draw.
fn standard_law(model: &ValueModel, shift: f64) -> DistributionArtifact {
    let arms = [
        [vec![8.0 + shift, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        [vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
    ];
    let columns: Vec<ScientificQuantity> =
        (0..2).flat_map(|a| model.attributes.iter().map(move |k| quantity(k, a))).collect();
    let mut draws = Vec::new();
    for i in 0..3 {
        for arm in &arms {
            for attribute in arm {
                draws.push(attribute[i]);
            }
        }
    }
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [3, columns.len()],
            weights: Some(vec![1.0, 1.0, 2.0]),
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn record(id: &str, a: [f64; 2], b: [f64; 2], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: "panel".to_owned(),
        options: [("a", a), ("b", b)]
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).to_owned(),
                belief: OptionBelief::Means(means.to_vec()),
            })
            .collect(),
        chosen: chosen.to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: format!("forecast-{id}") },
        observation_rows: vec![format!("{id}#row")],
    }
}

fn record_set(records: Vec<ChoiceRecord>) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: cost_model().attributes,
        population: "review panel".to_owned(),
        records,
        pooling: Pooling::Pooled { assumption_id: "assumption.one_panel".to_owned() },
    }
}

/// `w <= 1 + tolerance` from `r1` and `w >= (1 - tolerance) / 2` from `r2`.
fn settling_records() -> ChoiceRecordSet {
    record_set(vec![
        record("r1", [1.0, 1.0], [0.0, 0.0], "a"),
        record("r2", [1.0, 2.0], [0.0, 0.0], "b"),
    ])
}

/// Logit choices with consistency one and value `benefit - cost`.
fn simulated(n: usize, seed: u64) -> ChoiceRecordSet {
    let mut rng = Xorshift::new(seed);
    let records = (0..n)
        .map(|r| {
            let options: Vec<[f64; 2]> =
                (0..3).map(|_| [4.0 * rng.uniform(), 4.0 * rng.uniform()]).collect();
            let chosen = options
                .iter()
                .map(|o| o[0] - o[1] + rng.gumbel())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(0, |(i, _)| i);
            ChoiceRecord {
                id: format!("s{r}"),
                decider: "panel".to_owned(),
                options: options
                    .iter()
                    .enumerate()
                    .map(|(i, means)| ChoiceOption {
                        id: format!("o{i}"),
                        belief: OptionBelief::Means(means.to_vec()),
                    })
                    .collect(),
                chosen: format!("o{chosen}"),
                belief_source: BeliefSource::RecordedForecast { source_id: "forecast".into() },
                observation_rows: vec![format!("s{r}#row")],
            }
        })
        .collect();
    record_set(records)
}

fn quick() -> ChoiceModelOptions {
    ChoiceModelOptions { n_draws: 200, ..ChoiceModelOptions::default() }
}

fn tolerant() -> SetLearningOptions {
    SetLearningOptions { tolerance: 0.1, ..SetLearningOptions::default() }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn corrupt(bytes: &[u8]) -> Vec<u8> {
    let mut changed = bytes.to_vec();
    let middle = changed.len() / 2;
    changed[middle] ^= 0xff;
    changed
}

/// Write, check the retained identity binding and the corruption refusals, and read back.
fn round_trip(accepted: &AcceptedPreferences) -> AcceptedPreferencesArtifact {
    let artifact = AcceptedPreferencesArtifact::new(accepted.clone());
    assert_eq!(artifact.identity(), accepted.identity());
    let bytes = artifact.to_bytes("accepted-preferences").unwrap();
    let loaded = AcceptedPreferencesArtifact::from_bytes(&bytes, accepted.identity()).unwrap();
    assert_eq!(loaded, artifact);
    assert_eq!(loaded.accepted(), accepted);

    let error = AcceptedPreferencesArtifact::from_bytes(&bytes, "another identity").unwrap_err();
    assert_eq!(error, E::IdentityExpected);
    assert_eq!(error.detail(), reason_details::preference_artifact::IDENTITY_EXPECTED);
    let error =
        AcceptedPreferencesArtifact::from_bytes(&corrupt(&bytes), accepted.identity()).unwrap_err();
    assert!(matches!(error, E::Container(_)), "{error:?}");
    assert_eq!(error.detail(), reason_details::preference_artifact::CONTAINER_REFUSED);
    loaded
}

#[test]
fn accepted_preferences_of_every_basis_round_trip_and_replay() {
    let model = cost_model();
    let records = settling_records();

    let declared = AcceptedPreferences::declared(model.clone()).unwrap();
    let loaded = round_trip(&declared);
    loaded.replay(&records).unwrap();
    loaded.replay(&simulated(3, 1)).unwrap();

    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    let tolerance = PendingReview::ToleranceApplied { tolerance: 0.1 };
    assert_eq!(learned.pending(), vec![tolerance.clone()]);
    let reviewed = learned.acknowledge(&[tolerance]).unwrap();
    let loaded = round_trip(&reviewed);
    assert_eq!(loaded.accepted().version(), 2);
    assert!(loaded.accepted().pending().is_empty());
    loaded.replay(&records).unwrap();

    let records = simulated(300, 21);
    let estimated = estimate_accepted(&model, &records, None, quick()).unwrap();
    let loaded = round_trip(&estimated);
    let (PreferenceBasis::ChoiceEstimate(stored), PreferenceBasis::ChoiceEstimate(read)) =
        (estimated.basis(), loaded.accepted().basis())
    else {
        panic!("estimate basis")
    };
    for (s, r) in stored.measure.points.iter().zip(&read.measure.points) {
        assert_eq!(bits(s), bits(r));
    }
    assert_eq!(bits(&stored.coefficient_covariance), bits(&read.coefficient_covariance));
    assert_eq!(stored.consistency.to_bits(), read.consistency.to_bits());
    loaded.replay(&records).unwrap();
}

#[test]
fn a_replay_refuses_changed_records_and_a_changed_basis() {
    let model = cost_model();
    let records = settling_records();
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    let artifact = AcceptedPreferencesArtifact::new(learned.clone());

    // One record changed: `r1` now reads `w <= 1.2`.
    let mut changed = records.clone();
    changed.records[0] = record("r1", [1.2, 1.0], [0.0, 0.0], "a");
    let error = artifact.replay(&changed).unwrap_err();
    assert_eq!(error, E::ReplayRecordsDiffer);
    assert_eq!(error.detail(), reason_details::preference_artifact::REPLAY_RECORDS_DIFFER);

    // Records that no longer hold together: `r2` now reads `w >= 2`.
    let mut contradictory = records.clone();
    contradictory.records[1] = record("r2", [1.0, 0.5], [0.0, 0.0], "b");
    let error = artifact.replay(&contradictory).unwrap_err();
    assert!(matches!(&error, E::Accepted(AcceptedPreferencesError::Contradictory { .. })));
    assert_eq!(error.detail(), reason_details::accepted_preferences::CHOICES_CONTRADICTORY);

    // A stored range that the records do not reproduce.
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };
    let mut edited = (**set).clone();
    edited.ranges[0].learned.1 += 0.5;
    let edited = AcceptedPreferences::from_compatible_set(&model, edited, tolerant(), 0).unwrap();
    let error = AcceptedPreferencesArtifact::new(edited).replay(&records).unwrap_err();
    assert_eq!(error, E::ReplayMismatch { field: "ranges" });
    assert_eq!(error.detail(), reason_details::preference_artifact::REPLAY_MISMATCH);

    // An estimate: changed records, then a stored consistency the records do not reproduce.
    let records = simulated(300, 21);
    let estimated = estimate_accepted(&model, &records, None, quick()).unwrap();
    let mut changed = records.clone();
    let flipped = if changed.records[0].chosen == "o0" { "o1" } else { "o0" };
    changed.records[0].chosen = flipped.to_owned();
    let artifact = AcceptedPreferencesArtifact::new(estimated.clone());
    assert_eq!(artifact.replay(&changed), Err(E::ReplayRecordsDiffer));
    let PreferenceBasis::ChoiceEstimate(estimate) = estimated.basis() else { panic!("estimate") };
    let mut edited = (**estimate).clone();
    edited.consistency *= 1.01;
    let edited = AcceptedPreferences::from_estimate(&model, edited, quick(), 0).unwrap();
    let error = AcceptedPreferencesArtifact::new(edited).replay(&records).unwrap_err();
    assert_eq!(error, E::ReplayMismatch { field: "consistency" });

    // A stated count the records do not reproduce.
    let miscounted =
        AcceptedPreferences::from_estimate(&model, (**estimate).clone(), quick(), 3).unwrap();
    let error = AcceptedPreferencesArtifact::new(miscounted).replay(&records).unwrap_err();
    assert_eq!(error, E::ReplayMismatch { field: "stated_comparisons" });
}

#[test]
fn stated_comparisons_survive_the_wire_and_replay() {
    let model = cost_model();
    let mut records = settling_records();
    records.records[0].belief_source = BeliefSource::Stated {
        elicitation_id: "elicitation-1".to_owned(),
        assumption_id: "assumption.statements_reflect_preferences".to_owned(),
    };
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    let used = PendingReview::StatedComparisonsUsed { count: 1 };
    assert!(learned.pending().contains(&used));
    let reviewed = learned.acknowledge(&[used.clone()]).unwrap();
    let loaded = round_trip(&reviewed);
    assert_eq!(loaded.accepted().stated_comparisons(), 1);
    assert_eq!(loaded.accepted().acknowledged().last(), Some(&used));
    loaded.replay(&records).unwrap();
    // The same records read as recorded forecasts are other records.
    assert_eq!(loaded.replay(&settling_records()), Err(E::ReplayRecordsDiffer));
}

/// Write, check the identity bindings and corruption refusals, and read back.
fn decision_round_trip(
    artifact: &PreferenceDecisionArtifact,
    source: &DistributionArtifact,
) -> PreferenceDecisionArtifact {
    let digest = source_digest(source);
    assert_eq!(artifact.source_digest(), digest);
    let identity = artifact.contract_identity().to_owned();
    assert_eq!(identity, artifact.contract().identity().unwrap());
    let bytes = artifact.to_bytes("preference-decision").unwrap();
    let loaded = PreferenceDecisionArtifact::from_bytes(&bytes, &identity, &digest).unwrap();
    assert_eq!(&loaded, artifact);

    let error = PreferenceDecisionArtifact::from_bytes(&bytes, "another", &digest).unwrap_err();
    assert_eq!(error, E::ContractExpected);
    assert_eq!(error.detail(), reason_details::preference_artifact::CONTRACT_EXPECTED);
    let error = PreferenceDecisionArtifact::from_bytes(&bytes, &identity, "other").unwrap_err();
    assert_eq!(error, E::SourceExpected);
    assert_eq!(error.detail(), reason_details::preference_artifact::SOURCE_EXPECTED);
    let error =
        PreferenceDecisionArtifact::from_bytes(&corrupt(&bytes), &identity, &digest).unwrap_err();
    assert!(matches!(error, E::Container(_)), "{error:?}");
    // Bytes of the other kind are refused by the container.
    let other =
        AcceptedPreferencesArtifact::new(AcceptedPreferences::declared(cost_model()).unwrap())
            .to_bytes("accepted")
            .unwrap();
    assert!(matches!(
        PreferenceDecisionArtifact::from_bytes(&other, &identity, &digest),
        Err(E::Container(_))
    ));
    loaded
}

#[test]
fn decisions_of_both_routes_round_trip_and_replay() {
    let options = PreferenceEvaluationOptions::default();

    // Exact route, declared set.
    let source = standard_law(&cost_model(), 0.0);
    let exact = PreferenceDecisionArtifact::evaluate(
        contract(cost_model(), PreferencePolicy::RequireInvariantBestAction),
        &source,
        None,
        options,
    )
    .unwrap();
    assert_eq!(exact.route(), PreferenceRoute::Exact);
    assert!(exact.measure().is_none());
    assert_eq!(exact.options(), options);
    let loaded = decision_round_trip(&exact, &source);
    loaded.replay(&source).unwrap();
    let switch = &loaded.evaluation().exact().unwrap().switches[0];
    let threshold = switch.threshold.as_ref().unwrap();
    assert_eq!((threshold.parameter_id.as_str(), threshold.first_leads_above), ("w", false));
    let ranges = &loaded.evaluation().exact().unwrap().parameter_ranges;
    assert!(ranges.len() == 1 && ranges[0].0.abs() < 1e-9 && (ranges[0].1 - 3.0).abs() < 1e-9);
    assert_eq!(loaded.evaluation().statement(), exact.evaluation().statement());

    // Exact route, a never-beaten action: on `w` in `[0, 1.5]` A ties B at `w = 1.5`.
    let mut edge = cost_model();
    edge.preferences.parameters[0].upper = 1.5;
    let source_edge = standard_law(&edge, 0.0);
    let never = PreferenceDecisionArtifact::evaluate(
        contract(edge, PreferencePolicy::RequireInvariantBestAction),
        &source_edge,
        None,
        options,
    )
    .unwrap();
    let verdict = &never.evaluation().exact().unwrap().verdict;
    assert_eq!(
        verdict,
        &PreferenceVerdict::NeverBeaten { action: "A".into(), ties_with: vec!["B".into()] }
    );
    let loaded = decision_round_trip(&never, &source_edge);
    let actions = &loaded.evaluation().exact().unwrap().actions;
    assert!(actions[0].necessarily_best && !actions[0].uniquely_best);
    loaded.replay(&source_edge).unwrap();

    // Exact route, the measure of an accepted estimate under the weighted policy.
    let model = cost_model();
    let estimated = estimate_accepted(&model, &simulated(300, 21), None, quick()).unwrap();
    let weighted = PreferenceDecisionArtifact::evaluate(
        estimated.contract(parts(&model, PreferencePolicy::Weighted)),
        &source,
        estimated.measure().cloned(),
        options,
    )
    .unwrap();
    let loaded = decision_round_trip(&weighted, &source);
    let (Some(stored), Some(read)) = (weighted.measure(), loaded.measure()) else {
        panic!("measure")
    };
    for (s, r) in stored.points.iter().zip(&read.points) {
        assert_eq!(bits(s), bits(r));
    }
    assert!(matches!(
        loaded.evaluation().exact().unwrap().verdict,
        PreferenceVerdict::MeasureChoice { .. }
    ));
    loaded.replay(&source).unwrap();

    // Certified route.
    let shaped = contract(squared_model(), PreferencePolicy::RequireInvariantBestAction);
    let certified = PreferenceDecisionArtifact::evaluate(shaped, &source, None, options).unwrap();
    assert_eq!(certified.route(), PreferenceRoute::Certified);
    let loaded = decision_round_trip(&certified, &source);
    let (Some(stored), Some(read)) =
        (certified.evaluation().certified(), loaded.evaluation().certified())
    else {
        panic!("certified")
    };
    assert!(!stored.regions.is_empty());
    for (s, r) in stored.regions.iter().zip(&read.regions) {
        assert_eq!((bits(&s.lower), bits(&s.upper)), (bits(&r.lower), bits(&r.upper)));
    }
    assert_eq!(stored.undecided_fraction.to_bits(), read.undecided_fraction.to_bits());
    loaded.replay(&source).unwrap();
}

#[test]
#[allow(clippy::too_many_lines, reason = "one table of replay refusals")]
fn a_decision_replay_refuses_other_draws_and_any_changed_result() {
    let options = PreferenceEvaluationOptions::default();
    let source = standard_law(&cost_model(), 0.0);
    let c = contract(cost_model(), PreferencePolicy::RequireInvariantBestAction);
    let artifact = PreferenceDecisionArtifact::evaluate(c.clone(), &source, None, options).unwrap();

    let other = standard_law(&cost_model(), 0.5);
    let error = artifact.replay(&other).unwrap_err();
    assert_eq!(error, E::ReplaySourceDiffers);
    assert_eq!(error.detail(), reason_details::preference_artifact::REPLAY_SOURCE_DIFFERS);

    let PreferenceEvaluation::Exact(result) = artifact.evaluation() else { panic!("exact") };
    let mut flipped = (**result).clone();
    flipped.verdict = PreferenceVerdict::NecessarilyBest("A".into());
    let flipped = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(flipped)),
        &source,
    )
    .unwrap();
    assert_eq!(flipped.replay(&source), Err(E::ReplayMismatch { field: "verdict" }));

    let mut moved = (**result).clone();
    let range = moved.actions[0].value_range.unwrap();
    moved.actions[0].value_range = Some((range.0, range.1 + 1e-6));
    let moved = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(moved)),
        &source,
    )
    .unwrap();
    assert_eq!(moved.replay(&source), Err(E::ReplayMismatch { field: "value_range" }));

    // A unique-winner flag, a dominating rival or a threshold the evaluation does not give.
    let mut unique = (**result).clone();
    unique.actions[0].uniquely_best = true;
    let unique = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(unique)),
        &source,
    )
    .unwrap();
    assert_eq!(unique.replay(&source), Err(E::ReplayMismatch { field: "uniquely_best" }));
    let mut dominated = (**result).clone();
    dominated.actions[1].dominated_by = vec!["A".into()];
    let dominated = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(dominated)),
        &source,
    )
    .unwrap();
    assert_eq!(dominated.replay(&source), Err(E::ReplayMismatch { field: "dominated_by" }));
    let mut shifted = (**result).clone();
    if let Some(threshold) = shifted.switches[0].threshold.as_mut() {
        threshold.value += 0.25;
    }
    let shifted = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(shifted)),
        &source,
    )
    .unwrap();
    assert_eq!(shifted.replay(&source), Err(E::ReplayMismatch { field: "threshold" }));
    let mut narrowed = (**result).clone();
    narrowed.parameter_ranges[0].1 = 2.0;
    let narrowed = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(narrowed)),
        &source,
    )
    .unwrap();
    assert_eq!(narrowed.replay(&source), Err(E::ReplayMismatch { field: "parameter_ranges" }));

    // A difference far below the replay tolerance is accepted.
    let mut nudged = (**result).clone();
    nudged.actions[0].value_range = Some((range.0, range.1 * (1.0 + 1e-13)));
    let nudged = PreferenceDecisionArtifact::new(
        c.clone(),
        None,
        options,
        PreferenceEvaluation::Exact(Box::new(nudged)),
        &source,
    )
    .unwrap();
    nudged.replay(&source).unwrap();

    // A certified result whose box count was changed.
    let shaped = contract(squared_model(), PreferencePolicy::RequireInvariantBestAction);
    let certified =
        PreferenceDecisionArtifact::evaluate(shaped.clone(), &source, None, options).unwrap();
    let PreferenceEvaluation::Certified(result) = certified.evaluation() else {
        panic!("certified")
    };
    let mut recounted = (**result).clone();
    recounted.boxes_examined += 1;
    let recounted = PreferenceDecisionArtifact::new(
        shaped,
        None,
        options,
        PreferenceEvaluation::Certified(Box::new(recounted)),
        &source,
    )
    .unwrap();
    assert_eq!(recounted.replay(&source), Err(E::ReplayMismatch { field: "boxes_examined" }));
}

#[test]
fn building_a_decision_artifact_refuses_mismatched_or_invalid_parts() {
    let options = PreferenceEvaluationOptions::default();
    let source = standard_law(&cost_model(), 0.0);
    let c = contract(cost_model(), PreferencePolicy::RequireInvariantBestAction);
    let artifact = PreferenceDecisionArtifact::evaluate(c.clone(), &source, None, options).unwrap();

    // A result of one contract stored with another.
    let other = contract(cost_model(), PreferencePolicy::ReportOnly);
    let error = PreferenceDecisionArtifact::new(
        other,
        None,
        options,
        artifact.evaluation().clone(),
        &source,
    )
    .unwrap_err();
    assert_eq!(error, E::ResultContractMismatch);
    assert_eq!(error.detail(), reason_details::preference_artifact::RESULT_CONTRACT_MISMATCH);

    let mut single = c.clone();
    single.actions.truncate(1);
    let error = PreferenceDecisionArtifact::new(
        single,
        None,
        options,
        artifact.evaluation().clone(),
        &source,
    )
    .unwrap_err();
    assert!(matches!(&error, E::Contract(inner) if error.detail() == inner.detail()));

    let weighted = contract(cost_model(), PreferencePolicy::Weighted);
    let error = PreferenceDecisionArtifact::evaluate(weighted, &source, None, options).unwrap_err();
    assert!(matches!(&error, E::Evaluation(inner) if error.detail() == inner.detail()));

    let error = artifact.to_bytes("").unwrap_err();
    assert!(matches!(error, E::Encode(_)), "{error:?}");
    assert_eq!(error.detail(), reason_details::preference_artifact::ENCODE_FAILED);
}

#[test]
fn refusals_become_artifact_errors() {
    let source = standard_law(&cost_model(), 0.0);
    let artifact = PreferenceDecisionArtifact::evaluate(
        contract(cost_model(), PreferencePolicy::ReportOnly),
        &source,
        None,
        PreferenceEvaluationOptions::default(),
    )
    .unwrap();
    let bytes = artifact.to_bytes("preference-decision").unwrap();
    let error = PreferenceDecisionArtifact::from_bytes(
        &corrupt(&bytes),
        artifact.contract_identity(),
        artifact.source_digest(),
    )
    .unwrap_err();
    let E::Container(inner) = error.clone() else { panic!("{error:?}") };
    assert_eq!(IoError::from(error), inner);

    let refused = IoError::from(E::ReplayMismatch { field: "verdict" });
    let IoError::Refused { code, message } = refused else { panic!("{refused:?}") };
    assert_eq!(code, "decision_contract_unsatisfied");
    assert!(message.starts_with("preference_artifact.replay_mismatch: "), "{message}");
}
