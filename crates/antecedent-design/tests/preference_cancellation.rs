//! Cancellation of the preference routes: the exact route's linear programs, vertex bases and
//! draw share, the certified subdivision, the compatible-set learner's range programs and
//! conflict search, the choice-model fit and measure draws, and artifact replay. A token set
//! before the run stops it at once; a token that trips after `n` observations stops it in the
//! middle for every `n` short of a full run; a run with enough observations gives exactly the
//! result of the run without a context. A stopped run is the registered `cancelled_no_claim`
//! refusal and never a per-decider outcome or a verdict.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Debug;

use antecedent_contract::ScientificQuantity;
use antecedent_core::{CancellationToken, ExecutionContext, reason_details};
use antecedent_design::accepted_preferences::{estimate_accepted, learn_compatible_set_accepted};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::decision_contract::{ActionKind, DecisionCriterion, StructuralPolicy};
use antecedent_design::preference_artifact::{
    AcceptedPreferencesArtifact, PreferenceArtifactError, PreferenceDecisionArtifact,
};
use antecedent_design::preference_certified::{
    CertifiedOptions, PreferenceCertifiedError, evaluate_preference_certified,
    evaluate_preference_certified_with_context,
};
use antecedent_design::preference_choice_model::{
    ChoiceModelError, ChoiceModelOptions, estimate_choice_model, estimate_choice_model_with_context,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{
    PreferenceDecisionError, PreferenceDecisionOptions, evaluate_preference_contract,
    evaluate_preference_contract_with_context,
};
use antecedent_design::preference_evaluation::PreferenceEvaluationOptions;
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as P, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use antecedent_design::preference_set_learning::{
    SetLearningError, SetLearningOptions, SetOutcome, learn_compatible_set,
    learn_compatible_set_with_context,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

const UNITS: &str = "u";

/// A context whose token is the given one.
fn context(token: CancellationToken) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.cancellation = token;
    ctx
}

fn cancelled() -> ExecutionContext {
    let token = CancellationToken::new();
    token.cancel();
    context(token)
}

/// Runs `run` under tokens that trip after 0, 1, 2, ... observations until one run completes.
/// Every earlier run must be refused as cancelled, and the completed run must equal `expected`.
/// Returns the number of observations of a full run, which `at_least` bounds from below so the
/// cancellation is shown to land in the middle of the work, not only at its start.
fn trips_in_the_middle<T: PartialEq + Debug, R: Debug>(
    expected: &T,
    at_least: usize,
    is_cancelled: impl Fn(&R) -> bool,
    mut run: impl FnMut(&ExecutionContext) -> Result<T, R>,
) -> usize {
    for checks in 0..100_000 {
        match run(&context(CancellationToken::cancel_after_checks(checks))) {
            Ok(result) => {
                assert_eq!(&result, expected, "a completed run differs from the plain run");
                assert!(checks >= at_least, "only {checks} observations; expected {at_least}");
                return checks;
            }
            Err(error) => assert!(is_cancelled(&error), "after {checks} checks: {error:?}"),
        }
    }
    panic!("the run never completed");
}

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        regime_id: format!("do({name}=1)"),
        units: UNITS.into(),
        ..common::quantity(name)
    }
}

/// An exact finite law over the named columns; `rows[i]` is draw `i`.
fn law(names: &[&str], rows: &[Vec<f64>]) -> DistributionArtifact {
    let columns: Vec<ScientificQuantity> = names.iter().map(|name| quantity(name)).collect();
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
    let draws: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), names.len()],
            weights: None,
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

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "u/u".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn q(name: &str) -> ActionInput {
    ActionInput::Quantity(quantity(name))
}

fn action(id: &str, inputs: Vec<ActionInput>) -> PreferenceAction {
    PreferenceAction { id: id.into(), kind: ActionKind::Intervention, inputs }
}

fn contract(model: ValueModel, actions: Vec<PreferenceAction>) -> PreferenceContract {
    PreferenceContract {
        model,
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

/// `x0 + a x1 + b x2` over `a, b` in `[0, 1]` with `a + b <= 1.5`: affine, so the exact route
/// answers. `keep` leads everywhere, so the draw share enumerates the vertices of the set.
fn exact_contract() -> PreferenceContract {
    let value = P::sum(
        P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        P::product(P::Parameter(1), P::Attribute(2)),
    );
    let model = ValueModel {
        attributes: (0..3)
            .map(|k| Attribute { id: format!("x{k}"), units: UNITS.into() })
            .collect(),
        preferences: PreferenceSet {
            parameters: vec![parameter("a", 0.0, 1.0), parameter("b", 0.0, 1.0)],
            relations: vec![LinearRelation {
                id: "budget".into(),
                terms: vec![(0, 1.0), (1, 1.0)],
                sense: RelationSense::AtMost,
                bound: 1.5,
                origin: RelationOrigin::Declared,
            }],
        },
        value,
        utility_units: UNITS.into(),
    };
    let actions = vec![
        action("keep", vec![q("k0"), q("k1"), q("k2")]),
        action("switch", vec![q("s0"), q("s1"), q("s2")]),
        action("wait", vec![q("w0"), q("w1"), q("w2")]),
    ];
    contract(model, actions)
}

/// 2048 aligned draws, so the draw share reads the token more than once.
fn exact_law() -> DistributionArtifact {
    let rows: Vec<Vec<f64>> = (0..2048_u32)
        .map(|i| {
            let wobble = f64::from(i % 7) / 7.0;
            vec![10.0 + wobble, 2.0, 2.0, 1.0, 1.0 + wobble, 0.5, 0.0, 0.5, 1.0 - wobble]
        })
        .collect();
    law(&["k0", "k1", "k2", "s0", "s1", "s2", "w0", "w1", "w2"], &rows)
}

#[test]
fn the_exact_route_stops_between_programs_vertex_bases_and_draws() {
    let contract = exact_contract();
    let source = exact_law();
    let options = PreferenceDecisionOptions::default();
    let plain = evaluate_preference_contract(&contract, &source, None, options).unwrap();
    assert!(plain.draw_share_best_everywhere.is_some(), "the draw share enumerates vertices");

    let refused =
        evaluate_preference_contract_with_context(&contract, &source, None, options, &cancelled())
            .unwrap_err();
    assert_eq!(refused, PreferenceDecisionError::Cancelled);
    let refusal = refused.to_refusal();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(refusal.detail, reason_details::preference_decision::CANCELLED);

    // Ranges, margins and possibly-best programs for three actions, then the vertex bases and
    // two blocks of draws.
    trips_in_the_middle(
        &plain,
        12,
        |e: &PreferenceDecisionError| *e == PreferenceDecisionError::Cancelled,
        |ctx| evaluate_preference_contract_with_context(&contract, &source, None, options, ctx),
    );
}

/// Discount factor: value `x0 + d x1 + d^2 x2`; the subdivision has to split around the switch
/// at `d = 5/7`.
fn certified_contract() -> PreferenceContract {
    let value = P::sum(
        P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        P::product(P::product(P::Parameter(0), P::Parameter(0)), P::Attribute(2)),
    );
    let model = ValueModel {
        attributes: (0..3)
            .map(|k| Attribute { id: format!("x{k}"), units: UNITS.into() })
            .collect(),
        preferences: PreferenceSet {
            parameters: vec![parameter("d", 0.5, 1.0)],
            relations: vec![],
        },
        value,
        utility_units: UNITS.into(),
    };
    let fixed = |value| ActionInput::Fixed { value, units: UNITS.into() };
    contract(
        model,
        vec![
            action("soon", vec![fixed(0.0), q("soon_next"), fixed(0.0)]),
            action("late", vec![fixed(0.0), fixed(0.0), q("late_later")]),
        ],
    )
}

#[test]
fn the_certified_subdivision_stops_between_boxes() {
    let contract = certified_contract();
    let source = law(&["soon_next", "late_later"], &[vec![8.0, 12.0], vec![12.0, 16.0]]);
    let options = CertifiedOptions { max_boxes: 65, min_width_fraction: 0.0, tie_tolerance: 1e-9 };
    let plain = evaluate_preference_certified(&contract, &source, None, options).unwrap();
    assert!(plain.boxes_examined > 8, "the subdivision splits: {}", plain.boxes_examined);

    let refused =
        evaluate_preference_certified_with_context(&contract, &source, None, options, &cancelled())
            .unwrap_err();
    assert_eq!(refused, PreferenceCertifiedError::Cancelled);
    let refusal = refused.to_refusal();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(refusal.detail, reason_details::preference_certified::CANCELLED);

    // One observation per box taken from the queue, plus the root.
    let checks = trips_in_the_middle(
        &plain,
        4,
        |e: &PreferenceCertifiedError| *e == PreferenceCertifiedError::Cancelled,
        |ctx| evaluate_preference_certified_with_context(&contract, &source, None, options, ctx),
    );
    assert!(
        checks <= plain.boxes_examined + 1,
        "{checks} observations for {} boxes",
        plain.boxes_examined
    );
}

fn rate_model() -> ValueModel {
    ValueModel {
        attributes: (0..2)
            .map(|k| Attribute { id: format!("x{k}"), units: format!("unit_{k}") })
            .collect(),
        preferences: PreferenceSet {
            parameters: vec![PreferenceParameter {
                id: "t".into(),
                units: "utility/unit".into(),
                lower: -4.0,
                upper: 4.0,
                assumption_id: "declared_t".into(),
            }],
            relations: Vec::new(),
        },
        value: P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        utility_units: "utility".into(),
    }
}

fn record(id: &str, options: &[(&str, Vec<f64>)], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.into(),
        decider: "d".into(),
        options: options
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).into(),
                belief: OptionBelief::Means(means.clone()),
            })
            .collect(),
        chosen: chosen.into(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast".into() },
        observation_rows: vec![format!("{id}#row")],
    }
}

fn records(model: &ValueModel, records: Vec<ChoiceRecord>, pooling: Pooling) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "decision makers".into(),
        records,
        pooling,
    }
}

/// `t >= 1`, `t <= 3` and `t <= 1/2`: the first and last conflict, so the learner runs its
/// range programs, the smallest-conflict search and the tolerance program.
fn conflicting(model: &ValueModel) -> ChoiceRecordSet {
    records(
        model,
        vec![
            record("r1", &[("a", vec![0.0, 1.0]), ("b", vec![1.0, 0.0])], "a"),
            record("r2", &[("a", vec![0.0, 1.0]), ("b", vec![3.0, 0.0])], "b"),
            record("r3", &[("a", vec![0.0, 1.0]), ("b", vec![0.5, 0.0])], "b"),
        ],
        Pooling::Pooled { assumption_id: "assumption.shared".into() },
    )
}

#[test]
fn the_compatible_set_learner_stops_between_range_programs_and_in_the_conflict_search() {
    let model = rate_model();
    let set = conflicting(&model);
    let options = SetLearningOptions::default();
    let plain = learn_compatible_set(&model, &set, options).unwrap();
    assert!(matches!(plain.pooled, Some(SetOutcome::Contradictory(_))));

    let refused =
        learn_compatible_set_with_context(&model, &set, options, &cancelled()).unwrap_err();
    assert_eq!(refused, SetLearningError::Cancelled);
    let refusal = refused.to_refusal();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(refusal.detail, reason_details::preference_set_learning::CANCELLED);

    // The declared ranges, the pooled feasibility program, every subset the smallest-conflict
    // search tries and the tolerance program each read the token.
    trips_in_the_middle(
        &plain,
        6,
        |e: &SetLearningError| *e == SetLearningError::Cancelled,
        |ctx| learn_compatible_set_with_context(&model, &set, options, ctx),
    );

    // The accepted-preferences entry point and the artifact replay keep the same stop.
    let accepted = learn_compatible_set_accepted(
        &model,
        &records(&model, set.records[..2].to_vec(), set.pooling.clone()),
        None,
        options,
    )
    .unwrap();
    let narrowed = records(&model, set.records[..2].to_vec(), set.pooling.clone());
    let artifact = AcceptedPreferencesArtifact::new(accepted);
    artifact.replay(&narrowed).unwrap();
    let stopped = artifact.replay_with_context(&narrowed, &cancelled()).unwrap_err();
    assert_eq!(stopped, PreferenceArtifactError::Cancelled);
    assert_eq!(stopped.to_refusal().detail, reason_details::preference_artifact::CANCELLED);
    let token = CancellationToken::cancel_after_checks(2);
    assert_eq!(
        artifact.replay_with_context(&narrowed, &context(token)).unwrap_err(),
        PreferenceArtifactError::Cancelled
    );
    artifact.replay_with_context(&narrowed, &context(CancellationToken::new())).unwrap();
}

/// Small fixed-seed generator for simulating logit choices.
struct Lcg(u64);

impl Lcg {
    #[allow(clippy::cast_precision_loss, reason = "53-bit integers are exact in f64")]
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
}

/// `t * x1 + x0` choices with consistency one and `t = 2` over three options, from eight
/// deciders: the decider-clustered covariance of two coefficients needs more than two clusters,
/// since the per-decider score sums of two clusters cancel at the estimate.
fn logit_records(model: &ValueModel, n: usize, pooling: Pooling) -> ChoiceRecordSet {
    let mut rng = Lcg(17);
    let list = (0..n)
        .map(|index| {
            let means: Vec<Vec<f64>> =
                (0..3).map(|_| vec![4.0 * rng.uniform() - 2.0, 2.0 * rng.uniform()]).collect();
            let chosen = means
                .iter()
                .map(|option| option[0] + 2.0 * option[1] - (-rng.uniform().ln()).ln())
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map_or(0, |(position, _)| position);
            let named: Vec<(String, Vec<f64>)> = means
                .into_iter()
                .enumerate()
                .map(|(position, option)| (format!("o{position}"), option))
                .collect();
            let borrowed: Vec<(&str, Vec<f64>)> =
                named.iter().map(|(id, option)| (id.as_str(), option.clone())).collect();
            let mut made = record(&format!("r{index}"), &borrowed, &format!("o{chosen}"));
            made.decider = format!("d{}", index % 8);
            made
        })
        .collect();
    records(model, list, pooling)
}

#[test]
fn the_choice_model_stops_between_newton_steps_and_measure_draws() {
    let model = rate_model();
    let set =
        logit_records(&model, 400, Pooling::Pooled { assumption_id: "assumption.shared".into() });
    let options = ChoiceModelOptions { n_draws: 3000, ..ChoiceModelOptions::default() };
    let plain = estimate_choice_model(&model, &set, options).unwrap();
    assert!(plain.pooled.is_some());

    let refused =
        estimate_choice_model_with_context(&model, &set, options, &cancelled()).unwrap_err();
    assert_eq!(refused, ChoiceModelError::Cancelled);
    let refusal = refused.to_refusal();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(refusal.detail, reason_details::preference_choice_model::CANCELLED);

    // The fit's start and Newton steps, then three blocks of draws.
    trips_in_the_middle(
        &plain,
        5,
        |e: &ChoiceModelError| *e == ChoiceModelError::Cancelled,
        |ctx| estimate_choice_model_with_context(&model, &set, options, ctx),
    );

    // Per decider, a stop is the run's refusal, never one decider's outcome.
    let per_decider = logit_records(&model, 400, Pooling::PerDecider);
    for checks in 0..4 {
        let token = CancellationToken::cancel_after_checks(checks);
        assert_eq!(
            estimate_choice_model_with_context(&model, &per_decider, options, &context(token))
                .unwrap_err(),
            ChoiceModelError::Cancelled
        );
    }

    // Replay of an accepted estimate relearns under the context and keeps the stop.
    let accepted = estimate_accepted(&model, &set, None, options).unwrap();
    let artifact = AcceptedPreferencesArtifact::new(accepted);
    assert_eq!(
        artifact.replay_with_context(&set, &cancelled()).unwrap_err(),
        PreferenceArtifactError::Cancelled
    );
    let token = CancellationToken::cancel_after_checks(3);
    assert_eq!(
        artifact.replay_with_context(&set, &context(token)).unwrap_err(),
        PreferenceArtifactError::Cancelled
    );
    artifact.replay_with_context(&set, &context(CancellationToken::new())).unwrap();
}

#[test]
fn a_decision_artifact_replay_stops_in_the_re_evaluation() {
    let contract = exact_contract();
    let source = exact_law();
    let options = PreferenceEvaluationOptions::default();
    let artifact = PreferenceDecisionArtifact::evaluate(contract, &source, None, options).unwrap();
    artifact.replay(&source).unwrap();
    assert_eq!(
        artifact.replay_with_context(&source, &cancelled()).unwrap_err(),
        PreferenceArtifactError::Cancelled
    );
    for checks in [1, 5, 10] {
        let token = CancellationToken::cancel_after_checks(checks);
        assert_eq!(
            artifact.replay_with_context(&source, &context(token)).unwrap_err(),
            PreferenceArtifactError::Cancelled
        );
    }
    artifact.replay_with_context(&source, &context(CancellationToken::new())).unwrap();
}
