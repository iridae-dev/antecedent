//! The compiled decisions across a preference set and the compiled preference learners equal
//! the direct kernel calls they replace.
//!
//! Each compiled program (`TradeoffLawDecision`, `TradeoffCertifiedLawDecision`,
//! `TradeoffStructuralDecision`, `TradeoffCertifiedStructuralDecision`,
//! `PreferenceCompatibleSet`, `PreferenceChoiceModel`) calls the same design-crate kernel as the
//! direct entry on the same inputs, so every verdict, range, estimate and refusal agrees bit for
//! bit (compared both by `PartialEq` and by the `Debug` rendering, which distinguishes every
//! non-NaN bit pattern). Each test also reaches a hand-derived value through the compiled route
//! and checks the route and unit the program runs under.
//!
//! The standard law has `E[benefit, cost] = (10, 4)` under `A` and `(5.5, 1)` under `B`, and the
//! value is `benefit - w * cost`, so `v_A = 10 - 4w`, `v_B = 5.5 - w` and the switch point is
//! `w* = 1.5`.
//!
//! Calibration basis: the decisions carry no sampling interval. The choice model's interval is
//! the kernel's own (`ChoiceModelResult`), so the compiled estimate carries the same
//! calibration key (`PreferenceTradeOffFromChoices`) the direct estimate is measured under.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::claim_decision_programs::{ContractSource, SuppliedClaimSet};
use antecedent::analysis::preference_learning_programs::{
    TerminalError as LearningError, compile_choice_model, compile_choice_model_relearn,
    compile_compatible_set, compile_compatible_set_relearn,
};
use antecedent::analysis::tradeoff_decision_programs::{
    CompiledTradeoffDecision, TerminalError, TradeoffStructuralError, compile_tradeoff_decision,
    compile_tradeoff_decision_with_accepted, compile_tradeoff_structural_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::accepted_preferences::{
    AcceptedPreferences, PreferenceContractParts, decide_with_accepted_with_context,
    estimate_accepted_with_context, learn_compatible_set_accepted_with_context,
};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::decision_adapters::{
    ClaimProbability, SuppliedClaim, adapt_graph_dependent_claims,
};
use antecedent_design::decision_contract::{ActionKind, DecisionCriterion, StructuralPolicy};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::decision_robustness::AtomSupport;
use antecedent_design::decision_structural::AtomEvidence;
use antecedent_design::preference_artifact::PreferenceDecisionArtifact;
use antecedent_design::preference_choice_model::{
    CALIBRATION_QUERY, ChoiceModelOptions, estimate_choice_model_with_context,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{MeasureOrigin, PreferenceMeasure, PreferenceVerdict};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluationOptions, PreferenceRoute, decide_preferences_on_means_with_context,
    decide_preferences_structural_with_context, decide_preferences_with_context,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as P, PreferenceParameter, PreferenceSet, ValueModel, shape,
};
use antecedent_design::preference_set_learning::{
    SetLearningOptions, SetOutcome, learn_compatible_set_with_context,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn assert_same<T: PartialEq + std::fmt::Debug>(compiled: &T, direct: &T) {
    assert_eq!(compiled, direct);
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

// --- the standard decision -------------------------------------------------------------------

fn quantity(variable: &str, regime: &str, units: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn regime(action: usize) -> String {
    format!("do(a={action})")
}

fn attributes() -> Vec<Attribute> {
    vec![
        Attribute { id: "benefit".into(), units: "events".into() },
        Attribute { id: "cost".into(), units: "usd".into() },
    ]
}

/// `benefit - w * cost`, `w` in `[lower, upper]`.
fn model(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "w".into(),
                units: "events/usd".into(),
                lower,
                upper,
                assumption_id: "w-judgment".into(),
            }],
            vec![],
        ),
        value: P::difference(shape::rate(0), P::product(P::Parameter(0), shape::rate(1))),
        utility_units: "events".into(),
    }
}

fn actions(model: &ValueModel, functional: &str) -> Vec<PreferenceAction> {
    ["A", "B"]
        .iter()
        .enumerate()
        .map(|(a, id)| PreferenceAction {
            id: (*id).into(),
            kind: ActionKind::Intervention,
            inputs: model
                .attributes
                .iter()
                .map(|attribute| {
                    ActionInput::Quantity(quantity(
                        &attribute.id,
                        &regime(a),
                        &attribute.units,
                        functional,
                    ))
                })
                .collect(),
        })
        .collect()
}

fn contract(
    lower: f64,
    upper: f64,
    functional: &str,
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContract {
    let model = model(lower, upper);
    PreferenceContract {
        actions: actions(&model, functional),
        model,
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: policy,
    }
}

fn eu() -> DecisionCriterion {
    DecisionCriterion::PosteriorExpectedUtility
}

/// `A` and `B` on three weighted draws (weights 1, 1, 2); `E[benefit, cost]` is `(10, 4)` and
/// `(5.5, 1)`. `scale` multiplies every benefit draw.
fn law_scaled(scale: f64) -> DistributionArtifact {
    let arms = [[[8.0, 12.0, 10.0], [3.0, 5.0, 4.0]], [[9.0, 5.0, 4.0], [1.0, 1.0, 1.0]]];
    let mut columns = Vec::new();
    for (a, _) in arms.iter().enumerate() {
        for attribute in attributes() {
            columns.push(quantity(&attribute.id, &regime(a), &attribute.units, "outcome"));
        }
    }
    let mut draws = Vec::new();
    for i in 0..3 {
        for arm in &arms {
            draws.push(arm[0][i] * scale);
            draws.push(arm[1][i]);
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

fn law() -> DistributionArtifact {
    law_scaled(1.0)
}

fn means() -> MeanSource {
    let mut coordinates = Vec::new();
    for a in 0..2 {
        for attribute in attributes() {
            coordinates.push(quantity(&attribute.id, &regime(a), &attribute.units, "mean"));
        }
    }
    MeanSource {
        coordinates,
        means: vec![10.0, 4.0, 5.5, 1.0],
        estimate: MeanEstimate::Posterior,
        provider_id: "lab".into(),
        snapshot_id: "snap-1".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    }
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn unit_of(program: &CompiledTradeoffDecision) -> String {
    program.plan_view().unit_support().expect("the family is bound to a unit").unit().to_owned()
}

// --- decisions on a law or means -------------------------------------------------------------

#[test]
fn compiled_exact_decisions_equal_the_direct_kernels_on_a_law_and_on_means() {
    let ctx = ExecutionContext::for_tests(7);
    let policy = PreferencePolicy::RequireInvariantBestAction;

    // On a law: left of the switch point, A is necessarily best (v_A - v_B = 4.5 - 3w > 0).
    let declared = contract(0.0, 1.0, "outcome", eu(), policy);
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.route(), PreferenceRoute::Exact);
    assert_eq!(program.lowered().route(), "TradeoffLawDecision");
    assert_eq!(unit_of(&program), "preference_decision.exact");
    let compiled = program.execute(&ctx).unwrap();
    let direct = decide_preferences_with_context(&declared, &law(), None, options(), &ctx).unwrap();
    assert_same(&compiled, &direct);
    let exact = compiled.exact().expect("the exact route answered");
    assert_eq!(exact.verdict, PreferenceVerdict::NecessarilyBest("A".into()));

    // On means: the expectation of an affine value needs only the means.
    let declared = contract(0.0, 3.0, "mean", eu(), policy);
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Means(means()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.lowered().route(), "TradeoffLawDecision");
    let compiled = program.execute(&ctx).unwrap();
    let direct =
        decide_preferences_on_means_with_context(&declared, &means(), None, options(), &ctx)
            .unwrap();
    assert_same(&compiled, &direct);
    // Across w in [0, 3] the choice switches at w* = 1.5.
    let exact = compiled.exact().expect("the exact route answered");
    assert!(matches!(exact.verdict, PreferenceVerdict::PreferenceDependent { .. }));
    assert!(
        exact.switches.iter().any(|s| s.threshold.as_ref().is_some_and(|t| close(t.value, 1.5)))
    );
}

#[test]
fn a_contract_the_exact_route_falls_through_on_compiles_to_the_certified_family() {
    let ctx = ExecutionContext::for_tests(11);
    // The worst case of expected regret is not a linear program over the set: the route rule
    // says exact, the exact route refuses and the certified route answers.
    let declared =
        contract(0.0, 3.0, "outcome", DecisionCriterion::ExpectedRegret, PreferencePolicy::Maximin);
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.route(), PreferenceRoute::Certified);
    assert_eq!(program.lowered().route(), "TradeoffCertifiedLawDecision");
    assert_eq!(unit_of(&program), "preference_decision.certified");
    let compiled = program.execute(&ctx).unwrap();
    let direct = decide_preferences_with_context(&declared, &law(), None, options(), &ctx).unwrap();
    assert_same(&compiled, &direct);
    assert_eq!(compiled.route(), PreferenceRoute::Certified);

    // A criterion that is not an expectation goes to the certified route by the route rule.
    let declared = contract(
        0.0,
        2.0,
        "outcome",
        DecisionCriterion::Regret,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.lowered().route(), "TradeoffCertifiedLawDecision");
    let compiled = program.execute(&ctx);
    let direct = decide_preferences_with_context(&declared, &law(), None, options(), &ctx);
    assert_same(&compiled.map_err(|e| e.kernel().cloned()), &direct.map_err(Some));

    // On means, a contract that needs the certified route is refused by the kernel with the same
    // refusal.
    let declared =
        contract(0.0, 3.0, "mean", DecisionCriterion::ExpectedRegret, PreferencePolicy::Maximin);
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Means(means()),
        &ctx,
    )
    .unwrap();
    let Err(TerminalError::Kernel(compiled)) = program.execute(&ctx) else {
        panic!("a mean source cannot answer the certified route");
    };
    let direct =
        decide_preferences_on_means_with_context(&declared, &means(), None, options(), &ctx)
            .unwrap_err();
    assert_same(&compiled, &direct);
}

#[test]
fn an_invalid_contract_is_refused_at_compile_with_the_kernels_refusal() {
    let ctx = ExecutionContext::for_tests(3);
    let mut declared =
        contract(0.0, 1.0, "outcome", eu(), PreferencePolicy::RequireInvariantBestAction);
    declared.actions[1].id = "A".into();
    let Err(TerminalError::Kernel(compiled)) = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    ) else {
        panic!("a contract with a repeated action is refused");
    };
    let direct =
        decide_preferences_with_context(&declared, &law(), None, options(), &ctx).unwrap_err();
    assert_same(&compiled, &direct);
    assert_eq!(compiled.to_refusal(), direct.to_refusal());
}

#[test]
fn a_measure_and_options_are_part_of_the_program() {
    let ctx = ExecutionContext::for_tests(5);
    let declared = contract(0.0, 3.0, "outcome", eu(), PreferencePolicy::Weighted);
    let measure = PreferenceMeasure {
        points: vec![vec![0.5], vec![2.5]],
        weights: Some(vec![3.0, 1.0]),
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "panel-weights".into() },
        unresolved: vec![],
    };
    let compile = |measure: Option<PreferenceMeasure>, options| {
        compile_tradeoff_decision(
            declared.clone(),
            measure,
            options,
            ContractSource::Law(law()),
            &ctx,
        )
        .unwrap()
    };
    let program = compile(Some(measure.clone()), options());
    let compiled = program.execute(&ctx).unwrap();
    let direct =
        decide_preferences_with_context(&declared, &law(), Some(&measure), options(), &ctx)
            .unwrap();
    assert_same(&compiled, &direct);
    let mut shifted = measure.clone();
    shifted.weights = Some(vec![1.0, 3.0]);
    assert_ne!(program.identity(), compile(Some(shifted), options()).identity());
    let mut tighter = options();
    tighter.exact.tie_tolerance *= 2.0;
    assert_ne!(program.identity(), compile(Some(measure.clone()), tighter).identity());
    assert_eq!(program.identity(), compile(Some(measure), options()).identity());
}

#[test]
fn a_compiled_record_equals_the_directly_evaluated_record() {
    let ctx = ExecutionContext::for_tests(9);
    let declared =
        contract(0.0, 3.0, "outcome", eu(), PreferencePolicy::RequireInvariantBestAction);
    let program = compile_tradeoff_decision(
        declared.clone(),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    let compiled = program.record(&ctx).unwrap();
    let direct =
        PreferenceDecisionArtifact::evaluate_with_context(declared, &law(), None, options(), &ctx)
            .unwrap();
    assert_same(&compiled, &direct);
    assert_eq!(compiled.to_bytes("record").unwrap(), direct.to_bytes("record").unwrap());

    // A record binds joint draws; a grid of means is refused by the compiler.
    let declared = contract(0.0, 3.0, "mean", eu(), PreferencePolicy::RequireInvariantBestAction);
    let program =
        compile_tradeoff_decision(declared, None, options(), ContractSource::Means(means()), &ctx)
            .unwrap();
    assert!(matches!(program.record(&ctx), Err(TerminalError::Refused(_))));
}

#[test]
fn a_decision_with_accepted_preferences_equals_the_direct_kernel() {
    let ctx = ExecutionContext::for_tests(13);
    let accepted = AcceptedPreferences::declared(model(0.0, 1.0)).unwrap();
    let declared =
        contract(0.0, 1.0, "outcome", eu(), PreferencePolicy::RequireInvariantBestAction);
    let parts = || PreferenceContractParts {
        actions: declared.actions.clone(),
        criterion: declared.criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    };
    let program = compile_tradeoff_decision_with_accepted(
        &accepted,
        parts(),
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.lowered().route(), "TradeoffLawDecision");
    let compiled = program.execute(&ctx).unwrap();
    let direct =
        decide_with_accepted_with_context(&accepted, parts(), &law(), options(), &ctx).unwrap();
    assert_same(&compiled, &direct);
    // The accepted identity is declared: the same contract without it is another program.
    let plain = compile_tradeoff_decision(
        accepted.contract(parts()),
        None,
        options(),
        ContractSource::Law(law()),
        &ctx,
    )
    .unwrap();
    assert_ne!(program.identity(), plain.identity());
}

// --- decisions across structures -------------------------------------------------------------

fn structure(id: &str, scale: f64) -> SuppliedClaim {
    SuppliedClaim {
        id: id.into(),
        probability: ClaimProbability::Unspecified,
        evidence: AtomEvidence::Evaluated(Box::new(law_scaled(scale))),
        support: AtomSupport::supported(),
    }
}

fn structures() -> Vec<SuppliedClaim> {
    vec![structure("g1", 1.0), structure("g2", 0.9)]
}

#[test]
fn compiled_structural_decisions_equal_the_adapter_and_the_direct_kernel() {
    let ctx = ExecutionContext::for_tests(17);
    let mut declared =
        contract(0.0, 1.0, "outcome", eu(), PreferencePolicy::RequireInvariantBestAction);
    declared.structural_policy = StructuralPolicy::RequireInvariantBestAction;
    let program = compile_tradeoff_structural_decision(
        declared.clone(),
        None,
        options(),
        SuppliedClaimSet::GraphDependent(structures()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.route(), PreferenceRoute::Exact);
    assert_eq!(program.lowered().route(), "TradeoffStructuralDecision");
    let unit = program.plan_view().unit_support().expect("bound to a unit").unit().to_owned();
    assert_eq!(unit, "preference_decision.exact");
    let compiled = program.execute(&ctx).unwrap();
    let adapted = adapt_graph_dependent_claims(structures(), declared.structural_policy).unwrap();
    let direct = decide_preferences_structural_with_context(
        &declared,
        &adapted.atoms,
        None,
        options(),
        &ctx,
    )
    .unwrap();
    assert_same(&compiled, &direct);
    // Both structures keep A ahead on [0, 1] (v_A - v_B >= 0.9 * 4.5 - 3w > 0).
    assert_eq!(compiled.route(), PreferenceRoute::Exact);

    // The worst case of expected regret over the structures falls through to the certified
    // route, and so compiles to the certified family.
    declared.structural_policy = StructuralPolicy::Maximin;
    declared.criterion = DecisionCriterion::ExpectedRegret;
    let program = compile_tradeoff_structural_decision(
        declared.clone(),
        None,
        options(),
        SuppliedClaimSet::GraphDependent(structures()),
        &ctx,
    )
    .unwrap();
    assert_eq!(program.lowered().route(), "TradeoffCertifiedStructuralDecision");
    let compiled = program.execute(&ctx).map_err(|e| e.kernel().cloned());
    let adapted = adapt_graph_dependent_claims(structures(), declared.structural_policy).unwrap();
    let direct = decide_preferences_structural_with_context(
        &declared,
        &adapted.atoms,
        None,
        options(),
        &ctx,
    )
    .map_err(|e| Some(TradeoffStructuralError::Evaluation(e)));
    assert_same(&compiled, &direct);
    if let Ok(evaluation) = compiled {
        assert_eq!(evaluation.route(), PreferenceRoute::Certified);
    }
}

#[test]
fn claims_the_adapter_refuses_are_refused_first_as_on_the_direct_path() {
    let ctx = ExecutionContext::for_tests(19);
    // A repeated action makes the contract invalid; repeated claim ids make the claims invalid.
    let mut declared =
        contract(0.0, 1.0, "outcome", eu(), PreferencePolicy::RequireInvariantBestAction);
    declared.actions[1].id = "A".into();
    let repeated = vec![structure("g1", 1.0), structure("g1", 0.9)];
    let direct = adapt_graph_dependent_claims(repeated.clone(), declared.structural_policy)
        .expect_err("repeated claim ids are refused");
    let Err(TerminalError::Kernel(compiled)) = compile_tradeoff_structural_decision(
        declared,
        None,
        options(),
        SuppliedClaimSet::GraphDependent(repeated),
        &ctx,
    ) else {
        panic!("the compile is refused");
    };
    assert_same(&compiled, &TradeoffStructuralError::Adapter(direct));
}

// --- learning --------------------------------------------------------------------------------

/// `x0 + t * x1` with `t` in `[-4, 4]`.
fn rate_model() -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "x0".into(), units: "unit_0".into() },
            Attribute { id: "x1".into(), units: "unit_1".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "t0".into(),
                units: "utility/unit".into(),
                lower: -4.0,
                upper: 4.0,
                assumption_id: "declared_t0".into(),
            }],
            vec![],
        ),
        value: P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        utility_units: "utility".into(),
    }
}

fn record(id: &str, decider: &str, options: &[(&str, &[f64])], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.into(),
        decider: decider.into(),
        options: options
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).into(),
                belief: OptionBelief::Means(means.to_vec()),
            })
            .collect(),
        chosen: chosen.into(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast".into() },
        observation_rows: vec![format!("{id}#row")],
    }
}

/// `t >= 1` and `t <= 3` from one decider.
fn bracketing_records(model: &ValueModel) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "decision makers".into(),
        records: vec![
            record("r1", "d", &[("a", &[0.0, 1.0]), ("b", &[1.0, 0.0])], "a"),
            record("r2", "d", &[("a", &[0.0, 1.0]), ("b", &[3.0, 0.0])], "b"),
        ],
        pooling: Pooling::PerDecider,
    }
}

#[test]
fn the_compiled_compatible_set_equals_the_direct_learner() {
    let ctx = ExecutionContext::for_tests(23);
    let model = rate_model();
    let records = bracketing_records(&model);
    let options = SetLearningOptions::default();
    let program = compile_compatible_set(model.clone(), records.clone(), options, &ctx).unwrap();
    assert_eq!(program.lowered().route(), "PreferenceCompatibleSet");
    let unit = program.plan_view().unit_support().expect("bound to a unit").unit().to_owned();
    assert_eq!(unit, "preference_learning.compatible_set");
    let compiled = program.execute(&ctx).unwrap();
    let direct = learn_compatible_set_with_context(&model, &records, options, &ctx).unwrap();
    assert_same(&compiled, &direct);
    // Records on both sides bracket the parameter: t in [1, 3].
    let SetOutcome::Learned(learned) = &compiled.per_decider[0].outcome else {
        panic!("the records are consistent");
    };
    assert!(close(learned.ranges[0].learned.0, 1.0) && close(learned.ranges[0].learned.1, 3.0));

    // Accepting one decider's outcome is a reading of the same program.
    let compiled = program.accept(Some("d"), &ctx).unwrap();
    let direct =
        learn_compatible_set_accepted_with_context(&model, &records, Some("d"), options, &ctx)
            .unwrap();
    assert_same(&compiled, &direct);

    // Relearning revises accepted preferences: the next version, through the same learner.
    let declared = AcceptedPreferences::declared(model.clone()).unwrap();
    let relearn =
        compile_compatible_set_relearn(declared.clone(), records.clone(), options, &ctx).unwrap();
    assert_eq!(relearn.lowered().route(), "PreferenceCompatibleSet");
    assert_ne!(relearn.identity(), program.identity());
    let compiled = relearn.accept(Some("d"), &ctx).unwrap();
    let direct =
        declared.relearn_compatible_set_with_context(&records, Some("d"), options, &ctx).unwrap();
    assert_same(&compiled, &direct);
    assert_eq!(compiled.version(), 2);
}

#[test]
fn a_record_set_the_learner_refuses_is_refused_at_compile_with_its_error() {
    let ctx = ExecutionContext::for_tests(29);
    let model = rate_model();
    let mut records = bracketing_records(&model);
    records.records[1].id = "r1".into();
    let options = SetLearningOptions::default();
    let Err(LearningError::Kernel(compiled)) =
        compile_compatible_set(model.clone(), records.clone(), options, &ctx)
    else {
        panic!("repeated record ids are refused");
    };
    let direct = learn_compatible_set_with_context(&model, &records, options, &ctx).unwrap_err();
    assert_same(&compiled, &direct);
}

/// `per_benefit * benefit - cost`, four records on cost alone: the cheaper option is chosen
/// three times in four, so the consistency is `ln 3` with standard error `sqrt(4 / 3)`.
fn cost_only() -> (ValueModel, ChoiceRecordSet) {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "events".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "per_benefit".into(),
                units: "utility/unit".into(),
                lower: 0.0,
                upper: 50.0,
                assumption_id: "declared_per_benefit".into(),
            }],
            vec![],
        ),
        value: P::Sub(
            Box::new(P::Mul(Box::new(P::Parameter(0)), Box::new(P::Attribute(0)))),
            Box::new(P::Attribute(1)),
        ),
        utility_units: "usd".into(),
    };
    let records = (0..4_u8)
        .map(|r| {
            let benefit = f64::from(r);
            let chosen = if r >= 3 { "o1" } else { "o0" };
            let id = format!("r{r}");
            record(&id, "clinician", &[("o0", &[benefit, 0.0]), ("o1", &[benefit, 1.0])], chosen)
        })
        .collect();
    let records = ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "clinic patients".into(),
        records,
        pooling: Pooling::Pooled { assumption_id: "assumption.shared_preferences".into() },
    };
    (model, records)
}

#[test]
fn the_compiled_choice_model_equals_the_direct_estimator() {
    let ctx = ExecutionContext::for_tests(31);
    let (model, records) = cost_only();
    let options = ChoiceModelOptions { n_draws: 4000, ..ChoiceModelOptions::default() };
    let program = compile_choice_model(model.clone(), records.clone(), options, &ctx).unwrap();
    assert_eq!(program.lowered().route(), "PreferenceChoiceModel");
    let unit = program.plan_view().unit_support().expect("bound to a unit").unit().to_owned();
    assert_eq!(unit, "preference_learning.choice_model");
    let compiled = program.execute(&ctx).unwrap();
    let direct = estimate_choice_model_with_context(&model, &records, options, &ctx).unwrap();
    assert_same(&compiled, &direct);
    let pooled = compiled.pooled.as_ref().expect("a pooled estimate");
    assert!(close(pooled.consistency, 3.0_f64.ln()));
    assert!(close(pooled.consistency_standard_error, (4.0_f64 / 3.0).sqrt()));
    // The compiled estimate is measured under the key the direct estimate is.
    assert_eq!(CALIBRATION_QUERY, "PreferenceTradeOffFromChoices");

    let compiled = program.accept(None, &ctx).unwrap();
    let direct = estimate_accepted_with_context(&model, &records, None, options, &ctx).unwrap();
    assert_same(&compiled, &direct);

    let declared = AcceptedPreferences::declared(model).unwrap();
    let relearn =
        compile_choice_model_relearn(declared.clone(), records.clone(), options, &ctx).unwrap();
    assert_eq!(relearn.lowered().route(), "PreferenceChoiceModel");
    let compiled = relearn.accept(None, &ctx).unwrap();
    let direct = declared.relearn_estimate_with_context(&records, None, options, &ctx).unwrap();
    assert_same(&compiled, &direct);
}
