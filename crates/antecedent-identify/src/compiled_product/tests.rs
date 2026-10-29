//! Tests of the compiled identification product.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::IdentificationObligation;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, IdentificationStatus, ScientificWorldId, Value, VariableId,
};
use antecedent_expr::{CausalExprArena, DerivationMeta, ExprId, IdentifiedEstimand};

use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    search_bounded_diagnostic,
};

use super::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, ProductRefusal, RequiredFactor, UnresolvedObligation, claim_code,
    outcome_code,
};

const OUTCOMES: [IdentificationOutcome; 6] = [
    IdentificationOutcome::Identified,
    IdentificationOutcome::Unidentified,
    IdentificationOutcome::Unsupported,
    IdentificationOutcome::TimedOut,
    IdentificationOutcome::NumericallyInfeasible,
    IdentificationOutcome::Uncalibrated,
];

fn query() -> CausalQuery {
    CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ))
}

fn empty_product(
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    numerical_value: Option<f64>,
) -> CompiledIdentificationProduct {
    let mut parts = CompiledProductParts::new(outcome, claim_class);
    parts.numerical_value = numerical_value;
    let estimands =
        if outcome == IdentificationOutcome::Identified { vec![estimand()] } else { Vec::new() };
    CompiledIdentificationProduct::new(
        estimands,
        CausalExprArena::new(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts,
    )
    .expect("coherent product")
}

fn estimand() -> IdentifiedEstimand {
    IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    )
}

fn interaction_target() -> antecedent_core::ScientificTarget {
    interaction_target_with(
        1.0,
        antecedent_core::TargetPopulation::AllObserved,
        VariableId::from_raw(2),
    )
}

fn interaction_target_with(
    active: f64,
    population: antecedent_core::TargetPopulation,
    outcome: VariableId,
) -> antecedent_core::ScientificTarget {
    use antecedent_core::{
        Conditioning, GraphFamily, InferenceSelection, Intervention, InterventionContrast,
        InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits,
        ScientificTarget, TargetForm, TimeFrame, WorldAction, WorldBinding, WorldCollection,
    };
    let binding = WorldBinding::new(
        population,
        ObservationUnit::new("subject").unwrap(),
        OutcomeQuantity::mean(outcome, QuantityUnits::dimensionless()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let levels = [(0.0, 0.0), (active, 0.0), (0.0, active), (active, active)];
    let worlds = WorldCollection::new(levels.into_iter().enumerate().map(|(index, (a, b))| {
        InterventionWorld::new(
            ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            WorldAction::at_decision([
                Intervention::set(VariableId::from_raw(0), Value::f64(a)),
                Intervention::set(VariableId::from_raw(1), Value::f64(b)),
            ])
            .unwrap(),
            index == 0,
            binding.clone(),
            TimeFrame::static_decision(),
        )
        .unwrap()
    }))
    .unwrap();
    let contrast = InterventionContrast::interaction(
        worlds,
        ScientificWorldId::from_raw(3),
        ScientificWorldId::from_raw(1),
        ScientificWorldId::from_raw(2),
        ScientificWorldId::from_raw(0),
    )
    .unwrap();
    ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    )
}

fn supported_interaction_product() -> CompiledIdentificationProduct {
    let mut product = empty_product(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
        None,
    );
    product.world_obligations = (0..4)
        .map(|id| {
            IdentificationObligation::for_world(
                ScientificWorldId::from_raw(id),
                IdentificationOutcome::Identified,
                IdentificationClaimClass::IdentifiedClaim,
                Vec::new(),
                None,
            )
        })
        .collect();
    product.assembled = Some(IdentificationObligation::assembled(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
        Vec::new(),
        None,
    ));
    product.target_identity = Some(interaction_target().identity());
    product
}

#[test]
fn unsupported_target_binding_needs_no_fabricated_identification_and_cannot_rebind() {
    let target = interaction_target();
    let mut product = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        None,
    );
    product.world_obligations = (0..4)
        .map(|id| {
            IdentificationObligation::for_world(
                ScientificWorldId::from_raw(id),
                IdentificationOutcome::Unidentified,
                IdentificationClaimClass::ModelConditional,
                vec![],
                None,
            )
        })
        .collect();
    product.assembled = Some(IdentificationObligation::assembled(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        vec![],
        None,
    ));
    let bound = product.bind_target(&target).unwrap();
    assert!(bound.identified().is_empty());
    assert!(!bound.is_identified_claim());
    assert_eq!(bound.target_identity(), Some(target.identity()));
    assert!(bound.clone().bind_target(&target).is_ok());
    let changed = interaction_target_with(
        2.0,
        antecedent_core::TargetPopulation::AllObserved,
        VariableId::from_raw(2),
    );
    assert!(matches!(
        bound.bind_target(&changed),
        Err(ProductRefusal::TargetIdentityMismatch { .. })
    ));
}

#[test]
fn target_validation_requires_each_world_exactly_once_and_an_assembled_check() {
    let target = interaction_target();
    let product = supported_interaction_product();
    assert_eq!(product.validate_target(&target), Ok(()));
    let mut missing = product.clone();
    missing.world_obligations.pop();
    assert_eq!(
        missing.validate_target(&target),
        Err(ProductRefusal::MissingWorldObligation { world: ScientificWorldId::from_raw(3) })
    );
    let mut duplicate = product.clone();
    duplicate.world_obligations.push(product.world_obligations[0].clone());
    assert_eq!(
        duplicate.validate_target(&target),
        Err(ProductRefusal::DuplicateWorldObligation { world: ScientificWorldId::from_raw(0) })
    );
    let mut foreign = product.clone();
    foreign.world_obligations[0].world = Some(ScientificWorldId::from_raw(9));
    assert_eq!(
        foreign.validate_target(&target),
        Err(ProductRefusal::UnexpectedWorldObligation { world: ScientificWorldId::from_raw(9) })
    );
    let mut no_assembled = product;
    no_assembled.assembled = None;
    assert_eq!(
        no_assembled.validate_target(&target),
        Err(ProductRefusal::MissingAssembledObligation)
    );
}

#[test]
fn target_validation_never_promotes_feasible_unsupported_or_conditional_worlds() {
    let target = interaction_target();
    for outcome in OUTCOMES {
        for class in [
            IdentificationClaimClass::ModelConditional,
            IdentificationClaimClass::UnsupportedExtrapolation,
        ] {
            let mut product = supported_interaction_product();
            let world = &mut product.world_obligations[2];
            world.outcome = outcome;
            world.claim_class = class;
            world.numerical_value = Some(4.0);
            assert_eq!(
                product.validate_target(&target),
                Err(ProductRefusal::IdentifiedWithoutWorldSupport {
                    world: ScientificWorldId::from_raw(2),
                })
            );
            product.outcome = IdentificationOutcome::Unidentified;
            product.claim_class = IdentificationClaimClass::ModelConditional;
            assert!(product.assembled_is_identified_claim());
            // A product that no longer states an identified claim but whose assembled functional
            // still does is refused for the same unsupported world, not accepted.
            assert_eq!(
                product.validate_target(&target),
                Err(ProductRefusal::IdentifiedWithoutWorldSupport {
                    world: ScientificWorldId::from_raw(2),
                })
            );
            product.assembled.as_mut().unwrap().claim_class =
                IdentificationClaimClass::ModelConditional;
            assert_eq!(product.validate_target(&target), Ok(()));
            assert!(!product.is_identified_claim());
        }
    }
}

#[test]
fn target_validation_requires_whole_functional_support_for_an_identified_product() {
    let target = interaction_target();
    let mut product = supported_interaction_product();
    let assembled = product.assembled.as_mut().unwrap();
    assembled.outcome = IdentificationOutcome::Unidentified;
    assembled.claim_class = IdentificationClaimClass::ModelConditional;
    assembled.numerical_value = Some(1.0);
    assert_eq!(
        product.validate_target(&target),
        Err(ProductRefusal::IdentifiedWithoutAssembledSupport)
    );
}

fn assert_identified_world_collection(
    result: &IdentificationResult,
    parts: CompiledProductParts,
    target: &antecedent_core::ScientificTarget,
) {
    use antecedent_core::{ScientificTarget, TargetForm};
    // A collection of independently identified means is not an assembled contrast.
    let mut world_parts = parts;
    world_parts.assembled = None;
    let worlds = match target.form() {
        TargetForm::Contrast(contrast) => contrast.worlds().clone(),
        _ => unreachable!(),
    };
    let world_target = ScientificTarget::new(
        TargetForm::Worlds(worlds),
        target.assumptions().clone(),
        target.inference(),
    );
    let unbound_worlds =
        CompiledIdentificationProduct::retain(result, world_parts.clone()).unwrap();
    assert!(!unbound_worlds.is_identified_claim());
    let bound_worlds = CompiledIdentificationProduct::retain_for_target(
        result,
        world_parts.clone(),
        &world_target,
    )
    .unwrap();
    assert!(bound_worlds.is_identified_claim());
    assert_eq!(bound_worlds.validate_target(&world_target), Ok(()));
    assert!(matches!(
        CompiledIdentificationProduct::retain_for_target(result, world_parts, target),
        Err(ProductRefusal::MissingAssembledObligation)
    ));
}

#[test]
fn actual_joint_identification_is_bound_to_scientific_content_not_world_labels() {
    use antecedent_core::{
        GraphFamily, InferenceSelection, Intervention, ModelAssumptions, NodeRef,
        ResponseFunctional, ResponseQuery, ScientificTarget,
    };
    use antecedent_graph::Admg;

    let mut graph = Admg::empty();
    let a = graph.add_node(NodeRef::Static(VariableId::from_raw(0))).unwrap();
    let b = graph.add_node(NodeRef::Static(VariableId::from_raw(1))).unwrap();
    let y = graph.add_node(NodeRef::Static(VariableId::from_raw(2))).unwrap();
    graph.insert_directed(a, y).unwrap();
    graph.insert_directed(b, y).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
        ]),
    });
    let result = crate::GeneralizedAdjustmentIdentifier::new()
        .identify_joint_admg_response(&graph, &query)
        .unwrap();
    assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
    let supported = supported_interaction_product();
    let mut parts = CompiledProductParts::new(supported.outcome, supported.claim_class);
    parts.world_obligations = supported.world_obligations;
    parts.assembled = supported.assembled;
    let target = interaction_target();
    let product =
        CompiledIdentificationProduct::retain_for_target(&result, parts.clone(), &target).unwrap();
    assert_eq!(product.target_identity(), Some(target.identity()));
    assert_eq!(product.validate_target(&target), Ok(()));
    let bayesian = target.clone().with_inference(InferenceSelection::Bayesian);
    assert_eq!(product.validate_target(&bayesian), Ok(()));
    let changed = ScientificTarget::new(
        target.form().clone(),
        ModelAssumptions::new(GraphFamily::Admg, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    );
    assert_eq!(
        product.validate_target(&changed),
        Err(ProductRefusal::TargetIdentityMismatch {
            expected: target.identity(),
            actual: changed.identity(),
        })
    );
    let rebound =
        CompiledIdentificationProduct::retain_for_target(&result, parts.clone(), &changed).unwrap();
    assert_ne!(product, rebound);
    assert_ne!(product.identity(), rebound.identity());
    assert_identified_world_collection(&result, parts.clone(), &target);
    let legacy = CompiledIdentificationProduct::retain(&result, parts).unwrap();
    assert_eq!(legacy.target_identity(), None);
    assert_ne!(legacy.identity(), product.identity());
    assert_eq!(legacy.validate_target(&target), Err(ProductRefusal::UnboundScientificTarget));
    for changed in [
        interaction_target_with(
            2.0,
            antecedent_core::TargetPopulation::AllObserved,
            VariableId::from_raw(2),
        ),
        interaction_target_with(
            1.0,
            antecedent_core::TargetPopulation::Treated,
            VariableId::from_raw(2),
        ),
        interaction_target_with(
            1.0,
            antecedent_core::TargetPopulation::AllObserved,
            VariableId::from_raw(3),
        ),
    ] {
        assert_eq!(
            product.validate_target(&changed),
            Err(ProductRefusal::TargetIdentityMismatch {
                expected: target.identity(),
                actual: changed.identity(),
            })
        );
    }
}

#[test]
fn temporal_surface_requires_every_scheduled_cell_and_its_whole_functional() {
    use antecedent_core::{
        DoseTimeSurface, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, Intervention,
        InterventionWorld, ModelAssumptions, ScientificTarget, TargetForm, TemporalPolicy,
        TimeFrame, WorldAction, WorldCollection,
    };
    let source = interaction_target();
    let antecedent_core::TargetForm::Contrast(contrast) = source.form() else {
        unreachable!("interaction fixture")
    };
    let binding = contrast.worlds().worlds()[0].binding().clone();
    let make_target = |history| {
        let worlds = WorldCollection::new((0..4).map(|id| {
            let level = if id < 2 { 0.0 } else { 1.0 };
            let horizon = id % 2 + 1;
            InterventionWorld::new(
                ScientificWorldId::from_raw(id),
                WorldAction::schedule(
                    Intervention::set(VariableId::from_raw(0), Value::f64(level)),
                    TemporalPolicy::pulse(0),
                )
                .unwrap(),
                id == 0,
                binding.clone(),
                TimeFrame::temporal(
                    horizon,
                    history,
                    InitialLaw::named("observed_origin").unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        }))
        .unwrap();
        let surface = DoseTimeSurface::new(
            worlds,
            VariableId::from_raw(0),
            [0.0, 1.0],
            [1, 2],
            (0..4).map(ScientificWorldId::from_raw),
        )
        .unwrap();
        ScientificTarget::new(
            TargetForm::DoseTime(surface),
            ModelAssumptions::new(GraphFamily::TemporalDag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        )
    };
    let target = make_target(HistoryWindow::max_lag(2));
    let mut product = supported_interaction_product();
    product.target_identity = Some(target.identity());
    assert_eq!(product.validate_target(&target), Ok(()));
    product.assembled = None;
    assert_eq!(product.validate_target(&target), Err(ProductRefusal::MissingAssembledObligation));
    let changed = make_target(HistoryWindow::max_lag(1));
    assert!(matches!(
        product.validate_target(&changed),
        Err(ProductRefusal::TargetIdentityMismatch { .. })
    ));
    product.world_obligations.pop();
    assert_eq!(
        product.validate_target(&target),
        Err(ProductRefusal::MissingWorldObligation { world: ScientificWorldId::from_raw(3) })
    );
}

#[test]
fn mediation_cannot_acquire_authority_from_generic_world_validation() {
    use antecedent_core::{
        GraphFamily, InferenceSelection, Intervention, MediationArms, MediationComponent,
        MediationTarget, ModelAssumptions, ScientificTarget, TargetForm, TimeFrame,
    };
    let source = interaction_target();
    let TargetForm::Contrast(contrast) = source.form() else { unreachable!("interaction fixture") };
    let binding = contrast.worlds().worlds()[0].binding().clone();
    let arms = MediationArms::new(
        VariableId::from_raw(0),
        [VariableId::from_raw(1)],
        Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        Intervention::set(VariableId::from_raw(0), Value::f64(0.0)),
        MediationComponent::Total,
    )
    .unwrap();
    let mediation =
        MediationTarget::new(binding, TimeFrame::static_decision(), None, arms, []).unwrap();
    let target = ScientificTarget::new(
        TargetForm::Mediation(mediation),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    );
    let mut product = supported_interaction_product();
    product.target_identity = Some(target.identity());
    assert_eq!(
        product.validate_target(&target),
        Err(ProductRefusal::MediationRequiresTheoremScopedSupport)
    );
}

#[test]
fn six_identification_outcomes_do_not_compare_equal() {
    assert_eq!(OUTCOMES.len(), 6);
    let mut codes = OUTCOMES.map(outcome_code);
    codes.sort_unstable();
    assert_eq!(codes, [1, 2, 3, 4, 5, 6]);
    for (index, left) in OUTCOMES.iter().enumerate() {
        for right in &OUTCOMES[index + 1..] {
            assert_ne!(left, right);
        }
    }
}

#[test]
fn model_conditional_and_unsupported_extrapolation_are_not_identified_claims() {
    assert_ne!(
        IdentificationClaimClass::ModelConditional,
        IdentificationClaimClass::IdentifiedClaim
    );
    assert_ne!(
        IdentificationClaimClass::UnsupportedExtrapolation,
        IdentificationClaimClass::IdentifiedClaim
    );
    assert!(!IdentificationClaimClass::ModelConditional.is_identified_claim());
    assert!(!IdentificationClaimClass::UnsupportedExtrapolation.is_identified_claim());
    assert_ne!(
        claim_code(IdentificationClaimClass::ModelConditional),
        claim_code(IdentificationClaimClass::IdentifiedClaim)
    );
    assert_ne!(
        claim_code(IdentificationClaimClass::UnsupportedExtrapolation),
        claim_code(IdentificationClaimClass::IdentifiedClaim)
    );

    let conditional = empty_product(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::ModelConditional,
        Some(1.0),
    );
    let extrapolated = empty_product(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::UnsupportedExtrapolation,
        Some(2.0),
    );
    let identified = empty_product(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
        None,
    );
    assert!(!conditional.is_identified_claim());
    assert!(!extrapolated.is_identified_claim());
    assert!(identified.is_identified_claim());
    assert!(conditional.is_numerically_feasible());
}

#[test]
fn numerically_feasible_value_beside_unidentified_is_not_an_identified_claim() {
    let feasible = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Some(1.25),
    );
    assert_eq!(feasible.numerical_value(), Some(1.25));
    assert_eq!(feasible.outcome(), IdentificationOutcome::Unidentified);
    assert!(feasible.is_numerically_feasible());
    assert!(!feasible.is_identified_claim());
    assert_ne!(feasible.outcome(), IdentificationOutcome::Identified);
    assert_ne!(feasible.outcome(), IdentificationOutcome::NumericallyInfeasible);

    let infeasible = empty_product(
        IdentificationOutcome::NumericallyInfeasible,
        IdentificationClaimClass::ModelConditional,
        Some(1.25),
    );
    assert!(!infeasible.is_numerically_feasible());
    assert!(!infeasible.is_identified_claim());
}

fn refused(
    estimands: Vec<IdentifiedEstimand>,
    assumptions: AssumptionSet,
    parts: CompiledProductParts,
) -> ProductRefusal {
    CompiledIdentificationProduct::new(
        estimands,
        CausalExprArena::new(),
        DerivationTrace::default(),
        assumptions,
        Vec::new(),
        parts,
    )
    .expect_err("incoherent product is refused")
}

#[test]
fn an_outcome_that_disagrees_with_its_claim_or_estimands_is_refused() {
    // Identified with nothing retained.
    let identified_claim = || {
        CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        )
    };
    assert_eq!(
        refused(Vec::new(), AssumptionSet::new(), identified_claim()),
        ProductRefusal::IdentifiedWithoutEstimand
    );

    // Identified while an assumption it relies on is contradicted.
    let mut assumptions = AssumptionSet::new();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Positivity,
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Supported,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Consistency,
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Contradicted,
    });
    assert_eq!(
        refused(vec![estimand()], assumptions, identified_claim()),
        ProductRefusal::IdentifiedWithContradictedAssumption { index: 1 }
    );

    // An unidentified outcome may not state the identified-claim class.
    assert_eq!(
        refused(
            Vec::new(),
            AssumptionSet::new(),
            CompiledProductParts::new(
                IdentificationOutcome::Unidentified,
                IdentificationClaimClass::IdentifiedClaim,
            )
        ),
        ProductRefusal::UnidentifiedWithIdentifiedClaim
    );

    // A failure outcome may not state the identified-claim class or retain an estimand.
    for outcome in [
        IdentificationOutcome::Unsupported,
        IdentificationOutcome::TimedOut,
        IdentificationOutcome::NumericallyInfeasible,
        IdentificationOutcome::Uncalibrated,
    ] {
        assert_eq!(
            refused(
                Vec::new(),
                AssumptionSet::new(),
                CompiledProductParts::new(outcome, IdentificationClaimClass::IdentifiedClaim)
            ),
            ProductRefusal::IdentifiedClaimClassBesideFailure { outcome }
        );
        assert_eq!(
            refused(
                vec![estimand()],
                AssumptionSet::new(),
                CompiledProductParts::new(outcome, IdentificationClaimClass::ModelConditional)
            ),
            ProductRefusal::EstimandBesideNonIdentification { outcome }
        );
    }

    // The same rule covers world and assembled obligations.
    let mut with_world = CompiledProductParts::new(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
    );
    with_world.world_obligations = vec![IdentificationObligation::for_world(
        ScientificWorldId::from_raw(1),
        IdentificationOutcome::TimedOut,
        IdentificationClaimClass::IdentifiedClaim,
        Vec::new(),
        None,
    )];
    assert_eq!(
        refused(Vec::new(), AssumptionSet::new(), with_world),
        ProductRefusal::IdentifiedClaimClassBesideFailure {
            outcome: IdentificationOutcome::TimedOut
        }
    );
}

#[test]
fn assembled_contrast_is_not_identified_when_each_world_is_only_feasible() {
    let world = |raw: u32| {
        IdentificationObligation::for_world(
            ScientificWorldId::from_raw(raw),
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Vec::new(),
            Some(f64::from(raw)),
        )
    };
    let assembled = IdentificationObligation::assembled(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Vec::new(),
        Some(4.0),
    );
    let mut parts = CompiledProductParts::new(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
    );
    parts.world_obligations = vec![world(1), world(2)];
    parts.assembled = Some(assembled);
    let product = CompiledIdentificationProduct::new(
        vec![estimand()],
        CausalExprArena::new(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts,
    )
    .unwrap();

    assert!(
        product.world_obligations().iter().all(IdentificationObligation::is_numerically_feasible)
    );
    assert!(product.world_obligations().iter().all(|world| !world.is_identified_claim()));
    assert_eq!(
        product.assembled_obligation().map(IdentificationObligation::outcome),
        Some(IdentificationOutcome::Unidentified)
    );
    assert!(
        product
            .assembled_obligation()
            .is_some_and(IdentificationObligation::is_numerically_feasible,)
    );
    assert!(!product.assembled_is_identified_claim());
    assert!(!product.is_identified_claim());
}

fn identified_result_with_declared_positivity() -> IdentificationResult {
    let mut assumptions = AssumptionSet::new();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Positivity,
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    let mut derivation = DerivationTrace::default();
    derivation.push("backdoor", "adjustment set blocks back-door paths");
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    IdentificationResult::identified(
        query(),
        vec![estimand],
        CausalExprArena::new(),
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    )
}

#[test]
fn retain_keeps_functional_derivation_assumptions_factors_obligations_and_bounded_search() {
    let mut result = identified_result_with_declared_positivity();
    result
        .diagnostics
        .push(search_bounded_diagnostic("identify.example.search_bounded", "stopped"));
    result.diagnostics.push(Diagnostic::new(
        crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Warning,
        "capped",
    ));
    result.diagnostics.push(Diagnostic::new(
        crate::temporal_mag::HISTORY_CAPPED,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Warning,
        "history",
    ));
    result.diagnostics.push(Diagnostic::new(
        "identify.scientific.open_path",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "open",
    ));

    let mut parts = CompiledProductParts::new(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
    );
    parts.required_factors =
        vec![RequiredFactor::new("outcome_mean", Arc::from([VariableId::from_raw(1)]))];
    parts.unresolved_obligations = vec![UnresolvedObligation::new("support.overlap", "empty cell")];

    // A timeout cannot retain the identified functional of this result.
    let timed_out = CompiledProductParts::new(
        IdentificationOutcome::TimedOut,
        IdentificationClaimClass::ModelConditional,
    );
    assert_eq!(
        CompiledIdentificationProduct::retain(&result, timed_out).unwrap_err(),
        ProductRefusal::EstimandBesideNonIdentification {
            outcome: IdentificationOutcome::TimedOut
        }
    );
    // An identified outcome cannot be retained from a result that is not identified.
    let none = IdentificationResult::not_identified(
        query(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    );
    assert_eq!(
        CompiledIdentificationProduct::retain(&none, parts.clone()).unwrap_err(),
        ProductRefusal::IdentifiedFromUnidentifiedResult {
            status: IdentificationStatus::NotIdentified
        }
    );
    // An identified outcome may carry declared premises and must not carry an open obligation.
    assert_eq!(
        CompiledIdentificationProduct::retain(&result, parts.clone()).unwrap_err(),
        ProductRefusal::IdentifiedWithUnresolvedObligation { index: 0 }
    );
    let mut unidentified = parts.clone();
    unidentified.outcome = IdentificationOutcome::Unidentified;
    unidentified.claim_class = IdentificationClaimClass::ModelConditional;
    let carrying = CompiledIdentificationProduct::retain(&result, unidentified).unwrap();
    assert_eq!(carrying.unresolved_obligations().len(), 1);
    assert_eq!(carrying.unresolved_obligations()[0].code(), "support.overlap");
    assert!(!carrying.is_identified_claim());
    parts.unresolved_obligations.clear();
    let product = CompiledIdentificationProduct::retain(&result, parts).unwrap();

    assert_eq!(product.identified().len(), 1);
    assert_eq!(product.identified()[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    assert_eq!(product.identified()[0].functional, ExprId::from_raw(0));
    assert!(product.arena().is_empty());
    assert_eq!(product.derivation().steps.len(), 1);
    assert_eq!(product.derivation().steps[0].rule.as_ref(), "backdoor");
    assert_eq!(product.assumptions().len(), 1);
    assert_eq!(product.premises().count(), 1);
    assert!(product.unresolved_obligations().is_empty());
    assert_eq!(product.required_factors().len(), 1);
    assert_eq!(product.required_factors()[0].name(), "outcome_mean");
    assert_eq!(product.bounded_search_diagnostics().len(), 3);
    assert!(product.bounded_search_diagnostics().iter().all(|diagnostic| {
        diagnostic.code.as_ref().ends_with(".search_bounded")
            || diagnostic.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            || diagnostic.code.as_ref() == crate::temporal_mag::HISTORY_CAPPED
    }));
    assert!(product.is_identified_claim());
}

#[test]
fn scientific_identity_ignores_numerical_values_and_bounded_search_diagnostics() {
    let with_number = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Some(0.0),
    );
    let without_number = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        None,
    );
    assert_eq!(with_number, without_number);
    assert_eq!(with_number.identity(), without_number.identity());
    assert_ne!(with_number.numerical_value(), without_number.numerical_value());

    let negative_zero = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Some(-0.0),
    );
    assert_eq!(with_number, negative_zero);
    assert_eq!(with_number.identity(), negative_zero.identity());

    let mut parts = CompiledProductParts::new(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
    );
    parts.world_obligations = vec![IdentificationObligation::for_world(
        ScientificWorldId::from_raw(7),
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Vec::new(),
        Some(9.0),
    )];
    parts.assembled = Some(IdentificationObligation::assembled(
        IdentificationOutcome::TimedOut,
        IdentificationClaimClass::UnsupportedExtrapolation,
        Vec::new(),
        Some(8.0),
    ));
    let worlds = CompiledIdentificationProduct::new(
        Vec::new(),
        CausalExprArena::new(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts.clone(),
    )
    .unwrap();
    parts.world_obligations[0] = IdentificationObligation::for_world(
        ScientificWorldId::from_raw(7),
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        Vec::new(),
        None,
    );
    if let Some(assembled) = parts.assembled.as_mut() {
        *assembled = IdentificationObligation::assembled(
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::UnsupportedExtrapolation,
            Vec::new(),
            None,
        );
    }
    let worlds_without_numbers = CompiledIdentificationProduct::new(
        Vec::new(),
        CausalExprArena::new(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts,
    )
    .unwrap();
    assert_eq!(worlds, worlds_without_numbers);
    assert_eq!(worlds.identity(), worlds_without_numbers.identity());
}

#[test]
fn bounded_search_diagnostics_are_retained_and_omitted_from_identity() {
    let diagnostic = search_bounded_diagnostic("identify.example.search_bounded", "stopped");
    let with_search = CompiledIdentificationProduct::new(
        Vec::new(),
        CausalExprArena::new(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        vec![diagnostic],
        CompiledProductParts::new(
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::ModelConditional,
        ),
    )
    .unwrap();
    let without_search = empty_product(
        IdentificationOutcome::TimedOut,
        IdentificationClaimClass::ModelConditional,
        None,
    );
    assert_eq!(with_search, without_search);
    assert_eq!(with_search.identity(), without_search.identity());
    assert_eq!(with_search.bounded_search_diagnostics().len(), 1);
    assert!(without_search.bounded_search_diagnostics().is_empty());

    let mut derivation = DerivationTrace::default();
    derivation.push("id", "hedge");
    let different_derivation = CompiledIdentificationProduct::new(
        Vec::new(),
        CausalExprArena::new(),
        derivation,
        AssumptionSet::new(),
        Vec::new(),
        CompiledProductParts::new(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
        ),
    )
    .unwrap();
    let unidentified = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        None,
    );
    assert_ne!(different_derivation, unidentified);
    assert_ne!(different_derivation.identity(), unidentified.identity());
}

#[test]
fn functional_identity_ignores_expression_derivation_metadata() {
    let mut arena = CausalExprArena::new();
    let root = arena.backdoor_mean(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        &[VariableId::from_raw(2)],
        Value::f64(1.0),
    );
    let parts = CompiledProductParts::new(
        IdentificationOutcome::Identified,
        IdentificationClaimClass::IdentifiedClaim,
    );
    let functional = CompiledIdentificationProduct::new(
        vec![IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            root,
        )],
        arena.clone(),
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts.clone(),
    )
    .unwrap();
    arena.set_derivation(root, DerivationMeta::rule("renamed", None));
    let same_functional = CompiledIdentificationProduct::new(
        vec![IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            root,
        )],
        arena,
        DerivationTrace::default(),
        AssumptionSet::new(),
        Vec::new(),
        parts,
    )
    .unwrap();
    let empty = empty_product(
        IdentificationOutcome::Unidentified,
        IdentificationClaimClass::ModelConditional,
        None,
    );
    assert_eq!(functional, same_functional);
    assert_eq!(functional.identity(), same_functional.identity());
    assert_ne!(functional.identity(), empty.identity());
    assert!(functional.is_identified_claim());
}
