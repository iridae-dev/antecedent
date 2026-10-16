//! An inverse query reads a composed-mechanism response, a cohort-mean contrast and a
//! transported response as licensed forward sources, each with its own refusals and with
//! its identity carried into the result.
//!
//! Oracles.
//!
//! - Gaussian composition. The interventional means of the outcome under `do(t = 0, 1, 2)`
//!   are `1.0, 2.5, 4.0` (a hand-built line `1 + 1.5 t`); the utility is the mean itself.
//! - Finite composition. The outcome is `hi` (coded 1) with probability `1/4, 1/2, 3/4` under
//!   `do(t = 0, 1, 2)`. `E[Y]` and `P(Y >= 1)` are those probabilities; the left-inverse
//!   median `inf { x : F(x) >= 1/2 }` is `0, 0, 1` (`F_a1(0) = 1/2` reaches the level exactly).
//! - Cohort contrast. `tau(z) = 2 + 0.5 z` over `z = (-1, 0.5, 1)` with weights
//!   `(0.2, 0.3, 0.5)` has mean `2.225`; utilities `2 x - 1 = 3.45` and `4 x - 1.5 = 7.4`.
//! - Transported policy values. `7.5 - 10 p` for the inspecting policy and `0` for holding,
//!   at the target faulty probability `p = 0.2`: `5.5` and `0`.

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "dyadic oracles; one scenario per test"
)]

use antecedent_contract::provider_transport::TransportedInputPolicy;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_transported_source::{
    TransportReceipt, TransportedRefusal, TransportedRepresentation, TransportedResponse,
    TransportedRoute,
};
use antecedent_design::inverse_forward_source::{
    ComposedActionResponse, ComposedFamily, ComposedResponse, DETAIL_COMPOSED_LAW,
    DETAIL_COMPOSED_LEVELS, DETAIL_INTERVAL, DETAIL_POINT_ONLY_LAW,
    DETAIL_TRANSPORTED_NOT_ACCEPTED, ForwardRepresentation, ForwardSourceKind,
    ForwardSourceRefusal, InverseDemand, cohort_forward_source, composed_forward_source,
    composed_forward_source_accepting, evaluate_inverse_on_source, forward_claim_digest,
    seal_on_source, seal_per_action_on_source, transported_forward_source,
};
use antecedent_design::inverse_query::{
    Comparison, FeasibilityStatus, GridScope, InverseConstraint, InverseQuery, SelectionOutcome,
    SelectionRule,
};
use antecedent_design::inverse_query_artifact::{InverseQueryArtifact, decode_parts, encode_parts};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportedPolicyRequest,
    evaluate_transported_policy,
};
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions,
    CovariateShiftDeclaration, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    CompactExport, ExportInput, ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_io_core::{CompositionQueryWire, CompositionResultWire};

mod common;

use FeasibilityStatus::{Feasible, Infeasible};

fn mean_coordinate(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        regime_id: regime.into(),
        functional_id: functional.into(),
        ..common::quantity("y")
    }
}

fn identity_contract(coordinates: &[(&str, ScientificQuantity)]) -> DecisionContract {
    DecisionContract {
        actions: coordinates
            .iter()
            .map(|(id, coordinate)| DecisionAction {
                id: (*id).into(),
                kind: ActionKind::Intervention,
                inputs: vec![coordinate.clone()],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn query_on(
    contract: DecisionContract,
    grid: &[&str],
    constraints: Vec<InverseConstraint>,
) -> InverseQuery {
    InverseQuery {
        contract,
        grid_order: grid.iter().map(|id| (*id).to_owned()).collect(),
        grid_scope: GridScope::FiniteEnumeration,
        constraints,
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn mean_at_least(target: f64) -> InverseConstraint {
    InverseConstraint::TargetMean { target, comparison: Comparison::AtLeast }
}

// ---------------------------------------------------------------------------
// Composed mechanisms
// ---------------------------------------------------------------------------

const IDS: [&str; 3] = ["a0", "a1", "a2"];

fn gaussian_response(means: [f64; 3]) -> ComposedResponse {
    ComposedResponse {
        family: ComposedFamily::LinearGaussian,
        assumed_mechanisms: vec!["u".into()],
        transport: None,
        actions: IDS
            .iter()
            .zip(means)
            .enumerate()
            .map(|(k, (id, mean))| ComposedActionResponse {
                action_id: (*id).into(),
                coordinate: mean_coordinate(&format!("do(t={k})"), "mean"),
                premises_digest: format!("{k:064x}"),
                query: CompositionQueryWire::GaussianInterventionLaw {
                    interventions: vec![("t".into(), k as f64)],
                    outcome: "y".into(),
                },
                result: CompositionResultWire::GaussianLaw {
                    mean_bits: mean.to_bits(),
                    variance_bits: 0.25_f64.to_bits(),
                },
                level_values: vec![],
            })
            .collect(),
    }
}

fn gaussian_contract() -> DecisionContract {
    let coordinates: Vec<(&str, ScientificQuantity)> = IDS
        .iter()
        .enumerate()
        .map(|(k, id)| (*id, mean_coordinate(&format!("do(t={k})"), "mean")))
        .collect();
    identity_contract(&coordinates)
}

#[test]
fn a_gaussian_composition_answers_a_target_mean_and_carries_its_identity() {
    let source = composed_forward_source(&gaussian_response([1.0, 2.5, 4.0])).unwrap();
    assert_eq!(source.origin().kind, ForwardSourceKind::ComposedMechanism);
    assert_eq!(source.origin().representation, ForwardRepresentation::PointOnly);
    assert_eq!(source.origin().assumed_mechanisms, vec!["u".to_owned()]);
    assert!(source.origin().transport.is_none());
    let query = query_on(gaussian_contract(), &IDS, vec![mean_at_least(2.0)]);
    let sourced = evaluate_inverse_on_source(&query, &source, InverseDemand::default()).unwrap();
    let statuses: Vec<_> = sourced.result.actions.iter().map(|a| a.point.unwrap()).collect();
    assert_eq!(statuses, vec![Infeasible, Feasible, Feasible]);
    assert!((sourced.result.actions[1].point_values[0].value.unwrap() - 2.5).abs() < 1e-12);
    assert!((sourced.result.actions[2].point_values[0].value.unwrap() - 4.0).abs() < 1e-12);
    // The composition variance is not a utility standard error.
    assert_eq!(sourced.result.actions[1].point_values[0].standard_error, None);
    assert_eq!(sourced.result.selected.as_deref(), Some("a1"));
    assert_eq!(sourced.result.selection, SelectionOutcome::Selected);
    assert!(sourced.result.selection_certified);
    assert_eq!(sourced.origin, *source.origin());
}

#[test]
fn composed_identity_is_order_invariant_and_follows_the_premises() {
    let base = gaussian_response([1.0, 2.5, 4.0]);
    let mut reversed = base.clone();
    reversed.actions.reverse();
    let a = composed_forward_source(&base).unwrap();
    let b = composed_forward_source(&reversed).unwrap();
    assert_eq!(a.origin().source_identity, b.origin().source_identity);
    // A changed premises digest at an unchanged shape and unchanged numbers.
    let mut edited = base.clone();
    edited.actions[1].premises_digest = format!("{:064x}", 99);
    let c = composed_forward_source(&edited).unwrap();
    assert_ne!(a.origin().source_identity, c.origin().source_identity);
    assert_ne!(a.origin().claim_digest, c.origin().claim_digest);
    // A changed number changes the claim digest.
    let d = composed_forward_source(&gaussian_response([1.0, 2.75, 4.0])).unwrap();
    assert_ne!(a.origin().claim_digest, d.origin().claim_digest);
}

#[test]
fn a_gaussian_composition_refuses_laws_intervals_and_malformed_responses() {
    let source = composed_forward_source(&gaussian_response([1.0, 2.5, 4.0])).unwrap();
    let quantile = query_on(
        gaussian_contract(),
        &IDS,
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 1.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let error =
        evaluate_inverse_on_source(&quantile, &source, InverseDemand::default()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::ComposedLawNotMaterialized { needed: "quantile" });
    assert_eq!(error.code(), "decision_contract_unsatisfied");
    assert_eq!(error.detail(), DETAIL_COMPOSED_LAW);
    assert!(error.to_refusal().remedy.is_some());

    let probability = query_on(
        gaussian_contract(),
        &IDS,
        vec![InverseConstraint::ProbabilityThreshold {
            outcome_threshold: 1.0,
            tail: antecedent_design::decision_contract::Tail::Upper,
            probability: 0.5,
            comparison: Comparison::AtLeast,
        }],
    );
    let error =
        evaluate_inverse_on_source(&probability, &source, InverseDemand::default()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::ComposedLawNotMaterialized { needed: "probability" });

    let mean = query_on(gaussian_contract(), &IDS, vec![mean_at_least(2.0)]);
    let error = evaluate_inverse_on_source(&mean, &source, InverseDemand { interval_region: true })
        .unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::ComposedMechanism)
    );
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.detail(), DETAIL_INTERVAL);

    // A mean addressed by an outcome-law coordinate is refused before the engine.
    let mut outcome = gaussian_response([1.0, 2.5, 4.0]);
    outcome.actions[0].coordinate = mean_coordinate("do(t=0)", "outcome");
    assert!(matches!(
        composed_forward_source(&outcome),
        Err(ForwardSourceRefusal::ComposedCoordinateIsOutcomeLaw(_))
    ));
    // A query and a result of different kinds.
    let mut mismatched = gaussian_response([1.0, 2.5, 4.0]);
    mismatched.actions[0].result = CompositionResultWire::FiniteDistribution {
        probabilities: vec![("hi".into(), 1.0_f64.to_bits())],
    };
    assert!(matches!(
        composed_forward_source(&mismatched),
        Err(ForwardSourceRefusal::ComposedQueryResultMismatch(_))
    ));
    // A family that does not match its queries.
    let mut wrong_family = gaussian_response([1.0, 2.5, 4.0]);
    wrong_family.family = ComposedFamily::Tabular;
    assert!(matches!(
        composed_forward_source(&wrong_family),
        Err(ForwardSourceRefusal::ComposedFamilyMismatch(_))
    ));
    // A non-finite value.
    let mut non_finite = gaussian_response([1.0, f64::NAN, 4.0]);
    non_finite.actions[1].premises_digest = format!("{:064x}", 7);
    assert!(matches!(
        composed_forward_source(&non_finite),
        Err(ForwardSourceRefusal::ComposedInvalid(_))
    ));
}

fn finite_response(hi: [f64; 3]) -> ComposedResponse {
    ComposedResponse {
        family: ComposedFamily::Tabular,
        assumed_mechanisms: vec![],
        transport: None,
        actions: IDS
            .iter()
            .zip(hi)
            .enumerate()
            .map(|(k, (id, p))| ComposedActionResponse {
                action_id: (*id).into(),
                coordinate: mean_coordinate(&format!("do(t={k})"), "outcome"),
                premises_digest: format!("{:064x}", 100 + k),
                query: CompositionQueryWire::FiniteInterventionDistribution {
                    interventions: vec![("t".into(), k.to_string())],
                    outcome: "y".into(),
                },
                result: CompositionResultWire::FiniteDistribution {
                    probabilities: vec![
                        ("lo".into(), (1.0 - p).to_bits()),
                        ("hi".into(), p.to_bits()),
                    ],
                },
                level_values: vec![("lo".into(), 0.0), ("hi".into(), 1.0)],
            })
            .collect(),
    }
}

fn finite_contract() -> DecisionContract {
    let coordinates: Vec<(&str, ScientificQuantity)> = IDS
        .iter()
        .enumerate()
        .map(|(k, id)| (*id, mean_coordinate(&format!("do(t={k})"), "outcome")))
        .collect();
    identity_contract(&coordinates)
}

#[test]
fn a_finite_composition_answers_mean_quantile_and_probability_exactly() {
    let source = composed_forward_source(&finite_response([0.25, 0.5, 0.75])).unwrap();
    assert_eq!(source.origin().representation, ForwardRepresentation::Law);
    assert!(source.action_claims().is_some());
    let statuses = |constraint: InverseConstraint| -> Vec<FeasibilityStatus> {
        let query = query_on(finite_contract(), &IDS, vec![constraint]);
        evaluate_inverse_on_source(&query, &source, InverseDemand::default())
            .unwrap()
            .result
            .actions
            .iter()
            .map(|a| a.point.unwrap())
            .collect()
    };
    // E[Y] = 1/4, 1/2, 3/4 against 1/2 (the boundary is feasible within tolerance).
    assert_eq!(statuses(mean_at_least(0.5)), vec![Infeasible, Feasible, Feasible]);
    // Left-inverse medians are 0, 0, 1.
    assert_eq!(
        statuses(InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 1.0,
            comparison: Comparison::AtLeast
        }),
        vec![Infeasible, Infeasible, Feasible]
    );
    // P(Y >= 1) = 1/4, 1/2, 3/4 against 0.6.
    assert_eq!(
        statuses(InverseConstraint::ProbabilityThreshold {
            outcome_threshold: 1.0,
            tail: antecedent_design::decision_contract::Tail::Upper,
            probability: 0.6,
            comparison: Comparison::AtLeast,
        }),
        vec![Infeasible, Infeasible, Feasible]
    );
    let query = query_on(finite_contract(), &IDS, vec![mean_at_least(0.5)]);
    let result = evaluate_inverse_on_source(&query, &source, InverseDemand::default()).unwrap();
    assert!((result.result.actions[0].point_values[0].value.unwrap() - 0.25).abs() < 1e-12);
    // The law is exact given the declared mechanisms, which carry no calibration.
    assert_eq!(result.result.actions[0].point_values[0].standard_error, None);
    assert_eq!(result.result.point_source, None);
    assert!(result.result.grid_fully_decided);
    // An action the source does not answer is unevaluated, never interpolated.
    let mut partial = finite_response([0.25, 0.5, 0.75]);
    partial.actions.truncate(2);
    let source = composed_forward_source(&partial).unwrap();
    let result = evaluate_inverse_on_source(&query, &source, InverseDemand::default()).unwrap();
    assert_eq!(result.result.actions[2].point, Some(FeasibilityStatus::Unevaluated));
    assert!(!result.result.grid_fully_decided);
}

#[test]
fn a_finite_outcome_needs_its_levels_coded_exactly_once() {
    let mut missing = finite_response([0.25, 0.5, 0.75]);
    missing.actions[0].level_values.pop();
    let error = composed_forward_source(&missing).unwrap_err();
    assert!(matches!(error, ForwardSourceRefusal::ComposedLevelCoding { .. }));
    assert_eq!(error.detail(), DETAIL_COMPOSED_LEVELS);
    let mut extra = finite_response([0.25, 0.5, 0.75]);
    extra.actions[0].level_values.push(("mid".into(), 0.5));
    assert!(matches!(
        composed_forward_source(&extra),
        Err(ForwardSourceRefusal::ComposedLevelCoding { .. })
    ));
    let mut unnormalized = finite_response([0.25, 0.5, 0.75]);
    unnormalized.actions[1].result = CompositionResultWire::FiniteDistribution {
        probabilities: vec![("lo".into(), 0.5_f64.to_bits()), ("hi".into(), 0.75_f64.to_bits())],
    };
    assert!(matches!(
        composed_forward_source(&unnormalized),
        Err(ForwardSourceRefusal::ComposedInvalid(_))
    ));
}

#[test]
fn a_composed_source_seals_its_origin_and_a_consumer_re_derives_it() {
    let source = composed_forward_source(&gaussian_response([1.0, 2.5, 4.0])).unwrap();
    let query = query_on(gaussian_contract(), &IDS, vec![mean_at_least(2.0)]);
    let artifact = seal_on_source(query.clone(), &source, InverseDemand::default()).unwrap();
    assert_eq!(artifact.forward_origin(), Some(source.origin()));
    let bytes = artifact.to_bytes("composed-inverse").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.forward_origin(), Some(source.origin()));
    assert_eq!(consumed.identity(), artifact.identity());

    // The origin is part of the identity: the same query and claim without it differ.
    let bare = InverseQueryArtifact::new(
        query,
        antecedent_design::inverse_query::ForwardEvidence {
            point: source.claim().cloned(),
            ..antecedent_design::inverse_query::ForwardEvidence::default()
        },
    )
    .unwrap();
    assert_ne!(bare.identity().premises_digest, artifact.identity().premises_digest);
    assert_eq!(bare.identity().data_digest, artifact.identity().data_digest);

    // A rewritten origin is refused: the identity no longer matches.
    let (mut meta, contract, data) = decode_parts(&bytes).unwrap();
    meta.forward_origin.as_mut().unwrap().source_identity = "someone-elses-model".into();
    let forged = encode_parts(&meta, &contract, &data, "forged").unwrap();
    let error = InverseQueryArtifact::from_bytes(&forged, None).unwrap_err();
    assert!(error.to_string().contains("premises digest"), "{error}");

    // An origin that names another claim cannot be attached.
    let other = composed_forward_source(&gaussian_response([1.0, 2.75, 4.0])).unwrap();
    let bare = InverseQueryArtifact::new(
        query_on(gaussian_contract(), &IDS, vec![mean_at_least(2.0)]),
        antecedent_design::inverse_query::ForwardEvidence {
            point: source.claim().cloned(),
            ..antecedent_design::inverse_query::ForwardEvidence::default()
        },
    )
    .unwrap();
    let error = bare.with_forward_origin(other.origin().clone()).unwrap_err();
    assert!(error.to_string().contains("forward_origin_invalid"), "{error}");
}

#[test]
fn a_per_action_source_seals_one_artifact_per_action() {
    let source = composed_forward_source(&finite_response([0.25, 0.5, 0.75])).unwrap();
    let query = query_on(finite_contract(), &IDS, vec![mean_at_least(0.5)]);
    let error = seal_on_source(query.clone(), &source, InverseDemand::default()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::PerActionNotSealable);
    let sealed = seal_per_action_on_source(&query, &source, InverseDemand::default()).unwrap();
    assert_eq!(sealed.len(), 3);
    for (action, artifact) in &sealed {
        let origin = artifact.forward_origin().unwrap();
        let claim = artifact.evidence().point.as_ref().unwrap();
        assert_eq!(origin.claim_digest, forward_claim_digest(claim).unwrap());
        assert_eq!(artifact.query().grid_order, vec![action.clone()]);
        let bytes = artifact.to_bytes("per-action").unwrap();
        InverseQueryArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    }
    // a1 (E = 1/2) is feasible on its own artifact; a0 (E = 1/4) is not.
    assert_eq!(sealed[0].1.result().actions[0].point, Some(Infeasible));
    assert_eq!(sealed[1].1.result().actions[0].point, Some(Feasible));
}

// ---------------------------------------------------------------------------
// Cohort-mean contrast
// ---------------------------------------------------------------------------

fn wire(
    id: &str,
    role: &str,
    units: &str,
    functional: &str,
    regime: &str,
) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn retained_model() -> RetainedEffectModel {
    let export = CompactExport::build(ExportSpec {
        response: wire("y", "outcome", "mmHg", CATE_FUNCTIONAL_ID, "contrast:do(t=1)-do(t=0)"),
        inputs: vec![ExportInput {
            quantity: wire("z", "covariate", "1", "mean", "observed"),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: vec![2.0, 0.5],
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask: vec![],
    })
    .unwrap();
    RetainedEffectModel::from_compact_export(&export).unwrap()
}

fn cohort(population: &str) -> CohortBinding {
    CohortBinding::new(
        "clinic-2026-q1",
        population,
        vec!["r0".into(), "r1".into(), "r2".into()],
        vec!["z".into()],
        vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
        vec![0.2, 0.3, 0.5],
    )
    .unwrap()
}

fn cohort_source(
    population: &str,
    shift: Option<CovariateShiftDeclaration>,
) -> CohortDecisionSource {
    CohortDecisionSource::build(
        &retained_model(),
        &cohort(population),
        "mmHg",
        &CohortOptions { shift, expected_contrast_regime: None },
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
}

fn cohort_contract(coordinate: &ScientificQuantity, population: &str) -> DecisionContract {
    let action = |id: &str, slope: f64, offset: f64| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![coordinate.clone()],
        utility: UtilityExpr::sum(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(slope)),
            UtilityExpr::Const(offset),
        ),
    };
    DecisionContract {
        actions: vec![action("halve", 2.0, -1.0), action("treat", 4.0, -1.5)],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_cohort_contrast_is_a_point_source_with_its_own_law_refusal() {
    let source = cohort_source("target", None);
    let forward = cohort_forward_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap();
    assert_eq!(forward.origin().kind, ForwardSourceKind::CohortContrast);
    assert_eq!(forward.origin().source_identity, source.identity());
    assert_eq!(forward.origin().support, "supported");
    assert!(forward.origin().transport.is_none());
    let query = query_on(
        cohort_contract(source.quantity(), "target"),
        &["halve", "treat"],
        vec![mean_at_least(5.0)],
    );
    let result = evaluate_inverse_on_source(&query, &forward, InverseDemand::default()).unwrap();
    assert!((result.result.actions[0].point_values[0].value.unwrap() - 3.45).abs() < 1e-12);
    assert!((result.result.actions[1].point_values[0].value.unwrap() - 7.4).abs() < 1e-12);
    assert_eq!(result.result.selected.as_deref(), Some("treat"));
    assert_eq!(result.origin.source_identity, source.identity());

    let quantile = query_on(
        cohort_contract(source.quantity(), "target"),
        &["halve", "treat"],
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 5.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let error =
        evaluate_inverse_on_source(&quantile, &forward, InverseDemand::default()).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::PointOnlyLawRequested {
            kind: ForwardSourceKind::CohortContrast,
            needed: "quantile"
        }
    );
    assert_eq!(error.detail(), DETAIL_POINT_ONLY_LAW);
    let error =
        evaluate_inverse_on_source(&query, &forward, InverseDemand { interval_region: true })
            .unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::CohortContrast));
}

#[test]
fn a_shifted_cohort_is_a_transported_source_only_when_accepted() {
    let shift = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let source = cohort_source("clinic_b", Some(shift));
    let error =
        cohort_forward_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::TransportedNotAccepted);
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.detail(), DETAIL_TRANSPORTED_NOT_ACCEPTED);

    let forward =
        cohort_forward_source(&source, TransportedInputPolicy::AcceptsTransported).unwrap();
    assert_eq!(forward.origin().kind, ForwardSourceKind::TransportedResponse);
    assert_eq!(forward.origin().support, "extrapolative");
    let receipt = forward.origin().transport.as_ref().unwrap();
    assert_eq!(receipt.route, TransportedRoute::ShiftedCohort);
    assert_eq!(receipt.source_populations, vec!["target".to_owned()]);
    assert_eq!(receipt.target_population, "clinic_b");
    assert_eq!(receipt.transport_identities.len(), 1);
    // The model's standard error is retained as a diagnostic only.
    assert!(receipt.uncalibrated_standard_error.is_some());

    let query = query_on(
        cohort_contract(source.quantity(), "clinic_b"),
        &["halve", "treat"],
        vec![mean_at_least(5.0)],
    );
    let result = evaluate_inverse_on_source(&query, &forward, InverseDemand::default()).unwrap();
    assert!((result.result.actions[1].point_values[0].value.unwrap() - 7.4).abs() < 1e-12);
    assert_eq!(result.result.actions[1].point_values[0].standard_error, None);
    assert_eq!(result.origin.transport.as_ref().unwrap().target_population, "clinic_b");

    // The transport identity is sealed and survives a consumer's replay.
    let artifact = seal_on_source(query, &forward, InverseDemand::default()).unwrap();
    let bytes = artifact.to_bytes("shifted-cohort").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.forward_origin().unwrap().transport, forward.origin().transport);

    // A cohort in the model's own population is not a transported response.
    assert!(matches!(
        TransportedResponse::from_shifted_cohort(&cohort_source("target", None)),
        Err(antecedent_design::decision_transported_source::TransportedRefusal::NotTransported)
    ));
}

// ---------------------------------------------------------------------------
// Transported policy values
// ---------------------------------------------------------------------------

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

fn alarm_model() -> MeasurementModel {
    MeasurementModel {
        model_id: "inspection_alarm".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "alarm".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "category".into(),
        },
        construct: MeasurementVariable {
            id: "unit_condition".into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: "plant_a".into(),
            regime_id: "operations".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.75,
                specificity: 0.75,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion {
            coordinate_system_id: "alarm_panel".into(),
            version: 1,
        },
        evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
    }
}

fn commit(label: &str, immediate: [f64; 2]) -> ActionSpec {
    ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: immediate.to_vec(),
        delayed: None,
    }
}

fn inspection_model() -> FiniteSequentialModel {
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.5, 0.5],
        horizon: 2,
        actions: vec![
            ActionSpec {
                label: "inspect".into(),
                role: ActionRole::Continue,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: vec![1.0, 0.0, 0.0, 1.0],
                channel: Some(ObservationChannel { measurement: alarm_model() }),
                immediate_utility: vec![0.0, 0.0],
                delayed: None,
            },
            commit("ship", [10.0, -10.0]),
            commit("hold", [0.0, 0.0]),
        ],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(3),
    }
}

fn policy_value_coordinate(policy: &str) -> ScientificQuantity {
    ScientificQuantity {
        population_id: "plant_b".into(),
        regime_id: format!("policy:{policy}"),
        functional_id: "mean".into(),
        role: QuantityRole::Outcome,
        ..common::quantity("policy_value")
    }
}

fn transported_policies(faulty: f64) -> TransportedResponse {
    let source = inspection_model();
    let mut rows = vec![
        (ModelMechanism::InitialState, MechanismStatus::Shifted),
        (ModelMechanism::Utility, MechanismStatus::Invariant),
    ];
    for (action, spec) in source.actions.iter().enumerate() {
        rows.push((ModelMechanism::Transition { action }, MechanismStatus::Invariant));
        if spec.channel.is_some() {
            rows.push((ModelMechanism::Observation { action }, MechanismStatus::Invariant));
        }
    }
    let diagram = SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: rows,
    };
    let inspect = {
        let after = |observation: usize| {
            HistoryKey::from_steps([HistoryStep { action: INSPECT, observation }])
        };
        ContingentPolicy::new("inspect_then_ship_on_quiet")
            .decide(HistoryKey::root(), INSPECT)
            .decide(after(0), SHIP)
            .decide(after(1), HOLD)
    };
    let hold = ContingentPolicy::new("hold_always").decide(HistoryKey::root(), HOLD);
    let law = [1.0 - faulty, faulty];
    let ctx = ExecutionContext::for_tests(5);
    let members: Vec<TransportedResponse> = [inspect, hold]
        .iter()
        .map(|policy| {
            let request = TransportedPolicyRequest {
                source: &source,
                policy,
                target_population: "plant_b",
                target_initial: Some(&law),
                diagram: Some(&diagram),
                target_utility: None,
                limits: EvaluationLimits::default(),
            };
            let evaluation = evaluate_transported_policy(&request, &ctx).unwrap();
            TransportedResponse::from_transported_policy(
                &evaluation,
                policy_value_coordinate(policy.version()),
                "sequential_evaluation",
            )
            .unwrap()
        })
        .collect();
    TransportedResponse::merge_points(&members).unwrap()
}

fn policy_contract() -> DecisionContract {
    let action = |id: &str, policy: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Policy,
        inputs: vec![policy_value_coordinate(policy)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![
            action("inspect", "inspect_then_ship_on_quiet"),
            action("hold", "hold_always"),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "plant_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_transported_policy_value_is_a_licensed_inverse_source_with_its_diagram() {
    let response = transported_policies(0.2);
    let forward =
        transported_forward_source(&response, TransportedInputPolicy::AcceptsTransported).unwrap();
    let query = query_on(policy_contract(), &["inspect", "hold"], vec![mean_at_least(5.0)]);
    let result = evaluate_inverse_on_source(&query, &forward, InverseDemand::default()).unwrap();
    // 7.5 - 10 * 0.2 = 5.5 reaches 5; holding is worth 0.
    assert!((result.result.actions[0].point_values[0].value.unwrap() - 5.5).abs() < 1e-12);
    assert!(result.result.actions[1].point_values[0].value.unwrap().abs() < 1e-12);
    assert_eq!(result.result.feasible_actions, vec!["inspect".to_owned()]);
    assert_eq!(result.result.selected.as_deref(), Some("inspect"));
    let receipt = result.origin.transport.as_ref().unwrap();
    assert_eq!(receipt.route, TransportedRoute::Policy);
    assert_eq!(receipt.target_population, "plant_b");
    assert_eq!(receipt.source_populations, vec!["plant_a".to_owned()]);
    assert_eq!(result.origin.support, SupportStatus::Extrapolative.as_str());
    assert_eq!(result.origin.representation, ForwardRepresentation::PointOnly);

    // At p = 0.8 the same grid reverses: -0.5 misses the target.
    let worse = transported_policies(0.8);
    let forward =
        transported_forward_source(&worse, TransportedInputPolicy::AcceptsTransported).unwrap();
    let result = evaluate_inverse_on_source(&query, &forward, InverseDemand::default()).unwrap();
    assert!(result.result.feasible_actions.is_empty());
    assert!((result.result.actions[0].point_values[0].value.unwrap() + 0.5).abs() < 1e-12);
    // Different transported claims at one transport differ in identity.
    assert_ne!(
        transported_policies(0.2).receipt().canonical_identity(),
        worse.receipt().canonical_identity()
    );
}

#[test]
fn a_transported_response_is_refused_where_only_native_data_is_accepted() {
    let response = transported_policies(0.2);
    let error = transported_forward_source(&response, TransportedInputPolicy::NativeTargetOnly)
        .unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::TransportedNotAccepted);
    // A transported point refuses a law and an interval with its own typed refusal.
    let forward =
        transported_forward_source(&response, TransportedInputPolicy::AcceptsTransported).unwrap();
    let nonlinear = {
        let mut contract = policy_contract();
        contract.actions[0].utility =
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
        query_on(contract, &["inspect", "hold"], vec![mean_at_least(5.0)])
    };
    let error =
        evaluate_inverse_on_source(&nonlinear, &forward, InverseDemand::default()).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::PointOnlyLawRequested {
            kind: ForwardSourceKind::TransportedResponse,
            needed: "nonlinear_utility"
        }
    );
    let query = query_on(policy_contract(), &["inspect", "hold"], vec![mean_at_least(5.0)]);
    let error =
        evaluate_inverse_on_source(&query, &forward, InverseDemand { interval_region: true })
            .unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::TransportedResponse)
    );
}

fn mechanism_receipt() -> TransportReceipt {
    TransportReceipt {
        route: TransportedRoute::Mechanism,
        source_identity: format!("{:064x}", 7),
        source_populations: vec!["trial".to_owned()],
        target_population: "target".to_owned(),
        transport_identities: vec![format!("{:064x}", 9)],
        assumptions: vec!["outcome mechanism invariant across populations".to_owned()],
        support: "extrapolative".to_owned(),
        representation: TransportedRepresentation::PointOnly,
        uncalibrated_standard_error: None,
    }
}

#[test]
fn a_composition_with_carried_mechanisms_is_not_native_target_data() {
    let mut response = gaussian_response([1.0, 2.5, 4.0]);
    response.transport = Some(mechanism_receipt());

    // The default stance refuses a carried composition exactly as it refuses
    // every other transported source.
    let error = composed_forward_source(&response).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::TransportedNotAccepted);
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.detail(), DETAIL_TRANSPORTED_NOT_ACCEPTED);

    // Accepted explicitly, the origin names the carrying and is never `supported`.
    let forward =
        composed_forward_source_accepting(&response, TransportedInputPolicy::AcceptsTransported)
            .unwrap();
    assert_eq!(forward.origin().kind, ForwardSourceKind::ComposedMechanism);
    assert_eq!(forward.origin().support, "extrapolative");
    assert_eq!(forward.origin().transport, response.transport);

    // The numbers are the native composition's: carrying changes the label.
    let query = query_on(
        identity_contract(&[
            ("a0", mean_coordinate("do(t=0)", "mean")),
            ("a1", mean_coordinate("do(t=1)", "mean")),
            ("a2", mean_coordinate("do(t=2)", "mean")),
        ]),
        &IDS,
        vec![mean_at_least(2.5)],
    );
    let result = evaluate_inverse_on_source(&query, &forward, InverseDemand::default()).unwrap();
    assert!((result.result.actions[1].point_values[0].value.unwrap() - 2.5).abs() < 1e-12);
    assert_eq!(result.origin.transport.as_ref().unwrap().target_population, "target");

    // The carrying is part of the origin identity: the same numbers without
    // it have another identity.
    let native = composed_forward_source(&gaussian_response([1.0, 2.5, 4.0])).unwrap();
    assert_ne!(native.origin().canonical_identity(), forward.origin().canonical_identity());
    assert_eq!(native.origin().support, "supported");
}

#[test]
fn a_carried_composition_with_a_wrong_or_inflated_receipt_is_refused() {
    let mut response = gaussian_response([1.0, 2.5, 4.0]);
    let mut receipt = mechanism_receipt();
    receipt.route = TransportedRoute::Policy;
    response.transport = Some(receipt);
    let error =
        composed_forward_source_accepting(&response, TransportedInputPolicy::AcceptsTransported)
            .unwrap_err();
    assert!(matches!(error, ForwardSourceRefusal::ComposedInvalid(_)));

    let mut receipt = mechanism_receipt();
    receipt.support = "supported".to_owned();
    response.transport = Some(receipt);
    let error =
        composed_forward_source_accepting(&response, TransportedInputPolicy::AcceptsTransported)
            .unwrap_err();
    assert!(matches!(
        error,
        ForwardSourceRefusal::Transported(TransportedRefusal::SupportLabelInflated(_))
    ));
}

#[test]
fn one_shifted_cohort_is_refused_in_one_shape_by_every_consumer_that_does_not_accept_it() {
    use antecedent_design::cohort_decision_input::cohort_forward_claim;
    let shift = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let source = cohort_source("clinic_b", Some(shift));
    let inverse =
        cohort_forward_source(&source, TransportedInputPolicy::NativeTargetOnly).unwrap_err();
    let claim = cohort_forward_claim(&source).unwrap_err();
    assert_eq!(inverse.code(), claim.code);
    assert_eq!(inverse.detail(), claim.detail);
    // The decision consumer's refusal for a policy that does not admit the
    // label carries the same code.
    let decision = TransportedRefusal::SupportBelowPolicy {
        supplied: SupportStatus::Extrapolative,
        required: SupportStatus::Supported,
    };
    assert_eq!(decision.code(), inverse.code());
}
