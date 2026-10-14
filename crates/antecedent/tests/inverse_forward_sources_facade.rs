//! An inverse query reaches the composed, cohort and transported forward sources and
//! the associational observational route through the facade, each with its own
//! refusals and its identity carried into the result and the sealed artifact.
//!
//! Oracles.
//!
//! - Linear-Gaussian chain `X = 1 + 2 e_X`, `M = 0.5 + 2 X + e_M`,
//!   `Y = -1 + 0.5 X + 3 M + 0.5 e_Y`: `E[Y | do(X = 0, 1, 2)] = 0.5, 7, 13.5`.
//! - Finite chain (binary, X -> M -> Y, X -> Y): `P(Y = 1 | do(X = 0)) = 0.18` and
//!   `P(Y = 1 | do(X = 1)) = 0.72`.
//! - Eight observed units `(x, z, y)` with `y = 1` in rows 2, 5, 6 and 8 and `x = 1` in
//!   rows 5-8: `P(E) = 1/2`, `P(x = 1) = 1/2`, `P(E | x = 1) = 3/4`, `P(x = 1 | E) = 3/4`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "closed-form oracles; one scenario per test"
)]

use std::sync::Arc;

use antecedent::analysis::inverse_query::{
    Comparison, FeasibilityStatus, GridScope, InverseConstraint, InverseQuery, SelectionOutcome,
    SelectionRule, consume as consume_inverse, export as export_inverse,
};
use antecedent::analysis::inverse_sources::{
    ComposedActionQuery, ForwardRepresentation, ForwardSourceKind, ForwardSourceRefusal,
    InverseDemand, ObservationalScenarioError, ObservationalScenarioQuery, OutcomeEvent,
    ScenarioCell, ScenarioCondition, TransportDeclaration, TransportReceipt,
    TransportedInputPolicy, TransportedResponse, TransportedRoute, evaluate,
    evaluate_observational, inspect_observational, inspect_origin, seal, seal_per_action,
    transported_source, verify_observational,
};
use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource, StudyBuilder,
};
use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, MechanismDeclaration, MechanismOriginKind, QuantityRole,
    ScientificQuantity, TransportAdmission,
};
use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_transported_source::TransportedRepresentation;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{Domain, TabularRow};

use FeasibilityStatus::{Feasible, Infeasible};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-9, "{a} vs {b}");
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn role(variable: &str) -> QuantityRole {
    match variable {
        "X" => QuantityRole::Treatment,
        "M" => QuantityRole::Mediator,
        "Y" => QuantityRole::Outcome,
        _ => QuantityRole::Covariate,
    }
}

fn coordinate(variable: &str, units: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: role(variable),
        units: units.into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration(variable: &str, parents: &[&str], units: &str) -> MechanismDeclaration {
    MechanismDeclaration {
        output: coordinate(variable, units, "pop"),
        parents: parents.iter().map(|p| coordinate(p, units, "pop")).collect(),
        conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
        origin: MechanismOriginKind::Fitted,
        artifact_id: format!("fit:{variable}"),
        evidence_id: format!("data:{variable}"),
        latent_causes: Vec::new(),
    }
}

fn linear(variable: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismInput {
    let names: Vec<&str> = parents.iter().map(|(p, _)| *p).collect();
    let slot = MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(parents.iter().map(|(_, c)| *c).collect::<Vec<f64>>()),
        sigma,
    };
    MechanismInput {
        declaration: declaration(variable, &names, "u"),
        source: MechanismSource::from_fitted_slot(&slot, &names).unwrap(),
        output_support: Domain::Real,
        parent_domains: names.iter().map(|p| ((*p).to_owned(), Domain::Real)).collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

fn request_of(mechanisms: Vec<MechanismInput>) -> MechanismCompositionRequest {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.declaration.parents {
            edges.push((p.variable_id.clone(), m.declaration.output.variable_id.clone()));
        }
    }
    MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms,
        edges,
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn gaussian_chain() -> Vec<MechanismInput> {
    vec![
        linear("X", 1.0, 2.0, &[]),
        linear("M", 0.5, 1.0, &[("X", 2.0)]),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ]
}

fn table_input(variable: &str, parents: &[&str], rows: &[(&[&str], [f64; 2])]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    let mut input = MechanismInput {
        declaration: declaration(variable, parents, "category"),
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, level)| ((*parent).to_owned(), (*level).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|p| ((*p).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    };
    input.declaration.output.units = "category".into();
    input
}

fn binary_chain() -> MechanismCompositionRequest {
    let mut request = request_of(vec![
        table_input("X", &[], &[(&[], [0.6, 0.4])]),
        table_input("M", &["X"], &[(&["0"], [0.8, 0.2]), (&["1"], [0.3, 0.7])]),
        table_input(
            "Y",
            &["M", "X"],
            &[
                (&["0", "0"], [0.9, 0.1]),
                (&["1", "0"], [0.5, 0.5]),
                (&["0", "1"], [0.7, 0.3]),
                (&["1", "1"], [0.1, 0.9]),
            ],
        ),
    ]);
    request.admit_declared_mechanisms = true;
    request
}

/// The coordinate an action's utility reads: `functional` of `Y` under `do(X = arm)`.
fn y_under(arm: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "Y".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "u".into(),
        population_id: "pop".into(),
        regime_id: format!("do(X={arm})"),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn identity_contract(arms: &[&str], functional: &str) -> DecisionContract {
    DecisionContract {
        actions: arms
            .iter()
            .map(|arm| DecisionAction {
                id: format!("x{arm}"),
                kind: ActionKind::Intervention,
                inputs: vec![y_under(arm, functional)],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "u".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "pop".into(),
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

const GRID: [&str; 3] = ["x0", "x1", "x2"];

fn gaussian_actions() -> Vec<ComposedActionQuery> {
    ["0", "1", "2"]
        .iter()
        .map(|arm| ComposedActionQuery {
            action_id: format!("x{arm}"),
            coordinate: y_under(arm, "mean"),
            query: CompositionQueryWire::GaussianInterventionLaw {
                interventions: vec![("X".into(), arm.parse::<f64>().unwrap())],
                outcome: "Y".into(),
            },
            level_values: vec![],
        })
        .collect()
}

fn finite_actions(levels: Vec<(String, f64)>) -> Vec<ComposedActionQuery> {
    ["0", "1"]
        .iter()
        .map(|arm| ComposedActionQuery {
            action_id: format!("x{arm}"),
            coordinate: y_under(arm, "outcome"),
            query: CompositionQueryWire::FiniteInterventionDistribution {
                interventions: vec![("X".into(), (*arm).to_owned())],
                outcome: "Y".into(),
            },
            level_values: levels.clone(),
        })
        .collect()
}

fn native() -> TransportedInputPolicy {
    TransportedInputPolicy::NativeTargetOnly
}

// ---------------------------------------------------------------------------
// Composed mechanisms
// ---------------------------------------------------------------------------

#[test]
fn a_linear_gaussian_composition_supplies_a_replayed_point_per_action() {
    let prepared =
        StudyBuilder::mechanism_composition(request_of(gaussian_chain()), &ctx()).unwrap();
    let source =
        prepared.inverse_forward_source(&gaussian_actions(), None, native(), &ctx()).unwrap();
    assert_eq!(source.origin().kind, ForwardSourceKind::ComposedMechanism);
    assert_eq!(source.origin().representation, ForwardRepresentation::PointOnly);
    assert!(source.origin().assumed_mechanisms.is_empty());
    assert!(source.origin().transport.is_none());

    let query =
        query_on(identity_contract(&["0", "1", "2"], "mean"), &GRID, vec![mean_at_least(5.0)]);
    let sourced = evaluate(&query, &source, InverseDemand::default()).unwrap();
    let statuses: Vec<_> = sourced.result.actions.iter().map(|a| a.point.unwrap()).collect();
    assert_eq!(statuses, vec![Infeasible, Feasible, Feasible]);
    near(sourced.result.actions[0].point_values[0].value.unwrap(), 0.5);
    near(sourced.result.actions[1].point_values[0].value.unwrap(), 7.0);
    near(sourced.result.actions[2].point_values[0].value.unwrap(), 13.5);
    assert_eq!(sourced.result.selected.as_deref(), Some("x1"));
    assert_eq!(sourced.result.selection, SelectionOutcome::Selected);
    assert_eq!(sourced.origin, *source.origin());

    let inspection = inspect_origin(&sourced.origin);
    assert_eq!(inspection.kind, "composed_mechanism");
    assert_eq!(inspection.representation, "point_only");
    assert_eq!(inspection.interval_region, "not_licensed");
    assert!(!inspection.transported);
}

#[test]
fn a_changed_refit_changes_the_source_identity_and_the_values() {
    let base = StudyBuilder::mechanism_composition(request_of(gaussian_chain()), &ctx()).unwrap();
    let mut refit = gaussian_chain();
    // M = 1 + 3 X, so E[Y | do(X = 1)] = -1 + 0.5 + 3 * 4 = 11.5.
    refit[1] = linear("M", 1.0, 1.0, &[("X", 3.0)]);
    let refit = StudyBuilder::mechanism_composition(request_of(refit), &ctx()).unwrap();
    let a = base.inverse_forward_source(&gaussian_actions(), None, native(), &ctx()).unwrap();
    let b = refit.inverse_forward_source(&gaussian_actions(), None, native(), &ctx()).unwrap();
    assert_ne!(a.origin().source_identity, b.origin().source_identity);
    assert_ne!(a.origin().claim_digest, b.origin().claim_digest);
    let query =
        query_on(identity_contract(&["0", "1", "2"], "mean"), &GRID, vec![mean_at_least(5.0)]);
    let result = evaluate(&query, &b, InverseDemand::default()).unwrap();
    near(result.result.actions[1].point_values[0].value.unwrap(), 11.5);
}

#[test]
fn a_point_composition_refuses_laws_intervals_and_the_wrong_family() {
    let prepared =
        StudyBuilder::mechanism_composition(request_of(gaussian_chain()), &ctx()).unwrap();
    let source =
        prepared.inverse_forward_source(&gaussian_actions(), None, native(), &ctx()).unwrap();
    let contract = || identity_contract(&["0", "1", "2"], "mean");

    let quantile = query_on(
        contract(),
        &GRID,
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 1.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let error = evaluate(&quantile, &source, InverseDemand::default()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::ComposedLawNotMaterialized { needed: "quantile" });
    assert_eq!(error.code(), "decision_contract_unsatisfied");
    assert!(error.to_refusal().remedy.is_some());

    let mean = query_on(contract(), &GRID, vec![mean_at_least(5.0)]);
    let error = evaluate(&mean, &source, InverseDemand { interval_region: true }).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::ComposedMechanism)
    );
    assert_eq!(error.code(), "cell_not_licensed");

    // A finite-distribution query of a linear-Gaussian composition is refused by the
    // composition itself and keeps its registered code.
    let wrong = finite_actions(vec![("0".into(), 0.0), ("1".into(), 1.0)]);
    let error = prepared.inverse_forward_source(&wrong, None, native(), &ctx()).unwrap_err();
    assert!(matches!(error, ForwardSourceRefusal::Artifact { .. }), "{error:?}");
}

#[test]
fn a_finite_composition_supplies_the_exact_law_of_each_action() {
    let prepared = StudyBuilder::mechanism_composition(binary_chain(), &ctx()).unwrap();
    let levels = vec![("0".to_owned(), 0.0), ("1".to_owned(), 1.0)];
    let source =
        prepared.inverse_forward_source(&finite_actions(levels), None, native(), &ctx()).unwrap();
    assert_eq!(source.origin().representation, ForwardRepresentation::Law);
    // Every mechanism of the binary chain was written down, not estimated.
    assert_eq!(source.origin().assumed_mechanisms.len(), 3);
    assert!(source.action_claims().is_some());

    let grid = ["x0", "x1"];
    let contract = identity_contract(&["0", "1"], "outcome");
    // E[Y] = 0.18, 0.72 against 0.5.
    let mean = query_on(contract.clone(), &grid, vec![mean_at_least(0.5)]);
    let result = evaluate(&mean, &source, InverseDemand::default()).unwrap();
    near(result.result.actions[0].point_values[0].value.unwrap(), 0.18);
    near(result.result.actions[1].point_values[0].value.unwrap(), 0.72);
    assert_eq!(result.result.actions[0].point, Some(Infeasible));
    assert_eq!(result.result.actions[1].point, Some(Feasible));
    // The left-inverse median is 0 under x0 (F(0) = 0.82) and 1 under x1 (F(0) = 0.28).
    let median = query_on(
        contract.clone(),
        &grid,
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 1.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let result = evaluate(&median, &source, InverseDemand::default()).unwrap();
    assert_eq!(result.result.actions[0].point, Some(Infeasible));
    assert_eq!(result.result.actions[1].point, Some(Feasible));

    // Each action is its own intervention: no single claim exists to seal, so each
    // action is sealed separately and bound to its own claim.
    let error = seal(mean.clone(), &source, InverseDemand::default()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::PerActionNotSealable);
    let sealed = seal_per_action(&mean, &source, InverseDemand::default()).unwrap();
    assert_eq!(sealed.len(), 2);
    assert_ne!(sealed[0].1.identity(), sealed[1].1.identity());
}

#[test]
fn an_uncoded_finite_outcome_is_refused_with_its_level_coding() {
    let prepared = StudyBuilder::mechanism_composition(binary_chain(), &ctx()).unwrap();
    let error = prepared
        .inverse_forward_source(&finite_actions(vec![("1".into(), 1.0)]), None, native(), &ctx())
        .unwrap_err();
    assert!(matches!(error, ForwardSourceRefusal::ComposedLevelCoding { .. }), "{error:?}");
    assert_eq!(error.code(), "mechanism_composition_incoherent");
}

#[test]
fn a_composition_with_admitted_mechanisms_is_a_transported_claim() {
    let mut mechanisms = gaussian_chain();
    mechanisms[1].declaration.output.population_id = "source".into();
    mechanisms[1].declaration.parents[0].population_id = "source".into();
    let mut request = request_of(mechanisms);
    request.admissions = vec![TransportAdmission {
        variable_id: "M".into(),
        coordinate: AdmissionCoordinate::Population,
        source_id: "source".into(),
        target_id: "pop".into(),
        basis: AdmissionBasis::SelectionDiagram { targets: vec!["X".into()] },
    }];
    let prepared = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    // No receipt: the claim would say every mechanism is native to the target.
    let error = prepared
        .inverse_forward_source(
            &gaussian_actions(),
            None,
            TransportedInputPolicy::AcceptsTransported,
            &ctx(),
        )
        .unwrap_err();
    assert!(matches!(error, ForwardSourceRefusal::ComposedInvalid(_)), "{error:?}");

    let receipt = TransportReceipt {
        route: TransportedRoute::Mechanism,
        source_identity: "mechanism-transport".into(),
        source_populations: vec!["source".into()],
        target_population: "pop".into(),
        transport_identities: vec![format!("{:064x}", 7)],
        assumptions: vec!["M is invariant across populations".into()],
        support: "extrapolative".into(),
        representation: TransportedRepresentation::PointOnly,
        uncalibrated_standard_error: None,
    };
    // A consumer that accepts only native target data refuses it.
    let error = prepared
        .inverse_forward_source(&gaussian_actions(), Some(receipt.clone()), native(), &ctx())
        .unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::TransportedNotAccepted);
    assert_eq!(error.code(), "cell_not_licensed");

    let source = prepared
        .inverse_forward_source(
            &gaussian_actions(),
            Some(receipt),
            TransportedInputPolicy::AcceptsTransported,
            &ctx(),
        )
        .unwrap();
    assert_eq!(source.origin().support, "extrapolative");
    let inspection = inspect_origin(source.origin());
    assert!(inspection.transported);
    assert_eq!(inspection.source_populations, vec!["source".to_owned()]);
    assert_eq!(inspection.target_population.as_deref(), Some("pop"));
}

#[test]
fn a_sealed_composed_query_exports_and_a_consumer_re_derives_it() {
    let prepared =
        StudyBuilder::mechanism_composition(request_of(gaussian_chain()), &ctx()).unwrap();
    let source =
        prepared.inverse_forward_source(&gaussian_actions(), None, native(), &ctx()).unwrap();
    let query =
        query_on(identity_contract(&["0", "1", "2"], "mean"), &GRID, vec![mean_at_least(5.0)]);
    let artifact = seal(query, &source, InverseDemand::default()).unwrap();
    assert_eq!(artifact.forward_origin(), Some(source.origin()));
    let bytes = export_inverse(&artifact, "composed-inverse").unwrap();
    let consumed = consume_inverse(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.forward_origin(), Some(source.origin()));
    assert_eq!(consumed.identity(), artifact.identity());
}

// ---------------------------------------------------------------------------
// Transported responses
// ---------------------------------------------------------------------------

#[test]
fn a_transported_response_is_read_only_when_accepted_and_carries_its_receipt() {
    let declaration = TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "transport-1".into(),
        source_populations: vec!["trial".into()],
        target_population: "pop".into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: Some(0.3),
        provider_id: "transport-provider".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "contract-1".into(),
    };
    let response = TransportedResponse::declared_point(
        vec![y_under("a", "mean"), y_under("b", "mean")],
        vec![1.0, 3.0],
        &declaration,
    )
    .unwrap();
    // A consumer that accepts only native target data refuses it.
    let error = transported_source(&response, native()).unwrap_err();
    assert_eq!(error, ForwardSourceRefusal::TransportedNotAccepted);

    let source = transported_source(&response, TransportedInputPolicy::AcceptsTransported).unwrap();
    assert_eq!(source.origin().kind, ForwardSourceKind::TransportedResponse);
    assert_eq!(source.origin().support, "extrapolative");
    let inspection = inspect_origin(source.origin());
    assert!(inspection.transported);
    assert_eq!(inspection.target_population.as_deref(), Some("pop"));

    let grid = ["xa", "xb"];
    let query = query_on(identity_contract(&["a", "b"], "mean"), &grid, vec![mean_at_least(2.0)]);
    let sourced = evaluate(&query, &source, InverseDemand::default()).unwrap();
    assert_eq!(sourced.result.actions[0].point, Some(Infeasible));
    assert_eq!(sourced.result.actions[1].point, Some(Feasible));
    near(sourced.result.actions[1].point_values[0].value.unwrap(), 3.0);
    // The source model's standard error is a diagnostic: no feasibility inherits it.
    assert_eq!(sourced.result.actions[1].point_values[0].standard_error, None);
    assert_eq!(sourced.origin.transport.as_ref().unwrap().source_populations, vec!["trial"]);

    // A transported point supplies no law and no interval region.
    let quantile = query_on(
        identity_contract(&["a", "b"], "mean"),
        &grid,
        vec![InverseConstraint::TargetQuantile {
            p: 0.5,
            target: 1.0,
            comparison: Comparison::AtLeast,
        }],
    );
    let error = evaluate(&quantile, &source, InverseDemand::default()).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::PointOnlyLawRequested {
            kind: ForwardSourceKind::TransportedResponse,
            needed: "quantile",
        }
    );
    let error = evaluate(&query, &source, InverseDemand { interval_region: true }).unwrap_err();
    assert_eq!(
        error,
        ForwardSourceRefusal::IntervalNotLicensed(ForwardSourceKind::TransportedResponse)
    );

    // The sealed artifact binds the whole receipt: another transport at the same values
    // is another artifact.
    let artifact = seal(query.clone(), &source, InverseDemand::default()).unwrap();
    let mut other = declaration.clone();
    other.transport_identities = vec!["diagram-2".into()];
    let moved = TransportedResponse::declared_point(
        vec![y_under("a", "mean"), y_under("b", "mean")],
        vec![1.0, 3.0],
        &other,
    )
    .unwrap();
    let moved_source =
        transported_source(&moved, TransportedInputPolicy::AcceptsTransported).unwrap();
    let moved_artifact = seal(query, &moved_source, InverseDemand::default()).unwrap();
    assert_ne!(artifact.identity(), moved_artifact.identity());
}

// ---------------------------------------------------------------------------
// Observational scenarios
// ---------------------------------------------------------------------------

const X: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
const Z: [f64; 8] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const Y: [f64; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

fn covariate(name: &str) -> ScientificQuantity {
    ScientificQuantity { role: QuantityRole::Covariate, ..coordinate(name, "units", "target") }
}

fn outcome() -> ScientificQuantity {
    ScientificQuantity { role: QuantityRole::Outcome, ..coordinate("y", "units", "target") }
}

fn observed_law(meaning: DistributionMeaningWire) -> DistributionArtifact {
    let columns = [covariate("x"), covariate("z"), outcome()];
    let identity = DistributionIdentity::new(
        meaning,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "observed-units".into(),
            provider_id: "enumerated".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .unwrap();
    let mut draws = Vec::with_capacity(24);
    for row in 0..8 {
        draws.extend([X[row], Z[row], Y[row]]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [8, 3],
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

fn scenario_query() -> ObservationalScenarioQuery {
    ObservationalScenarioQuery {
        outcome: outcome(),
        event: OutcomeEvent::AtLeast(1.0),
        scenarios: vec![ScenarioCell {
            id: "x_is_1".into(),
            conditions: vec![ScenarioCondition {
                coordinate: covariate("x"),
                lower: 1.0,
                upper: 1.0,
            }],
        }],
    }
}

#[test]
fn observational_scenarios_are_associational_and_re_derive() {
    let law = observed_law(DistributionMeaningWire::EmpiricalOutcome);
    let result = evaluate_observational(&scenario_query(), &law).unwrap();
    near(result.event_mass, 0.5);
    let x1 = &result.scenarios[0];
    near(x1.mass, 0.5);
    near(x1.event_given_scenario.unwrap(), 0.75);
    near(x1.scenario_given_event, 0.75);
    assert!(!result.ranks_actions);
    assert!(!result.is_intervention_effect);
    verify_observational(&result, &scenario_query(), &law).unwrap();

    let inspection = inspect_observational(&result);
    assert_eq!(inspection.standing, "associational");
    assert!(!inspection.ranks_actions);
    assert!(!inspection.is_intervention_effect);
    near(inspection.scenarios[0].1.unwrap(), 0.75);

    // A law whose meaning is interventional answers an intervention, not an observed
    // population.
    let interventional = observed_law(DistributionMeaningWire::InterventionalPredictive);
    let error = evaluate_observational(&scenario_query(), &interventional).unwrap_err();
    assert!(matches!(error, ObservationalScenarioError::MeaningNotObservational(_)));
    assert_eq!(error.code(), "distribution_meaning_mismatch");

    // A stored result of another law does not replay.
    let mut stored = result;
    stored.event_mass = 0.25;
    assert_eq!(
        verify_observational(&stored, &scenario_query(), &law),
        Err(ObservationalScenarioError::ResultMismatch)
    );
}
