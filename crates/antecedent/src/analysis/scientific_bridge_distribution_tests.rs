//! The facade of a bridge of draws: prepare, derive, inspect, export and
//! independently consume, the interval gate, and the bridge as a stage of a
//! propagation chain that continues into a transport and a utility.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeEndpoint, BridgeMapping, BridgeScope,
    MappingUncertainty, QuantityClass, SourceMappingDependence, TrialEffectPair,
    TrialLevelValidation, TransportabilityPremise,
};
use antecedent_contract::propagation_chain::{ChainReading, StageKind, TableConstruction};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_io::distribution_artifact::{
    DistributionCalibration, DistributionIdentity, DistributionMetadata, DistributionProvenance,
    DistributionTrust, DrawAlignment,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainStageWire, ChainUnaryWire,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use antecedent_stats::normal_ppf;

use super::*;
use crate::analysis::bridge_transport_decision::{
    compile_assessed_bridge_robust_decision, compile_assessed_bridge_transport_decision,
    compile_bridge_transport_decision, consume_assessed_bridge_robust_decision,
    consume_assessed_bridge_transport_decision, consume_bridge_transport_decision,
};
use crate::analysis::bridge_transport_premise_assessment::{
    BridgeTransportPremiseAssessmentRequest, compile_bridge_transport_premise_assessment,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use crate::analysis::propagation_chain::PropagationChain;
use crate::analysis::composition::{
    BundleBuilder, NodeKind, compile_bridge_bundle_recomputation,
    consume_portable_bridge_bundle_recomputation,
};

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn draws_table(
    rows: usize,
    mean: f64,
    sd: f64,
    semantic: DistributionMeaningWire,
) -> DistributionArtifact {
    let quantities = [quantity("trial:biomarker", "mmol_per_l"), quantity("trial:cost", "usd")];
    let identity = DistributionIdentity::new(
        semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study-1".into(),
            provider_id: "posterior.provider".into(),
            rng_id: "seed:11".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::with_capacity(rows * 2);
    for i in 0..rows {
        draws.push(mean + sd * normal_ppf((i as f64 + 0.5) / rows as f64));
        draws.push(
            0.2 + 0.3 * normal_ppf(((i * 487) % rows) as f64 / rows as f64 + 0.5 / rows as f64),
        );
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("table")
}

fn contract(
    mapping: BridgeMapping,
    uncertainty: Option<MappingUncertainty>,
    dependence: Option<SourceMappingDependence>,
) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Posterior,
        target_value: ValueKind::Posterior,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping,
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: uncertainty,
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

#[test]
fn predictive_transport_bridge_and_nonlinear_utility_compile_together() {
    let mut declaration = contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 }, None, None,
    );
    declaration.source_value = ValueKind::PredictiveLaw;
    declaration.target_value = ValueKind::PredictiveLaw;
    let established = BridgeScope {
        population_id: "validation-adults".into(), regime_id: "do(t=1)".into(),
    };
    let target = BridgeScope {
        population_id: "adults".into(), regime_id: "do(t=1)".into(),
    };
    declaration.evidence_scope = established.clone();
    declaration.transport = Some(TransportabilityPremise {
        id: "validation-to-target".into(),
        statement: "the trial-level surrogate relation holds in the target population".into(),
        established_in: established, applies_to: target,
        evidence: Some("selection-diagram-review".into()),
    });
    let outcome = declaration.target.quantity.clone();
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risk-sensitive".into(), kind: ActionKind::External,
                inputs: vec![outcome.clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            },
            DecisionAction {
                id: "linear".into(), kind: ActionKind::External,
                inputs: vec![outcome], utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(), target_population: "adults".into(),
        horizon: 0, structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let ctx = ExecutionContext::for_tests(31);
    let plan = compile_bridge_transport_decision(
        declaration, draws_table(64, 1.0, 0.3, DistributionMeaningWire::PosteriorPredictive),
        9, decision,
    ).expect("bridge, transport and utility compiler");
    assert_eq!(plan.plan_view().route().as_str(), "BridgeTransportDecision");
    let result = plan.execute(&ctx).expect("transported bridge utility");
    assert!(result.bridge.inspect().transported);
    assert_eq!(result.decision.decision.n_draws, 64);
    let bytes = plan.export(&result, "bridge-transport-utility", &ctx)
        .expect("complete composite artifact");
    let replay = consume_bridge_transport_decision(
        &bytes, plan.lowered().identity(), &ctx,
    ).expect("independent bridge and decision replay");
    assert_eq!(replay.decision.decision, result.decision.decision);
}

#[test]
#[ignore = "source-only matched validation and decision replay; scientific records are unearned"]
fn matched_trial_validation_and_bridge_decision_replay_together() {
    let trials = |offset: f64| TrialLevelValidation {
        trials: (0..20).map(|index| {
            let x = -1.0 + 2.0 * index as f64 / 19.0;
            TrialEffectPair { surrogate_effect: x,
                terminal_effect: 0.5 + 2.0 * x + offset
                    + (if index % 2 == 0 { 0.01 } else { -0.01 }),
            }
        }).collect(),
        min_trials: 3, min_r_squared: 0.0,
    };
    let source = trials(0.0);
    let premise = TransportabilityPremise {
        id: "validation-to-target".into(),
        statement: "trial surrogate relation transfers".into(),
        established_in: BridgeScope {
            population_id: "validation-adults".into(), regime_id: "do(t=1)".into(),
        },
        applies_to: BridgeScope {
            population_id: "adults".into(), regime_id: "do(t=1)".into(),
        },
        evidence: Some("independent-target-validation".into()),
    };
    let mut declaration = contract(BridgeMapping::TrialLevelLinear(source.clone()), None, None);
    declaration.source_value = ValueKind::PredictiveLaw;
    declaration.target_value = ValueKind::PredictiveLaw;
    declaration.evidence_scope = premise.established_in.clone();
    declaration.transport = Some(premise.clone());
    let outcome = declaration.target.quantity.clone();
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "use-bridge".into(), kind: ActionKind::External,
                inputs: vec![outcome.clone()], utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "risk-sensitive".into(), kind: ActionKind::External,
                inputs: vec![outcome],
                utility: UtilityExpr::product(
                    UtilityExpr::Input(0), UtilityExpr::Input(0),
                ),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(), target_population: "adults".into(),
        horizon: 0, structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let ctx = ExecutionContext::for_tests(43);
    let assessment = compile_bridge_transport_premise_assessment(
        BridgeTransportPremiseAssessmentRequest {
            premise, source, target: trials(0.05),
            source_trial_ids: (0..20).map(|i| format!("source-trial-{i}")).collect(),
            target_trial_ids: (0..20).map(|i| format!("target-trial-{i}")).collect(),
            surrogate_grid: vec![-0.8, 0.0, 0.8], equivalence_margin: 0.2,
        },
    ).unwrap();
    let robust_bridge = compile_bridge_transport_decision(
        declaration.clone(),
        draws_table(64, 0.0, 0.1, DistributionMeaningWire::PosteriorPredictive),
        11, decision.clone(),
    ).unwrap();
    let robust_assessed = compile_assessed_bridge_transport_decision(
        assessment.clone(), robust_bridge,
    ).unwrap();
    let robust_contract = DecisionContract {
        criterion: DecisionCriterion::MaximinOverStructures, ..decision.clone()
    };
    let bridge = compile_bridge_transport_decision(
        declaration, draws_table(64, 1.0, 0.3, DistributionMeaningWire::PosteriorPredictive),
        11, decision,
    ).unwrap();
    let plan = compile_assessed_bridge_transport_decision(assessment, bridge).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "AssessedBridgeTransportDecision");
    let result = plan.execute(&ctx).unwrap();
    assert!(result.assessment.equivalence_screen_passed);
    assert_eq!(result.assessment.calibration, "not_measured");
    let bytes = plan.export(&result, "assessed-bridge-decision", &ctx).unwrap();
    let replay = consume_assessed_bridge_transport_decision(
        &bytes, result.lowered_program_identity, &ctx,
    ).unwrap();
    assert_eq!(replay.assessment, result.assessment);
    assert_eq!(replay.bridge.decision.decision, result.bridge.decision.decision);
    let robust = compile_assessed_bridge_robust_decision(
        robust_assessed, robust_contract,
    ).unwrap();
    let bounds = robust.execute(&ctx).unwrap();
    assert_eq!(bounds.calibration, "not_measured");
    assert_eq!(bounds.robust.actions.len(), 2);
    let portable = robust.export(&bounds, "assessed-bridge-robust", &ctx).unwrap();
    let replayed = consume_assessed_bridge_robust_decision(
        &portable, robust.lowered().identity(), &ctx,
    ).unwrap();
    assert_eq!(replayed.assessed.assessment, bounds.assessed.assessment);
    assert_eq!(replayed.robust, bounds.robust);
}

#[test]
#[ignore = "source-only live bridge bundle terminal; no calibration earned"]
fn predictive_bridge_bundle_recomputes_a_live_terminal() {
    let mut declaration = contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 }, None, None,
    );
    declaration.source_value = ValueKind::PredictiveLaw;
    declaration.target_value = ValueKind::PredictiveLaw;
    let source = draws_table(64, 1.0, 0.3, DistributionMeaningWire::PosteriorPredictive);
    let prepared = PreparedScientificBridgeDistribution::prepare(declaration.clone()).unwrap();
    let result = prepared.derive(&source, None, 41).unwrap();
    let bridge_artifact = result.export("bridge-bundle-terminal").unwrap();
    let mut builder = BundleBuilder::new();
    let source_bytes = source.to_bytes("bridge-bundle-source-node").unwrap();
    builder.add_artifact(Some("source"), NodeKind::Distribution, &source_bytes).unwrap();
    builder.add_artifact(
        Some("terminal"), NodeKind::ScientificBridge, &bridge_artifact.bytes,
    ).unwrap();
    builder.connect("source", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let expected_bundle = bundle.identity().to_owned();
    let ctx = ExecutionContext::for_tests(41);
    let plan = compile_bridge_bundle_recomputation(
        bundle, "terminal", declaration, source, 41,
    ).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "BridgeBundleRecomputation");
    let receipt = plan.execute(&ctx).unwrap();
    assert_eq!(receipt.terminal_identity, bridge_artifact.identity);
    let bytes = plan.export_portable(&receipt, &ctx).unwrap();
    let replay = consume_portable_bridge_bundle_recomputation(
        &bytes, &expected_bundle, plan.lowered().identity(), &ctx,
    ).unwrap();
    assert_eq!(replay, receipt);
}

fn uncertain() -> BridgeContract {
    contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 },
        Some(MappingUncertainty {
            id: "calibration-fit".into(),
            intercept_variance: 0.01,
            slope_variance: 0.04,
            intercept_slope_covariance: -0.005,
        }),
        Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.01,
            evidence: "shared-sample".into(),
        }),
    )
}

fn code(refusal: &ScientificBridgeRefusal) -> (&'static str, &str) {
    (refusal.code, refusal.detail.as_str())
}

#[test]
fn a_bridge_of_draws_is_prepared_derived_exported_and_consumed_independently() {
    let prepared = PreparedScientificBridgeDistribution::prepare(uncertain()).expect("prepare");
    assert!(!prepared.transported());
    let table = draws_table(3_000, 1.0, 0.3, DistributionMeaningWire::CausalFunctionalPosterior);
    let result = prepared.derive(&table, None, 7).expect("derive");
    let inspection = result.inspect();
    assert_eq!(inspection.n_draws, 3_000);
    assert_eq!(inspection.reading, "uncalibrated_spread");
    assert_eq!(inspection.value_kind, ValueKind::Posterior);
    assert_eq!(inspection.contract_identity, prepared.contract_identity().to_hex());
    assert!(inspection.unavailable.iter().any(|item| item == "calibrated_interval"));
    // The covariance of the slope and the source moves the mean by the covariance: 0.5 + 2 + 0.01.
    assert!((inspection.mean - 2.51).abs() < 0.05, "{}", inspection.mean);
    // The band above cannot see the 0.01 covariance (Monte Carlo error is about 0.012), so
    // the declared dependence is checked by a symmetric pair on the same seed: the mean is
    // linear in the covariance with slope one, so +0.01 and -0.01 differ by exactly 0.02.
    let with_covariance = |covariance: f64| {
        let declared = contract(
            BridgeMapping::Affine { intercept: 0.5, slope: 2.0 },
            Some(MappingUncertainty {
                id: "calibration-fit".into(),
                intercept_variance: 0.01,
                slope_variance: 0.04,
                intercept_slope_covariance: -0.005,
            }),
            Some(SourceMappingDependence::SlopeCovariance {
                covariance,
                evidence: "shared-sample".into(),
            }),
        );
        PreparedScientificBridgeDistribution::prepare(declared)
            .unwrap()
            .derive(&table, None, 7)
            .unwrap()
            .inspect()
            .mean
    };
    let (plus, minus) = (with_covariance(0.01), with_covariance(-0.01));
    assert!((plus - minus - 0.02).abs() < 1e-6, "{plus} {minus}");
    assert!((plus - inspection.mean).abs() < 1e-12, "the declared covariance is the one used");
    // The bridged table is the source with the bridged column replaced, under the same alignment.
    assert_eq!(
        result.distribution().alignment_identity().unwrap(),
        table.alignment_identity().unwrap()
    );
    assert_eq!(result.source().draws(), table.draws());

    // A bridged distribution is a posterior and never a mean.
    let refusal = result.require_value_kind(ValueKind::Mean).unwrap_err();
    assert_eq!(
        code(&refusal),
        (
            "bridge_substitution_refused",
            antecedent_core::reason_details::scientific_bridge::VALUE_KIND_SUBSTITUTION
        )
    );
    assert!(result.require_value_kind(ValueKind::Posterior).is_ok());
    // No interval without a level, and none outside a measured design.
    let refusal = result.interval().unwrap_err();
    assert_eq!(
        code(&refusal),
        (
            "route_not_supported",
            antecedent_core::reason_details::scientific_bridge::INTERVAL_NOT_SUPPORTED
        )
    );
    let refusal = result.bridged_interval(0.95).unwrap_err();
    assert_eq!(
        code(&refusal),
        (
            "route_not_supported",
            antecedent_core::reason_details::scientific_bridge::INTERVAL_SCOPE_OUTSIDE_MEASURED
        )
    );

    // Export, then consume as a fresh consumer that retained the identity.
    let export = result.export("draws-bridge").expect("export");
    assert_eq!(scientific_bridge_artifact_shape(&export.bytes).unwrap(), "draws");
    let consumed =
        consume_scientific_bridge_distribution_artifact(&export.bytes, Some(&export.identity))
            .expect("consume");
    assert_eq!(consumed.identity, export.identity);
    assert_eq!(consumed.bridged.claim().draws_digest, result.bridged().claim().draws_digest);
    for (a, b) in consumed.bridged.mapped_column().iter().zip(result.bridged().mapped_column()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    // The consumer of a bridged mean does not read a bridge of draws, and a flipped byte or
    // another retained identity is refused.
    assert!(consume_scientific_bridge_artifact(&export.bytes, None).is_err());
    let mut flipped = export.bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    assert!(consume_scientific_bridge_distribution_artifact(&flipped, None).is_err());
    assert!(
        consume_scientific_bridge_distribution_artifact(&export.bytes, Some(&"0".repeat(64)))
            .is_err()
    );
}

#[test]
fn only_a_posterior_or_predictive_law_is_prepared_and_an_associational_claim_is_never_a_source() {
    // A mean-valued contract is the point route's.
    let mut mean = uncertain();
    mean.source_value = ValueKind::Mean;
    mean.target_value = ValueKind::Mean;
    let table = draws_table(500, 1.0, 0.3, DistributionMeaningWire::CausalFunctionalPosterior);
    // A mean names no draws: the table's kind is refused as a substitution.
    let prepared = PreparedScientificBridgeDistribution::prepare(mean).unwrap();
    let refusal = prepared.derive(&table, None, 1).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::scientific_bridge::VALUE_KIND_SUBSTITUTION
    );
    // A set or a joint law is carried by no route.
    let mut set = uncertain();
    set.source_value = ValueKind::IdentifiedSet;
    set.target_value = ValueKind::IdentifiedSet;
    let refusal = PreparedScientificBridgeDistribution::prepare(set).unwrap_err();
    assert_eq!(
        code(&refusal),
        (
            "bridge_mapping_not_supported",
            antecedent_core::reason_details::scientific_bridge::VALUE_KIND_NOT_CARRIED
        )
    );
    // An uncertain map without a declared dependence is refused before any draw.
    let mut undeclared = uncertain();
    undeclared.source_dependence = None;
    let refusal = PreparedScientificBridgeDistribution::prepare(undeclared).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::scientific_bridge::SOURCE_DEPENDENCE_MISSING
    );
    // An associational source claim is never a bridge source.
    let prepared = PreparedScientificBridgeDistribution::prepare(uncertain()).unwrap();
    let refusal = prepared.derive(&table, Some(ClaimKind::Associational), 1).unwrap_err();
    assert_eq!(
        code(&refusal),
        (
            "bridge_mapping_not_supported",
            antecedent_core::reason_details::scientific_bridge::SOURCE_NOT_DRAWS
        )
    );
}

#[test]
fn an_exact_map_publishes_its_interval_as_an_exact_transform_and_a_line_of_few_trials_does_not() {
    let table = draws_table(4_000, 1.0, 0.3, DistributionMeaningWire::CausalFunctionalPosterior);
    let exact = contract(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 }, None, None);
    let result = PreparedScientificBridgeDistribution::prepare(exact)
        .unwrap()
        .derive(&table, None, 3)
        .unwrap();
    let interval =
        result.bridged_interval(0.95).expect("an exact transform of the source interval");
    assert_eq!(interval.supported_by, "exact_transform_of_declared_source_interval");
    let sd = result.bridged().source_variance().sqrt();
    let half = antecedent_stats::normal_ppf(0.975) * 2.0 * sd;
    assert!((interval.upper - interval.lower - 2.0 * half).abs() < 1e-9);
    assert!((interval.source_standard_error - sd).abs() < 1e-12);

    // Five validation trials are outside every measured design: reported, annotated.
    let trials = [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]
        .iter()
        .map(|(s, t)| TrialEffectPair { surrogate_effect: *s, terminal_effect: *t })
        .collect();
    let line = contract(
        BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 5,
            min_r_squared: 0.9,
        }),
        None,
        Some(SourceMappingDependence::Independent { evidence: "disjoint".into() }),
    );
    let near_two =
        draws_table(20_000, 2.0, 0.05, DistributionMeaningWire::CausalFunctionalPosterior);
    let bridged = PreparedScientificBridgeDistribution::prepare(line)
        .unwrap()
        .derive(&near_two, None, 3)
        .unwrap();
    let reported = bridged.bridged_interval(0.95).expect("reported without a measured design");
    assert_ne!(
        reported.calibration.status,
        antecedent_io_core::contract_section::CalibrationStatus::Calibrated
    );
}

#[test]
fn a_chain_of_bridge_transport_and_utility_propagates_the_mapping_and_the_source_jointly() {
    let rows = 4_001;
    let table = draws_table(rows, 1.0, 0.3, DistributionMeaningWire::CausalFunctionalPosterior);
    let result = PreparedScientificBridgeDistribution::prepare(uncertain())
        .unwrap()
        .derive(&table, None, 9)
        .unwrap();
    let parts = result.chain_stage("terminal").expect("an affine bridge is a stage of a chain");
    let stages = vec![
        parts.stage.clone(),
        ChainStageWire::new(
            "transported",
            StageKind::Transport,
            &["terminal"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(0.9), ChainCoefficientWire::fixed(0.0)),
        ),
        ChainStageWire::new(
            "utility",
            StageKind::DecisionUtility,
            &["transported"],
            ChainOpWire::unary(ChainUnaryWire::exponential_utility(0.5)),
        ),
    ];
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "utility",
        parts.sources.clone(),
        stages,
    );
    let ctx = ExecutionContext::for_tests(31);
    let chain = PropagationChain::seal(draft, &ctx).expect("a supported chain");
    let mapped = chain.stage("terminal").expect("bridged stage");
    assert!((mapped.value("mean").unwrap() - result.mean()).abs() < 1e-9);
    // The utility is the expectation of the utility of each joint draw of the bridged and
    // transported value, not the utility of its mean.
    let expectation: f64 = result
        .bridged()
        .mapped_column()
        .iter()
        .map(|y| 1.0 - (-0.5 * (0.9 * y)).exp())
        .sum::<f64>()
        / rows as f64;
    let utility = chain.answer().expect("answer");
    assert!((utility.value("mean").unwrap() - expectation).abs() < 1e-9);
    let of_the_mean = 1.0 - (-0.5 * 0.9 * result.mean()).exp();
    assert!((expectation - of_the_mean).abs() > 1e-4);
    // The chain is exported and consumed independently, replaying every stage.
    let bytes = chain.export().expect("export");
    let consumed = crate::analysis::propagation_chain::consume_propagation_chain(
        &bytes,
        Some(chain.identity()),
        &ctx,
    )
    .expect("consume");
    assert_eq!(consumed.output, "utility");
}

#[test]
fn a_predictive_bridge_handoff_compiles_as_a_predictive_law() {
    let mut declaration = uncertain();
    declaration.source_value = ValueKind::PredictiveLaw;
    declaration.target_value = ValueKind::PredictiveLaw;
    let table = draws_table(1_001, 1.0, 0.3, DistributionMeaningWire::PosteriorPredictive);
    let result = PreparedScientificBridgeDistribution::prepare(declaration)
        .expect("predictive bridge")
        .derive(&table, None, 9)
        .expect("predictive mapped draws");
    let parts = result.chain_stage("terminal").expect("aligned bridge stage");
    assert_eq!(parts.reading, ChainReading::Predictive);
    assert_eq!(parts.sources[0].meaning.as_deref(), Some("predictive"));
    let ctx = ExecutionContext::for_tests(31);
    let plan = result.compile_chain_stage("terminal", &ctx)
        .expect("compiler accepts checked predictive handoff");
    let chain = plan.execute(&ctx).expect("retained bridge program executes");
    assert_eq!(chain.label(), "predictive_propagation");
    let output = chain.stage("terminal").expect("mapped output");
    assert!((output.value("mean").unwrap() - result.mean()).abs() < 1e-9);
}

#[test]
#[ignore = "source-only target posterior predictive bridge decision; reliability not earned"]
fn independent_target_posterior_and_source_innovation_replay_together() {
    use crate::analysis::bridge_transport_decision::{
        BridgeTargetPosteriorPremises, compile_assessed_bridge_posterior_decision,
        consume_assessed_bridge_posterior_decision,
    };
    let trials = |offset: f64| TrialLevelValidation {
        trials: (0..20).map(|index| {
            let x = -1.0 + 2.0 * index as f64 / 19.0;
            TrialEffectPair { surrogate_effect: x,
                terminal_effect: 0.5 + 2.0 * x + offset
                    + (if index % 2 == 0 { 0.01 } else { -0.01 }),
            }
        }).collect(),
        min_trials: 3, min_r_squared: 0.0,
    };
    let source = trials(0.0);
    let premise = TransportabilityPremise {
        id: "validation-to-target".into(),
        statement: "trial surrogate relation transfers".into(),
        established_in: BridgeScope {
            population_id: "validation-adults".into(), regime_id: "do(t=1)".into(),
        },
        applies_to: BridgeScope {
            population_id: "adults".into(), regime_id: "do(t=1)".into(),
        },
        evidence: Some("independent-target-validation".into()),
    };
    let mut declaration = contract(BridgeMapping::TrialLevelLinear(source.clone()), None, None);
    declaration.source_value = ValueKind::PredictiveLaw;
    declaration.target_value = ValueKind::PredictiveLaw;
    declaration.evidence_scope = premise.established_in.clone();
    declaration.transport = Some(premise.clone());
    let outcome = declaration.target.quantity.clone();
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "use-bridge".into(), kind: ActionKind::External,
                inputs: vec![outcome.clone()], utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "risk-sensitive".into(), kind: ActionKind::External,
                inputs: vec![outcome],
                utility: UtilityExpr::product(
                    UtilityExpr::Input(0), UtilityExpr::Input(0),
                ),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::ExpectedRegret,
        constraints: Vec::new(), target_population: "adults".into(),
        horizon: 0, structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let ctx = ExecutionContext::for_tests(43);
    let assessment = compile_bridge_transport_premise_assessment(
        BridgeTransportPremiseAssessmentRequest {
            premise, source, target: trials(0.05),
            source_trial_ids: (0..20).map(|i| format!("source-trial-{i}")).collect(),
            target_trial_ids: (0..20).map(|i| format!("target-trial-{i}")).collect(),
            surrogate_grid: vec![-0.8, 0.0, 0.8], equivalence_margin: 0.2,
        },
    ).unwrap();
    let source_draws = draws_table(64, 0.0, 0.1, DistributionMeaningWire::PosteriorPredictive);
    let mut metadata = source_draws.metadata().clone();
    metadata.weights = Some((0..64).map(f64::from).collect());
    let weighted_source = DistributionArtifact::new(metadata, source_draws.draws().to_vec()).unwrap();
    let bridge = compile_bridge_transport_decision(declaration, weighted_source, 11, decision).unwrap();
    let assessed = compile_assessed_bridge_transport_decision(assessment, bridge).unwrap();
    let plan = compile_assessed_bridge_posterior_decision(assessed, BridgeTargetPosteriorPremises {
        validation_design: "independent-target-trials".into(),
        target_validation_independent: true, source_innovation_transfers: true,
    }, &ctx).unwrap();
    let result = plan.execute(&ctx).unwrap();
    assert_eq!(result.calibration, "not_measured");
    let weights = result.decision.chain.output_law().unwrap().1.unwrap();
    assert_eq!(weights.len(), 63);
    assert!(f64::from_bits(weights[0]) < f64::from_bits(weights[62]));
    let artifact = plan.export(&result, "target-posterior-bridge", &ctx).unwrap();
    let replay = consume_assessed_bridge_posterior_decision(&artifact, plan.lowered().identity(), &ctx).unwrap();
    assert_eq!(result.decision.chain.identity(), replay.decision.chain.identity());
    assert_eq!(result.decision.decision, replay.decision.decision);
}
