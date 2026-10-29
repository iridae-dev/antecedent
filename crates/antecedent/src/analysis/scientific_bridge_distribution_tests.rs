//! The facade of a bridge of draws: prepare, derive, inspect, export and
//! independently consume, the interval gate, and the bridge as a stage of a
//! propagation chain that continues into a transport and a utility.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeEndpoint, BridgeMapping, BridgeScope,
    MappingUncertainty, QuantityClass, SourceMappingDependence, TrialEffectPair,
    TrialLevelValidation,
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
use crate::analysis::propagation_chain::PropagationChain;

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
            calibration: DistributionCalibration::Unmeasured,
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
        ("bridge_substitution_refused", "scientific_bridge.value_kind_substitution")
    );
    assert!(result.require_value_kind(ValueKind::Posterior).is_ok());
    // No interval without a level, and none outside a measured design.
    let refusal = result.interval().unwrap_err();
    assert_eq!(code(&refusal), ("cell_not_licensed", "scientific_bridge.interval_not_licensed"));
    let refusal = result.bridged_interval(0.95).unwrap_err();
    assert_eq!(
        code(&refusal),
        ("cell_not_licensed", "scientific_bridge.interval_scope_outside_measured")
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
    assert_eq!(refusal.detail, "scientific_bridge.value_kind_substitution");
    // A set or a joint law is carried by no route.
    let mut set = uncertain();
    set.source_value = ValueKind::IdentifiedSet;
    set.target_value = ValueKind::IdentifiedSet;
    let refusal = PreparedScientificBridgeDistribution::prepare(set).unwrap_err();
    assert_eq!(code(&refusal), ("cell_not_licensed", "scientific_bridge.value_kind_not_carried"));
    // An uncertain map without a declared dependence is refused before any draw.
    let mut undeclared = uncertain();
    undeclared.source_dependence = None;
    let refusal = PreparedScientificBridgeDistribution::prepare(undeclared).unwrap_err();
    assert_eq!(refusal.detail, "scientific_bridge.source_dependence_missing");
    // An associational source claim is never a bridge source.
    let prepared = PreparedScientificBridgeDistribution::prepare(uncertain()).unwrap();
    let refusal = prepared.derive(&table, Some(ClaimKind::Associational), 1).unwrap_err();
    assert_eq!(code(&refusal), ("cell_not_licensed", "scientific_bridge.source_not_draws"));
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
    assert_eq!(interval.licensed_by, "exact_transform_of_declared_source_interval");
    let sd = result.bridged().source_variance().sqrt();
    let half = antecedent_stats::normal_ppf(0.975) * 2.0 * sd;
    assert!((interval.upper - interval.lower - 2.0 * half).abs() < 1e-9);
    assert!((interval.source_standard_error - sd).abs() < 1e-12);

    // Five validation trials are outside every measured design.
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
    let refusal = bridged.bridged_interval(0.95).unwrap_err();
    assert_eq!(refusal.detail, "scientific_bridge.interval_scope_outside_measured");
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
    let chain = PropagationChain::seal(draft, &ctx).expect("a licensed chain");
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
