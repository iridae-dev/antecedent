//! Unit tests of the rules of a bridge that carries draws: licensing, the
//! coefficient law of each mapping, the conditional draw it implies and the
//! identity of the derived claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::identity::{ContractIdentities, SemanticDigest};

use super::super::super::kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
use super::super::super::mapping::{BridgeMapping, TrialEffectPair, TrialLevelValidation};
use super::super::{BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeScope};
use super::{BridgeDrawsSource, DrawMapping, MappedDrawsSummary};
use crate::claim::{ClaimKind, DomainStatus};
use crate::quantity::{QuantityRole, ScientificQuantity};

use super::super::{MappingUncertainty, SourceMappingDependence};

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "log_hazard".into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn validation() -> BridgeAssumption {
    BridgeAssumption {
        id: "validation".into(),
        kind: AssumptionKind::SurrogateValidation,
        statement: "the surrogate is validated at the trial level".into(),
        evidence: Some("meta-analysis:ckd-2024".into()),
    }
}

fn contract(
    mapping: BridgeMapping,
    kind: ValueKind,
    uncertainty: Option<MappingUncertainty>,
    dependence: Option<SourceMappingDependence>,
) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: kind,
        target_value: kind,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping,
        assumptions: vec![validation()],
        transport: None,
        mapping_uncertainty: uncertainty,
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

fn affine(
    kind: ValueKind,
    uncertainty: Option<MappingUncertainty>,
    dependence: Option<SourceMappingDependence>,
) -> BridgeContract {
    contract(BridgeMapping::Affine { intercept: 0.5, slope: 2.0 }, kind, uncertainty, dependence)
}

fn uncertainty(intercept: f64, slope: f64, covariance: f64) -> MappingUncertainty {
    MappingUncertainty {
        id: "calibration-fit".into(),
        intercept_variance: intercept,
        slope_variance: slope,
        intercept_slope_covariance: covariance,
    }
}

fn independent() -> SourceMappingDependence {
    SourceMappingDependence::Independent { evidence: "disjoint-samples".into() }
}

fn trial_line() -> BridgeMapping {
    BridgeMapping::TrialLevelLinear(TrialLevelValidation {
        trials: [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]
            .iter()
            .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                surrogate_effect: *surrogate_effect,
                terminal_effect: *terminal_effect,
            })
            .collect(),
        min_trials: 5,
        min_r_squared: 0.9,
    })
}

fn identities() -> ContractIdentities {
    ContractIdentities::new(digest(1), digest(2), None, None, digest(3), digest(4), digest(5))
}

fn draws_source<'a>(
    quantity: &'a ScientificQuantity,
    identities: &'a ContractIdentities,
    kind: Option<ValueKind>,
    claim_kind: ClaimKind,
    joint: bool,
) -> BridgeDrawsSource<'a> {
    BridgeDrawsSource {
        quantity,
        value_kind: kind,
        meaning: "bootstrap",
        claim_kind,
        claim_id: digest(9),
        identities,
        execution: None,
        evidence: &[],
        evaluated: DomainStatus::Evaluated,
        joint,
        alignment_identity: "alignment-a",
    }
}

fn code(error: &super::super::BridgeRefusal) -> (&'static str, &str) {
    (error.code, error.detail.as_str())
}

#[test]
fn a_posterior_is_carried_only_by_the_draws_check() {
    let bridge = affine(ValueKind::Posterior, None, None);
    assert!(bridge.check_draws_licensing().is_ok());
    let refused = bridge.check_licensing().unwrap_err();
    assert_eq!(code(&refused), ("cell_not_licensed", "scientific_bridge.value_kind_not_carried"));
    // A mean still goes through the point check and is not a draws bridge's business.
    let mean = affine(ValueKind::Mean, None, None);
    assert!(mean.check_licensing().is_ok());
    // An identified set or a joint law is carried by neither.
    for kind in [ValueKind::IdentifiedSet, ValueKind::JointOutcomeLaw] {
        let refused = affine(kind, None, None).check_draws_licensing().unwrap_err();
        assert_eq!(
            code(&refused),
            ("cell_not_licensed", "scientific_bridge.value_kind_not_carried")
        );
    }
    // The source and target kinds still have to agree.
    let mut mixed = affine(ValueKind::Posterior, None, None);
    mixed.target_value = ValueKind::PredictiveLaw;
    assert_eq!(
        code(&mixed.check_draws_licensing().unwrap_err()).1,
        "scientific_bridge.value_kind_substitution"
    );
}

#[test]
fn an_uncertain_map_declares_its_dependence_and_no_map_counts_its_uncertainty_twice() {
    let some = Some(uncertainty(0.01, 0.04, 0.0));
    let refused = affine(ValueKind::Posterior, some.clone(), None).check_draws_licensing();
    assert_eq!(code(&refused.unwrap_err()).1, "scientific_bridge.source_dependence_missing");
    let refused = affine(ValueKind::Posterior, None, Some(independent())).check_draws_licensing();
    assert_eq!(code(&refused.unwrap_err()).1, "scientific_bridge.mapping_uncertainty_missing");
    assert!(
        affine(ValueKind::Posterior, some.clone(), Some(independent()))
            .check_draws_licensing()
            .is_ok()
    );

    // A trial-level line takes its uncertainty from its trials, so declaring it again is refused.
    let line = contract(trial_line(), ValueKind::Posterior, some, Some(independent()));
    assert_eq!(
        code(&line.check_draws_licensing().unwrap_err()).1,
        "scientific_bridge.mapping_uncertainty_invalid"
    );
    let line = contract(trial_line(), ValueKind::Posterior, None, None);
    assert_eq!(
        code(&line.check_draws_licensing().unwrap_err()).1,
        "scientific_bridge.source_dependence_missing"
    );
    let line = contract(trial_line(), ValueKind::Posterior, None, Some(independent()));
    assert!(line.check_draws_licensing().is_ok());
}

#[test]
fn the_innovation_reproduces_the_declared_moments() {
    let law = super::AffineDrawLaw {
        intercept_mean: 0.5,
        slope_mean: 2.0,
        intercept_variance: 0.09,
        slope_variance: 0.04,
        intercept_slope_covariance: -0.02,
        source_slope_covariance: 0.01,
        degrees_of_freedom: None,
        residual_variance: 0.0,
        domain: None,
    };
    let source_variance = 0.25;
    let innovation = law.innovation(source_variance).unwrap();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    // Var(b) = loading^2 V + sd^2 and Cov(b, M) = loading V.
    assert!(close(
        innovation.slope_loading.powi(2) * source_variance + innovation.slope_sd.powi(2),
        law.slope_variance
    ));
    assert!(close(innovation.slope_loading * source_variance, law.source_slope_covariance));
    // Cov(a, b) = on_slope * slope_sd and Var(a) = on_slope^2 + intercept_sd^2.
    assert!(close(
        innovation.intercept_on_slope * innovation.slope_sd,
        law.intercept_slope_covariance
    ));
    assert!(close(
        innovation.intercept_on_slope.powi(2) + innovation.intercept_sd.powi(2),
        law.intercept_variance
    ));
}

#[test]
fn a_covariance_no_joint_law_has_is_refused() {
    let law = super::AffineDrawLaw {
        intercept_mean: 0.0,
        slope_mean: 1.0,
        intercept_variance: 0.01,
        slope_variance: 0.01,
        intercept_slope_covariance: 0.0,
        source_slope_covariance: 0.02,
        degrees_of_freedom: None,
        residual_variance: 0.0,
        domain: None,
    };
    // 0.02^2 = 0.0004 exceeds 0.01 * 0.01 = 0.0001.
    let refused = law.innovation(0.01).unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.dependence_not_a_covariance");
    // A covariance with draws that do not vary is not a statement about anything.
    let refused = law.innovation(0.0).unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.draws_unusable");
    let refused = law.innovation(f64::NAN).unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.source_variance_invalid");
}

#[test]
fn public_affine_law_invalid_variances_and_covariances_refuse() {
    let law = super::AffineDrawLaw {
        intercept_mean: 0.0,
        slope_mean: 1.0,
        intercept_variance: 1.0,
        slope_variance: 1.0,
        intercept_slope_covariance: 0.0,
        source_slope_covariance: 0.0,
        degrees_of_freedom: None,
        residual_variance: 0.0,
        domain: None,
    };
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let mut x = law.clone();
        x.intercept_variance = bad;
        assert_eq!(
            code(&x.innovation(1.0).unwrap_err()).1,
            "scientific_bridge.mapping_uncertainty_invalid"
        );
        let mut x = law.clone();
        x.slope_variance = bad;
        assert_eq!(
            code(&x.innovation(1.0).unwrap_err()).1,
            "scientific_bridge.mapping_uncertainty_invalid"
        );
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut x = law.clone();
        x.intercept_slope_covariance = bad;
        assert_eq!(
            code(&x.innovation(1.0).unwrap_err()).1,
            "scientific_bridge.mapping_uncertainty_invalid"
        );
        let mut x = law.clone();
        x.source_slope_covariance = bad;
        assert_eq!(
            code(&x.innovation(1.0).unwrap_err()).1,
            "scientific_bridge.source_dependence_invalid"
        );
    }
    let mut zero = law.clone();
    zero.intercept_variance = 0.0;
    zero.slope_variance = 0.0;
    let innovation = zero.innovation(0.0).unwrap();
    assert_eq!((innovation.intercept_sd, innovation.slope_sd), (0.0, 0.0));
    let mut large = law;
    large.slope_variance = 1e300;
    large.source_slope_covariance = 1e300;
    assert!(large.innovation(1e300).is_ok());
}

#[test]
fn a_trial_line_draws_its_coefficients_from_the_least_squares_covariance() {
    let bridge = contract(trial_line(), ValueKind::Posterior, None, Some(independent()));
    let plan = bridge
        .plan_distribution(
            &draws_source(
                &bridge.source.quantity,
                &identities(),
                Some(ValueKind::Posterior),
                ClaimKind::Mixture,
                true,
            ),
            7,
        )
        .unwrap();
    let DrawMapping::Affine(law) = plan.mapping() else { panic!("a line is an affine law") };
    // Hand computation: x = 0..4, y = 0.1, 1.1, 1.9, 3.2, 3.9.
    let (xbar, ybar) = (2.0_f64, 2.04_f64);
    let sxx = 10.0_f64;
    let sxy: f64 = [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]
        .iter()
        .map(|(x, y): &(f64, f64)| (x - xbar) * (y - ybar))
        .sum();
    let syy: f64 = [0.1_f64, 1.1, 1.9, 3.2, 3.9].iter().map(|y| (y - ybar).powi(2)).sum();
    let slope = sxy / sxx;
    let s2 = (syy - sxy * sxy / sxx) / 3.0;
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(law.slope_mean, slope));
    assert!(close(law.intercept_mean, ybar - slope * xbar));
    assert!(close(law.slope_variance, s2 / sxx));
    assert!(close(law.intercept_variance, s2 * (0.2 + xbar * xbar / sxx)));
    assert!(close(law.intercept_slope_covariance, -s2 * xbar / sxx));
    assert!(close(law.residual_variance, s2));
    assert_eq!(law.degrees_of_freedom, Some(3.0));
    assert_eq!(law.domain, Some((0.0, 4.0)));
    // The residual belongs to the law of a new trial, not to the posterior of the mean line.
    assert!(!plan.includes_residual());
}

#[test]
fn a_line_of_too_few_trials_has_no_finite_coefficient_variance() {
    let mapping = BridgeMapping::TrialLevelLinear(TrialLevelValidation {
        trials: [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2)]
            .iter()
            .map(|(s, t)| TrialEffectPair { surrogate_effect: *s, terminal_effect: *t })
            .collect(),
        min_trials: 4,
        min_r_squared: 0.9,
    });
    let bridge = contract(mapping, ValueKind::Posterior, None, Some(independent()));
    let refused = bridge
        .plan_distribution(
            &draws_source(
                &bridge.source.quantity,
                &identities(),
                Some(ValueKind::Posterior),
                ClaimKind::Mixture,
                true,
            ),
            1,
        )
        .unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.validation_trials_too_few");
}

#[test]
fn only_aligned_draws_of_the_declared_kind_and_quantity_are_bridged() {
    let bridge = affine(ValueKind::Posterior, None, None);
    let ids = identities();
    let source = |kind, claim_kind, joint| {
        draws_source(&bridge.source.quantity, &ids, kind, claim_kind, joint)
    };
    assert!(
        bridge
            .plan_distribution(&source(Some(ValueKind::Posterior), ClaimKind::Mixture, true), 1)
            .is_ok()
    );
    // An associational claim is never a bridge source.
    let refused = bridge
        .plan_distribution(&source(Some(ValueKind::Posterior), ClaimKind::Associational, true), 1)
        .unwrap_err();
    assert_eq!(code(&refused), ("cell_not_licensed", "scientific_bridge.source_not_draws"));
    assert_eq!(refused.supplied.as_deref(), Some("associational"));
    // Draws of another meaning are neither a posterior nor a predictive law.
    let refused = bridge.plan_distribution(&source(None, ClaimKind::Mixture, true), 1).unwrap_err();
    assert_eq!(
        code(&refused),
        ("distribution_meaning_mismatch", "scientific_bridge.draws_meaning_mismatch")
    );
    // A predictive law is not a posterior.
    let refused = bridge
        .plan_distribution(&source(Some(ValueKind::PredictiveLaw), ClaimKind::Mixture, true), 1)
        .unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.value_kind_substitution");
    // Independent marginals have no row pairing for the bridged draws to keep.
    let refused = bridge
        .plan_distribution(&source(Some(ValueKind::Posterior), ClaimKind::Mixture, false), 1)
        .unwrap_err();
    assert_eq!(code(&refused), ("joint_law_required", "scientific_bridge.draws_not_aligned"));
    // Draws about another quantity.
    let other = quantity("trial:other");
    let refused = bridge
        .plan_distribution(
            &draws_source(&other, &ids, Some(ValueKind::Posterior), ClaimKind::Mixture, true),
            1,
        )
        .unwrap_err();
    assert_eq!(code(&refused).1, "scientific_bridge.quantity_mismatch");
}

#[test]
fn the_derived_claim_is_one_realization_of_the_bridge() {
    let bridge =
        affine(ValueKind::PredictiveLaw, Some(uncertainty(0.01, 0.04, 0.0)), Some(independent()));
    let ids = identities();
    let evidence: Vec<Arc<str>> = vec![Arc::from("snapshot:panel-2024")];
    let source = BridgeDrawsSource {
        evidence: &evidence,
        ..draws_source(
            &bridge.source.quantity,
            &ids,
            Some(ValueKind::PredictiveLaw),
            ClaimKind::Mixture,
            true,
        )
    };
    let summary = |byte: u8| MappedDrawsSummary {
        n_draws: 1000,
        mean: 3.0,
        variance: 1.5,
        draws_digest: digest(byte),
    };
    let first = bridge.plan_distribution(&source, 11).unwrap().finish(&summary(1)).unwrap();
    let again = bridge.plan_distribution(&source, 11).unwrap().finish(&summary(1)).unwrap();
    assert_eq!(first, again);
    // Another realization of the mapping's coefficients, or other mapped draws, is another claim.
    let other_stream = bridge.plan_distribution(&source, 12).unwrap().finish(&summary(1)).unwrap();
    let other_draws = bridge.plan_distribution(&source, 11).unwrap().finish(&summary(2)).unwrap();
    assert_ne!(first.claim.claim_id, other_stream.claim.claim_id);
    assert_ne!(first.claim.claim_id, other_draws.claim.claim_id);
    assert_ne!(first.claim.claim_id, first.source_claim_id);
    // The source is separately inspectable, the alignment is the source's and nothing is calibrated.
    assert_eq!(first.alignment_identity, "alignment-a");
    assert_eq!(first.claim.kind, ClaimKind::Mixture);
    assert_eq!(first.value_kind, ValueKind::PredictiveLaw);
    let evidence: Vec<&str> = first.claim.evidence.iter().map(|item| &**item).collect();
    assert!(evidence.contains(&"snapshot:panel-2024"));
    assert!(evidence.contains(&"alignment:alignment-a"));
    assert!(evidence.contains(&"bridge_reading:uncalibrated_spread"));
    assert!(evidence.contains(&"mapping_uncertainty:calibration-fit"));
    assert!(
        first
            .receipt
            .unresolved
            .iter()
            .any(|item| &**item == "mapping_independent_of_sibling_columns")
    );
    assert!(
        first.receipt.unavailable_operations.iter().any(|item| &**item == "calibrated_interval")
    );
    // The bridged draws are a predictive law, never a mean.
    assert_eq!(
        code(&first.require_value_kind(ValueKind::Mean).unwrap_err()).1,
        "scientific_bridge.value_kind_substitution"
    );
    assert!(first.require_value_kind(ValueKind::PredictiveLaw).is_ok());
    assert_eq!(
        code(&first.request_interval().unwrap_err()).1,
        "scientific_bridge.interval_not_licensed"
    );
}
