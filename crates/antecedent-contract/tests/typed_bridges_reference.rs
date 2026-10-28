//! Typed-bridge point mappings and their propagated variances against exact rational references.
//!
//! `oracle_data/typed_bridges_reference.py` (output frozen in `typed_bridges_reference.txt`)
//! evaluates, with `fractions.Fraction` on the binary64 inputs taken as exact rationals, the
//! trial-level least-squares line (slope, intercept, coefficient of determination, residual
//! variance on k - 2 degrees of freedom), the coefficient moments of the fitted mean line, the
//! first-order variance of an affine map with declared intercept and slope variances and a declared
//! slope-source covariance, the Rogan-Gladen inversion and its forward map with the delta-method
//! variances of the rates, and the second-moment expectation of a square. Trial sets cover a
//! surrogate offset of 1e6, effects 1e-6 apart, tied surrogate effects, three trials and twelve
//! trials with a negative slope; values are taken at both ends of the validated range. The
//! exponential map is checked against the normal moment-generating function and is refused for a
//! law known only by two moments.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AnalyticLaw, AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
    BridgeMapping, BridgeScope, BridgeSource, MappingUncertainty, NonlinearMap, QuantityClass,
    SourceMappingDependence, TrialEffectPair, TrialLevelValidation, ValueKind,
    derive_bridged_claim_propagating, nonlinear_expectation,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ContractIdentities, SemanticDigest};

const REFERENCE: &str = include_str!("oracle_data/typed_bridges_reference.txt");

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(
    variable: &str,
    units: &str,
    functional: &str,
    population: &str,
    regime: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(value: f64, units: &str) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from(units)),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        [Arc::<str>::from("prior:p0")],
    )
}

fn assumption(id: &str, kind: AssumptionKind) -> BridgeAssumption {
    BridgeAssumption {
        id: id.into(),
        kind,
        statement: format!("{id} holds"),
        evidence: Some("study:fixture".into()),
    }
}

fn endpoint(q: ScientificQuantity, class: QuantityClass) -> BridgeEndpoint {
    BridgeEndpoint { quantity: q, class }
}

fn dependence(covariance: f64) -> SourceMappingDependence {
    if covariance == 0.0 {
        SourceMappingDependence::Independent { evidence: "disjoint-samples".into() }
    } else {
        SourceMappingDependence::SlopeCovariance { covariance, evidence: "joint-study".into() }
    }
}

fn num(fields: &[&str], at: usize) -> f64 {
    fields[at].parse().unwrap_or_else(|_| panic!("field {at} of {fields:?}"))
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    // A value that is exactly zero in binary64 arithmetic but one part in 1e17 away in exact
    // arithmetic (0.1 + 0.9 - 1) is within the absolute floor.
    let scale = want.abs();
    assert!((got - want).abs() <= rel * scale + 1e-15, "{what}: got {got} want {want}");
}

fn lines(kind: &str) -> Vec<Vec<&'static str>> {
    REFERENCE
        .lines()
        .filter(|l| l.starts_with(kind))
        .map(|l| l.split_whitespace().collect())
        .collect()
}

fn propagate(
    contract: &BridgeContract,
    source_quantity: &ScientificQuantity,
    units: &str,
    value: f64,
    se: f64,
    nonlinear: Option<NonlinearMap>,
) -> antecedent_contract::bridge::BridgedClaim {
    let source_claim = claim(value, units);
    let source = BridgeSource { claim: &source_claim, quantity: source_quantity };
    derive_bridged_claim_propagating(contract, &source, Some(se), nonlinear).expect("derived")
}

#[test]
fn trial_level_fit_range_and_mean_line_variance_match_the_exact_reference() {
    for f in lines("trial") {
        let name = f[1];
        let (x0, se, cov) = (num(&f, 2), num(&f, 3), num(&f, 4));
        let trials: Vec<TrialEffectPair> = f[f.len() - 1]
            .split(',')
            .map(|p| {
                let (x, y) = p.split_once(':').unwrap();
                TrialEffectPair {
                    surrogate_effect: x.parse().unwrap(),
                    terminal_effect: y.parse().unwrap(),
                }
            })
            .collect();
        let source_q =
            quantity("trial:biomarker", "mmol_per_l", "average_effect", "adults", "do(t=1)");
        let contract = BridgeContract {
            kind: BridgeKind::SurrogateTerminalOutcome,
            source: endpoint(source_q.clone(), QuantityClass::SurrogateEndpoint),
            target: endpoint(
                quantity("trial:survival", "log_hazard", "average_effect", "adults", "do(t=1)"),
                QuantityClass::TerminalOutcome,
            ),
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: BridgeScope {
                population_id: "adults".into(),
                regime_id: "do(t=1)".into(),
            },
            mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
                trials: trials.clone(),
                min_trials: 3,
                min_r_squared: 0.5,
            }),
            assumptions: vec![assumption("validation", AssumptionKind::SurrogateValidation)],
            transport: None,
            mapping_uncertainty: None,
            source_dependence: Some(dependence(cov)),
            estimand_bridge: None,
        };
        let bridged = propagate(&contract, &source_q, "mmol_per_l", x0, se, None);
        let fit = bridged.fit.expect("fit");
        // Slope and intercept are conditioned by the surrogate offset; the relative error of the
        // centered two-pass sums stays well inside 1e-9 on every set.
        close(fit.slope, num(&f, 5), 1e-9, &format!("{name} slope"));
        close(fit.intercept, num(&f, 6), 1e-9, &format!("{name} intercept"));
        close(fit.r_squared, num(&f, 7), 1e-9, &format!("{name} r2"));
        close(bridged.value, num(&f, 8), 1e-9, &format!("{name} value"));
        close(fit.surrogate_min, num(&f, 12), 0.0, &format!("{name} min"));
        close(fit.surrogate_max, num(&f, 13), 0.0, &format!("{name} max"));
        let propagation = bridged.propagation.as_ref().expect("propagation");
        close(propagation.mean, num(&f, 9), 1e-9, &format!("{name} mean"));
        close(propagation.variance, num(&f, 10), 1e-7, &format!("{name} variance"));
        // The same line through a declared square map.
        let squared =
            propagate(&contract, &source_q, "mmol_per_l", x0, se, Some(NonlinearMap::Square));
        let nonlinear = squared.propagation.unwrap().nonlinear.expect("square");
        close(nonlinear.expectation, num(&f, 11), 1e-7, &format!("{name} E[x^2]"));
    }
}

#[test]
fn affine_propagation_matches_the_exact_variance_with_declared_dependence() {
    for f in lines("affine") {
        let name = f[1];
        let (intercept, slope, x0, se) = (num(&f, 2), num(&f, 3), num(&f, 4), num(&f, 5));
        let (va, vb, cab, cov) = (num(&f, 6), num(&f, 7), num(&f, 8), num(&f, 9));
        let source_q = quantity("survey:intent", "probability", "uptake", "panel", "observational");
        let contract = BridgeContract {
            kind: BridgeKind::PreferenceRealizedBehavior,
            source: endpoint(source_q.clone(), QuantityClass::StatedPreference),
            target: endpoint(
                quantity("behavior:uptake", "probability", "uptake", "panel", "observational"),
                QuantityClass::RealizedBehavior,
            ),
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: BridgeScope {
                population_id: "panel".into(),
                regime_id: "observational".into(),
            },
            mapping: BridgeMapping::Affine { intercept, slope },
            assumptions: vec![assumption("linkage", AssumptionKind::BehavioralLinkage)],
            transport: None,
            mapping_uncertainty: Some(MappingUncertainty {
                id: "calibration".into(),
                intercept_variance: va,
                slope_variance: vb,
                intercept_slope_covariance: cab,
            }),
            source_dependence: Some(dependence(cov)),
            estimand_bridge: None,
        };
        let bridged = propagate(&contract, &source_q, "probability", x0, se, None);
        close(bridged.value, num(&f, 10), 1e-12, &format!("{name} value"));
        let propagation = bridged.propagation.as_ref().expect("propagation");
        close(propagation.mean, num(&f, 11), 1e-12, &format!("{name} mean"));
        close(propagation.variance, num(&f, 12), 1e-9, &format!("{name} variance"));
        let squared =
            propagate(&contract, &source_q, "probability", x0, se, Some(NonlinearMap::Square));
        close(
            squared.propagation.unwrap().nonlinear.unwrap().expectation,
            num(&f, 13),
            1e-9,
            &format!("{name} E[x^2]"),
        );
    }
}

fn model(sens: f64, spec: f64, var_se: f64, var_sp: f64) -> MeasurementModel {
    MeasurementModel {
        model_id: "screening-flag".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: sens,
                specificity: spec,
                sensitivity_variance: var_se,
                specificity_variance: var_sp,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::ValidationSample {
            sample_id: "validation".into(),
            units: 400,
        },
    }
}

fn correction(
    model: MeasurementModel,
    source: (QuantityClass, ScientificQuantity),
    target: (QuantityClass, ScientificQuantity),
) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: endpoint(source.1, source.0),
        target: endpoint(target.1, target.0),
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
        },
        assumptions: vec![BridgeAssumption::for_measurement_model("flag-model", &model)],
        mapping: BridgeMapping::MeasurementCorrection(Box::new(model)),
        transport: None,
        mapping_uncertainty: None,
        source_dependence: Some(SourceMappingDependence::Independent {
            evidence: "disjoint".into(),
        }),
        estimand_bridge: None,
    }
}

#[test]
fn rogan_gladen_inversion_and_forward_map_match_the_exact_reference_with_rate_variances() {
    for f in lines("correction") {
        let name = f[1];
        let (sens, spec, vse, vsp) = (num(&f, 2), num(&f, 3), num(&f, 4), num(&f, 5));
        let (q, se) = (num(&f, 6), num(&f, 7));
        let measured =
            quantity("schema:flag", "probability", "prevalence", "clinic-a", "observational");
        let latent =
            quantity("construct:disease", "probability", "prevalence", "clinic-a", "observational");
        let backward = correction(
            model(sens, spec, vse, vsp),
            (QuantityClass::MeasuredOutcome, measured.clone()),
            (QuantityClass::LatentConstruct, latent.clone()),
        );
        let got = propagate(&backward, &measured, "probability", q, se, None);
        close(got.value, num(&f, 8), 1e-12, &format!("{name} inversion"));
        close(
            got.propagation.unwrap().variance,
            num(&f, 9),
            1e-9,
            &format!("{name} inversion variance"),
        );
        let forward = correction(
            model(sens, spec, vse, vsp),
            (QuantityClass::LatentConstruct, latent.clone()),
            (QuantityClass::MeasuredOutcome, measured),
        );
        let got = propagate(&forward, &latent, "probability", q, se, None);
        close(got.value, num(&f, 10), 1e-12, &format!("{name} forward"));
        close(
            got.propagation.unwrap().variance,
            num(&f, 11),
            1e-9,
            &format!("{name} forward variance"),
        );
    }
}

#[test]
fn the_exponential_map_uses_the_normal_mgf_and_is_refused_for_a_two_moment_law() {
    let normal = AnalyticLaw::Normal { mean: 0.4, variance: 0.09 };
    let moment = nonlinear_expectation(NonlinearMap::Exponential, &normal).unwrap();
    // exp(mu + sigma^2 / 2) with mu = 0.4 and sigma^2 = 0.09.
    close(moment.expectation, (0.4_f64 + 0.045).exp(), 1e-14, "mgf");
    close(moment.map_of_mean, 0.4_f64.exp(), 1e-14, "map of mean");
    let two = AnalyticLaw::SecondMoment { mean: 0.4, variance: 0.09 };
    assert!(nonlinear_expectation(NonlinearMap::Exponential, &two).is_err());
    let square = nonlinear_expectation(NonlinearMap::Square, &two).unwrap();
    close(square.expectation, 0.16 + 0.09, 1e-14, "square");
    assert!(
        nonlinear_expectation(NonlinearMap::Square, &AnalyticLaw::MeanOnly { mean: 1.0 }).is_err()
    );
}
