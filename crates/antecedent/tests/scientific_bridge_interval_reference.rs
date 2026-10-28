//! The interval of a bridged value against exact-rational sums and 50-digit quantiles.
//!
//! `oracle_data/scientific_bridge_interval_reference.py` (output frozen in
//! `scientific_bridge_interval_reference.txt`) forms the validation-trial line and its residual
//! variance from `fractions.Fraction` sums, evaluates the normal quantile (`sqrt(2) erfinv`) and the
//! Student-t quantile (bisection on the distribution function through the Gauss hypergeometric
//! function) at 50 digits, and writes the trial-level prediction interval, the affine transform of
//! the source interval and the delta-method interval of the Rogan-Gladen correction in both
//! directions, clipped to [0, 1]. Cases cover levels 0.5 to 0.9999, a line through three trials
//! (one degree of freedom), twenty trials, a surrogate offset of 1e6, tied surrogate effects, ends of
//! the validated range, a zero source standard error, a negative slope and clipping at both ends.
//! Requests outside the measured scope stay refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::PreparedScientificBridge;
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, TrialEffectPair, TrialLevelValidation, ValueKind,
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
use antecedent_io::bridge_interval::construct_bridged_claim_interval;

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

fn derive(
    contract: &BridgeContract,
    value: f64,
    units: &str,
    standard_error: Option<f64>,
) -> antecedent::analysis::scientific_bridge::ScientificBridgeResult {
    let envelope = claim(value, units);
    let source = BridgeSource { claim: &envelope, quantity: &contract.source.quantity };
    PreparedScientificBridge::prepare(contract.clone())
        .expect("licensed")
        .derive_with_standard_error(&source, standard_error)
        .expect("derived")
}

const REFERENCE: &str = include_str!("oracle_data/scientific_bridge_interval_reference.txt");

fn rows(kind: &str) -> Vec<Vec<&'static str>> {
    REFERENCE
        .lines()
        .filter(|l| l.starts_with(kind))
        .map(|l| l.split_whitespace().collect())
        .collect()
}

fn num(fields: &[&str], at: usize) -> f64 {
    fields[at].parse().unwrap_or_else(|_| panic!("field {at} of {fields:?}"))
}

fn near(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-12);
    assert!((got - want).abs() <= rel * scale, "{what}: got {got} want {want}");
}

fn surrogate(trials: Vec<TrialEffectPair>) -> BridgeContract {
    let mut contract = surrogate_template();
    contract.mapping = BridgeMapping::TrialLevelLinear(TrialLevelValidation {
        trials,
        min_trials: 3,
        min_r_squared: 0.5,
    });
    contract
}

fn surrogate_template() -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity(
                "trial:biomarker",
                "mmol_per_l",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "trial:survival",
                "log_hazard",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::Identity,
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn affine(intercept: f64, slope: f64) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::PreferenceRealizedBehavior,
        source: BridgeEndpoint {
            quantity: quantity("survey:intent", "probability", "uptake", "panel", "observational"),
            class: QuantityClass::StatedPreference,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "behavior:uptake",
                "probability",
                "uptake",
                "panel",
                "observational",
            ),
            class: QuantityClass::RealizedBehavior,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "panel".into(),
            regime_id: "observational".into(),
        },
        mapping: BridgeMapping::Affine { intercept, slope },
        assumptions: vec![BridgeAssumption {
            id: "linkage".into(),
            kind: AssumptionKind::BehavioralLinkage,
            statement: "stated intent maps to uptake by the declared line".into(),
            evidence: Some("linkage-study:2023".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn correction(
    forward: bool,
    sensitivity: f64,
    specificity: f64,
    variance_se: f64,
    variance_sp: f64,
) -> BridgeContract {
    let model = MeasurementModel {
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
                sensitivity,
                specificity,
                sensitivity_variance: variance_se,
                specificity_variance: variance_sp,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::ValidationSample {
            sample_id: "validation-sample".into(),
            units: 400,
        },
    };
    let measured = BridgeEndpoint {
        quantity: quantity("schema:flag", "probability", "prevalence", "clinic-a", "observational"),
        class: QuantityClass::MeasuredOutcome,
    };
    let latent = BridgeEndpoint {
        quantity: quantity(
            "construct:disease",
            "probability",
            "prevalence",
            "clinic-a",
            "observational",
        ),
        class: QuantityClass::LatentConstruct,
    };
    let (source, target) = if forward { (latent, measured) } else { (measured, latent) };
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source,
        target,
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
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn compare(
    name: &str,
    built: &antecedent_io::bridge_interval::BridgedClaimInterval,
    fields: [f64; 4],
) {
    let [value, sd, lower, upper] = fields;
    near(built.value, value, 1e-9, &format!("{name} value"));
    near(built.standard_error, sd, 1e-9, &format!("{name} standard error"));
    // The quantile is the library's rational approximation; the reference is exact to 50 digits.
    let scale = upper - lower;
    assert!(
        (built.lower - lower).abs() <= 1e-7 * scale.abs().max(1e-9),
        "{name} lower {} vs {lower}",
        built.lower
    );
    assert!(
        (built.upper - upper).abs() <= 1e-7 * scale.abs().max(1e-9),
        "{name} upper {} vs {upper}",
        built.upper
    );
}

#[test]
fn trial_level_prediction_intervals_match_the_exact_line_and_t_quantile() {
    for f in rows("trial") {
        let name = f[1];
        let (x0, se, level) = (num(&f, 2), num(&f, 3), num(&f, 4));
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
        let contract = surrogate(trials);
        let result = derive(&contract, x0, "mmol_per_l", Some(se));
        let built = construct_bridged_claim_interval(result.bridged(), Some(se), level)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        compare(name, &built.interval, [num(&f, 5), num(&f, 6), num(&f, 7), num(&f, 8)]);
        assert_eq!(built.interval.degrees_of_freedom, Some(num(&f, 9)));
    }
}

#[test]
fn affine_intervals_are_the_exact_transform_of_the_source_interval() {
    for f in rows("affine") {
        let name = f[1];
        let (a, b, x0, se, level) = (num(&f, 2), num(&f, 3), num(&f, 4), num(&f, 5), num(&f, 6));
        let result = derive(&affine(a, b), x0, "probability", Some(se));
        let built = construct_bridged_claim_interval(result.bridged(), Some(se), level).unwrap();
        compare(name, &built.interval, [num(&f, 7), num(&f, 8), num(&f, 9), num(&f, 10)]);
        let published = result.bridged_interval(level).expect("an exact transform needs no record");
        assert_eq!(published.licensed_by, "exact_transform_of_declared_source_interval");
    }
}

#[test]
fn correction_intervals_match_the_delta_method_in_both_directions_and_clip_to_the_unit_interval() {
    for f in rows("correction") {
        let name = f[1];
        let (sens, spec, vse, vsp) = (num(&f, 2), num(&f, 3), num(&f, 4), num(&f, 5));
        let (q, se, level) = (num(&f, 6), num(&f, 7), num(&f, 8));
        let forward = f[9] == "forward";
        let result = derive(&correction(forward, sens, spec, vse, vsp), q, "probability", Some(se));
        let built = construct_bridged_claim_interval(result.bridged(), Some(se), level).unwrap();
        compare(name, &built.interval, [num(&f, 10), num(&f, 11), num(&f, 12), num(&f, 13)]);
        assert!(built.interval.lower >= 0.0 && built.interval.upper <= 1.0);
    }
}

#[test]
fn rates_outside_the_measured_range_and_degenerate_lines_are_refused() {
    // Youden index 0.25 is below the measured minimum of 0.3.
    let weak =
        derive(&correction(false, 0.6, 0.65, 0.0004, 0.0004), 0.4, "probability", Some(0.02));
    assert_eq!(
        construct_bridged_claim_interval(weak.bridged(), Some(0.02), 0.95)
            .expect_err("refused")
            .detail,
        "scientific_bridge.interval_scope_outside_measured"
    );
    // A declared rate standard error of 0.06 exceeds the measured 0.05.
    let noisy =
        derive(&correction(false, 0.9, 0.8, 0.0036, 0.0004), 0.31, "probability", Some(0.02));
    assert_eq!(
        construct_bridged_claim_interval(noisy.bridged(), Some(0.02), 0.95)
            .expect_err("refused")
            .detail,
        "scientific_bridge.interval_scope_outside_measured"
    );
    // A perfect line with no source standard error has no prediction spread.
    let line: Vec<TrialEffectPair> = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect();
    let exact = derive(&surrogate(line), 1.0, "mmol_per_l", Some(0.0));
    assert_eq!(
        construct_bridged_claim_interval(exact.bridged(), Some(0.0), 0.95)
            .expect_err("refused")
            .detail,
        "scientific_bridge.interval_scope_outside_measured"
    );
}
