//! Repeated-sampling coverage of the interval of a bridged value.
//!
//! The interval carries the source estimate's declared standard error through the
//! bridge's mapping together with the mapping's own estimation uncertainty. Two
//! mappings estimate something and are measured; an identity or declared affine map is
//! an exact monotone transform of the source interval and has nothing to measure.
//!
//! * `surrogate_terminal_outcome` (trial-level least-squares line): `k` validation
//!   trials with surrogate effects `x_i ~ U(0.5, 2.0)` and terminal effects
//!   `0.1 + 0.9 x_i + 0.15 e_i` (the contract treats the trial effects as exact points).
//!   A new trial has true surrogate effect `x ~ U(0.9, 1.6)`, true terminal effect
//!   `0.1 + 0.9 x + 0.15 e`, and the source claim reports `x + 0.1 z`. The scored
//!   interval is the prediction interval for that new trial's terminal effect. `k` is
//!   `grid_n(20)`: 15, 20 and 40 trials (the short-series grid of the shared harness).
//! * `latent_measured_outcome` (binary-misclassification correction): sensitivity 0.9 and
//!   specificity 0.8 estimated from `m` validation units each, `m = grid_n(200)`: 100,
//!   200 and 400; a recorded prevalence from 1000 units. The scored interval is the
//!   delta-method interval of the latent prevalence (clipped to `[0, 1]`).
//!
//! Boundaries (recorded under their own functional, never gating a route): a latent
//! prevalence of 0.02, where the corrected value sits at the edge of its parameter space
//! and the clipped interval is one-sided; and a validation set whose Youden index is
//! small is outside the measured scope and refused, so is not a coverage cell.
//!
//! The scored interval is `construct_bridged_claim_interval`, the unlicensed
//! construction, and every replicate is bound under the key it builds, the key the public
//! `bridged_claim_interval` resolves a record with.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::neg_cmp_op_on_partial_ord, reason = "a NaN rate must fail the range test")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, TrialEffectPair, TrialLevelValidation, ValueKind,
    derive_bridged_claim,
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
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed, uniform,
};

const LEVEL: f64 = 0.95;
const SOURCE_SE_TRIAL: f64 = 0.1;
const SOURCE_UNITS: usize = 1000;

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

fn surrogate_contract(trials: Vec<TrialEffectPair>) -> BridgeContract {
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
        mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 3,
            min_r_squared: 0.3,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:fixture".into()),
        }],
        transport: None,

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn correction_contract(
    sensitivity: f64,
    specificity: f64,
    validation_units: u64,
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
                sensitivity_variance: sensitivity * (1.0 - sensitivity) / validation_units as f64,
                specificity_variance: specificity * (1.0 - specificity) / validation_units as f64,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::ValidationSample {
            sample_id: "validation-sample".into(),
            units: validation_units,
        },
    };
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: BridgeEndpoint {
            quantity: quantity(
                "schema:flag",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "construct:disease",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::LatentConstruct,
        },
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

struct Scored {
    construction: Construction,
    scope: ScopeFacts,
    interval: (f64, f64),
    truth: f64,
}

fn score_contract(
    contract: &BridgeContract,
    source_value: f64,
    source_units: &str,
    standard_error: f64,
    truth: f64,
    functional_suffix: &str,
) -> Option<Scored> {
    let envelope = claim(source_value, source_units);
    let source = BridgeSource { claim: &envelope, quantity: &contract.source.quantity };
    let bridged = derive_bridged_claim(contract, &source).ok()?;
    let built = construct_bridged_claim_interval(&bridged, Some(standard_error), LEVEL).ok()?;
    let key = built.basis.key;
    Some(Scored {
        construction: Construction {
            query: key.query,
            graph_class: key.graph_class,
            structure: key.structure,
            modality: key.modality,
            inference: key.inference,
            estimator: key.estimator,
            interval_method: key.interval_method,
            se_kind: key.se_kind,
            dependence: key.dependence,
            posterior: key.posterior,
            functional: format!("{}{functional_suffix}", key.functional),
            identification: key.identification,
            reported_level: key.level,
        },
        scope: ScopeFacts {
            row_count: built.basis.scope.row_count,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
        interval: (built.interval.lower, built.interval.upper),
        truth,
    })
}

fn trial_level_replicate(trials: usize, seed: u64) -> Option<Scored> {
    let mut noise = gaussian(seed);
    let mut unit = uniform(seed, 1);
    let (alpha, beta, tau) = (0.1, 0.9, 0.15);
    let pairs: Vec<TrialEffectPair> = (0..trials)
        .map(|_| {
            let x = 0.5 + 1.5 * unit();
            TrialEffectPair {
                surrogate_effect: x,
                terminal_effect: alpha + beta * x + tau * noise(),
            }
        })
        .collect();
    let x_true = 0.9 + 0.7 * unit();
    let theta = alpha + beta * x_true + tau * noise();
    let x_hat = x_true + SOURCE_SE_TRIAL * noise();
    score_contract(&surrogate_contract(pairs), x_hat, "mmol_per_l", SOURCE_SE_TRIAL, theta, "")
}

fn binomial(unit: &mut impl FnMut() -> f64, n: usize, p: f64) -> f64 {
    (0..n).filter(|_| unit() < p).count() as f64 / n as f64
}

fn correction_replicate(
    validation_units: usize,
    prevalence: f64,
    functional_suffix: &str,
    seed: u64,
) -> Option<Scored> {
    let (sensitivity, specificity) = (0.9_f64, 0.8_f64);
    let mut unit = uniform(seed, 2);
    let sensitivity_hat = binomial(&mut unit, validation_units, sensitivity);
    let specificity_hat = binomial(&mut unit, validation_units, specificity);
    let recorded_rate = sensitivity * prevalence + (1.0 - specificity) * (1.0 - prevalence);
    let recorded = binomial(&mut unit, SOURCE_UNITS, recorded_rate);
    let source_se = (recorded * (1.0 - recorded) / SOURCE_UNITS as f64).sqrt();
    if !(source_se > 0.0) {
        return None;
    }
    score_contract(
        &correction_contract(sensitivity_hat, specificity_hat, validation_units as u64),
        recorded,
        "probability",
        source_se,
        prevalence,
        functional_suffix,
    )
}

fn run(
    test: &'static str,
    dgp: &'static str,
    interval_method: &'static str,
    base: u64,
    replicate: impl Fn(u64) -> Option<Scored> + Sync,
    boundary: bool,
) {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: interval_method }, LEVEL);
    let scored = map_replicates(n_sim(), |r| replicate(stream_seed(grid_seed(base), r)));
    for outcome in scored {
        match outcome {
            Some(one) => {
                tally.bind_validation(&one.construction, one.scope, "not_applicable");
                tally.record(Some(one.interval), one.truth);
            }
            None => tally.skip(),
        }
    }
    if boundary {
        tally.emit_named_boundary();
    } else {
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn surrogate_trial_level_prediction_interval_l95() {
    let trials = grid_n(20);
    run(
        "surrogate_trial_level_prediction_interval_l95",
        "linear_trial_level_relation_residual_sd_0_15_new_trial_source_se_0_1",
        "student_t_prediction_interval",
        0x6b00_0000,
        |seed| trial_level_replicate(trials, seed),
        false,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn latent_measured_correction_prevalence_0_2_l95() {
    let units = grid_n(200);
    run(
        "latent_measured_correction_prevalence_0_2_l95",
        "binary_misclassification_se_0_9_sp_0_8_estimated_from_validation_sample_prevalence_0_2",
        "normal_delta_method_clipped",
        0x6c00_0000,
        |seed| correction_replicate(units, 0.2, "", seed),
        false,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn latent_measured_correction_prevalence_0_5_l95() {
    let units = grid_n(200);
    run(
        "latent_measured_correction_prevalence_0_5_l95",
        "binary_misclassification_se_0_9_sp_0_8_estimated_from_validation_sample_prevalence_0_5",
        "normal_delta_method_clipped",
        0x6d00_0000,
        |seed| correction_replicate(units, 0.5, "", seed),
        false,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn latent_measured_correction_low_prevalence_boundary_l95() {
    let units = grid_n(200);
    run(
        "latent_measured_correction_low_prevalence_boundary_l95",
        "binary_misclassification_se_0_9_sp_0_8_estimated_from_validation_sample_prevalence_0_02",
        "normal_delta_method_clipped",
        0x6e00_0000,
        |seed| correction_replicate(units, 0.02, ";boundary=low_prevalence", seed),
        true,
    );
}
