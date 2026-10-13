//! Binary outcome misclassification: corrected prevalence and risk difference.
//!
//! The non-differential route applies one sensitivity and specificity to both
//! exposure arms and keeps the covariance the shared estimates induce between
//! them. The differential route applies arm-specific rates from independent
//! validation samples. A model declares which; the other route refuses it.

use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, CorrectedFunctionalRecord, DifferentialAxis, ErrorStructure, MeasurementModel,
    MeasurementParameters, MeasurementRoute, ParameterUncertainty,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, CorrectedProportion, CorrectedRiskDifference, TwoByTwoCounts,
    corrected_prevalence, corrected_risk_difference_differential,
    corrected_risk_difference_nondifferential,
};

use super::{
    DataBinding, MeasurementError, RecordParts, build_record, check_binding, check_cancelled,
    component, digest, model_error, require_route, stats_error,
};

/// A corrected prevalence and its bound record.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedBinaryPrevalence {
    /// Corrected proportion and variance components.
    pub result: CorrectedProportion,
    /// Observed positives.
    pub events: u64,
    /// Units observed.
    pub total: u64,
    /// The corrected prevalence, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
}

/// Two exposure arms' observed outcome counts, by arm identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RiskDifferenceInput<'a> {
    /// Arm identity of the exposed group.
    pub exposed_arm_id: &'a str,
    /// Arm identity of the unexposed group.
    pub unexposed_arm_id: &'a str,
    /// Observed events and totals by arm.
    pub table: TwoByTwoCounts,
}

/// A corrected risk difference and its bound record.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedBinaryRiskDifference {
    /// Corrected risk difference and variance components.
    pub result: CorrectedRiskDifference,
    /// Observed counts.
    pub table: TwoByTwoCounts,
    /// Whether the differential (per-arm) rates were applied.
    pub differential: bool,
    /// The corrected risk difference, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
}

fn rates_of(entry: &BinaryArmRates) -> Result<BinaryErrorRates, MeasurementError> {
    BinaryErrorRates::new(
        entry.sensitivity,
        entry.specificity,
        entry.sensitivity_variance,
        entry.specificity_variance,
    )
    .map_err(|e| stats_error(&e))
}

fn parameters(
    model: &MeasurementModel,
) -> Result<(&[BinaryArmRates], f64), MeasurementError> {
    match &model.parameters {
        MeasurementParameters::Binary { rates, min_youden } => Ok((rates, *min_youden)),
        _ => Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.invalid_model",
            "binary route needs binary misclassification parameters",
        )),
    }
}

fn uncertainty_of(entries: &[&BinaryArmRates]) -> ParameterUncertainty {
    if entries.iter().any(|e| e.sensitivity_variance > 0.0 || e.specificity_variance > 0.0) {
        ParameterUncertainty::Propagated
    } else {
        ParameterUncertainty::DeclaredExact
    }
}

/// Corrected prevalence of a binary latent condition from observed positives.
///
/// For a model declared differential by arm, `binding.arm_id` selects the
/// arm's rates; a missing or unknown arm refuses rather than falling back to
/// another arm's rates.
///
/// # Errors
/// A malformed model, binding, route or structure mismatch, an unknown arm,
/// Youden's index at or below the model's minimum, or cancellation refuse.
pub fn correct_binary_prevalence(
    model: &MeasurementModel,
    binding: &DataBinding,
    events: u64,
    total: u64,
    ctx: &ExecutionContext,
) -> Result<CorrectedBinaryPrevalence, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::BinaryMisclassification)?;
    let (rates, min_youden) = parameters(model)?;
    let entry = match model.structure {
        ErrorStructure::NonDifferential => &rates[0],
        ErrorStructure::Differential(DifferentialAxis::Arm) => {
            let wanted = binding.arm_id.as_deref();
            rates.iter().find(|r| r.arm_id.as_deref() == wanted && wanted.is_some()).ok_or_else(
                || {
                    MeasurementError::new(
                        antecedent_core::reason_code!("measurement_binding_mismatch"),
                        "measurement_model.arm_mismatch",
                        format!(
                            "the model is differential by arm and has no rates for arm `{}`",
                            wanted.unwrap_or("all_arms")
                        ),
                    )
                },
            )?
        }
        ErrorStructure::Differential(_) => {
            return Err(MeasurementError::new(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "measurement_model.differential_axis_unsupported",
                "only error that differs by arm has a bounded route; other axes are refused, not pooled",
            ));
        }
        _ => {
            return Err(MeasurementError::new(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "measurement_model.differential_axis_unsupported",
                "unrecognized error structure",
            ));
        }
    };
    let rates = rates_of(entry)?;
    let result = corrected_prevalence(events, total, &rates, min_youden).map_err(|e| stats_error(&e))?;
    let identity = digest(&[("events", &[events as f64]), ("total", &[total as f64])]);
    let record = build_record(RecordParts {
        model,
        functional_id: "latent_prevalence",
        point: result.corrected,
        naive_point: Some(result.observed),
        variance: Some(result.variance),
        components: vec![
            component("sampling_of_observed_proportion", result.variance_sampling),
            component("sensitivity_specificity_uncertainty", result.variance_rates),
        ],
        rows: total,
        data_identity: identity,
        parameter_uncertainty: uncertainty_of(&[entry]),
    });
    Ok(CorrectedBinaryPrevalence { result, events, total, record })
}

/// Corrected risk difference for a misclassified binary outcome across two
/// exposure arms.
///
/// A model that is `NonDifferential` applies its shared rates to both arms
/// (one validation sample, so the arms' corrections covary); a model that is
/// differential by arm applies each arm's own rates. The model's arm binding
/// must be all arms: a model established for one arm cannot correct a
/// contrast between two.
///
/// # Errors
/// A malformed model, binding, route or structure mismatch, an arm with no
/// rates, a model bound to a single arm, Youden's index at or below the
/// minimum, or cancellation refuse.
pub fn correct_binary_risk_difference(
    model: &MeasurementModel,
    binding: &DataBinding,
    input: &RiskDifferenceInput<'_>,
    ctx: &ExecutionContext,
) -> Result<CorrectedBinaryRiskDifference, MeasurementError> {
    check_cancelled(ctx)?;
    // A contrast spans arms: a model established for a single arm cannot
    // correct it, and the data binding names no single arm.
    if matches!(model.binding.arm, ArmBinding::Arm(_)) {
        model.validate().map_err(|error| model_error(&error))?;
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_binding_mismatch"),
            "measurement_model.arm_mismatch",
            "a model established for one arm cannot correct a contrast between arms",
        ));
    }
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::BinaryMisclassification)?;
    let (rates, min_youden) = parameters(model)?;
    if input.exposed_arm_id == input.unexposed_arm_id {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "measurement_model.input_invalid",
            "the exposed and unexposed arms are the same arm",
        ));
    }
    let (result, differential, entries) = match model.structure {
        ErrorStructure::NonDifferential => {
            let shared = rates_of(&rates[0])?;
            (
                corrected_risk_difference_nondifferential(&input.table, &shared, min_youden)
                    .map_err(|e| stats_error(&e))?,
                false,
                vec![&rates[0]],
            )
        }
        ErrorStructure::Differential(DifferentialAxis::Arm) => {
            let find = |arm: &str| {
                rates.iter().find(|r| r.arm_id.as_deref() == Some(arm)).ok_or_else(|| {
                    MeasurementError::new(
                        antecedent_core::reason_code!("measurement_binding_mismatch"),
                        "measurement_model.arm_mismatch",
                        format!("the model is differential by arm and has no rates for arm `{arm}`"),
                    )
                })
            };
            let exposed = find(input.exposed_arm_id)?;
            let unexposed = find(input.unexposed_arm_id)?;
            (
                corrected_risk_difference_differential(
                    &input.table,
                    &rates_of(exposed)?,
                    &rates_of(unexposed)?,
                    min_youden,
                )
                .map_err(|e| stats_error(&e))?,
                true,
                vec![exposed, unexposed],
            )
        }
        _ => {
            return Err(MeasurementError::new(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "measurement_model.differential_axis_unsupported",
                "only error that differs by arm has a bounded route; other axes are refused, not pooled",
            ));
        }
    };
    let t = input.table;
    let identity = digest(&[(
        "table",
        &[
            t.exposed_events as f64,
            t.exposed_total as f64,
            t.unexposed_events as f64,
            t.unexposed_total as f64,
        ],
    )]);
    let record = build_record(RecordParts {
        model,
        functional_id: "risk_difference",
        point: result.corrected,
        naive_point: Some(result.naive),
        variance: Some(result.variance),
        components: vec![
            component("sampling_of_observed_proportions", result.variance_sampling),
            component("sensitivity_specificity_uncertainty", result.variance_rates),
        ],
        rows: t.exposed_total + t.unexposed_total,
        data_identity: identity,
        parameter_uncertainty: uncertainty_of(&entries),
    });
    Ok(CorrectedBinaryRiskDifference { result, table: input.table, differential, record })
}
