//! Exact posterior of the arm means of an adaptive repeated-measures trial, as an explicit
//! uncalibrated estimator.
//!
//! A declared [`RepeatedMeasuresProcedure`] fixes the within-unit structure (measurements per
//! unit, marginal variance and the declared correlation or hierarchy) and the assignment rule.
//! Given the completed units and one normal prior per arm mean, [`RepeatedMeasuresProcedure::posterior`]
//! returns the conjugate posterior of every arm mean and of the declared contrast of the final
//! analysis (the same plus and minus arms the frequentist final analysis contrasts), from
//! [`antecedent_stats::adaptive_inference::repeated_measures_posterior`].
//!
//! # What the posterior conditions on
//!
//! The posterior conditions on the observed units. When the assignment of a unit depends on the
//! data of earlier units and on independent randomization only, its probability factors out of
//! the likelihood of the arm means (the likelihood principle), so the posterior is the same
//! under any such policy and is reported with no correction for the adaptation. The
//! probabilities of the realized assignments are retained as selection evidence and are not used
//! by the posterior. This is a property of the posterior only: the frequentist analysis of the
//! same trial is calibrated to the whole procedure and is a separate report. An assignment that
//! depends on an unobserved quantity is refused with `cell_not_licensed`
//! (`adaptive_inference.selection_not_ignorable`).
//!
//! # What it does not say
//!
//! The within-unit variance and correlation are declared, not estimated, and the arm-mean priors
//! are declared, not defaulted: a prior alone never identifies the contrast, and an arm without a
//! completed unit is refused. The report is unlicensed and its repeated-sampling calibration is
//! unmeasured; both facts are retained in the artifact.
//!
//! The retained report carries the completed units, so the loader recomputes every reported
//! number from them under the declared procedure and refuses a payload that does not reproduce.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use antecedent_stats::adaptive_inference::{
    NormalMeanPrior, RepeatedMeasuresPosterior, RepeatedMeasuresPosteriorRequest as StatsRequest,
    SelectionMechanism, UnitAssignment, UnitRecord, repeated_measures_posterior,
};
use serde::{Deserialize, Serialize};

use super::adaptive_inference::AdaptiveInferenceRefusal;
use super::adaptive_procedure_report::{RepeatedMeasuresProcedure, refusal_of};

/// Estimator id of the repeated-measures arm-contrast posterior.
pub const REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID: &str =
    "bayesian.repeated_measures_arm_contrast";

/// Calibration standing carried by the report: repeated-sampling calibration is unmeasured.
const UNCALIBRATED_STANDING: &str = "unmeasured";

/// Format name of the retained artifact envelope.
pub const REPEATED_MEASURES_POSTERIOR_ARTIFACT_FORMAT: &str =
    "antecedent.repeated_measures_posterior";

/// Layout version of the envelope. Another version is refused, not migrated silently.
pub const REPEATED_MEASURES_POSTERIOR_ARTIFACT_VERSION: u32 = 1;

const SELECTION_FIXED: &str = "fixed";
const SELECTION_OBSERVED_HISTORY: &str = "depends_on_observed_history";
const DEFAULT_CREDIBLE_LEVEL: f64 = 0.95;

/// One completed unit: its arm, its measurements and the probability its realized arm had.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorUnit {
    /// The unit's arm.
    pub arm: usize,
    /// The unit's measurements, in order.
    pub measurements: Vec<f64>,
    /// The probability the policy assigned to the realized arm; retained as evidence only.
    pub assignment_probability: f64,
}

/// A posterior request against a declared procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresPosteriorSpec {
    /// One normal prior per arm mean; declared, never defaulted.
    pub priors: Vec<NormalMeanPrior>,
    /// The completed units.
    pub units: Vec<PosteriorUnit>,
    /// What the assignment depended on. `None` takes it from the declared assignment rule: a
    /// fixed rule is `Fixed`, an adaptive rule depends on the observed history. Declaring
    /// `DependsOnUnobserved` is refused; declaring `Fixed` for an adaptive procedure is invalid.
    pub selection: Option<SelectionMechanism>,
    /// Credible level in `(0, 1)`; `None` uses `0.95`.
    pub credible_level: Option<f64>,
}

/// A declared prior on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PriorWire {
    /// Prior mean of the arm mean.
    pub mean: f64,
    /// Prior variance of the arm mean.
    pub variance: f64,
}

/// Posterior of one arm mean on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArmPosteriorWire {
    /// Units observed under the arm.
    pub units: u64,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
}

/// Posterior of the declared contrast on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastPosteriorWire {
    /// Posterior mean of the plus arm mean minus the minus arm mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Equal-tailed credible interval.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive.
    pub probability_positive: f64,
}

/// The retained result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepeatedMeasuresPosteriorReport {
    /// Always [`REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID`].
    pub estimator_id: String,
    /// `posterior_equal_tailed_credible_interval`.
    pub uncertainty: String,
    /// What the contrast is the contrast of.
    pub estimand: String,
    /// The likelihood the posterior is exact under.
    pub likelihood: String,
    /// The assumptions the numbers rest on.
    pub assumptions: Vec<String>,
    /// Identity (hex) of the declared procedure the posterior was computed under.
    pub procedure_identity: String,
    /// Name of the procedure's analysis route.
    pub route: String,
    /// Arm entering the contrast with a plus sign.
    pub plus: u64,
    /// Arm entering the contrast with a minus sign.
    pub minus: u64,
    /// The declared prior of each arm mean.
    pub priors: Vec<PriorWire>,
    /// Credible level of the interval.
    pub credible_level: f64,
    /// Posterior of each arm mean.
    pub arms: Vec<ArmPosteriorWire>,
    /// Posterior of the declared contrast.
    pub contrast: ContrastPosteriorWire,
    /// Variance of one generalized-least-squares unit summary under the declared structure.
    pub unit_summary_variance: f64,
    /// `fixed` or `depends_on_observed_history`.
    pub selection: String,
    /// Whether the assignment adapted to observed data.
    pub adaptive: bool,
    /// The premise under which the posterior needs no correction for the adaptation.
    pub selection_premise: String,
    /// The probability of each unit's realized arm, retained and unused by the posterior.
    pub assignment_probabilities: Vec<f64>,
    /// Arm of each completed unit.
    pub unit_arms: Vec<u64>,
    /// Measurements of each completed unit.
    pub unit_measurements: Vec<Vec<f64>>,
    /// Always `false`: no support license covers this estimator.
    pub licensed: bool,
    /// Always `false`: repeated-sampling calibration is unmeasured.
    pub calibration_measured: bool,
    /// Calibration standing; `unmeasured`.
    pub calibration: String,
}

fn invalid(message: impl Into<String>) -> AdaptiveInferenceRefusal {
    AdaptiveInferenceRefusal {
        code: reason_code!("invalid_argument"),
        detail: "adaptive_inference.input_invalid".to_owned(),
        message: message.into(),
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1.0e-9 * (1.0 + b.abs())
}

fn selection_tag(selection: SelectionMechanism) -> Result<&'static str, AdaptiveInferenceRefusal> {
    match selection {
        SelectionMechanism::Fixed => Ok(SELECTION_FIXED),
        SelectionMechanism::DependsOnObservedHistory => Ok(SELECTION_OBSERVED_HISTORY),
        _ => Err(invalid(
            "a report never records an assignment that depends on an unobserved quantity",
        )),
    }
}

fn selection_of(tag: &str) -> Result<SelectionMechanism, AdaptiveInferenceRefusal> {
    match tag {
        SELECTION_FIXED => Ok(SelectionMechanism::Fixed),
        SELECTION_OBSERVED_HISTORY => Ok(SelectionMechanism::DependsOnObservedHistory),
        _ => Err(invalid("unknown selection mechanism in the report")),
    }
}

fn assumptions() -> Vec<String> {
    [
        "within-unit measurements are Gaussian around the arm mean with the declared variance and \
         correlation (declared, not estimated)",
        "units are independent given the arm means",
        "the assignment of a unit depends only on observed earlier data and independent \
         randomization, so the likelihood of the arm means is the same under any such policy",
        "the arm-mean priors are declared; a prior alone does not identify the contrast",
        "the posterior is not the frequentist whole-procedure analysis and its repeated-sampling \
         calibration is unmeasured",
    ]
    .map(str::to_owned)
    .to_vec()
}

fn units_of(units: &[PosteriorUnit]) -> Vec<UnitRecord<'_>> {
    units
        .iter()
        .map(|u| UnitRecord {
            arm: u.arm,
            measurements: &u.measurements,
            assignment_probability: u.assignment_probability,
        })
        .collect()
}

fn compute(
    procedure: &RepeatedMeasuresProcedure,
    priors: &[NormalMeanPrior],
    units: &[PosteriorUnit],
    selection: SelectionMechanism,
    level: f64,
) -> Result<RepeatedMeasuresPosterior, AdaptiveInferenceRefusal> {
    let spec = procedure.spec();
    let records = units_of(units);
    repeated_measures_posterior(&StatsRequest {
        structure: spec.structure(),
        selection,
        priors,
        units: &records,
        plus: spec.analysis().plus,
        minus: spec.analysis().minus,
        credible_level: level,
    })
    .map_err(|error| refusal_of(&error))
}

fn declared_selection(
    procedure: &RepeatedMeasuresProcedure,
    requested: Option<SelectionMechanism>,
) -> Result<SelectionMechanism, AdaptiveInferenceRefusal> {
    let declared = match procedure.spec().assignment() {
        UnitAssignment::Fixed { .. } => SelectionMechanism::Fixed,
        _ => SelectionMechanism::DependsOnObservedHistory,
    };
    match requested {
        None => Ok(declared),
        Some(SelectionMechanism::Fixed) if declared != SelectionMechanism::Fixed => Err(invalid(
            "the procedure declares an adaptive assignment rule, so its selection cannot be \
             declared fixed",
        )),
        Some(selection) => Ok(selection),
    }
}

impl RepeatedMeasuresProcedure {
    /// The exact posterior of the arm means and of the declared contrast under the procedure's
    /// within-unit structure.
    ///
    /// # Errors
    /// `cell_not_licensed` with `adaptive_inference.selection_not_ignorable` for an assignment
    /// that depends on an unobserved quantity, `adaptive_inference.arm_not_observed` for a
    /// contrast arm without a completed unit, and `adaptive_inference.structure_incomplete` for
    /// an inadmissible declared structure; `invalid_argument` for malformed units, priors, level
    /// or a selection that contradicts the declared assignment rule.
    pub fn posterior(
        &self,
        spec: &RepeatedMeasuresPosteriorSpec,
    ) -> Result<RepeatedMeasuresPosteriorReport, AdaptiveInferenceRefusal> {
        let selection = declared_selection(self, spec.selection)?;
        let level = spec.credible_level.unwrap_or(DEFAULT_CREDIBLE_LEVEL);
        let posterior = compute(self, &spec.priors, &spec.units, selection, level)?;
        let analysis = self.spec().analysis();
        Ok(RepeatedMeasuresPosteriorReport {
            estimator_id: REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID.to_owned(),
            uncertainty: "posterior_equal_tailed_credible_interval".to_owned(),
            estimand: "difference of the plus and minus arm means of the population mean \
                       measurement, under the declared within-unit dependence"
                .to_owned(),
            likelihood: posterior.likelihood.to_owned(),
            assumptions: assumptions(),
            procedure_identity: self.identity().to_owned(),
            route: self.spec().route_name().to_owned(),
            plus: analysis.plus as u64,
            minus: analysis.minus as u64,
            priors: spec
                .priors
                .iter()
                .map(|p| PriorWire { mean: p.mean, variance: p.variance })
                .collect(),
            credible_level: posterior.credible_level,
            arms: posterior
                .arms
                .iter()
                .map(|a| ArmPosteriorWire {
                    units: a.units as u64,
                    mean: a.mean,
                    standard_deviation: a.standard_deviation,
                })
                .collect(),
            contrast: ContrastPosteriorWire {
                mean: posterior.contrast.mean,
                standard_deviation: posterior.contrast.standard_deviation,
                interval: posterior.contrast.interval,
                probability_positive: posterior.contrast.probability_positive,
            },
            unit_summary_variance: posterior.unit_summary_variance,
            selection: selection_tag(selection)?.to_owned(),
            adaptive: posterior.evidence.adaptive,
            selection_premise: posterior.evidence.premise.to_owned(),
            assignment_probabilities: posterior.evidence.assignment_probabilities,
            unit_arms: spec.units.iter().map(|u| u.arm as u64).collect(),
            unit_measurements: spec.units.iter().map(|u| u.measurements.clone()).collect(),
            licensed: false,
            calibration_measured: false,
            calibration: UNCALIBRATED_STANDING.to_owned(),
        })
    }
}

impl RepeatedMeasuresPosteriorReport {
    /// Recompute every reported number from the retained units under `procedure`.
    ///
    /// # Errors
    /// An `invalid_argument` refusal naming the first inconsistency: another estimator or
    /// procedure, a claimed license or measured calibration, or a number the retained units do
    /// not reproduce.
    pub fn validate(
        &self,
        procedure: &RepeatedMeasuresProcedure,
    ) -> Result<(), AdaptiveInferenceRefusal> {
        if self.estimator_id != REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID {
            return Err(invalid("not a repeated-measures posterior report"));
        }
        if self.licensed || self.calibration_measured || self.calibration != UNCALIBRATED_STANDING {
            return Err(invalid("a repeated-measures posterior is unlicensed and uncalibrated"));
        }
        if self.procedure_identity != procedure.identity() {
            return Err(invalid("the report was computed under a different declared procedure"));
        }
        let analysis = procedure.spec().analysis();
        if self.plus != analysis.plus as u64 || self.minus != analysis.minus as u64 {
            return Err(invalid("the report's contrast arms differ from the procedure's"));
        }
        let rows = self.unit_arms.len();
        if rows == 0
            || self.unit_measurements.len() != rows
            || self.assignment_probabilities.len() != rows
        {
            return Err(invalid("unit arms and measurements do not align"));
        }
        let units = self
            .unit_arms
            .iter()
            .zip(&self.unit_measurements)
            .zip(&self.assignment_probabilities)
            .map(|((arm, measurements), probability)| {
                Ok(PosteriorUnit {
                    arm: usize::try_from(*arm).map_err(|_| invalid("unit arm out of range"))?,
                    measurements: measurements.clone(),
                    assignment_probability: *probability,
                })
            })
            .collect::<Result<Vec<_>, AdaptiveInferenceRefusal>>()?;
        let priors: Vec<NormalMeanPrior> = self
            .priors
            .iter()
            .map(|p| NormalMeanPrior { mean: p.mean, variance: p.variance })
            .collect();
        let selection = selection_of(&self.selection)?;
        let expected = compute(procedure, &priors, &units, selection, self.credible_level)?;
        let arms_match = expected.arms.len() == self.arms.len()
            && expected.arms.iter().zip(&self.arms).all(|(e, a)| {
                e.units as u64 == a.units
                    && near(a.mean, e.mean)
                    && near(a.standard_deviation, e.standard_deviation)
            });
        let c = &expected.contrast;
        let contrast_match = near(self.contrast.mean, c.mean)
            && near(self.contrast.standard_deviation, c.standard_deviation)
            && near(self.contrast.interval[0], c.interval[0])
            && near(self.contrast.interval[1], c.interval[1])
            && near(self.contrast.probability_positive, c.probability_positive);
        if !arms_match
            || !contrast_match
            || !near(self.unit_summary_variance, expected.unit_summary_variance)
            || self.adaptive != expected.evidence.adaptive
            || self.selection_premise != expected.evidence.premise
        {
            return Err(invalid("reported posterior is not the posterior of the retained units"));
        }
        Ok(())
    }
}

/// Write a repeated-measures posterior report as a retained artifact.
///
/// # Errors
/// The report fails its own validation under `procedure`, or cannot be serialized.
pub fn encode_repeated_measures_posterior_report(
    report: &RepeatedMeasuresPosteriorReport,
    procedure: &RepeatedMeasuresProcedure,
) -> Result<String, AdaptiveInferenceRefusal> {
    report.validate(procedure)?;
    seal_report_without_validation(report)
}

/// Load a retained repeated-measures posterior artifact and recompute it under `procedure`.
///
/// # Errors
/// A refusal of the envelope (format, version, estimator, digest, claimed license) or of the
/// report validation (another procedure, numbers the retained units do not reproduce).
pub fn load_repeated_measures_posterior_report(
    text: &str,
    procedure: &RepeatedMeasuresProcedure,
) -> Result<RepeatedMeasuresPosteriorReport, AdaptiveInferenceRefusal> {
    let report = open_envelope(text)?;
    report.validate(procedure)?;
    Ok(report)
}

/// One completed unit on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorUnitWire {
    /// The unit's arm.
    pub arm: u64,
    /// The unit's measurements, in order.
    pub measurements: Vec<f64>,
    /// The probability the policy assigned to the realized arm.
    pub assignment_probability: f64,
}

/// A posterior request on the wire, for the hosts that pass JSON.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepeatedMeasuresPosteriorRequestWire {
    /// One normal prior per arm mean.
    pub priors: Vec<PriorWire>,
    /// The completed units.
    pub units: Vec<PosteriorUnitWire>,
    /// `fixed`, `depends_on_observed_history` or `depends_on_unobserved`; absent takes the
    /// declared assignment rule's.
    #[serde(default)]
    pub selection: Option<String>,
    /// Credible level in `(0, 1)`; absent uses `0.95`.
    #[serde(default)]
    pub credible_level: Option<f64>,
}

impl RepeatedMeasuresPosteriorRequestWire {
    /// The request as the typed specification.
    ///
    /// # Errors
    /// `invalid_argument` for an unknown selection mechanism or an arm that is not an index.
    pub fn into_spec(self) -> Result<RepeatedMeasuresPosteriorSpec, AdaptiveInferenceRefusal> {
        let selection = match self.selection.as_deref() {
            None => None,
            Some(SELECTION_FIXED) => Some(SelectionMechanism::Fixed),
            Some(SELECTION_OBSERVED_HISTORY) => Some(SelectionMechanism::DependsOnObservedHistory),
            Some("depends_on_unobserved") => Some(SelectionMechanism::DependsOnUnobserved),
            Some(_) => return Err(invalid("unknown selection mechanism")),
        };
        let units = self
            .units
            .into_iter()
            .map(|u| {
                Ok(PosteriorUnit {
                    arm: usize::try_from(u.arm).map_err(|_| invalid("unit arm out of range"))?,
                    measurements: u.measurements,
                    assignment_probability: u.assignment_probability,
                })
            })
            .collect::<Result<Vec<_>, AdaptiveInferenceRefusal>>()?;
        Ok(RepeatedMeasuresPosteriorSpec {
            priors: self
                .priors
                .iter()
                .map(|p| NormalMeanPrior { mean: p.mean, variance: p.variance })
                .collect(),
            units,
            selection,
            credible_level: self.credible_level,
        })
    }
}

/// The retained envelope: format, layout version, estimator id, the standing flags, the report and
/// a digest of all of them, so a loader refuses an edited payload, another estimator and a layout
/// it does not know. A forger who recomputes the digest is caught by the report's recomputation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    format: String,
    version: u32,
    estimator_id: String,
    licensed: bool,
    calibration_measured: bool,
    payload: serde_json::Value,
    digest: String,
}

fn digest_of(licensed: bool, calibration_measured: bool, payload: &serde_json::Value) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REPEATED_MEASURES_POSTERIOR_ARTIFACT_FORMAT.as_bytes());
    hasher.update(&REPEATED_MEASURES_POSTERIOR_ARTIFACT_VERSION.to_le_bytes());
    hasher.update(REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID.as_bytes());
    hasher.update(&[u8::from(licensed), u8::from(calibration_measured)]);
    hasher.update(payload.to_string().as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn contains_null(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Array(items) => items.iter().any(contains_null),
        serde_json::Value::Object(map) => map.values().any(contains_null),
        _ => false,
    }
}

/// Wrap a report in the envelope without validating it against a procedure. This exists so a
/// test can build a resealed forgery; the producer path is
/// [`encode_repeated_measures_posterior_report`].
///
/// # Errors
/// A report with a non-finite number, which JSON cannot represent.
#[doc(hidden)]
pub fn seal_report_without_validation(
    report: &RepeatedMeasuresPosteriorReport,
) -> Result<String, AdaptiveInferenceRefusal> {
    let payload = serde_json::to_value(report).map_err(|e| invalid(format!("report: {e}")))?;
    if contains_null(&payload) {
        return Err(invalid("the report holds a non-finite number the artifact cannot represent"));
    }
    let digest = digest_of(false, false, &payload);
    serde_json::to_string(&Envelope {
        format: REPEATED_MEASURES_POSTERIOR_ARTIFACT_FORMAT.to_owned(),
        version: REPEATED_MEASURES_POSTERIOR_ARTIFACT_VERSION,
        estimator_id: REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID.to_owned(),
        licensed: false,
        calibration_measured: false,
        payload,
        digest,
    })
    .map_err(|e| invalid(format!("artifact: {e}")))
}

fn open_envelope(text: &str) -> Result<RepeatedMeasuresPosteriorReport, AdaptiveInferenceRefusal> {
    let envelope: Envelope =
        serde_json::from_str(text).map_err(|e| invalid(format!("artifact is malformed: {e}")))?;
    if envelope.format != REPEATED_MEASURES_POSTERIOR_ARTIFACT_FORMAT
        || envelope.version != REPEATED_MEASURES_POSTERIOR_ARTIFACT_VERSION
        || envelope.estimator_id != REPEATED_MEASURES_POSTERIOR_ESTIMATOR_ID
    {
        return Err(invalid("artifact is not a repeated-measures posterior of this layout"));
    }
    if envelope.licensed || envelope.calibration_measured {
        return Err(invalid(
            "an uncalibrated artifact cannot claim a license or a measured calibration",
        ));
    }
    if digest_of(envelope.licensed, envelope.calibration_measured, &envelope.payload)
        != envelope.digest
    {
        return Err(invalid("artifact digest does not match its payload"));
    }
    serde_json::from_value(envelope.payload)
        .map_err(|e| invalid(format!("artifact payload is not a posterior report: {e}")))
}
