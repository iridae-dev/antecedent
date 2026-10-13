//! Adaptive inference result artifact (`adaptive_inference_result_v1`).
//!
//! One inferential result from an adaptive experiment, with the monitoring
//! contract it was computed under:
//!
//! * the embedded `adaptive_monitoring_contract_v1` metadata (the procedure,
//!   the retained history, the stopping rule, the paradigm, the boundaries and
//!   the observation accounting);
//! * the request: which look, which route and its parameters; and
//! * the stored values, the method name, the point-only claim, the
//!   **unmeasured** calibration status and the closed guarantee.
//!
//! A consumer trusts none of the stored values. It replays the embedded
//! contract through the contract checks (so a missing element, an inconsistent
//! record or a paradigm the cell is not licensed for is refused), requires the
//! contract's paradigm to be the request's, and recomputes the result from the
//! retained history with the estimators. A resealed edit of the stopping rule,
//! of a probability, of an outcome, of the paradigm or of a stored value is
//! refused, and a consumer that retained the identity independently refuses a
//! consistently resealed result of another experiment.
//!
//! What this does **not** say: no interval, credible interval, significance
//! level or coverage statement for the whole procedure (closed:
//! `cell_not_licensed`), and no claim that the result transfers to another
//! assignment policy, outcome structure, monitoring strategy or stopping rule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, BoundarySidedness, InferenceParadigm, SpendingKind, StoppingSpec,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::adaptive_inference::{
    AdaptiveInferenceError, ArmCounts, Estimand, GroupSequentialPlan, OutcomeModelPrior,
    PosteriorRule, Sidedness, SpendingFunction, WeightScheme, WeightedHistory, WeightedOptions,
    adaptively_weighted_estimate, anytime_valid_arm_sequences, bayesian_posterior,
    expected_loss_stopping, group_sequential_decision, standardized_difference,
};
use antecedent_io_core::adaptive_monitoring_contract_artifact::{
    AdaptiveMonitoringContractArtifact, AdaptiveMonitoringContractError, MonitoringContractMeta,
};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE: &str = "adaptive_inference_result_v1";
/// Calibration status: no whole-procedure coverage record measures any route.
pub const ADAPTIVE_INFERENCE_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy.
pub const MAX_ADAPTIVE_INFERENCE_RESULT_BYTES: usize = 256 * 1024 * 1024;
/// Relative tolerance when comparing a stored value with its recomputation
/// (`exp`, `ln` and `erfc` may differ in the last bits across platforms).
pub const ADAPTIVE_RESULT_REPLAY_TOLERANCE: f64 = 1e-9;

const ARTIFACT_KIND: &str = crate::artifact_kinds::ADAPTIVE_INFERENCE_RESULT;
const META_SECTION: &str = "adaptive_inference_result_meta";

crate::sectioned_artifact_error!(
    /// Why an adaptive inference result artifact was refused or could not be
    /// produced.
    AdaptiveInferenceResultArtifactError,
    "adaptive inference result",
    "adaptive_inference",
    "result",
    true
);

type Error = AdaptiveInferenceResultArtifactError;

fn malformed(text: impl Into<String>) -> Error {
    Error::Malformed(text.into())
}

impl From<AdaptiveMonitoringContractError> for Error {
    fn from(error: AdaptiveMonitoringContractError) -> Self {
        match error {
            AdaptiveMonitoringContractError::Refused { code, detail, message } => {
                let message = if message.starts_with(&detail) {
                    message
                } else {
                    format!("{detail}: {message}")
                };
                Self::Refused { code, message }
            }
            AdaptiveMonitoringContractError::IdentityMismatch { field } => {
                Self::IdentityMismatch { field }
            }
            AdaptiveMonitoringContractError::ReplayMismatch(what) => Self::ResultMismatch(what),
            AdaptiveMonitoringContractError::UnsupportedVersion { version } => {
                Self::UnsupportedVersion { version }
            }
            other => Self::Malformed(other.to_string()),
        }
    }
}

// -- the request ------------------------------------------------------------------------------

/// What is asked of an adaptive experiment's retained history.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveResultSpec {
    /// Beta-Bernoulli posteriors of every arm after `look`.
    BayesianPosterior {
        /// Look index.
        look: u32,
        /// Prior first shape (at least one).
        prior_alpha: f64,
        /// Prior second shape (at least one).
        prior_beta: f64,
    },
    /// The contract's posterior stopping rule evaluated after `look`.
    PosteriorStopping {
        /// Look index.
        look: u32,
        /// Prior first shape.
        prior_alpha: f64,
        /// Prior second shape.
        prior_beta: f64,
    },
    /// An adaptively weighted estimate after `look`.
    AdaptiveWeighting {
        /// Look index.
        look: u32,
        /// Arm entering with a plus sign (or the arm of an arm mean).
        plus_arm: u32,
        /// Arm entering with a minus sign; absent for an arm mean.
        minus_arm: Option<u32>,
        /// `uniform` or `variance_stabilizing`.
        scheme: String,
        /// Prior mean of the predictable outcome model.
        prior_mean: f64,
        /// Pseudo-observations behind the prior mean.
        prior_strength: f64,
        /// Weight subjects by their number of measurements.
        size_weighted: bool,
    },
    /// Confidence sequences for bounded arm means after `look`.
    ConfidenceSequence {
        /// Look index.
        look: u32,
        /// Two-sided miscoverage.
        alpha: f64,
        /// Sample size the boundary is tuned for.
        tuning_samples: f64,
        /// Contrast arms `(plus, minus)`, when asked.
        contrast: Option<(u32, u32)>,
    },
    /// The contract's group-sequential rule applied to the looks up to `look`.
    GroupSequential {
        /// Last look performed.
        look: u32,
        /// Control arm.
        control_arm: u32,
        /// Treatment arm.
        treatment_arm: u32,
    },
}

impl AdaptiveResultSpec {
    /// The paradigm this request belongs to.
    #[must_use]
    pub const fn paradigm(&self) -> InferenceParadigm {
        match self {
            Self::BayesianPosterior { .. } => InferenceParadigm::BayesianPosterior,
            Self::PosteriorStopping { .. } => InferenceParadigm::DecisionTheoreticStopping,
            Self::AdaptiveWeighting { .. } => InferenceParadigm::FrequentistAdaptiveWeighting,
            Self::ConfidenceSequence { .. } => InferenceParadigm::AnytimeValidFrequentist,
            Self::GroupSequential { .. } => InferenceParadigm::FrequentistGroupSequential,
        }
    }

    /// The look the request is about.
    #[must_use]
    pub const fn look(&self) -> u32 {
        match self {
            Self::BayesianPosterior { look, .. }
            | Self::PosteriorStopping { look, .. }
            | Self::AdaptiveWeighting { look, .. }
            | Self::ConfidenceSequence { look, .. }
            | Self::GroupSequential { look, .. } => *look,
        }
    }
}

/// The values of one result, in the route's fixed order.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveResult {
    /// The paradigm.
    pub paradigm: InferenceParadigm,
    /// The method that produced the values.
    pub method: &'static str,
    /// Values in the order documented on [`compute_adaptive_result`].
    pub values: Vec<f64>,
    /// Detail of the closed guarantee that goes with the values.
    pub closed_guarantee: &'static str,
}

/// A refusal to compute a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdaptiveResultError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
}

impl std::fmt::Display for AdaptiveResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for AdaptiveResultError {}

impl From<AdaptiveInferenceError> for AdaptiveResultError {
    fn from(error: AdaptiveInferenceError) -> Self {
        Self { code: error.code, detail: error.detail.to_owned(), message: error.message }
    }
}

fn contract_refusal(refusal: &antecedent_contract::ExternalRefusal) -> AdaptiveResultError {
    AdaptiveResultError {
        code: refusal.code,
        detail: refusal.detail.clone(),
        message: refusal.remedy.unwrap_or("").to_owned(),
    }
}

fn request_invalid(message: &str) -> AdaptiveResultError {
    AdaptiveResultError {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "adaptive_inference.input_invalid".to_owned(),
        message: message.to_owned(),
    }
}

// -- computation ------------------------------------------------------------------------------

struct LookData {
    arms: usize,
    units: Vec<usize>,
    assigned: Vec<usize>,
    probabilities: Vec<f64>,
}

fn look_data(
    contract: &AdaptiveDesignContract,
    look: usize,
) -> Result<LookData, AdaptiveResultError> {
    if look >= contract.looks() {
        return Err(request_invalid("the look is not one of the contract's interim analyses"));
    }
    let declaration = contract.declaration();
    let arms = contract.arms();
    let units = contract.analysis_units(look);
    let rows = declaration.assignment_probabilities.as_ref().expect("validated contract");
    let mut assigned = Vec::with_capacity(units.len());
    let mut probabilities = Vec::with_capacity(units.len() * arms);
    for &unit in &units {
        assigned.push(declaration.assigned_arm[unit] as usize);
        probabilities.extend_from_slice(&rows[unit]);
    }
    Ok(LookData { arms, units, assigned, probabilities })
}

fn unit_outcome(contract: &AdaptiveDesignContract, unit: usize) -> f64 {
    let measurements = &contract.declaration().outcomes[unit];
    measurements.iter().sum::<f64>() / measurements.len() as f64
}

fn arm_counts(contract: &AdaptiveDesignContract, data: &LookData) -> Vec<ArmCounts> {
    let mut counts = vec![ArmCounts { successes: 0, failures: 0 }; data.arms];
    for (&unit, &arm) in data.units.iter().zip(&data.assigned) {
        // Beta-Bernoulli cells declare single {0, 1} outcomes.
        if unit_outcome(contract, unit) > 0.5 {
            counts[arm].successes += 1;
        } else {
            counts[arm].failures += 1;
        }
    }
    counts
}

fn estimand_of(
    plus: u32,
    minus: Option<u32>,
    arms: usize,
) -> Result<Estimand, AdaptiveResultError> {
    let in_range = |arm: u32| (arm as usize) < arms;
    match minus {
        None if in_range(plus) => Ok(Estimand::ArmMean(plus as usize)),
        Some(minus) if in_range(plus) && in_range(minus) && plus != minus => {
            Ok(Estimand::Contrast { plus: plus as usize, minus: minus as usize })
        }
        _ => Err(request_invalid("the estimand names arms outside the declared arms")),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per paradigm keeps each route's data preparation next to its estimator call"
)]
/// Compute the result a request asks of a contract.
///
/// The route is authorized first: the paradigm must be one the contract's
/// procedure cell is licensed for, the exposure must equal the assignment and
/// the delay must not depend on the outcome. Value order:
///
/// * Bayesian posterior: per arm `alpha, beta, mean, variance,
///   probability_best, expected_loss`.
/// * Posterior stopping: `best_arm, probability_best, expected_loss, stop`.
/// * Adaptive weighting: `estimate, variance, units, effective_units,
///   max_weight_share`.
/// * Confidence sequence: per arm `count, mean, lower, upper`, then the
///   contrast `lower, upper` when asked.
/// * Group sequential: `stopped_at` (`-1` when no look crossed), then per
///   performed look `critical, statistic, spent`.
///
/// # Errors
/// A refused authorization, a request outside the contract, a floor failure,
/// an outcome outside a route's range, or cancellation.
pub fn compute_adaptive_result(
    contract: &AdaptiveDesignContract,
    spec: &AdaptiveResultSpec,
    ctx: &ExecutionContext,
) -> Result<AdaptiveResult, AdaptiveResultError> {
    let paradigm = spec.paradigm();
    contract.authorize(paradigm).map_err(|r| contract_refusal(&r))?;
    let look = spec.look() as usize;
    let data = look_data(contract, look)?;
    let declaration = contract.declaration();
    match spec {
        AdaptiveResultSpec::BayesianPosterior { prior_alpha, prior_beta, .. } => {
            let counts = arm_counts(contract, &data);
            let posterior = bayesian_posterior(*prior_alpha, *prior_beta, &counts, ctx)?;
            let values = posterior
                .arms
                .iter()
                .flat_map(|a| {
                    [a.alpha, a.beta, a.mean, a.variance, a.probability_best, a.expected_loss]
                })
                .collect();
            Ok(AdaptiveResult {
                paradigm,
                method: "beta_bernoulli_conjugate_posterior",
                values,
                closed_guarantee: posterior.guarantee.detail,
            })
        }
        AdaptiveResultSpec::PosteriorStopping { prior_alpha, prior_beta, .. } => {
            let rule = match declaration.stopping_rule.map(|r| r.spec) {
                Some(StoppingSpec::PosteriorProbabilityBest { threshold }) => {
                    PosteriorRule::ProbabilityBest { threshold }
                }
                Some(StoppingSpec::PosteriorExpectedLoss { loss_threshold }) => {
                    PosteriorRule::ExpectedLoss { loss_threshold }
                }
                _ => {
                    return Err(request_invalid(
                        "the contract's stopping rule is not a posterior rule",
                    ));
                }
            };
            let counts = arm_counts(contract, &data);
            let posterior = bayesian_posterior(*prior_alpha, *prior_beta, &counts, ctx)?;
            let decision = expected_loss_stopping(&posterior, rule)?;
            Ok(AdaptiveResult {
                paradigm,
                method: "posterior_expected_loss_stopping",
                values: vec![
                    decision.best_arm as f64,
                    decision.probability_best,
                    decision.expected_loss,
                    f64::from(u8::from(decision.stop)),
                ],
                closed_guarantee: decision.guarantee.detail,
            })
        }
        AdaptiveResultSpec::AdaptiveWeighting {
            plus_arm,
            minus_arm,
            scheme,
            prior_mean,
            prior_strength,
            size_weighted,
            ..
        } => {
            let scheme = match scheme.as_str() {
                "uniform" => WeightScheme::Uniform,
                "variance_stabilizing" => WeightScheme::VarianceStabilizing,
                _ => return Err(request_invalid("an unknown weighting scheme")),
            };
            let estimand = estimand_of(*plus_arm, *minus_arm, data.arms)?;
            let outcomes: Vec<f64> =
                data.units.iter().map(|&u| unit_outcome(contract, u)).collect();
            let multiplier: Option<Vec<f64>> = size_weighted.then(|| {
                data.units.iter().map(|&u| declaration.outcomes[u].len() as f64).collect()
            });
            let history = WeightedHistory {
                arms: data.arms,
                assigned: &data.assigned,
                probabilities: &data.probabilities,
                outcomes: &outcomes,
                multiplier: multiplier.as_deref(),
            };
            let options = WeightedOptions {
                scheme,
                prior: OutcomeModelPrior { mean: *prior_mean, strength: *prior_strength },
                probability_floor: declaration.probability_floor,
            };
            let fit = adaptively_weighted_estimate(&history, estimand, &options, ctx)?;
            Ok(AdaptiveResult {
                paradigm,
                method: "adaptively_weighted_augmented_inverse_probability",
                values: vec![
                    fit.estimate,
                    fit.uncalibrated_variance,
                    fit.units as f64,
                    fit.effective_units,
                    fit.max_weight_share,
                ],
                closed_guarantee: fit.guarantee.detail,
            })
        }
        AdaptiveResultSpec::ConfidenceSequence { alpha, tuning_samples, contrast, .. } => {
            let mut by_arm: Vec<Vec<f64>> = vec![Vec::new(); data.arms];
            for (&unit, &arm) in data.units.iter().zip(&data.assigned) {
                by_arm[arm].push(unit_outcome(contract, unit));
            }
            let contrast = contrast.map(|(p, m)| (p as usize, m as usize));
            let sequences =
                anytime_valid_arm_sequences(&by_arm, contrast, *alpha, *tuning_samples, ctx)?;
            let mut values: Vec<f64> = sequences
                .arms
                .iter()
                .flat_map(|a| [a.count as f64, a.mean, a.lower, a.upper])
                .collect();
            if let Some((lower, upper)) = sequences.contrast {
                values.extend([lower, upper]);
            }
            Ok(AdaptiveResult {
                paradigm,
                method: "mixture_hoeffding_confidence_sequence",
                values,
                closed_guarantee: sequences.guarantee.detail,
            })
        }
        AdaptiveResultSpec::GroupSequential { control_arm, treatment_arm, .. } => {
            let rule = declaration.stopping_rule.map(|r| r.spec);
            let Some(StoppingSpec::AlphaSpending { spending, total_alpha, sidedness }) = rule
            else {
                return Err(request_invalid("the contract's stopping rule is not alpha spending"));
            };
            if control_arm == treatment_arm
                || *control_arm as usize >= data.arms
                || *treatment_arm as usize >= data.arms
            {
                return Err(request_invalid("control and treatment must be two declared arms"));
            }
            let looks = declaration.interim_analyses.as_ref().expect("validated contract");
            let plan = GroupSequentialPlan {
                fractions: looks.iter().map(|l| l.information_fraction).collect(),
                alpha: total_alpha,
                spending: match spending {
                    SpendingKind::PocockType => SpendingFunction::PocockType,
                    _ => SpendingFunction::OBrienFlemingType,
                },
                sidedness: match sidedness {
                    BoundarySidedness::OneSidedUpper => Sidedness::OneSidedUpper,
                    _ => Sidedness::TwoSidedSymmetric,
                },
            };
            let mut statistics = Vec::with_capacity(look + 1);
            for performed in 0..=look {
                let at = look_data(contract, performed)?;
                let pick = |arm: u32| -> Vec<f64> {
                    at.units
                        .iter()
                        .zip(&at.assigned)
                        .filter(|(_, a)| **a == arm as usize)
                        .map(|(&u, _)| unit_outcome(contract, u))
                        .collect()
                };
                statistics
                    .push(standardized_difference(&pick(*control_arm), &pick(*treatment_arm))?);
            }
            let outcome = group_sequential_decision(&plan, &statistics, ctx)?;
            let mut values = vec![outcome.stopped_at.map_or(-1.0, |k| k as f64)];
            let spent = outcome.spend_increments.iter();
            for ((critical, statistic), spend) in
                outcome.critical.iter().zip(&outcome.statistics).zip(spent)
            {
                values.extend([*critical, *statistic, *spend]);
            }
            Ok(AdaptiveResult {
                paradigm,
                method: "lan_demets_alpha_spending_group_sequential",
                values,
                closed_guarantee: outcome.guarantee.detail,
            })
        }
    }
}

// -- wire ------------------------------------------------------------------------------------

/// The request on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecWire {
    /// Route kind.
    pub kind: String,
    /// Look index.
    pub look: u32,
    /// First arm (plus arm, control arm).
    pub first_arm: Option<u32>,
    /// Second arm (minus arm, treatment arm).
    pub second_arm: Option<u32>,
    /// Real parameters as `f64` bits, in the route's fixed order.
    pub parameter_bits: Vec<u64>,
    /// Weighting scheme name.
    pub scheme: Option<String>,
    /// Whether subjects are weighted by their measurement counts.
    pub size_weighted: bool,
}

/// Stored metadata of a result artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveResultMeta {
    /// Artifact major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The embedded monitoring contract.
    pub contract: MonitoringContractMeta,
    /// The request.
    pub spec: SpecWire,
    /// Method name.
    pub method: String,
    /// Stored values as `f64` bits.
    pub value_bits: Vec<u64>,
    /// Claim class: always `point_only`.
    pub claim: String,
    /// Calibration status: always `unmeasured`.
    pub calibration: String,
    /// Detail of the closed guarantee.
    pub closed_guarantee: String,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

fn spec_to_wire(spec: &AdaptiveResultSpec) -> Result<SpecWire, Error> {
    let mut wire = SpecWire {
        kind: String::new(),
        look: spec.look(),
        first_arm: None,
        second_arm: None,
        parameter_bits: Vec::new(),
        scheme: None,
        size_weighted: false,
    };
    match spec {
        AdaptiveResultSpec::BayesianPosterior { prior_alpha, prior_beta, .. } => {
            "bayesian_posterior".clone_into(&mut wire.kind);
            wire.parameter_bits = vec![prior_alpha.to_bits(), prior_beta.to_bits()];
        }
        AdaptiveResultSpec::PosteriorStopping { prior_alpha, prior_beta, .. } => {
            "posterior_stopping".clone_into(&mut wire.kind);
            wire.parameter_bits = vec![prior_alpha.to_bits(), prior_beta.to_bits()];
        }
        AdaptiveResultSpec::AdaptiveWeighting {
            plus_arm,
            minus_arm,
            scheme,
            prior_mean,
            prior_strength,
            size_weighted,
            ..
        } => {
            "adaptive_weighting".clone_into(&mut wire.kind);
            wire.first_arm = Some(*plus_arm);
            wire.second_arm = *minus_arm;
            wire.parameter_bits = vec![prior_mean.to_bits(), prior_strength.to_bits()];
            wire.scheme = Some(scheme.clone());
            wire.size_weighted = *size_weighted;
        }
        AdaptiveResultSpec::ConfidenceSequence { alpha, tuning_samples, contrast, .. } => {
            "confidence_sequence".clone_into(&mut wire.kind);
            wire.parameter_bits = vec![alpha.to_bits(), tuning_samples.to_bits()];
            if let Some((plus, minus)) = contrast {
                wire.first_arm = Some(*plus);
                wire.second_arm = Some(*minus);
            }
        }
        AdaptiveResultSpec::GroupSequential { control_arm, treatment_arm, .. } => {
            "group_sequential".clone_into(&mut wire.kind);
            wire.first_arm = Some(*control_arm);
            wire.second_arm = Some(*treatment_arm);
        }
    }
    Ok(wire)
}

fn spec_from_wire(wire: &SpecWire) -> Result<AdaptiveResultSpec, Error> {
    let real = |index: usize| {
        wire.parameter_bits
            .get(index)
            .map(|b| f64::from_bits(*b))
            .ok_or_else(|| malformed("a request parameter is missing"))
    };
    let arm = |value: Option<u32>| value.ok_or_else(|| malformed("a request arm is missing"));
    let look = wire.look;
    Ok(match wire.kind.as_str() {
        "bayesian_posterior" => AdaptiveResultSpec::BayesianPosterior {
            look,
            prior_alpha: real(0)?,
            prior_beta: real(1)?,
        },
        "posterior_stopping" => AdaptiveResultSpec::PosteriorStopping {
            look,
            prior_alpha: real(0)?,
            prior_beta: real(1)?,
        },
        "adaptive_weighting" => AdaptiveResultSpec::AdaptiveWeighting {
            look,
            plus_arm: arm(wire.first_arm)?,
            minus_arm: wire.second_arm,
            scheme: wire
                .scheme
                .clone()
                .ok_or_else(|| malformed("a weighting request names its scheme"))?,
            prior_mean: real(0)?,
            prior_strength: real(1)?,
            size_weighted: wire.size_weighted,
        },
        "confidence_sequence" => AdaptiveResultSpec::ConfidenceSequence {
            look,
            alpha: real(0)?,
            tuning_samples: real(1)?,
            contrast: match (wire.first_arm, wire.second_arm) {
                (Some(p), Some(m)) => Some((p, m)),
                _ => None,
            },
        },
        "group_sequential" => AdaptiveResultSpec::GroupSequential {
            look,
            control_arm: arm(wire.first_arm)?,
            treatment_arm: arm(wire.second_arm)?,
        },
        _ => return Err(Error::UnsupportedSemantics("request kind")),
    })
}

fn compute_identity(meta: &AdaptiveResultMeta) -> String {
    let mut writer = CanonicalWriter::new("antecedent.adaptive_inference.result");
    writer.word(u64::from(meta.version));
    writer.text(&meta.feature);
    writer.text(&meta.contract.identity);
    let spec = &meta.spec;
    writer.text(&spec.kind).word(u64::from(spec.look));
    writer.opt_word(spec.first_arm.map(u64::from));
    writer.opt_word(spec.second_arm.map(u64::from));
    writer.word(spec.parameter_bits.len() as u64);
    for bit in &spec.parameter_bits {
        writer.word(*bit);
    }
    writer.opt_text(spec.scheme.as_deref()).word(u64::from(spec.size_weighted));
    writer.text(&meta.method);
    writer.word(meta.value_bits.len() as u64);
    for bit in &meta.value_bits {
        writer.word(*bit);
    }
    writer.text(&meta.claim).text(&meta.calibration).text(&meta.closed_guarantee);
    writer.finish_hex()
}

fn close(stored: f64, recomputed: f64) -> bool {
    if stored.is_nan() || recomputed.is_nan() {
        return stored.is_nan() && recomputed.is_nan();
    }
    (stored - recomputed).abs() <= ADAPTIVE_RESULT_REPLAY_TOLERANCE * recomputed.abs().max(1.0)
}

fn result_refusal(error: AdaptiveResultError) -> Error {
    Error::Refused { code: error.code, message: format!("{}: {}", error.detail, error.message) }
}

fn check(
    meta: &AdaptiveResultMeta,
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<AdaptiveInferenceResultArtifact, Error> {
    if meta.version != ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION {
        return Err(Error::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE {
        return Err(Error::UnsupportedSemantics("feature marker"));
    }
    if meta.claim != "point_only" || meta.calibration != ADAPTIVE_INFERENCE_CALIBRATION {
        return Err(Error::UnsupportedSemantics("claim or calibration status"));
    }
    let contract_artifact = AdaptiveMonitoringContractArtifact::from_meta(&meta.contract, None)?;
    let spec = spec_from_wire(&meta.spec)?;
    if contract_artifact.paradigm() != spec.paradigm() {
        return Err(Error::ResultMismatch("paradigm"));
    }
    let recomputed = compute_adaptive_result(contract_artifact.contract(), &spec, ctx)
        .map_err(result_refusal)?;
    if recomputed.method != meta.method || recomputed.closed_guarantee != meta.closed_guarantee {
        return Err(Error::ResultMismatch("method or closed guarantee"));
    }
    let stored: Vec<f64> = meta.value_bits.iter().map(|b| f64::from_bits(*b)).collect();
    if stored.len() != recomputed.values.len()
        || stored.iter().zip(&recomputed.values).any(|(s, r)| !close(*s, *r))
    {
        return Err(Error::ResultMismatch("values"));
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Error::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Error::IdentityMismatch { field: "retained_identity" });
    }
    Ok(AdaptiveInferenceResultArtifact {
        contract: contract_artifact,
        spec,
        result: recomputed,
        identity,
    })
}

fn encode_meta(meta: &AdaptiveResultMeta, artifact_id: &str) -> Result<Vec<u8>, Error> {
    if artifact_id.trim().is_empty() {
        return Err(Error::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Error::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "adaptive_inference_result",
            max_bytes: MAX_ADAPTIVE_INFERENCE_RESULT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, AdaptiveInferenceResultArtifactError))
}

fn decode_meta(bytes: &[u8]) -> Result<AdaptiveResultMeta, Error> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_ADAPTIVE_INFERENCE_RESULT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, AdaptiveInferenceResultArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Error::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version| {
        if version == ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Error::UnsupportedVersion { version })
        }
    })
}

// -- public type -----------------------------------------------------------------------------

/// A validated result: the replayed monitoring contract, the request and the
/// recomputed values.
#[derive(Clone, Debug)]
pub struct AdaptiveInferenceResultArtifact {
    contract: AdaptiveMonitoringContractArtifact,
    spec: AdaptiveResultSpec,
    result: AdaptiveResult,
    identity: String,
}

impl AdaptiveInferenceResultArtifact {
    /// Compute a result and seal it with its contract.
    ///
    /// # Errors
    /// A refused authorization or request, a floor failure, or a contract that
    /// does not seal.
    pub fn seal(
        contract: &AdaptiveDesignContract,
        spec: &AdaptiveResultSpec,
        ctx: &ExecutionContext,
    ) -> Result<(Self, AdaptiveResultMeta), Error> {
        let result = compute_adaptive_result(contract, spec, ctx).map_err(result_refusal)?;
        let (_, contract_meta) =
            AdaptiveMonitoringContractArtifact::seal(contract, spec.paradigm())?;
        let mut meta = AdaptiveResultMeta {
            version: ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION,
            feature: ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE.to_owned(),
            contract: contract_meta,
            spec: spec_to_wire(spec)?,
            method: result.method.to_owned(),
            value_bits: result.values.iter().map(|v| v.to_bits()).collect(),
            claim: "point_only".to_owned(),
            calibration: ADAPTIVE_INFERENCE_CALIBRATION.to_owned(),
            closed_guarantee: result.closed_guarantee.to_owned(),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        let artifact = check(&meta, None, ctx)?;
        Ok((artifact, meta))
    }

    /// The replayed monitoring contract: the procedure, the stopping rule and
    /// the paradigm a fresh consumer reads.
    #[must_use]
    pub const fn contract(&self) -> &AdaptiveMonitoringContractArtifact {
        &self.contract
    }

    /// The request.
    #[must_use]
    pub const fn spec(&self) -> &AdaptiveResultSpec {
        &self.spec
    }

    /// The recomputed result.
    #[must_use]
    pub const fn result(&self) -> &AdaptiveResult {
        &self.result
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize a sealed result through the checksummed container.
    ///
    /// # Errors
    /// A result that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        contract: &AdaptiveDesignContract,
        spec: &AdaptiveResultSpec,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        let (_, meta) = Self::seal(contract, spec, ctx)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by replay and recomputation.
    ///
    /// # Errors
    /// Corruption, another major version, a contract refusal, a result that
    /// does not replay, or an identity that does not match the retained one.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        check(&decode_meta(bytes)?, expected_identity, ctx)
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &AdaptiveResultMeta,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        check(meta, expected_identity, ctx)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut AdaptiveResultMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Decode stored metadata without validating it, for tests that edit it.
    ///
    /// # Errors
    /// Corruption, an unsupported layout or version.
    #[doc(hidden)]
    pub fn decode_unchecked(bytes: &[u8]) -> Result<AdaptiveResultMeta, Error> {
        decode_meta(bytes)
    }
}
