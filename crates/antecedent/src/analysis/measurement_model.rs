//! Corrected functionals under a declared measurement model: prepare,
//! estimate, export and independently consume.
//!
//! A recorded column is not the scientific construct it is recorded for. This
//! module routes one request through the layers that own each rule and adds no
//! scientific rule of its own:
//!
//! 1. **Prepare** ([`StudyBuilder::measurement_model`]) checks, before any
//!    number is formed, that the model is well formed, that its construct is
//!    the one requested (a proxy is never the construct), that the population,
//!    regime, arm and representation of the analysis are the ones the model
//!    was established for, that the declared error structure is the one the
//!    analysis assumes, and that the route licenses the downstream operation.
//!    These are `antecedent_contract::measurement` checks; the data's own
//!    binding is derived from the same fields, so the two cannot disagree.
//! 2. **Estimate** ([`PreparedMeasurementModel::estimate`]) runs the corrected
//!    functional of `antecedent_estimate::measurement` and keeps the result
//!    bound to the model identity. The standard error is an uncalibrated
//!    delta-method estimate that includes the uncertainty of the estimated
//!    measurement parameters where the record says it was propagated. No
//!    interval is produced: [`MeasurementEstimate::interval`] refuses with
//!    `cell_not_licensed` until the whole measurement-plus-estimation method
//!    has a measured coverage record.
//! 3. **Export** ([`MeasurementEstimate::export`]) writes the model, its
//!    identities and the result with its sufficient inputs.
//! 4. **Consume** ([`consume_measurement_model_artifact`]) recomputes the
//!    point and standard error from those inputs with `antecedent-stats` only,
//!    against a model identity the consumer retained independently of the
//!    bytes, and refuses a stored number that does not recompute.
//!
//! # Lifecycle
//!
//! * [`PreparedMeasurementModel::refresh`] with new outcome-only data of the
//!   same question **keeps** the model, its identities, the binding, the
//!   route, the downstream operation and the provider trust label, and
//!   **recomputes** the point, the standard error, the variance components,
//!   the data identity and the replay inputs. It **refuses** data of another
//!   family or question (another route, functional or arm pair) and never
//!   changes validation parameters: those are not data.
//! * [`PreparedMeasurementModel::with_validation_parameters`] is the only way
//!   the numbers a model was established with change. It **rebuilds** the
//!   measurement and evidence identities (so the model identity), re-runs
//!   every prepare check on the new model and recomputes; a result bound to
//!   the old identity is not carried over. It **refuses** a parameter set of
//!   another route, a structure that disagrees with the parameter layout, and
//!   a different provider object (which needs its own verification through a
//!   new preparation).
//! * A refresh that refuses returns the error and leaves the prepared state it
//!   was called on untouched: both operations take `&self` and return a new
//!   value.
//!
//! # Representation mappings
//!
//! A representation change needs an explicit mapping from the model's
//! representation to the requested one. A mapping that is a relabeling carries
//! values only; the correction then runs on the values as recorded and the
//! estimate reports the mapping identity it relied on. A mapping never carries
//! intervention meaning unless it declares the evidence for it, and the
//! coordinates of a learned vector representation are not intervenable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::measurement::{
    LatentPosteriorTrust, MeasurementOperation, MeasurementRequest, check_latent_posterior,
    check_measurement_request, route_card,
};
use antecedent_contract::{
    ExternalRefusal, ExternalScientificObject, ExternalTrustState, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_core::identity::SemanticDigest;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, ErrorStructure, MeasurementIdentities, MeasurementInference,
    MeasurementModel, MeasurementParameters, MeasurementRoute, ParameterEvidence,
    ParameterUncertainty, RepresentationMapping, RepresentationVersion,
};
use antecedent_estimate::measurement::{
    CategoricalFunctional, CorrectedBinaryPrevalence, CorrectedBinaryRiskDifference,
    CorrectedCategoricalOutcome, CorrectedLinearEffect, CorrectedPosteriorLinearEffect,
    DataBinding, LinearMeasuredData, MeasurementError, PosteriorCombination, PosteriorLinearData,
    RiskDifferenceInput, correct_binary_prevalence, correct_binary_risk_difference,
    correct_categorical, correct_classical_linear_effect, correct_latent_posterior_linear_effect,
    correct_repeated_linear_effect, interval_not_licensed,
};
use antecedent_io::IoError;
use antecedent_io::measurement_model_wire::{
    CategoricalReplayFunctional, MeasurementModelArtifact, MeasurementReplay, MeasurementResult,
    ReplayedValues, replay_values,
};
use antecedent_stats::measurement_error::{ObservedClassSummary, TwoByTwoCounts};

use super::StudyBuilder;

/// One refusal as plain data: registered reason code, namespaced
/// `measurement_model.*` detail and the explanation of what was refused and
/// what would license it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementModelRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `measurement_model.<slot>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
    /// The declaration or evidence that would license the request, when known.
    pub remedy: Option<String>,
}

impl std::fmt::Display for MeasurementModelRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for MeasurementModelRefusal {}

impl From<MeasurementModelRefusal> for IoError {
    fn from(refusal: MeasurementModelRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl From<MeasurementModelRefusal> for crate::CausalError {
    fn from(refusal: MeasurementModelRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

impl MeasurementModelRefusal {
    fn from_estimate(error: &MeasurementError) -> Self {
        Self {
            code: error.code,
            detail: error.detail.to_owned(),
            message: error.message.clone(),
            remedy: None,
        }
    }

    fn from_contract(refusal: &ExternalRefusal) -> Self {
        let mut message = format!("refused at stage `{}`", refusal.stage);
        if let Some(offending) = &refusal.offending {
            message.push_str(&format!("; offending `{offending}`"));
        }
        if let Some(expected) = &refusal.expected {
            message.push_str(&format!("; expected `{expected}`"));
        }
        if let Some(supplied) = &refusal.supplied {
            message.push_str(&format!("; supplied `{supplied}`"));
        }
        if let Some(remedy) = refusal.remedy {
            message.push_str(&format!("; {remedy}"));
        }
        Self {
            code: refusal.code,
            detail: refusal.detail.clone(),
            message,
            remedy: refusal.remedy.map(str::to_owned),
        }
    }

    /// The refusal of an artifact error. A semantic refusal of the artifact
    /// layer (identity, binding or recomputation) is `artifact_refused` under
    /// `measurement_binding_mismatch`; bytes that are corrupt, truncated or of
    /// another layout are `artifact_invalid`.
    fn from_io(error: &IoError) -> Self {
        match error {
            IoError::Refused { message, .. } => Self {
                code: antecedent_core::reason_code!("measurement_binding_mismatch"),
                detail: "measurement_model.artifact_refused".to_owned(),
                message: message.clone(),
                remedy: None,
            },
            other => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "measurement_model.artifact_invalid".to_owned(),
                message: other.to_string(),
                remedy: None,
            },
        }
    }

    fn cancelled() -> Self {
        Self {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            detail: "measurement_model.cancelled".to_owned(),
            message: "cancelled before a corrected functional was formed; this is not a verdict on the data"
                .to_owned(),
            remedy: None,
        }
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), MeasurementModelRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(MeasurementModelRefusal::cancelled());
    }
    Ok(())
}

/// What the analysis says about the data and the question it asks of a
/// measurement model.
///
/// The construct, population, regime, arm and representation are stated once;
/// the contract request and the data binding of the estimator are both derived
/// from them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementAnalysisBinding {
    /// Variable the analysis treats as the construct (never the indicator).
    pub construct_id: String,
    /// Population the data describe and the analysis is about.
    pub population_id: String,
    /// Regime the data were collected under.
    pub regime_id: String,
    /// Arm of the data; `None` when the data span arms.
    pub arm_id: Option<String>,
    /// Representation the analysis expects the indicator in.
    pub representation: RepresentationVersion,
    /// Explicit mapping from the model's representation to `representation`,
    /// supplied when they differ.
    pub representation_mapping: Option<RepresentationMapping>,
    /// Error structure the analysis assumes; non-differential by default.
    pub assumed_structure: ErrorStructure,
}

impl MeasurementAnalysisBinding {
    /// A binding for the construct named by `quantity`, in the quantity's own
    /// population and regime, assuming non-differential error.
    #[must_use]
    pub fn for_construct(
        quantity: &ScientificQuantity,
        arm_id: Option<String>,
        representation: RepresentationVersion,
    ) -> Self {
        Self {
            construct_id: quantity.variable_id.clone(),
            population_id: quantity.population_id.clone(),
            regime_id: quantity.regime_id.clone(),
            arm_id,
            representation,
            representation_mapping: None,
            assumed_structure: ErrorStructure::NonDifferential,
        }
    }

    /// This binding assuming `structure` instead of non-differential error.
    #[must_use]
    pub fn assuming(mut self, structure: ErrorStructure) -> Self {
        self.assumed_structure = structure;
        self
    }

    /// This binding with an explicit representation mapping.
    #[must_use]
    pub fn with_mapping(mut self, mapping: RepresentationMapping) -> Self {
        self.representation_mapping = Some(mapping);
        self
    }
}

/// A scalar functional of corrected categories, named by level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CategoricalTarget {
    /// The proportion of one true category.
    Proportion(String),
    /// The mean outcome among units in one true category.
    OutcomeMean(String),
    /// The difference of outcome means `first - second`.
    OutcomeContrast {
        /// Minuend category.
        first: String,
        /// Subtrahend category.
        second: String,
    },
}

impl CategoricalTarget {
    fn functional(&self) -> CategoricalFunctional<'_> {
        match self {
            Self::Proportion(level) => CategoricalFunctional::Proportion(level),
            Self::OutcomeMean(level) => CategoricalFunctional::OutcomeMean(level),
            Self::OutcomeContrast { first, second } => {
                CategoricalFunctional::OutcomeContrast { first, second }
            }
        }
    }

    fn replay(&self) -> CategoricalReplayFunctional {
        match self {
            Self::Proportion(level) => CategoricalReplayFunctional::Proportion(level.clone()),
            Self::OutcomeMean(level) => CategoricalReplayFunctional::OutcomeMean(level.clone()),
            Self::OutcomeContrast { first, second } => CategoricalReplayFunctional::OutcomeContrast {
                first: first.clone(),
                second: second.clone(),
            },
        }
    }
}

/// The outcome-side data of one corrected functional. Each variant is one
/// bounded route; the model's validation parameters are not data and are not
/// carried here.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeasurementInput {
    /// Classical continuous error: outcome, error-prone regressor and
    /// error-free covariates of a linear outcome model.
    ClassicalLinear {
        /// Outcome.
        outcome: Vec<f64>,
        /// Error-prone regressor; the row mean of replicates when replicated.
        regressor: Vec<f64>,
        /// Error-free covariates, column-major `n x covariate_count`.
        covariates: Vec<f64>,
        /// Number of covariate columns.
        covariate_count: usize,
        /// Replicates averaged into each regressor value; `None` means one each.
        replicate_counts: Option<Vec<u32>>,
    },
    /// Repeated measurements: the error variance comes from the sample's own
    /// replicates.
    RepeatedLinear {
        /// Outcome.
        outcome: Vec<f64>,
        /// Every unit's replicates back to back.
        replicate_values: Vec<f64>,
        /// Replicate count of each unit.
        replicate_counts: Vec<u32>,
        /// Error-free covariates, column-major.
        covariates: Vec<f64>,
        /// Number of covariate columns.
        covariate_count: usize,
    },
    /// Binary misclassification: corrected prevalence from observed positives.
    BinaryPrevalence {
        /// Observed positives.
        events: u64,
        /// Units observed.
        total: u64,
    },
    /// Binary misclassification: corrected risk difference across two arms.
    BinaryRiskDifference {
        /// Arm identity of the exposed group.
        exposed_arm_id: String,
        /// Arm identity of the unexposed group.
        unexposed_arm_id: String,
        /// Observed events and totals by arm.
        table: TwoByTwoCounts,
    },
    /// Probabilistic categorical assignment: observed-label summaries in level
    /// order and the functional of the corrected categories to report.
    Categorical {
        /// Observed-label summaries, one per level of the model.
        classes: Vec<ObservedClassSummary>,
        /// Which functional of the corrected categories.
        target: CategoricalTarget,
    },
    /// An external latent posterior consumed as joint draws.
    LatentPosterior {
        /// Outcome.
        outcome: Vec<f64>,
        /// Joint posterior draws, draw-major.
        draws: Vec<f64>,
        /// Number of joint draws.
        draw_count: usize,
        /// Error-free covariates, column-major.
        covariates: Vec<f64>,
        /// Number of covariate columns.
        covariate_count: usize,
    },
}

impl MeasurementInput {
    /// The downstream operation this input's corrected functional enters.
    #[must_use]
    pub fn operation(&self) -> MeasurementOperation {
        match self {
            Self::ClassicalLinear { .. }
            | Self::RepeatedLinear { .. }
            | Self::LatentPosterior { .. } => MeasurementOperation::LinearAdjustedEffect,
            Self::BinaryPrevalence { .. } => MeasurementOperation::LatentPrevalence,
            Self::BinaryRiskDifference { .. } => MeasurementOperation::RiskDifference,
            Self::Categorical { target, .. } => match target {
                CategoricalTarget::Proportion(_) => MeasurementOperation::LatentPrevalence,
                CategoricalTarget::OutcomeMean(_) | CategoricalTarget::OutcomeContrast { .. } => {
                    MeasurementOperation::CategoryOutcomeMeans
                }
            },
        }
    }

    /// Whether `other` asks the same question of new data: the same route
    /// family, and for a contrast or a category functional the same arms or
    /// levels.
    fn same_question(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ClassicalLinear { .. }, Self::ClassicalLinear { .. })
            | (Self::RepeatedLinear { .. }, Self::RepeatedLinear { .. })
            | (Self::BinaryPrevalence { .. }, Self::BinaryPrevalence { .. })
            | (Self::LatentPosterior { .. }, Self::LatentPosterior { .. }) => true,
            (
                Self::BinaryRiskDifference { exposed_arm_id: a, unexposed_arm_id: b, .. },
                Self::BinaryRiskDifference { exposed_arm_id: c, unexposed_arm_id: d, .. },
            ) => a == c && b == d,
            (
                Self::Categorical { target: first, .. },
                Self::Categorical { target: second, .. },
            ) => first == second,
            _ => false,
        }
    }
}

/// The estimator's own result for one route, kept so a caller can read the
/// parts the record summarizes (moments, covariance, components).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MeasurementDetail {
    /// Classical or repeated-measurement linear effect.
    Linear(CorrectedLinearEffect),
    /// Linear effect over a latent posterior.
    PosteriorLinear(CorrectedPosteriorLinearEffect),
    /// Corrected prevalence.
    BinaryPrevalence(CorrectedBinaryPrevalence),
    /// Corrected risk difference.
    BinaryRiskDifference(CorrectedBinaryRiskDifference),
    /// Corrected categories and the functional reported.
    Categorical(CorrectedCategoricalOutcome, CategoricalTarget),
}

/// A measurement model, the analysis binding and the data, checked.
///
/// Built by [`StudyBuilder::measurement_model`]. Holds nothing that has not
/// passed the prepare checks; every method that changes it returns a new value
/// and leaves this one untouched.
#[derive(Clone, Debug)]
pub struct PreparedMeasurementModel {
    model: MeasurementModel,
    binding: MeasurementAnalysisBinding,
    input: MeasurementInput,
    trust: Option<LatentPosteriorTrust>,
}

/// A corrected functional bound to its model, with the inputs to replay it.
#[derive(Clone, Debug)]
pub struct MeasurementEstimate {
    model: MeasurementModel,
    binding: MeasurementAnalysisBinding,
    record: CorrectedFunctionalRecord,
    replay: MeasurementReplay,
    provider_trust: Option<LatentPosteriorTrust>,
    detail: MeasurementDetail,
    operation: MeasurementOperation,
}

/// The new prepared state and result of a refresh.
#[derive(Clone, Debug)]
pub struct MeasurementRefresh {
    /// The refreshed prepared state.
    pub prepared: PreparedMeasurementModel,
    /// The recomputed result, bound to the (possibly rebuilt) model identity.
    pub estimate: MeasurementEstimate,
}

fn request_of(
    binding: &MeasurementAnalysisBinding,
    operation: MeasurementOperation,
) -> MeasurementRequest<'_> {
    MeasurementRequest {
        construct_id: &binding.construct_id,
        population_id: &binding.population_id,
        regime_id: &binding.regime_id,
        arm_id: binding.arm_id.as_deref(),
        representation: &binding.representation,
        assumed_structure: binding.assumed_structure,
        operation,
    }
}

fn check_request(
    model: &MeasurementModel,
    binding: &MeasurementAnalysisBinding,
    operation: MeasurementOperation,
) -> Result<(), MeasurementModelRefusal> {
    check_measurement_request(
        model,
        &request_of(binding, operation),
        binding.representation_mapping.as_ref(),
    )
    .map_err(|refusal| MeasurementModelRefusal::from_contract(&refusal))
}

fn check_provider(
    model: &MeasurementModel,
    provider: Option<(&ExternalScientificObject, &ExternalTrustState)>,
) -> Result<Option<LatentPosteriorTrust>, MeasurementModelRefusal> {
    match (model.route, provider) {
        (MeasurementRoute::ExternalLatentPosterior, None) => Err(MeasurementModelRefusal {
            code: antecedent_core::reason_code!("measurement_posterior_not_trusted"),
            detail: "measurement_model.posterior_identity".to_owned(),
            message: "the model names an external provider posterior and none was supplied; \
                      supply the exact provider object with an attestation or verification receipt"
                .to_owned(),
            remedy: Some("supply the exact provider object the model names".to_owned()),
        }),
        (_, Some((object, trust))) => check_latent_posterior(model, object, trust)
            .map(Some)
            .map_err(|refusal| MeasurementModelRefusal::from_contract(&refusal)),
        (_, None) => Ok(None),
    }
}

impl StudyBuilder {
    /// Prepare a corrected functional under a declared measurement model.
    ///
    /// Every prepare check runs here, before any number is formed: the model
    /// is well formed; the requested construct is the model's construct and
    /// the measured column is never offered in its place; the population,
    /// regime, arm and representation match (a changed representation needs
    /// the explicit mapping on `binding`); the assumed error structure is the
    /// declared one; the route licenses the operation `input` enters; and, for
    /// the posterior route, the provider object is the exact one the model
    /// names and is attested or verified, never natively trusted.
    ///
    /// # Errors
    /// The first failed check as a [`MeasurementModelRefusal`], or
    /// cancellation.
    pub fn measurement_model(
        model: MeasurementModel,
        binding: MeasurementAnalysisBinding,
        input: MeasurementInput,
        provider: Option<(&ExternalScientificObject, &ExternalTrustState)>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMeasurementModel, MeasurementModelRefusal> {
        check_cancelled(ctx)?;
        check_request(&model, &binding, input.operation())?;
        let trust = check_provider(&model, provider)?;
        Ok(PreparedMeasurementModel { model, binding, input, trust })
    }
}

impl PreparedMeasurementModel {
    /// The checked model.
    #[must_use]
    pub const fn model(&self) -> &MeasurementModel {
        &self.model
    }

    /// The analysis binding the model was checked against.
    #[must_use]
    pub const fn binding(&self) -> &MeasurementAnalysisBinding {
        &self.binding
    }

    /// The retained outcome-side data.
    #[must_use]
    pub const fn input(&self) -> &MeasurementInput {
        &self.input
    }

    /// The separately frozen identities of the model.
    #[must_use]
    pub fn identities(&self) -> MeasurementIdentities {
        self.model.identities()
    }

    /// The model identity as lowercase hex, the value a consumer retains.
    #[must_use]
    pub fn model_identity_hex(&self) -> String {
        self.model.identities().model.to_hex()
    }

    /// The operation the retained data's functional enters.
    #[must_use]
    pub fn operation(&self) -> MeasurementOperation {
        self.input.operation()
    }

    /// How far the provider object was trusted, for the posterior route.
    #[must_use]
    pub const fn provider_trust(&self) -> Option<LatentPosteriorTrust> {
        self.trust
    }

    /// The assumptions, allowed operations and refused operations of the
    /// model's route.
    #[must_use]
    pub fn route_card(&self) -> antecedent_contract::measurement::MeasurementRouteCard {
        route_card(self.model.route)
    }

    /// Whether the route licenses `operation` for this binding.
    ///
    /// # Errors
    /// The coded refusal naming what would license the operation. Conditioning
    /// on the construct inside an identification proof, intervening on the
    /// representation, transporting to another population, a nonlinear outcome
    /// model and an interval are refused for every route.
    pub fn check_operation(
        &self,
        operation: MeasurementOperation,
    ) -> Result<(), MeasurementModelRefusal> {
        check_request(&self.model, &self.binding, operation)
    }

    /// An interval for the corrected functional.
    ///
    /// # Errors
    /// Always `cell_not_licensed`: no coverage record measures the whole
    /// measurement-plus-estimation method.
    pub fn interval(&self) -> Result<std::convert::Infallible, MeasurementModelRefusal> {
        Err(MeasurementModelRefusal::from_estimate(&interval_not_licensed()))
    }

    /// Form the corrected functional from the retained data.
    ///
    /// # Errors
    /// The estimator's coded refusals (reliability, Youden index, confusion
    /// conditioning, replicates or draws below the model's minimum, rank
    /// deficiency) or cancellation. Nothing is returned partially.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MeasurementEstimate, MeasurementModelRefusal> {
        check_cancelled(ctx)?;
        // The data are in the model's coordinates; a declared relabeling
        // mapping carries values across unchanged, so the estimator's own
        // binding check sees the model's representation.
        let data_binding = DataBinding {
            population_id: self.binding.population_id.clone(),
            regime_id: self.binding.regime_id.clone(),
            arm_id: self.binding.arm_id.clone(),
            representation: self.model.representation.clone(),
        };
        let from_estimate = |error: MeasurementError| MeasurementModelRefusal::from_estimate(&error);
        let (record, replay, detail) = match &self.input {
            MeasurementInput::ClassicalLinear {
                outcome,
                regressor,
                covariates,
                covariate_count,
                replicate_counts,
            } => {
                let data = LinearMeasuredData {
                    outcome,
                    regressor,
                    covariates,
                    covariate_count: *covariate_count,
                    replicate_counts: replicate_counts.as_deref(),
                };
                let effect =
                    correct_classical_linear_effect(&self.model, &data_binding, &data, ctx)
                        .map_err(from_estimate)?;
                let replay = MeasurementReplay::Linear {
                    moments: effect.moments.clone(),
                    coefficient_index: 1,
                };
                (effect.record.clone(), replay, MeasurementDetail::Linear(effect))
            }
            MeasurementInput::RepeatedLinear {
                outcome,
                replicate_values,
                replicate_counts,
                covariates,
                covariate_count,
            } => {
                let effect = correct_repeated_linear_effect(
                    &self.model,
                    &data_binding,
                    outcome,
                    replicate_values,
                    replicate_counts,
                    covariates,
                    *covariate_count,
                    ctx,
                )
                .map_err(from_estimate)?;
                let replay = MeasurementReplay::Linear {
                    moments: effect.moments.clone(),
                    coefficient_index: 1,
                };
                (effect.record.clone(), replay, MeasurementDetail::Linear(effect))
            }
            MeasurementInput::BinaryPrevalence { events, total } => {
                let corrected =
                    correct_binary_prevalence(&self.model, &data_binding, *events, *total, ctx)
                        .map_err(from_estimate)?;
                let replay = MeasurementReplay::BinaryPrevalence {
                    events: *events,
                    total: *total,
                    arm_id: self.binding.arm_id.clone(),
                };
                (corrected.record.clone(), replay, MeasurementDetail::BinaryPrevalence(corrected))
            }
            MeasurementInput::BinaryRiskDifference { exposed_arm_id, unexposed_arm_id, table } => {
                let corrected = correct_binary_risk_difference(
                    &self.model,
                    &data_binding,
                    &RiskDifferenceInput {
                        exposed_arm_id,
                        unexposed_arm_id,
                        table: *table,
                    },
                    ctx,
                )
                .map_err(from_estimate)?;
                let replay = MeasurementReplay::BinaryRiskDifference {
                    exposed_arm_id: exposed_arm_id.clone(),
                    unexposed_arm_id: unexposed_arm_id.clone(),
                    table: *table,
                };
                (
                    corrected.record.clone(),
                    replay,
                    MeasurementDetail::BinaryRiskDifference(corrected),
                )
            }
            MeasurementInput::Categorical { classes, target } => {
                let corrected = correct_categorical(&self.model, &data_binding, classes, ctx)
                    .map_err(from_estimate)?;
                let record = corrected.record(&target.functional()).map_err(from_estimate)?;
                let replay = MeasurementReplay::Categorical {
                    classes: classes.clone(),
                    functional: target.replay(),
                };
                (record, replay, MeasurementDetail::Categorical(corrected, target.clone()))
            }
            MeasurementInput::LatentPosterior {
                outcome,
                draws,
                draw_count,
                covariates,
                covariate_count,
            } => {
                let data = PosteriorLinearData {
                    outcome,
                    draws,
                    draw_count: *draw_count,
                    covariates,
                    covariate_count: *covariate_count,
                };
                let effect =
                    correct_latent_posterior_linear_effect(&self.model, &data_binding, &data, ctx)
                        .map_err(from_estimate)?;
                let replay = match (effect.combination, &effect.imputations, &effect.moments) {
                    (PosteriorCombination::MultipleImputation, Some(combined), _) => {
                        MeasurementReplay::MultipleImputation {
                            coefficients: combined.draw_coefficients.clone(),
                            covariances: combined.draw_covariances.clone(),
                            coefficient_index: 1,
                        }
                    }
                    (_, _, Some(moments)) => MeasurementReplay::Linear {
                        moments: moments.clone(),
                        coefficient_index: 1,
                    },
                    _ => {
                        return Err(MeasurementModelRefusal {
                            code: antecedent_core::reason_code!("invalid_argument"),
                            detail: "measurement_model.input_invalid".to_owned(),
                            message: "the posterior route returned no replayable inputs".to_owned(),
                            remedy: None,
                        });
                    }
                };
                (effect.record.clone(), replay, MeasurementDetail::PosteriorLinear(effect))
            }
        };
        Ok(MeasurementEstimate {
            model: self.model.clone(),
            binding: self.binding.clone(),
            record,
            replay,
            provider_trust: self.trust,
            detail,
            operation: self.input.operation(),
        })
    }

    /// Recompute with new outcome-only data of the same question under the
    /// same validation parameters.
    ///
    /// Keeps the model and its identities; recomputes the point, standard
    /// error and data identity. The prepared state this is called on is not
    /// changed, whether the refresh succeeds or refuses.
    ///
    /// # Errors
    /// Data of another route family, functional or arm pair
    /// (`route_not_supported`), any estimator refusal on the new data, or
    /// cancellation.
    pub fn refresh(
        &self,
        input: MeasurementInput,
        ctx: &ExecutionContext,
    ) -> Result<MeasurementRefresh, MeasurementModelRefusal> {
        check_cancelled(ctx)?;
        if !self.input.same_question(&input) {
            return Err(MeasurementModelRefusal {
                code: antecedent_core::reason_code!("route_not_supported"),
                detail: "measurement_model.route_mismatch".to_owned(),
                message: "refresh keeps the question: new data of another route family, \
                          functional or arm pair is a different request and needs a new preparation"
                    .to_owned(),
                remedy: None,
            });
        }
        let prepared = Self {
            model: self.model.clone(),
            binding: self.binding.clone(),
            input,
            trust: self.trust,
        };
        let estimate = prepared.estimate(ctx)?;
        Ok(MeasurementRefresh { prepared, estimate })
    }

    /// Rebuild the model with new validation parameters and evidence and
    /// recompute on the retained data.
    ///
    /// The measurement and evidence identities, and so the model identity,
    /// change; every prepare check runs again on the new model. The route,
    /// indicator, construct, binding, representation and declared structure
    /// are not parameters and do not change here.
    ///
    /// # Errors
    /// Parameters of another route (`route_not_supported`), a model that is
    /// not well formed under the new parameters, a different provider object
    /// (`measurement_posterior_not_trusted`: it needs its own verification
    /// through a new preparation), any estimator refusal, or cancellation.
    pub fn with_validation_parameters(
        &self,
        parameters: MeasurementParameters,
        evidence: ParameterEvidence,
        ctx: &ExecutionContext,
    ) -> Result<MeasurementRefresh, MeasurementModelRefusal> {
        check_cancelled(ctx)?;
        if parameters.route() != self.model.route {
            return Err(MeasurementModelRefusal {
                code: antecedent_core::reason_code!("route_not_supported"),
                detail: "measurement_model.route_mismatch".to_owned(),
                message: format!(
                    "the model declares route `{}`; the supplied parameters belong to `{}`",
                    self.model.route.as_str(),
                    parameters.route().as_str()
                ),
                remedy: None,
            });
        }
        if let (
            MeasurementParameters::ExternalPosterior { reference: old, .. },
            MeasurementParameters::ExternalPosterior { reference: new, .. },
        ) = (&self.model.parameters, &parameters)
        {
            if old != new {
                return Err(MeasurementModelRefusal {
                    code: antecedent_core::reason_code!("measurement_posterior_not_trusted"),
                    detail: "measurement_model.posterior_identity".to_owned(),
                    message: format!(
                        "the provider object changed from `{}/{}` to `{}/{}`; the new object needs \
                         its own attestation or verification through a new preparation",
                        old.provider_id, old.object_id, new.provider_id, new.object_id
                    ),
                    remedy: Some("supply the exact provider object the model names".to_owned()),
                });
            }
        }
        let mut model = self.model.clone();
        model.parameters = parameters;
        model.evidence = evidence;
        check_request(&model, &self.binding, self.input.operation())?;
        let prepared = Self {
            model,
            binding: self.binding.clone(),
            input: self.input.clone(),
            trust: self.trust,
        };
        let estimate = prepared.estimate(ctx)?;
        Ok(MeasurementRefresh { prepared, estimate })
    }
}

impl MeasurementEstimate {
    /// The corrected functional, bound to the model identity.
    #[must_use]
    pub const fn record(&self) -> &CorrectedFunctionalRecord {
        &self.record
    }

    /// The model the result is bound to.
    #[must_use]
    pub const fn model(&self) -> &MeasurementModel {
        &self.model
    }

    /// The estimator's own result for the route.
    #[must_use]
    pub const fn detail(&self) -> &MeasurementDetail {
        &self.detail
    }

    /// The corrected point value.
    #[must_use]
    pub const fn point(&self) -> f64 {
        self.record.point
    }

    /// The uncorrected value of the same functional, when one exists.
    #[must_use]
    pub const fn naive_point(&self) -> Option<f64> {
        self.record.naive_point
    }

    /// The delta-method standard error. It is uncalibrated: no coverage has
    /// been measured and no interval follows from it.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.record.standard_error
    }

    /// What the uncertainty means: always [`MeasurementInference::PointOnly`]
    /// or [`MeasurementInference::UncalibratedStandardError`].
    #[must_use]
    pub const fn inference(&self) -> MeasurementInference {
        self.record.inference
    }

    /// Whether the measurement parameters' uncertainty is in the variance.
    #[must_use]
    pub const fn parameter_uncertainty(&self) -> ParameterUncertainty {
        self.record.parameter_uncertainty
    }

    /// The identity of the model this result is bound to.
    #[must_use]
    pub const fn model_identity(&self) -> SemanticDigest {
        self.record.model_identity
    }

    /// The route that produced the result.
    #[must_use]
    pub const fn route(&self) -> MeasurementRoute {
        self.record.route
    }

    /// The downstream operation the result enters.
    #[must_use]
    pub const fn operation(&self) -> MeasurementOperation {
        self.operation
    }

    /// How far the provider object was trusted, for the posterior route.
    #[must_use]
    pub const fn provider_trust(&self) -> Option<LatentPosteriorTrust> {
        self.provider_trust
    }

    /// The representation mapping the result relied on, when the analysis
    /// representation differed from the model's.
    #[must_use]
    pub fn representation_mapping_id(&self) -> Option<&str> {
        self.binding.representation_mapping.as_ref().map(|m| m.mapping_id.as_str())
    }

    /// An interval for the corrected functional.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `measurement_model.interval_closed`.
    pub fn interval(&self) -> Result<std::convert::Infallible, MeasurementModelRefusal> {
        Err(MeasurementModelRefusal::from_estimate(&interval_not_licensed()))
    }

    /// Value and variance of a linear functional of the corrected coefficient
    /// vector (intercept first, the latent regressor second, then the
    /// covariates). The variance is the quadratic form under the joint
    /// covariance, which includes the measurement-parameter uncertainty where
    /// it was propagated.
    ///
    /// # Errors
    /// A route that has no coefficient vector refuses with the operation's
    /// contract refusal; a weight vector of the wrong width refuses.
    pub fn downstream_linear_functional(
        &self,
        weights: &[f64],
    ) -> Result<(f64, f64), MeasurementModelRefusal> {
        check_request(
            &self.model,
            &self.binding,
            MeasurementOperation::LinearFunctionalOfCoefficients,
        )?;
        match &self.detail {
            MeasurementDetail::Linear(effect) => effect
                .downstream_linear_functional(weights)
                .map_err(|error| MeasurementModelRefusal::from_estimate(&error)),
            MeasurementDetail::PosteriorLinear(effect) => {
                let d = effect.coefficients.len();
                if weights.len() != d || weights.iter().any(|w| !w.is_finite()) {
                    return Err(MeasurementModelRefusal {
                        code: antecedent_core::reason_code!("invalid_argument"),
                        detail: "measurement_model.input_invalid".to_owned(),
                        message: "functional weights do not match the design".to_owned(),
                        remedy: None,
                    });
                }
                let value: f64 =
                    weights.iter().zip(&effect.coefficients).map(|(w, b)| w * b).sum();
                let mut variance = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        variance += weights[i] * effect.covariance[i * d + j] * weights[j];
                    }
                }
                Ok((value, variance))
            }
            _ => Err(MeasurementModelRefusal {
                code: antecedent_core::reason_code!("cell_not_licensed"),
                detail: "measurement_model.operation_not_licensed".to_owned(),
                message: "this route has no coefficient vector to take a functional of".to_owned(),
                remedy: None,
            }),
        }
    }

    /// Write the model, its identities and the result with the sufficient
    /// inputs a consumer recomputes it from.
    ///
    /// # Errors
    /// A result that does not replay under its own model (an exporter never
    /// writes a number its consumer would refuse), or a blank artifact id.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, MeasurementModelRefusal> {
        let result = MeasurementResult {
            record: self.record.clone(),
            replay: self.replay.clone(),
            provider_trust: self.provider_trust.map(|t| t.as_str().to_owned()),
        };
        MeasurementModelArtifact::new(&self.model, Some(&result))
            .and_then(|artifact| artifact.to_bytes(artifact_id))
            .map_err(|error| MeasurementModelRefusal::from_io(&error))
    }
}

/// A consumed measurement-model artifact: the model, the stored record and the
/// values an independent recomputation produced.
#[derive(Clone, Debug)]
pub struct ConsumedMeasurement {
    artifact: MeasurementModelArtifact,
    model: MeasurementModel,
    result: Option<MeasurementResult>,
    replayed: Option<ReplayedValues>,
}

impl ConsumedMeasurement {
    /// The model the bytes carry; its identity equals the one the consumer
    /// retained.
    #[must_use]
    pub const fn model(&self) -> &MeasurementModel {
        &self.model
    }

    /// The stored corrected functional, when the artifact carries one.
    #[must_use]
    pub fn record(&self) -> Option<&CorrectedFunctionalRecord> {
        self.result.as_ref().map(|r| &r.record)
    }

    /// The point, variance and row count recomputed from the stored sufficient
    /// inputs with `antecedent-stats` only.
    #[must_use]
    pub const fn replayed(&self) -> Option<&ReplayedValues> {
        self.replayed.as_ref()
    }

    /// Trust label of an external posterior (`externally_attested` or
    /// `exact_request_verified`); `None` for native routes.
    #[must_use]
    pub fn provider_trust(&self) -> Option<&str> {
        self.result.as_ref().and_then(|r| r.provider_trust.as_deref())
    }

    /// Whether the stored result was computed from data with this identity.
    #[must_use]
    pub fn covers_data_identity(&self, identity: &SemanticDigest) -> bool {
        self.artifact.covers_data_identity(identity)
    }

    /// Check that the stored result may be used for `binding` and
    /// `operation`: the consumer's own population, regime, arm,
    /// representation, assumed structure and downstream operation, against the
    /// model the bytes carry.
    ///
    /// # Errors
    /// The contract refusal naming the mismatch.
    pub fn check_use(
        &self,
        binding: &MeasurementAnalysisBinding,
        operation: MeasurementOperation,
    ) -> Result<(), MeasurementModelRefusal> {
        check_request(&self.model, binding, operation)
    }
}

/// Independently consume a measurement-model artifact.
///
/// `expected_model_identity_hex` is the 64-character model identity the
/// consumer retained independently of the bytes. A swapped sensitivity or
/// specificity, population, regime, arm or representation version is a
/// different identity and is refused; so is a stored point or standard error
/// that does not recompute from its stored inputs.
///
/// # Errors
/// Corrupt, truncated, oversized or incompatible bytes, an identity that is
/// not the expected one (`measurement_binding_mismatch`), or a result that
/// does not recompute.
pub fn consume_measurement_model_artifact(
    bytes: &[u8],
    expected_model_identity_hex: &str,
) -> Result<ConsumedMeasurement, MeasurementModelRefusal> {
    let artifact = MeasurementModelArtifact::from_bytes(bytes, expected_model_identity_hex)
        .map_err(|error| MeasurementModelRefusal::from_io(&error))?;
    let model =
        artifact.model().map_err(|error| MeasurementModelRefusal::from_io(&error))?;
    let result =
        artifact.result().map_err(|error| MeasurementModelRefusal::from_io(&error))?;
    let replayed = match &result {
        Some(r) => Some(
            replay_values(&model, &r.replay)
                .map_err(|error| MeasurementModelRefusal::from_io(&error))?,
        ),
        None => None,
    };
    Ok(ConsumedMeasurement { artifact, model, result, replayed })
}
