//! Composing independently fitted mechanisms into one executable joint model:
//! prepare, execute, export and independently consume.
//!
//! Individually valid conditional mechanisms do not imply a valid joint model.
//! This module routes a *declared* composition through the layers that own
//! each rule and adds none of its own:
//!
//! 1. **Prepare** ([`StudyBuilder::mechanism_composition`]) checks, before any
//!    number is formed, the semantic coordinates of every mechanism with
//!    `antecedent_contract::check_mechanism_composition` (a parent must be a
//!    realization of the same quantity its producer yields; a mechanism from
//!    another population or regime needs a declared admission), binds the
//!    plan to the numbers of each mechanism, and compiles the result under the
//!    model-level coherence checker (acyclicity, parents equal conditioning
//!    set, support overlap, shared latents, provenance, aligned uncertainty).
//!    Every defect is returned, each with a registered code, a namespaced
//!    detail and the declaration that would license the composition.
//! 2. **Execute** runs the closed-form joint law and `do()` law, structural
//!    sampling and the first-order delta-method variance of a linear contrast
//!    of interventional means (linear-Gaussian), or exact enumeration and
//!    truncated factorization (finite tables).
//! 3. **Export** seals the premises and one query into
//!    `mechanism_composition_v1`; **consume** recomputes the premises digest,
//!    recompiles under the coherence checker and replays the query to
//!    identical result bits.
//!
//! # Declared, not native
//!
//! A mechanism built from a fitted registry slot is labelled
//! [`MechanismOriginKind::Fitted`]; one built from coefficients or tables the
//! analyst wrote down is labelled [`MechanismOriginKind::Declared`], is
//! refused unless the request acknowledges it, and is listed by
//! [`PreparedMechanismComposition::declared_mechanisms`]. The facade sets the
//! label from the source of the numbers; a declaration cannot claim a fit it
//! does not have. A slot carries no parameter covariance and no fit identity:
//! the caller supplies the identities of the fit and its evidence, and a
//! reported covariance, if any.
//!
//! # Uncertainty
//!
//! Separately fitted mechanisms are not independent by default. A variance
//! over more than one mechanism needs an aligned joint law, an explicit
//! cross-mechanism covariance block for every pair (a zero block is a declared
//! independence for that pair), or independence by design with its
//! justification; otherwise the refusal is `joint_law_required`. The variance
//! is a parameter-covariance variance only: it omits second-order terms, holds
//! residual scales fixed and carries no coverage claim. The interval route is
//! closed ([`PreparedMechanismComposition::interval`]).
//!
//! # Lifecycle
//!
//! [`PreparedMechanismComposition::refresh_mechanisms`] takes mechanisms
//! refitted on new outcome-only data. It **keeps** the plan: the declared
//! graph, units, transforms, populations and regimes, admissions, latent
//! declarations, fitted domains and supports, and the kind of declared
//! dependence. It **recomputes** the numbers of the refitted mechanisms, their
//! fit and evidence identities, their reported covariance and the joint model,
//! and runs every prepare check again. It **refuses** a refit that changes a
//! mechanism's parents, family or variable, and a refit under an aligned joint
//! law or cross-covariance declared for the earlier fit: those dependence
//! declarations describe the earlier estimates and are `unknown` again until
//! the caller declares the new one. The call takes `&self` and returns a new
//! value, so a refused refresh leaves the prepared composition unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::{
    DependenceDeclaration, ExternalRefusal, MechanismCompositionDeclaration,
    MechanismCompositionPlan, MechanismDeclaration, MechanismOriginKind, TransportAdmission,
    check_mechanism_composition,
};
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_io::mechanism_composition_artifact::{
    MechanismCompositionArtifactError, MechanismCompositionArtifactWire, MechanismNumbers,
    lower_plan, refusals_for_findings,
};
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CompositionError,
    CrossCovarianceBlock, Domain, FiniteLaw, FunctionalUncertainty, JointGaussian, MeanContrast,
    MechanismFamily, MechanismForm, MechanismUncertainty, ParameterId, PosteriorDependence,
    SampleTable, TabularRow, UncertaintyMeaning, check_composition,
};

pub use antecedent_io::mechanism_composition_artifact::{
    CompositionQueryWire, CompositionResultWire,
};

use super::StudyBuilder;

/// The refusals of a composition request: every defect found, each with a
/// registered reason code, a namespaced `mechanism_composition.*` detail, the
/// offending subject, expected and supplied semantics and a remedy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismCompositionRefusal {
    /// Every refusal, in the fixed order the checkers found them. Never empty.
    pub refusals: Vec<ExternalRefusal>,
}

impl MechanismCompositionRefusal {
    fn one(refusal: ExternalRefusal) -> Self {
        Self { refusals: vec![refusal] }
    }

    /// A refusal whose subject is described by `subject`.
    fn of(
        code: &'static str,
        stage: &'static str,
        detail: &str,
        subject: impl Into<String>,
    ) -> Self {
        Self::one(ExternalRefusal::new(code, stage, detail).with_offending(subject.into()))
    }

    /// The code of the first refusal.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.refusals.first().map_or(antecedent_core::reason_code!("invalid_argument"), |r| r.code)
    }

    /// The namespaced detail of every refusal.
    #[must_use]
    pub fn details(&self) -> Vec<&str> {
        self.refusals.iter().map(|r| r.detail.as_str()).collect()
    }
}

impl std::fmt::Display for MechanismCompositionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusals.first() {
            Some(first) => {
                write!(
                    f,
                    "{}{}: {}",
                    antecedent_core::reason_code::PREFIX,
                    first.code,
                    first.detail
                )?;
                if let Some(subject) = &first.offending {
                    write!(f, " ({subject})")?;
                }
                if self.refusals.len() > 1 {
                    write!(f, " and {} more", self.refusals.len() - 1)?;
                }
                Ok(())
            }
            None => f.write_str("mechanism composition refused"),
        }
    }
}

impl std::error::Error for MechanismCompositionRefusal {}

impl From<MechanismCompositionRefusal> for IoError {
    fn from(refusal: MechanismCompositionRefusal) -> Self {
        Self::Refused { code: refusal.code(), message: refusal.to_string() }
    }
}

impl From<MechanismCompositionRefusal> for crate::CausalError {
    fn from(refusal: MechanismCompositionRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

fn model_refusal(error: CompositionError) -> MechanismCompositionRefusal {
    match error {
        CompositionError::Incoherent { findings } => {
            MechanismCompositionRefusal { refusals: refusals_for_findings(&findings) }
        }
        CompositionError::Unsupported { message } => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("route_not_supported"),
            "compose",
            "mechanism_composition.request_unsupported",
            message,
        ),
        CompositionError::Cancelled => cancelled(),
        CompositionError::Numerical { message } => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "compose",
            "mechanism_composition.numerical_failure",
            message,
        ),
        other => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("route_not_supported"),
            "compose",
            "mechanism_composition.request_unsupported",
            other.to_string(),
        ),
    }
}

fn cancelled() -> MechanismCompositionRefusal {
    MechanismCompositionRefusal::of(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "compose",
        "mechanism_composition.cancelled",
        "cancelled before a composed result was formed; this is not a verdict on the mechanisms",
    )
}

fn artifact_refusal(error: MechanismCompositionArtifactError) -> MechanismCompositionRefusal {
    match error {
        MechanismCompositionArtifactError::Incoherent { refusals } => {
            MechanismCompositionRefusal { refusals }
        }
        MechanismCompositionArtifactError::PremisesMismatch => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("mechanism_composition_incoherent"),
            "consume",
            "mechanism_composition.premises_mismatch",
            "the premises and query do not hash to the stored or the retained identity",
        ),
        MechanismCompositionArtifactError::ResultMismatch => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("mechanism_composition_incoherent"),
            "consume",
            "mechanism_composition.result_mismatch",
            "the stored result is not the replay of the stored premises and query",
        ),
        MechanismCompositionArtifactError::Replay(text)
            if text.contains("composition execution cancelled") =>
        {
            cancelled()
        }
        other => MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "consume",
            "mechanism_composition.artifact_invalid",
            other.to_string(),
        ),
    }
}

/// Where a mechanism's numbers came from. The facade derives the mechanism's
/// origin label from this, so a declaration cannot claim a fit it does not
/// have.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MechanismSource {
    /// A linear-Gaussian mechanism fitted by the library's mechanism registry.
    FittedLinear {
        /// Intercept.
        intercept: f64,
        /// Idiosyncratic residual standard deviation.
        sigma: f64,
        /// `(parent variable, coefficient)`.
        coefficients: Vec<(String, f64)>,
    },
    /// Coefficients the analyst wrote down. Labelled declared, not native.
    DeclaredLinear {
        /// Intercept.
        intercept: f64,
        /// Idiosyncratic residual standard deviation.
        sigma: f64,
        /// `(parent variable, coefficient)`.
        coefficients: Vec<(String, f64)>,
    },
    /// A conditional probability table the analyst wrote down. Labelled
    /// declared, not native.
    DeclaredTable {
        /// Levels of the produced variable.
        levels: Vec<String>,
        /// One row per parent configuration.
        rows: Vec<TabularRow>,
    },
}

impl MechanismSource {
    /// Map a fitted linear-Gaussian registry slot onto a source.
    ///
    /// `parents` names the slot's coefficients in the order of the fit's
    /// parent gather plan; the slot itself carries only positions.
    ///
    /// # Errors
    /// A slot that is not a plain linear-Gaussian mechanism (a basis
    /// expansion, a discrete or constant slot, a vacant or pending slot), or a
    /// parent list of another length than the slot's coefficients.
    pub fn from_fitted_slot(
        slot: &MechanismSlot,
        parents: &[&str],
    ) -> Result<Self, MechanismCompositionRefusal> {
        match slot {
            MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
                if coeffs.len() != parents.len() {
                    return Err(MechanismCompositionRefusal::of(
                        antecedent_core::reason_code!("invalid_argument"),
                        "compose",
                        "mechanism_composition.shape_coefficient_parent_mismatch",
                        format!(
                            "the slot has {} coefficients and {} parents were named",
                            coeffs.len(),
                            parents.len()
                        ),
                    ));
                }
                Ok(Self::FittedLinear {
                    intercept: *intercept,
                    sigma: *sigma,
                    coefficients: parents
                        .iter()
                        .zip(coeffs.iter())
                        .map(|(p, c)| ((*p).to_owned(), *c))
                        .collect(),
                })
            }
            _ => Err(MechanismCompositionRefusal::of(
                antecedent_core::reason_code!("mechanism_composition_incoherent"),
                "compose",
                "mechanism_composition.slot_not_linear_gaussian",
                "only a plain linear-Gaussian slot composes; a basis expansion, discrete, \
                 constant, vacant or pending slot is outside the bounded family",
            )),
        }
    }

    fn origin(&self) -> MechanismOriginKind {
        match self {
            Self::FittedLinear { .. } => MechanismOriginKind::Fitted,
            Self::DeclaredLinear { .. } | Self::DeclaredTable { .. } => {
                MechanismOriginKind::Declared
            }
        }
    }

    fn family(&self) -> MechanismFamily {
        match self {
            Self::FittedLinear { .. } | Self::DeclaredLinear { .. } => {
                MechanismFamily::LinearGaussian
            }
            Self::DeclaredTable { .. } => MechanismFamily::Tabular,
        }
    }

    fn form(&self) -> MechanismForm {
        match self {
            Self::FittedLinear { intercept, sigma, coefficients }
            | Self::DeclaredLinear { intercept, sigma, coefficients } => {
                MechanismForm::LinearGaussian {
                    intercept: *intercept,
                    sigma: *sigma,
                    coefficients: coefficients.clone(),
                }
            }
            Self::DeclaredTable { levels, rows } => {
                MechanismForm::Tabular { levels: levels.clone(), rows: rows.clone() }
            }
        }
    }
}

/// One mechanism of a composition: its semantic coordinates, its numbers and
/// what the fit reported about their uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismInput {
    /// The mechanism in semantic coordinates: the quantity produced and the
    /// quantities consumed, fit and evidence identities, declared shared
    /// latent causes. Its origin label is overwritten from `source`.
    pub declaration: MechanismDeclaration,
    /// The numbers and where they came from.
    pub source: MechanismSource,
    /// Support of the produced variable.
    pub output_support: Domain,
    /// Domain each parent was fitted on, by parent variable.
    pub parent_domains: Vec<(String, Domain)>,
    /// Marginal parameter covariance the fit reported, when it did.
    pub uncertainty: Option<MechanismUncertainty>,
    /// Loading on each declared shared latent, by latent identity.
    pub latent_loadings: Vec<(String, f64)>,
}

/// What is declared about the dependence between the mechanisms' estimates.
#[derive(Clone, Debug, PartialEq)]
pub enum DependenceInput {
    /// Nothing declared; no uncertainty claim over several mechanisms.
    Unknown,
    /// Estimated from disjoint, independently generated evidence.
    IndependentByDesign {
        /// Why the estimates are independent.
        justification: String,
    },
    /// One aligned joint parameter law.
    AlignedJointLaw {
        /// Identity of the joint law.
        law_id: String,
        /// What the law means.
        meaning: UncertaintyMeaning,
        /// Parameters in the order of `covariance`.
        parameters: Vec<ParameterId>,
        /// Row-major covariance over `parameters`.
        covariance: Vec<f64>,
    },
    /// Explicit covariance between every pair of mechanisms; a zero block is a
    /// declared independence for that pair.
    CrossCovariance {
        /// One block per unordered pair of mechanisms.
        blocks: Vec<CrossCovarianceBlock>,
    },
}

impl DependenceInput {
    fn declaration(&self) -> DependenceDeclaration {
        match self {
            Self::Unknown => DependenceDeclaration::Unknown,
            Self::IndependentByDesign { justification } => {
                DependenceDeclaration::IndependentByDesign { justification: justification.clone() }
            }
            Self::AlignedJointLaw { law_id, .. } => {
                DependenceDeclaration::AlignedJointLaw { law_id: law_id.clone() }
            }
            Self::CrossCovariance { .. } => DependenceDeclaration::CrossCovariance,
        }
    }

    fn law(&self) -> PosteriorDependence {
        match self {
            Self::Unknown => PosteriorDependence::Unknown,
            Self::IndependentByDesign { justification } => {
                PosteriorDependence::IndependentByDesign { justification: justification.clone() }
            }
            Self::AlignedJointLaw { meaning, parameters, covariance, .. } => {
                PosteriorDependence::Joint {
                    meaning: *meaning,
                    parameters: parameters.clone(),
                    covariance: covariance.clone(),
                }
            }
            Self::CrossCovariance { blocks } => {
                PosteriorDependence::CrossCovariance { blocks: blocks.clone() }
            }
        }
    }
}

/// A declared composition: semantic coordinates, numbers and dependence.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismCompositionRequest {
    /// Target population identity.
    pub target_population_id: String,
    /// Target regime identity.
    pub target_regime_id: String,
    /// The mechanisms.
    pub mechanisms: Vec<MechanismInput>,
    /// Declared causal graph as `(parent variable, child variable)`.
    pub edges: Vec<(String, String)>,
    /// Declared admissions of mechanisms from other contexts.
    pub admissions: Vec<TransportAdmission>,
    /// Declared dependence between the mechanisms' estimates.
    pub dependence: DependenceInput,
    /// Acknowledge mechanisms that were declared, not estimated.
    pub admit_declared_mechanisms: bool,
}

/// The mechanism a refit replaces.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismRefit {
    /// Variable whose mechanism was refitted.
    pub variable_id: String,
    /// The refitted numbers; must be of the same family and name the same
    /// parents.
    pub source: MechanismSource,
    /// Identity of the new fit.
    pub artifact_id: String,
    /// Identity of the new evidence (a refit on new data has a new one).
    pub evidence_id: String,
    /// The refit's reported marginal parameter covariance, when it reports one.
    pub uncertainty: Option<MechanismUncertainty>,
}

#[derive(Clone)]
enum Executable {
    Gaussian(ComposedLinearGaussian),
    Finite(ComposedFinite),
}

/// A composition that compiled: one executable joint model with the plan it
/// was compiled under.
#[derive(Clone)]
pub struct PreparedMechanismComposition {
    request: MechanismCompositionRequest,
    plan: MechanismCompositionPlan,
    spec: ComposedModelSpec,
    executable: Executable,
}

impl std::fmt::Debug for PreparedMechanismComposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedMechanismComposition")
            .field("plan_identity", &self.plan.identity_hex())
            .field("family", &self.family())
            .field("variables", &self.variables())
            .finish_non_exhaustive()
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), MechanismCompositionRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok(())
}

fn build(
    request: &MechanismCompositionRequest,
) -> Result<(MechanismCompositionPlan, ComposedModelSpec, Executable), MechanismCompositionRefusal>
{
    let declaration = MechanismCompositionDeclaration {
        target_population_id: request.target_population_id.clone(),
        target_regime_id: request.target_regime_id.clone(),
        mechanisms: request
            .mechanisms
            .iter()
            .map(|m| {
                let mut declaration = m.declaration.clone();
                declaration.origin = m.source.origin();
                declaration
            })
            .collect(),
        edges: request.edges.clone(),
        admissions: request.admissions.clone(),
        dependence: request.dependence.declaration(),
        admit_declared_mechanisms: request.admit_declared_mechanisms,
    };
    let plan = check_mechanism_composition(&declaration)
        .map_err(|refusals| MechanismCompositionRefusal { refusals })?;
    let mut numbers: BTreeMap<String, MechanismNumbers> = BTreeMap::new();
    for input in &request.mechanisms {
        numbers.insert(
            input.declaration.output.variable_id.clone(),
            MechanismNumbers {
                form: input.source.form(),
                output_support: input.output_support.clone(),
                parent_domains: input.parent_domains.clone(),
                uncertainty: input.uncertainty.clone(),
                latent_loadings: input.latent_loadings.clone(),
            },
        );
    }
    let spec = lower_plan(&plan, &numbers, request.dependence.law()).map_err(|error| {
        MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("invalid_argument"),
            "compose",
            "mechanism_composition.numbers_unbound",
            error.to_string(),
        )
    })?;
    let all_linear =
        request.mechanisms.iter().all(|m| m.source.family() == MechanismFamily::LinearGaussian);
    let all_tabular =
        request.mechanisms.iter().all(|m| m.source.family() == MechanismFamily::Tabular);
    let executable = if all_linear {
        Executable::Gaussian(ComposedLinearGaussian::compile(&spec).map_err(model_refusal)?)
    } else if all_tabular {
        Executable::Finite(ComposedFinite::compile(&spec).map_err(model_refusal)?)
    } else {
        // The model checker names the defect; a mixed composition always has
        // at least that finding, so this refusal is never empty.
        let findings = check_composition(&spec);
        return Err(MechanismCompositionRefusal { refusals: refusals_for_findings(&findings) });
    };
    Ok((plan, spec, executable))
}

impl StudyBuilder {
    /// Prepare a declared composition of independently fitted mechanisms.
    ///
    /// # Errors
    /// Every semantic refusal of the declared coordinates, or, once those
    /// hold, every coherence finding of the numbers (cycle, parents versus
    /// conditioning set, support gap, dropped or unloaded shared latent,
    /// misaligned or contradicted uncertainty, missing provenance, mixed
    /// families), or cancellation.
    pub fn mechanism_composition(
        request: MechanismCompositionRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMechanismComposition, MechanismCompositionRefusal> {
        check_cancelled(ctx)?;
        let (plan, spec, executable) = build(&request)?;
        Ok(PreparedMechanismComposition { request, plan, spec, executable })
    }
}

impl PreparedMechanismComposition {
    /// The semantically checked plan, in canonical order.
    #[must_use]
    pub const fn plan(&self) -> &MechanismCompositionPlan {
        &self.plan
    }

    /// The canonical model-level specification the composition compiled from.
    #[must_use]
    pub const fn spec(&self) -> &ComposedModelSpec {
        &self.spec
    }

    /// Canonical identity of the plan (never a formatted rendering).
    #[must_use]
    pub fn plan_identity_hex(&self) -> String {
        self.plan.identity_hex()
    }

    /// The family the composition executes in.
    #[must_use]
    pub fn family(&self) -> MechanismFamily {
        match self.executable {
            Executable::Gaussian(_) => MechanismFamily::LinearGaussian,
            Executable::Finite(_) => MechanismFamily::Tabular,
        }
    }

    /// Variables of the joint model, in canonical order.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        match &self.executable {
            Executable::Gaussian(model) => model.variables(),
            Executable::Finite(model) => model.variables(),
        }
    }

    /// Variables whose mechanism was written down by the analyst rather than
    /// estimated. Every claim from this composition rests on them as assumed.
    #[must_use]
    pub fn declared_mechanisms(&self) -> Vec<&str> {
        self.plan
            .mechanisms
            .iter()
            .filter(|m| m.origin == MechanismOriginKind::Declared)
            .map(|m| m.variable_id.as_str())
            .collect()
    }

    /// An interval for any functional of this composition.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `mechanism_composition.interval_route_closed`:
    /// the delta-method variance carries no measured coverage.
    pub fn interval(&self) -> Result<std::convert::Infallible, MechanismCompositionRefusal> {
        Err(MechanismCompositionRefusal::of(
            antecedent_core::reason_code!("cell_not_licensed"),
            "uncertainty",
            "mechanism_composition.interval_route_closed",
            "the first-order parameter-covariance variance has no measured coverage; the point \
             and the labelled variance are the result",
        ))
    }

    /// Variables whose mechanism was fitted in another population or regime and
    /// admitted under a declared basis. Empty states that every mechanism is
    /// native to the composition's target.
    pub(super) fn admitted_variables(&self) -> Vec<&str> {
        self.request.admissions.iter().map(|a| a.variable_id.as_str()).collect()
    }

    pub(super) fn gaussian(&self) -> Result<&ComposedLinearGaussian, MechanismCompositionRefusal> {
        match &self.executable {
            Executable::Gaussian(model) => Ok(model),
            Executable::Finite(_) => Err(MechanismCompositionRefusal::of(
                antecedent_core::reason_code!("route_not_supported"),
                "compose",
                "mechanism_composition.request_unsupported",
                "a Gaussian query was asked of a finite-table composition",
            )),
        }
    }

    fn finite(&self) -> Result<&ComposedFinite, MechanismCompositionRefusal> {
        match &self.executable {
            Executable::Finite(model) => Ok(model),
            Executable::Gaussian(_) => Err(MechanismCompositionRefusal::of(
                antecedent_core::reason_code!("route_not_supported"),
                "compose",
                "mechanism_composition.request_unsupported",
                "a finite-law query was asked of a linear-Gaussian composition",
            )),
        }
    }

    /// The joint Gaussian law under `do(interventions)`; an empty list is the
    /// observational law of the composition.
    ///
    /// # Errors
    /// A finite-table composition, an unknown or repeated variable, a value
    /// outside the domain a consumer was fitted on.
    pub fn intervention_law(
        &self,
        interventions: &[(&str, f64)],
    ) -> Result<JointGaussian, MechanismCompositionRefusal> {
        self.gaussian()?.law(interventions).map_err(model_refusal)
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]`, exact given the declared
    /// mechanisms.
    ///
    /// # Errors
    /// As [`Self::intervention_law`], or an unknown outcome.
    pub fn mean_contrast(
        &self,
        contrast: &MeanContrast,
    ) -> Result<f64, MechanismCompositionRefusal> {
        self.gaussian()?.mean_contrast(contrast).map_err(model_refusal)
    }

    /// The first-order delta-method variance of a contrast of interventional
    /// means from the declared parameter law, with the gradient and how the
    /// dependence between mechanisms was licensed.
    ///
    /// # Errors
    /// `joint_law_required` when the dependence between more than one
    /// mechanism is unknown or a cross-covariance pair is missing, a marginal
    /// the fit did not report, an assembled law that is not positive
    /// semidefinite, and the errors of [`Self::mean_contrast`].
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
    ) -> Result<FunctionalUncertainty, MechanismCompositionRefusal> {
        if let antecedent_contract::UncertaintyRoute::Closed(refusal) =
            self.plan.uncertainty_route()
        {
            return Err(MechanismCompositionRefusal::one(refusal));
        }
        // A product of coefficients along a path is nonlinear in the parameters:
        // a declared independence of the estimates is not a joint law for it.
        let arms: Vec<Vec<String>> = contrast
            .arms
            .iter()
            .map(|arm| arm.interventions.iter().map(|(name, _)| name.clone()).collect())
            .collect();
        self.plan
            .check_contrast_uncertainty(&contrast.outcome, &arms)
            .map_err(MechanismCompositionRefusal::one)?;
        self.gaussian()?.mean_contrast_uncertainty(contrast).map_err(model_refusal)
    }

    /// Draw `n_rows` units from the (possibly intervened) joint model. The
    /// noise of each variable is keyed by its name, so a variable's draws do
    /// not depend on the other variables or their order.
    ///
    /// # Errors
    /// As [`Self::intervention_law`], a size outside the bound, or
    /// cancellation.
    pub fn sample(
        &self,
        interventions: &[(&str, f64)],
        n_rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<SampleTable, MechanismCompositionRefusal> {
        check_cancelled(ctx)?;
        self.gaussian()?.sample(interventions, n_rows, ctx).map_err(model_refusal)
    }

    /// The exact finite joint law under `do(interventions)` by truncated
    /// factorization.
    ///
    /// # Errors
    /// A linear-Gaussian composition, an unknown variable or level, or
    /// cancellation.
    pub fn finite_law(
        &self,
        interventions: &[(&str, &str)],
        ctx: &ExecutionContext,
    ) -> Result<FiniteLaw, MechanismCompositionRefusal> {
        check_cancelled(ctx)?;
        self.finite()?.law(interventions, ctx).map_err(model_refusal)
    }

    /// Replace the mechanisms named by `refits` with their refits and
    /// recompile. See the module documentation for what is kept, recomputed
    /// and refused. `dependence` redeclares the dependence law when the
    /// earlier one was an aligned joint law or a cross-covariance (those
    /// describe the earlier estimates); `None` keeps the earlier declaration
    /// when it is still valid.
    ///
    /// # Errors
    /// A refit of an unknown variable, of another family or with other parents
    /// (the mechanism is a different one), a retained joint law or
    /// cross-covariance for a refitted mechanism, any prepare refusal on the
    /// new numbers, or cancellation.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "refits are consumed as one replacement set: the caller hands over the new fits and keeps no alias"
    )]
    pub fn refresh_mechanisms(
        &self,
        refits: Vec<MechanismRefit>,
        dependence: Option<DependenceInput>,
        ctx: &ExecutionContext,
    ) -> Result<Self, MechanismCompositionRefusal> {
        check_cancelled(ctx)?;
        let mut request = self.request.clone();
        for refit in &refits {
            let Some(slot) = request
                .mechanisms
                .iter_mut()
                .find(|m| m.declaration.output.variable_id == refit.variable_id)
            else {
                return Err(MechanismCompositionRefusal::of(
                    antecedent_core::reason_code!("quantity_semantics_mismatch"),
                    "compose",
                    "mechanism_composition.identity_declaration_unknown_variable",
                    format!("`{}` has no mechanism in this composition", refit.variable_id),
                ));
            };
            if slot.source.family() != refit.source.family() {
                return Err(MechanismCompositionRefusal::of(
                    antecedent_core::reason_code!("mechanism_composition_incoherent"),
                    "compose",
                    "mechanism_composition.family_mixed",
                    format!("the refit of `{}` changes its mechanism family", refit.variable_id),
                ));
            }
            slot.source.clone_from(&refit.source);
            slot.declaration.artifact_id.clone_from(&refit.artifact_id);
            slot.declaration.evidence_id.clone_from(&refit.evidence_id);
            slot.uncertainty.clone_from(&refit.uncertainty);
        }
        if let Some(new) = dependence {
            request.dependence = new;
        } else {
            {
                let described_the_earlier_fit = matches!(
                    request.dependence,
                    DependenceInput::AlignedJointLaw { .. }
                        | DependenceInput::CrossCovariance { .. }
                );
                if described_the_earlier_fit && !refits.is_empty() {
                    return Err(MechanismCompositionRefusal::one(
                        ExternalRefusal::new(
                            antecedent_core::reason_code!("joint_law_required"),
                            "compose",
                            "mechanism_composition.posterior_dependence_unknown",
                        )
                        .with_offending("composition")
                        .with_expected("a dependence law for the refitted estimates")
                        .with_supplied("the law declared for the earlier fit")
                        .with_remedy(
                            "a joint law or cross-covariance describes the estimates it was \
                             declared for; declare the dependence of the refitted mechanisms",
                        ),
                    ));
                }
            }
        }
        let (plan, spec, executable) = build(&request)?;
        Ok(Self { request, plan, spec, executable })
    }

    /// Seal the premises and one query into a `mechanism_composition_v1`
    /// artifact: the query is executed and its result stored as bit patterns.
    ///
    /// # Errors
    /// An incoherent composition for this query (an uncertainty claim without
    /// a joint law is refused), a failed execution, or cancellation.
    pub fn export(
        &self,
        query: CompositionQueryWire,
        ctx: &ExecutionContext,
    ) -> Result<MechanismCompositionExport, MechanismCompositionRefusal> {
        check_cancelled(ctx)?;
        let wire = MechanismCompositionArtifactWire::seal(&self.spec, query, ctx)
            .map_err(artifact_refusal)?;
        let bytes = wire.export().map_err(artifact_refusal)?;
        Ok(MechanismCompositionExport {
            bytes,
            premises_digest: wire.premises_digest.clone(),
            result: wire.result,
        })
    }
}

/// An exported composition.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismCompositionExport {
    /// The artifact bytes.
    pub bytes: Vec<u8>,
    /// Digest of the premises and the query; a consumer that retains it
    /// independently refuses a consistently resealed edit.
    pub premises_digest: String,
    /// The stored result.
    pub result: CompositionResultWire,
}

/// A consumed `mechanism_composition_v1` artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedComposition {
    /// The identity that was recomputed and verified.
    pub premises_digest: String,
    /// The family the composition executed in.
    pub family: MechanismFamily,
    /// The replayed result, bit-identical to the stored one.
    pub result: CompositionResultWire,
}

impl ConsumedComposition {
    /// Mean and variance of a Gaussian intervention law.
    #[must_use]
    pub fn gaussian_law(&self) -> Option<(f64, f64)> {
        match &self.result {
            CompositionResultWire::GaussianLaw { mean_bits, variance_bits } => {
                Some((f64::from_bits(*mean_bits), f64::from_bits(*variance_bits)))
            }
            _ => None,
        }
    }

    /// Point, parameter-covariance variance, dependence label and method of a
    /// contrast of interventional means.
    #[must_use]
    pub fn gaussian_functional(&self) -> Option<(f64, f64, &str, &str)> {
        match &self.result {
            CompositionResultWire::GaussianFunctional {
                point_bits,
                variance_bits,
                dependence,
                method,
            } => Some((
                f64::from_bits(*point_bits),
                f64::from_bits(*variance_bits),
                dependence.as_str(),
                method.as_str(),
            )),
            _ => None,
        }
    }

    /// `(level, probability)` of a finite intervention distribution.
    #[must_use]
    pub fn finite_distribution(&self) -> Option<Vec<(String, f64)>> {
        match &self.result {
            CompositionResultWire::FiniteDistribution { probabilities } => {
                Some(probabilities.iter().map(|(l, p)| (l.clone(), f64::from_bits(*p))).collect())
            }
            _ => None,
        }
    }
}

/// Independently consume a mechanism-composition artifact: recompute the
/// premises digest, recompile under the coherence checker and replay the query
/// to identical result bits.
///
/// `expected_premises_digest` is the digest the consumer holds independently
/// of the bytes.
///
/// # Errors
/// Corrupt or unsupported bytes, a premises digest that is not the stored or
/// the retained one, premises that no longer compose (with every structured
/// refusal), or a stored result that does not replay.
pub fn consume_mechanism_composition_artifact(
    bytes: &[u8],
    expected_premises_digest: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedComposition, MechanismCompositionRefusal> {
    check_cancelled(ctx)?;
    // A readable container of another kind is a recognised artifact this consumer
    // does not read; the composition artifact itself is not a container.
    let expected =
        antecedent_io::mechanism_composition_artifact::MECHANISM_COMPOSITION_ARTIFACT_FEATURE;
    if let Some(kind) = antecedent_design::inverse_query_artifact::recognised_artifact_kind(bytes)
        .filter(|kind| kind != expected)
    {
        return Err(MechanismCompositionRefusal::one(
            ExternalRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "consume",
                "mechanism_composition.artifact_kind_mismatch",
            )
            .with_offending(format!("expected {expected}, supplied {kind}"))
            .with_expected(expected)
            .with_supplied(kind.clone())
            .with_remedy(antecedent_design::inverse_query_artifact::artifact_kind_remedy(&kind)),
        ));
    }
    let wire = MechanismCompositionArtifactWire::decode(bytes).map_err(artifact_refusal)?;
    let consumed = wire.consume(expected_premises_digest, ctx).map_err(artifact_refusal)?;
    Ok(ConsumedComposition {
        premises_digest: consumed.premises_digest,
        family: consumed.family,
        result: consumed.result,
    })
}
