// SPDX-License-Identifier: MIT OR Apache-2.0

//! Named recipes and the provider capabilities each inference requires.
//!
//! The recipes are linear/Gaussian, finite discrete, generalized linear,
//! g-computation, and orthogonal score. A recipe names one inference. A
//! coherent component either declares its own inference or takes the study
//! default; the default is used only when the component declares none. Two
//! components of one study may therefore name different inferences.
//!
//! A provider is accepted only for capabilities its declaration lists. The
//! provider name is not a capability. A Bayesian recipe is a likelihood:
//! a provider whose only capability is prediction is refused, and so is
//! [`PredictionOnlyBoundary`]. That boundary is the prediction contract of
//! this crate. An orthogonal-score nuisance and a frequentist g-computation
//! plug-in mean may attach it, and the attachment stays prediction-only.
//!
//! A likelihood or an update also probes the provider body. The probe is the
//! log-density or update call the recipe names. A declaration that lists the
//! capability is not that call; the body must answer it. The probe does not
//! fit a model. This module declares and accepts; [`execution`] runs. A
//! [`BoundRecipe`] becomes a [`execution::RecipeRun`] that evaluates the
//! likelihood, updates the posterior, solves the estimating equations, forms the
//! orthogonal score and integrates exactly, re-checking the provider on every
//! call. Row selections are physical `u64` indices. They are not converted to
//! `usize`.
//!
//! Accepting a provider is not identification and is not calibration. It does
//! not open a support-matrix route.

use std::collections::BTreeSet;
use std::fmt;

pub mod execution;

use antecedent_core::{
    CapabilityRequest, FailedCapabilityAttempt, InferenceSelection, ProviderCapability,
    ProviderDeclaration, ProviderId, is_blank,
};

/// Name frequentist, Bayesian, or generalized Bayesian on a component, or omit it.
pub const NAME_COMPONENT_INFERENCE: &str = concat!(
    "name frequentist, bayesian, or generalized Bayesian, ",
    "or omit the declaration to use the study default",
);

/// Name frequentist, Bayesian, or generalized Bayesian as the study default.
pub const NAME_STUDY_DEFAULT: &str =
    "name frequentist, bayesian, or generalized Bayesian as the study default";

/// Name frequentist, Bayesian, or generalized Bayesian on the recipe.
pub const NAME_RECIPE_INFERENCE: &str =
    "name frequentist, bayesian, or generalized Bayesian on the recipe";

/// Keep a prediction-only provider on the prediction boundary.
pub const KEEP_PREDICTION_BOUNDARY: &str = concat!(
    "keep the prediction-only provider on the prediction boundary ",
    "and declare a likelihood provider",
);

/// Declare the capabilities the recipe lists. A provider name is not one of them.
pub const DECLARE_REQUIRED_CAPABILITIES: &str = concat!(
    "declare the capabilities the recipe requires; ",
    "the provider name is not a capability",
);

/// Select the recipe that names the component's declared inference.
pub const SELECT_DECLARED_INFERENCE: &str = concat!(
    "select the recipe that names the component's declared inference; ",
    "the study default does not replace a declaration",
);

/// Select the recipe that names the study default, or declare a component inference.
pub const SELECT_STUDY_DEFAULT: &str = concat!(
    "select the recipe that names the study default, ",
    "or declare an inference on the component",
);

/// Bind a provider that declares this recipe's capabilities.
pub const BIND_MODEL_PROVIDER: &str = "bind a provider that declares this recipe's capabilities";

/// Answer the log-density or update call. A declaration is not that call.
pub const ANSWER_CAPABILITY_CALL: &str = concat!(
    "answer the log-density or update call the recipe probes; ",
    "listing the capability does not answer it",
);

const ESTIMATING_EQUATIONS: &[ProviderCapability] = &[ProviderCapability::EstimatingEquations];

const DENSITY_UPDATING: &[ProviderCapability] =
    &[ProviderCapability::Density, ProviderCapability::Updating];

const DISCRETE_FREQUENTIST: &[ProviderCapability] = &[ProviderCapability::ExactIntegration];

const DISCRETE_BAYESIAN: &[ProviderCapability] = &[
    ProviderCapability::Density,
    ProviderCapability::Updating,
    ProviderCapability::ExactIntegration,
];

const GLM_FREQUENTIST: &[ProviderCapability] =
    &[ProviderCapability::DifferentiableLogDensity, ProviderCapability::EstimatingEquations];

const GCOMP_BAYESIAN: &[ProviderCapability] =
    &[ProviderCapability::Prediction, ProviderCapability::Density, ProviderCapability::Updating];

const ORTHOGONAL: &[ProviderCapability] =
    &[ProviderCapability::EstimatingEquations, ProviderCapability::InfluenceContributions];

const PREDICTION_BOUNDARY_LABEL: &str = "prediction boundary";

/// Where an inference selection was required.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum InferenceSite {
    /// [`StudyDefault`].
    StudyDefault,
    /// A [`CoherentComponent`] declaration.
    Component,
    /// A [`Recipe`].
    Recipe,
}

impl InferenceSite {
    const fn next_attempt(self) -> &'static str {
        match self {
            Self::StudyDefault => NAME_STUDY_DEFAULT,
            Self::Component => NAME_COMPONENT_INFERENCE,
            Self::Recipe => NAME_RECIPE_INFERENCE,
        }
    }
}

impl fmt::Display for InferenceSite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::StudyDefault => "a study default",
            Self::Component => "a component inference",
            Self::Recipe => "a recipe",
        })
    }
}

/// The five named recipes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecipeKind {
    /// Linear or Gaussian conditional model.
    LinearGaussian,
    /// Finite discrete conditional law.
    FiniteDiscrete,
    /// Generalized linear model.
    Glm,
    /// G-computation of an interventional mean.
    GComputation,
    /// Orthogonal score, with cross-fit nuisances kept on the prediction boundary.
    OrthogonalScore,
}

impl RecipeKind {
    /// The five recipes, in report order.
    pub const ALL: [Self; 5] = [
        Self::LinearGaussian,
        Self::FiniteDiscrete,
        Self::Glm,
        Self::GComputation,
        Self::OrthogonalScore,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::FiniteDiscrete => "finite_discrete",
            Self::Glm => "glm",
            Self::GComputation => "g_computation",
            Self::OrthogonalScore => "orthogonal_score",
        }
    }

    /// Whether `inference` is one this recipe names.
    #[must_use]
    pub const fn supports(self, inference: InferenceSelection) -> bool {
        self.capabilities(inference).is_some()
    }

    /// Recipe of this kind naming the component's resolved inference.
    ///
    /// The study default is read only when the component declares none.
    ///
    /// # Errors
    ///
    /// [`RecipeError::UnsupportedRealization`] when this kind does not name the
    /// resolved inference. The component is not relabeled to a supported one.
    #[must_use = "an unsupported component inference is refused"]
    pub const fn for_component(
        self,
        component: &CoherentComponent<'_>,
        study: StudyDefault,
    ) -> Result<Recipe, RecipeError> {
        Recipe::new(self, component.resolve(study))
    }

    const fn capabilities(
        self,
        inference: InferenceSelection,
    ) -> Option<&'static [ProviderCapability]> {
        match (self, inference) {
            (
                Self::LinearGaussian,
                InferenceSelection::Frequentist | InferenceSelection::GeneralizedBayesian,
            )
            | (Self::FiniteDiscrete | Self::Glm, InferenceSelection::GeneralizedBayesian)
            | (Self::GComputation, InferenceSelection::Frequentist) => Some(ESTIMATING_EQUATIONS),
            (Self::LinearGaussian | Self::Glm, InferenceSelection::Bayesian) => {
                Some(DENSITY_UPDATING)
            }
            (Self::FiniteDiscrete, InferenceSelection::Frequentist) => Some(DISCRETE_FREQUENTIST),
            (Self::FiniteDiscrete, InferenceSelection::Bayesian) => Some(DISCRETE_BAYESIAN),
            (Self::Glm, InferenceSelection::Frequentist) => Some(GLM_FREQUENTIST),
            (Self::GComputation, InferenceSelection::Bayesian) => Some(GCOMP_BAYESIAN),
            (Self::OrthogonalScore, InferenceSelection::Frequentist) => Some(ORTHOGONAL),
            _ => None,
        }
    }

    const fn supported_label(self) -> &'static str {
        match self {
            Self::LinearGaussian | Self::FiniteDiscrete | Self::Glm => {
                "frequentist, bayesian, or generalized_bayesian"
            }
            Self::GComputation => "frequentist or bayesian",
            Self::OrthogonalScore => "frequentist",
        }
    }
}

impl fmt::Display for RecipeKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A recipe and the inference it names.
///
/// [`Self::required_capabilities`] is fixed by the kind and the inference.
/// Construction does not fit the recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Recipe {
    kind: RecipeKind,
    inference: InferenceSelection,
    required: &'static [ProviderCapability],
}

impl Recipe {
    /// Recipe of `kind` that names `inference`.
    ///
    /// # Errors
    ///
    /// [`RecipeError::InferenceNotNamed`] when `inference` is not frequentist,
    /// Bayesian, or generalized Bayesian. [`RecipeError::UnsupportedRealization`]
    /// when this kind does not name that inference. Orthogonal score names
    /// frequentist only. G-computation names frequentist or Bayesian.
    #[must_use = "an unsupported inference is refused"]
    pub const fn new(kind: RecipeKind, inference: InferenceSelection) -> Result<Self, RecipeError> {
        if !is_named_inference(inference) {
            return Err(RecipeError::InferenceNotNamed {
                site: InferenceSite::Recipe,
                next_attempt: InferenceSite::Recipe.next_attempt(),
            });
        }
        let Some(required) = kind.capabilities(inference) else {
            return Err(RecipeError::UnsupportedRealization {
                kind,
                inference,
                supported: kind.supported_label(),
            });
        };
        Ok(Self { kind, inference, required })
    }

    /// Recipe kind.
    #[must_use]
    pub const fn kind(self) -> RecipeKind {
        self.kind
    }

    /// Inference this recipe names.
    #[must_use]
    pub const fn inference(self) -> InferenceSelection {
        self.inference
    }

    /// Capabilities a provider must have declared.
    ///
    /// Linear/Gaussian frequentist and generalized Bayesian, finite-discrete
    /// generalized Bayesian, and frequentist g-computation require estimating
    /// equations. Linear/Gaussian and generalized-linear Bayesian recipes require
    /// density and updating. Finite-discrete frequentist requires exact integration;
    /// its Bayesian recipe also requires density and updating. Frequentist
    /// generalized linear requires a differentiable log density and estimating
    /// equations. Bayesian g-computation requires prediction, density, and
    /// updating from that provider. Orthogonal score requires estimating
    /// equations and influence contributions.
    #[must_use]
    pub const fn required_capabilities(self) -> &'static [ProviderCapability] {
        self.required
    }

    /// Whether this inference is an ordinary likelihood.
    ///
    /// Bayesian recipes are. Frequentist and generalized-Bayesian recipes are not.
    #[must_use]
    pub const fn uses_likelihood(self) -> bool {
        matches!(self.inference, InferenceSelection::Bayesian)
    }

    /// Accept `provider` when it declares every required capability and answers
    /// each log-density or update call this recipe names.
    ///
    /// A Bayesian recipe refuses a prediction-only provider before the
    /// capability list is walked. The declaration check is not the probe. The
    /// probe then asks the body for each required log density or update. It
    /// does not fit a model.
    ///
    /// # Errors
    ///
    /// [`RecipeError::PredictionOnlyAsLikelihood`] when this recipe is a
    /// likelihood and the provider declares only prediction.
    /// [`RecipeError::MissingCapability`] when a required capability is absent.
    /// [`RecipeError::UnansweredCall`] when the body does not answer a required
    /// log density or update, including when the declaration listed it.
    #[must_use = "a provider that lacks the recipe's capabilities is refused"]
    pub fn accept_provider(
        &self,
        provider: &ProviderDeclaration,
    ) -> Result<AcceptedProvider, RecipeError> {
        if self.uses_likelihood() && prediction_only(provider) {
            return Err(prediction_only_likelihood(self.kind, provider));
        }
        require_capabilities(provider, self.required)?;
        Ok(AcceptedProvider { provider: provider.id(), recipe: *self })
    }

    /// Accept `provider` as this recipe's likelihood.
    ///
    /// # Errors
    ///
    /// [`RecipeError::NotALikelihood`] when the recipe is frequentist or
    /// generalized Bayesian. [`RecipeError::PredictionOnlyAsLikelihood`] when
    /// the provider declares only prediction. [`RecipeError::MissingCapability`]
    /// when a required capability is absent. [`RecipeError::UnansweredCall`]
    /// when the body does not answer a required log density or update.
    #[must_use = "a prediction-only provider is not a likelihood"]
    pub fn accept_as_likelihood(
        &self,
        provider: &ProviderDeclaration,
    ) -> Result<LikelihoodRole, RecipeError> {
        if !self.uses_likelihood() {
            return Err(RecipeError::NotALikelihood { kind: self.kind, inference: self.inference });
        }
        let accepted = self.accept_provider(provider)?;
        Ok(LikelihoodRole { provider: accepted.provider(), recipe: *self })
    }

    /// Attach the prediction boundary for a nuisance or a plug-in mean.
    ///
    /// Orthogonal score attaches a nuisance. Frequentist g-computation attaches
    /// the plug-in conditional mean. Both stay prediction-only. A Bayesian
    /// recipe refuses the boundary as a likelihood. Any other recipe refuses
    /// it as the statistical model.
    ///
    /// # Errors
    ///
    /// [`RecipeError::PredictionBoundaryAsLikelihood`] for a Bayesian recipe.
    /// [`RecipeError::PredictionBoundaryRefused`] when this recipe does not
    /// attach a prediction. [`RecipeError::EmptyRows`] or
    /// [`RecipeError::DuplicateRow`] when `rows` is empty or repeats an index.
    #[must_use = "the prediction boundary is not a likelihood"]
    pub fn attach_prediction_boundary<'a>(
        &self,
        rows: &'a [u64],
    ) -> Result<PredictionAttachment<'a>, RecipeError> {
        let use_as = self.boundary_use()?;
        checked_rows(PREDICTION_BOUNDARY_LABEL, rows)?;
        Ok(PredictionAttachment { use_as, rows })
    }

    /// Bind this recipe to `component` when their inferences agree.
    ///
    /// Agreement uses [`CoherentComponent::resolve`]: a declared component
    /// inference wins over `study`. The provider must pass [`Self::accept_provider`].
    ///
    /// # Errors
    ///
    /// [`RecipeError::InferenceMismatch`] when the resolved inference is not the
    /// one this recipe names. Provider errors are those of [`Self::accept_provider`].
    #[must_use = "a component inference that the recipe does not name is refused"]
    pub fn bind<'a>(
        &self,
        component: &'a CoherentComponent<'a>,
        study: StudyDefault,
        provider: &ProviderDeclaration,
    ) -> Result<BoundRecipe<'a>, RecipeError> {
        let resolved = component.resolve(study);
        if resolved != self.inference {
            let next_attempt = if component.declared().is_some() {
                SELECT_DECLARED_INFERENCE
            } else {
                SELECT_STUDY_DEFAULT
            };
            return Err(RecipeError::InferenceMismatch {
                component: component.name().to_owned(),
                declared: component.declared(),
                study_default: study.inference(),
                resolved,
                kind: self.kind,
                recipe_inference: self.inference,
                next_attempt,
            });
        }
        let accepted = self.accept_provider(provider)?;
        Ok(BoundRecipe {
            recipe: *self,
            provider: accepted.provider(),
            component: component.name(),
            rows: component.rows(),
        })
    }

    fn boundary_use(&self) -> Result<BoundaryUse, RecipeError> {
        match (self.kind, self.inference) {
            (RecipeKind::OrthogonalScore, InferenceSelection::Frequentist) => {
                Ok(BoundaryUse::Nuisance)
            }
            (RecipeKind::GComputation, InferenceSelection::Frequentist) => {
                Ok(BoundaryUse::PlugInMean)
            }
            _ if self.uses_likelihood() => Err(RecipeError::PredictionBoundaryAsLikelihood {
                capability: ProviderCapability::Prediction,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }),
            _ => Err(RecipeError::PredictionBoundaryRefused {
                kind: self.kind,
                inference: self.inference,
                next_attempt: BIND_MODEL_PROVIDER,
            }),
        }
    }
}

/// Study inference used when a component declares none.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StudyDefault {
    inference: InferenceSelection,
}

impl StudyDefault {
    /// Study default `inference`.
    ///
    /// # Errors
    ///
    /// [`RecipeError::InferenceNotNamed`] when `inference` is not frequentist,
    /// Bayesian, or generalized Bayesian.
    #[must_use = "an unnamed study default is refused"]
    pub const fn new(inference: InferenceSelection) -> Result<Self, RecipeError> {
        if !is_named_inference(inference) {
            return Err(RecipeError::InferenceNotNamed {
                site: InferenceSite::StudyDefault,
                next_attempt: InferenceSite::StudyDefault.next_attempt(),
            });
        }
        Ok(Self { inference })
    }

    /// The default.
    #[must_use]
    pub const fn inference(self) -> InferenceSelection {
        self.inference
    }
}

/// One coherent statistical component and the rows it uses.
///
/// `rows` are physical row indices. Duplicate indices are refused. The slice
/// is stored as given.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CoherentComponent<'a> {
    name: &'a str,
    inference: Option<InferenceSelection>,
    rows: &'a [u64],
}

impl<'a> CoherentComponent<'a> {
    /// Component `name`, an optional declared inference, and physical `rows`.
    ///
    /// `None` means the study default applies. `Some` is used as declared.
    ///
    /// # Errors
    ///
    /// [`RecipeError::BlankComponentName`] when `name` is empty or whitespace.
    /// [`RecipeError::InferenceNotNamed`] when the declaration is not a named
    /// inference. [`RecipeError::EmptyRows`] when `rows` is empty.
    /// [`RecipeError::DuplicateRow`] when an index repeats.
    #[must_use = "a blank name, unnamed inference, or illegal row selection is refused"]
    pub fn new(
        name: &'a str,
        inference: Option<InferenceSelection>,
        rows: &'a [u64],
    ) -> Result<Self, RecipeError> {
        if is_blank(name) {
            return Err(RecipeError::BlankComponentName);
        }
        if let Some(inference) = inference {
            named(inference, InferenceSite::Component)?;
        }
        checked_rows(name, rows)?;
        Ok(Self { name, inference, rows })
    }

    /// Component name.
    #[must_use]
    pub const fn name(self) -> &'a str {
        self.name
    }

    /// Declared inference, if the component has one.
    #[must_use]
    pub const fn declared(self) -> Option<InferenceSelection> {
        self.inference
    }

    /// Physical row indices, in declaration order.
    #[must_use]
    pub const fn rows(self) -> &'a [u64] {
        self.rows
    }

    /// This component's inference.
    ///
    /// A declaration is returned unchanged. The study default is returned only
    /// when [`Self::declared`] is `None`.
    #[must_use]
    pub const fn resolve(self, study: StudyDefault) -> InferenceSelection {
        match self.inference {
            Some(inference) => inference,
            None => study.inference,
        }
    }
}

/// Prediction contract of this crate.
///
/// The only capability is [`ProviderCapability::Prediction`]. Likelihood use
/// is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PredictionOnlyBoundary {
    capability: ProviderCapability,
}

impl PredictionOnlyBoundary {
    /// The prediction boundary.
    #[must_use]
    pub const fn new() -> Self {
        Self { capability: ProviderCapability::Prediction }
    }

    /// The only capability. Always prediction.
    #[must_use]
    pub const fn capability(self) -> ProviderCapability {
        self.capability
    }

    /// Refuse this boundary as a likelihood.
    ///
    /// # Errors
    ///
    /// Always [`RecipeError::PredictionBoundaryAsLikelihood`].
    #[must_use = "the prediction boundary is not a likelihood"]
    pub const fn as_likelihood(self) -> Result<LikelihoodRole, RecipeError> {
        Err(RecipeError::PredictionBoundaryAsLikelihood {
            capability: self.capability(),
            next_attempt: KEEP_PREDICTION_BOUNDARY,
        })
    }
}

impl Default for PredictionOnlyBoundary {
    fn default() -> Self {
        Self::new()
    }
}

/// How a prediction attachment is used. Neither variant is a likelihood.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BoundaryUse {
    /// Cross-fit nuisance for an orthogonal score.
    Nuisance,
    /// Plug-in conditional mean for frequentist g-computation.
    PlugInMean,
}

/// Provider accepted for a recipe.
///
/// The probe may have asked the body to answer a log density or an update.
/// No model was fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AcceptedProvider {
    provider: ProviderId,
    recipe: Recipe,
}

impl AcceptedProvider {
    /// Accepted provider.
    #[must_use]
    pub const fn provider(self) -> ProviderId {
        self.provider
    }

    /// Recipe the provider was accepted for.
    #[must_use]
    pub const fn recipe(self) -> Recipe {
        self.recipe
    }
}

/// Likelihood role of a Bayesian recipe's provider.
///
/// Produced only by [`Recipe::accept_as_likelihood`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LikelihoodRole {
    provider: ProviderId,
    recipe: Recipe,
}

impl LikelihoodRole {
    /// Provider accepted as the likelihood.
    #[must_use]
    pub const fn provider(self) -> ProviderId {
        self.provider
    }

    /// Bayesian recipe.
    #[must_use]
    pub const fn recipe(self) -> Recipe {
        self.recipe
    }
}

/// Recipe bound to one component and one accepted provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundRecipe<'a> {
    recipe: Recipe,
    provider: ProviderId,
    component: &'a str,
    rows: &'a [u64],
}

impl<'a> BoundRecipe<'a> {
    /// Bound recipe.
    #[must_use]
    pub const fn recipe(self) -> Recipe {
        self.recipe
    }

    /// Inference the recipe names. This is the component's resolved inference.
    #[must_use]
    pub const fn inference(self) -> InferenceSelection {
        self.recipe.inference()
    }

    /// Accepted provider.
    #[must_use]
    pub const fn provider(self) -> ProviderId {
        self.provider
    }

    /// Component name.
    #[must_use]
    pub const fn component(self) -> &'a str {
        self.component
    }

    /// Physical row indices, borrowed from the component.
    #[must_use]
    pub const fn rows(self) -> &'a [u64] {
        self.rows
    }
}

/// Prediction-boundary attachment. [`Self::as_likelihood`] is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PredictionAttachment<'a> {
    use_as: BoundaryUse,
    rows: &'a [u64],
}

impl<'a> PredictionAttachment<'a> {
    /// Nuisance or plug-in mean.
    #[must_use]
    pub const fn use_as(self) -> BoundaryUse {
        self.use_as
    }

    /// Always prediction.
    #[must_use]
    pub const fn role(self) -> ProviderCapability {
        match self.use_as {
            BoundaryUse::Nuisance | BoundaryUse::PlugInMean => ProviderCapability::Prediction,
        }
    }

    /// Physical row indices the attachment was given.
    #[must_use]
    pub const fn rows(self) -> &'a [u64] {
        self.rows
    }

    /// Refuse the attachment as a likelihood.
    ///
    /// # Errors
    ///
    /// Always [`RecipeError::PredictionBoundaryAsLikelihood`].
    #[must_use = "a prediction attachment is not a likelihood"]
    pub const fn as_likelihood(self) -> Result<LikelihoodRole, RecipeError> {
        Err(RecipeError::PredictionBoundaryAsLikelihood {
            capability: self.role(),
            next_attempt: KEEP_PREDICTION_BOUNDARY,
        })
    }
}

/// Why a recipe, component, provider, or prediction boundary was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecipeError {
    /// The component name was empty or whitespace.
    BlankComponentName,
    /// The row selection was empty.
    EmptyRows {
        /// Component name, or the prediction-boundary label.
        label: String,
    },
    /// A physical row index appeared twice.
    DuplicateRow {
        /// Component name, or the prediction-boundary label.
        label: String,
        /// Index that repeated.
        row: u64,
    },
    /// The selection was not frequentist, Bayesian, or generalized Bayesian.
    InferenceNotNamed {
        /// Where the selection was required.
        site: InferenceSite,
        /// [`InferenceSite`] next attempt.
        next_attempt: &'static str,
    },
    /// This recipe does not name that inference.
    UnsupportedRealization {
        /// Recipe kind.
        kind: RecipeKind,
        /// Inference that was asked for.
        inference: InferenceSelection,
        /// Inferences this kind does name.
        supported: &'static str,
    },
    /// The component's resolved inference is not the recipe's inference.
    InferenceMismatch {
        /// Component name.
        component: String,
        /// Declaration. `None` when the study default was the selection.
        declared: Option<InferenceSelection>,
        /// Study default that was offered.
        study_default: InferenceSelection,
        /// Inference [`CoherentComponent::resolve`] returned.
        resolved: InferenceSelection,
        /// Recipe kind.
        kind: RecipeKind,
        /// Inference the recipe names.
        recipe_inference: InferenceSelection,
        /// Next attempt.
        next_attempt: &'static str,
    },
    /// A prediction-only provider was offered as a likelihood.
    PredictionOnlyAsLikelihood {
        /// Provider id.
        provider: ProviderId,
        /// Provider name. Not a capability.
        provider_name: String,
        /// Bayesian recipe that refused it.
        kind: RecipeKind,
        /// [`KEEP_PREDICTION_BOUNDARY`].
        next_attempt: &'static str,
    },
    /// The prediction boundary was offered as a likelihood.
    PredictionBoundaryAsLikelihood {
        /// Capability the boundary has. Prediction.
        capability: ProviderCapability,
        /// [`KEEP_PREDICTION_BOUNDARY`].
        next_attempt: &'static str,
    },
    /// The body did not answer a log density or an update the recipe names.
    ///
    /// The declaration may already list that capability. The list is not the call.
    UnansweredCall {
        /// Provider id.
        provider: ProviderId,
        /// Provider name. Not a capability.
        provider_name: String,
        /// Log density or update the probe asked for.
        capability: ProviderCapability,
        /// [`ANSWER_CAPABILITY_CALL`].
        next_attempt: &'static str,
    },
    /// A required capability was not declared.
    MissingCapability {
        /// Provider id.
        provider: ProviderId,
        /// Provider name. Not a capability.
        provider_name: String,
        /// First required capability the declaration does not contain.
        capability: ProviderCapability,
        /// Earliest capability the declaration does contain, if it contains any.
        legal_next: Option<ProviderCapability>,
        /// [`DECLARE_REQUIRED_CAPABILITIES`].
        next_attempt: &'static str,
    },
    /// A frequentist or generalized-Bayesian recipe was asked to be a likelihood.
    NotALikelihood {
        /// Recipe kind.
        kind: RecipeKind,
        /// Inference the recipe names.
        inference: InferenceSelection,
    },
    /// The prediction boundary was offered as the statistical model.
    PredictionBoundaryRefused {
        /// Recipe kind.
        kind: RecipeKind,
        /// Inference the recipe names.
        inference: InferenceSelection,
        /// [`BIND_MODEL_PROVIDER`].
        next_attempt: &'static str,
    },
}

impl fmt::Display for RecipeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlankComponentName => formatter.write_str("component name is blank"),
            Self::EmptyRows { label } => write!(formatter, "{label} has no rows"),
            Self::DuplicateRow { label, row } => write!(formatter, "{label} repeats row {row}"),
            Self::InferenceNotNamed { site, next_attempt } => {
                write!(formatter, "{site} must name frequentist, bayesian,")?;
                write!(formatter, " or generalized Bayesian")?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::UnsupportedRealization { kind, inference, supported } => {
                write!(formatter, "recipe {kind} does not name ")?;
                write!(formatter, "{}", inference_label(*inference))?;
                write!(formatter, "; it names {supported}")
            }
            Self::InferenceMismatch {
                component,
                declared,
                study_default,
                resolved,
                kind,
                recipe_inference,
                next_attempt,
            } => {
                write!(formatter, "component {component} selects ")?;
                write!(formatter, "{}", inference_label(*resolved))?;
                write!(formatter, "; declared ")?;
                write!(formatter, "{}", option_label(*declared))?;
                write!(formatter, "; study default ")?;
                write!(formatter, "{}", inference_label(*study_default))?;
                write!(formatter, "; recipe {kind} names ")?;
                write!(formatter, "{}", inference_label(*recipe_inference))?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::PredictionOnlyAsLikelihood { provider, provider_name, kind, next_attempt } => {
                write!(formatter, "provider {provider} ({provider_name}) is prediction-only")?;
                write!(formatter, " and is not a likelihood for {kind}")?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::PredictionBoundaryAsLikelihood { capability, next_attempt } => {
                write!(formatter, "prediction-only boundary ({capability}) is not a likelihood")?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::UnansweredCall { provider, provider_name, capability, next_attempt } => {
                write!(formatter, "provider {provider} ({provider_name})")?;
                write!(formatter, " does not answer {capability}; ")?;
                write!(formatter, "a declaration is not that call")?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::MissingCapability {
                provider,
                provider_name,
                capability,
                legal_next,
                next_attempt,
            } => {
                write!(formatter, "provider {provider} ({provider_name})")?;
                write!(formatter, " does not declare {capability}; ")?;
                write_legal_next(formatter, *legal_next)?;
                write!(formatter, "; next attempt: {next_attempt}")
            }
            Self::NotALikelihood { kind, inference } => {
                write!(formatter, "recipe {kind} names ")?;
                write!(formatter, "{}", inference_label(*inference))?;
                formatter.write_str(" and is not a likelihood")
            }
            Self::PredictionBoundaryRefused { kind, inference, next_attempt } => {
                write!(formatter, "prediction-only boundary is not the model for {kind} (")?;
                write!(formatter, "{}", inference_label(*inference))?;
                write!(formatter, "); next attempt: {next_attempt}")
            }
        }
    }
}

impl std::error::Error for RecipeError {}

const fn is_named_inference(inference: InferenceSelection) -> bool {
    matches!(
        inference,
        InferenceSelection::Frequentist
            | InferenceSelection::Bayesian
            | InferenceSelection::GeneralizedBayesian
    )
}

const fn named(inference: InferenceSelection, site: InferenceSite) -> Result<(), RecipeError> {
    if is_named_inference(inference) {
        Ok(())
    } else {
        Err(RecipeError::InferenceNotNamed { site, next_attempt: site.next_attempt() })
    }
}

const fn inference_label(inference: InferenceSelection) -> &'static str {
    match inference {
        InferenceSelection::Unspecified => "unspecified",
        InferenceSelection::Frequentist => "frequentist",
        InferenceSelection::Bayesian => "bayesian",
        InferenceSelection::GeneralizedBayesian => "generalized_bayesian",
        _ => "unnamed",
    }
}

const fn option_label(inference: Option<InferenceSelection>) -> &'static str {
    match inference {
        Some(inference) => inference_label(inference),
        None => "none",
    }
}

const fn prediction_only(provider: &ProviderDeclaration) -> bool {
    let capabilities = provider.capabilities();
    capabilities.len() == 1 && capabilities.contains(ProviderCapability::Prediction)
}

fn prediction_only_likelihood(kind: RecipeKind, provider: &ProviderDeclaration) -> RecipeError {
    RecipeError::PredictionOnlyAsLikelihood {
        provider: provider.id(),
        provider_name: provider.name().to_owned(),
        kind,
        next_attempt: KEEP_PREDICTION_BOUNDARY,
    }
}

const fn is_log_density_or_update(capability: ProviderCapability) -> bool {
    matches!(
        capability,
        ProviderCapability::Density
            | ProviderCapability::DifferentiableLogDensity
            | ProviderCapability::Updating
    )
}

/// Check every required capability against the declaration, and each required
/// log density or update against the body as well.
///
/// A log density or update is compiled with
/// [`ProviderDeclaration::compile_verified`], so a declared call the body does
/// not supply is refused. Other capabilities have no body call to probe, so the
/// declaration is the only evidence and the plain compile is the check.
fn require_capabilities(
    provider: &ProviderDeclaration,
    required: &[ProviderCapability],
) -> Result<(), RecipeError> {
    let missing = |capability: ProviderCapability, failed: FailedCapabilityAttempt| {
        RecipeError::MissingCapability {
            provider: failed.provider(),
            provider_name: failed.provider_name().to_owned(),
            capability,
            legal_next: failed.legal_next(),
            next_attempt: DECLARE_REQUIRED_CAPABILITIES,
        }
    };
    // Every declaration is checked before any call is probed, so a missing declaration is
    // reported ahead of an unanswered call.
    for &capability in required {
        provider
            .compile(CapabilityRequest::new(capability))
            .map_err(|failed| missing(capability, failed))?;
    }
    for &capability in required.iter().filter(|&&capability| is_log_density_or_update(capability)) {
        if provider.compile_verified(CapabilityRequest::new(capability)).is_err() {
            return Err(RecipeError::UnansweredCall {
                provider: provider.id(),
                provider_name: provider.name().to_owned(),
                capability,
                next_attempt: ANSWER_CAPABILITY_CALL,
            });
        }
    }
    Ok(())
}

fn checked_rows(label: &str, rows: &[u64]) -> Result<(), RecipeError> {
    if rows.is_empty() {
        return Err(RecipeError::EmptyRows { label: label.to_owned() });
    }
    let mut seen = BTreeSet::new();
    for &row in rows {
        if !seen.insert(row) {
            return Err(RecipeError::DuplicateRow { label: label.to_owned(), row });
        }
    }
    Ok(())
}

fn write_legal_next(
    formatter: &mut fmt::Formatter<'_>,
    legal_next: Option<ProviderCapability>,
) -> fmt::Result {
    match legal_next {
        Some(capability) => write!(formatter, "a legal next request is {capability}"),
        None => formatter.write_str("the declaration supports nothing"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANSWER_CAPABILITY_CALL, BIND_MODEL_PROVIDER, BoundaryUse, CoherentComponent,
        DECLARE_REQUIRED_CAPABILITIES, InferenceSite, KEEP_PREDICTION_BOUNDARY,
        NAME_COMPONENT_INFERENCE, NAME_RECIPE_INFERENCE, NAME_STUDY_DEFAULT,
        PredictionOnlyBoundary, Recipe, RecipeError, RecipeKind, SELECT_DECLARED_INFERENCE,
        SELECT_STUDY_DEFAULT, StudyDefault,
    };
    use antecedent_core::{
        CapabilityRequest, InferenceSelection, ProviderCapability, ProviderDeclaration, ProviderId,
    };

    fn declare(id: u64, name: &str, capabilities: &[ProviderCapability]) -> ProviderDeclaration {
        ProviderDeclaration::new(ProviderId::from_raw(id), name, capabilities.iter().copied())
            .unwrap()
    }

    fn component<'a>(
        name: &'a str,
        inference: Option<InferenceSelection>,
        rows: &'a [u64],
    ) -> CoherentComponent<'a> {
        CoherentComponent::new(name, inference, rows).unwrap()
    }

    #[test]
    fn recipes_name_inference_and_required_capabilities() {
        let cases = [
            (
                RecipeKind::LinearGaussian,
                InferenceSelection::Frequentist,
                &[ProviderCapability::EstimatingEquations][..],
            ),
            (
                RecipeKind::LinearGaussian,
                InferenceSelection::Bayesian,
                &[ProviderCapability::Density, ProviderCapability::Updating][..],
            ),
            (
                RecipeKind::LinearGaussian,
                InferenceSelection::GeneralizedBayesian,
                &[ProviderCapability::EstimatingEquations][..],
            ),
            (
                RecipeKind::FiniteDiscrete,
                InferenceSelection::Frequentist,
                &[ProviderCapability::ExactIntegration][..],
            ),
            (
                RecipeKind::FiniteDiscrete,
                InferenceSelection::Bayesian,
                &[
                    ProviderCapability::Density,
                    ProviderCapability::Updating,
                    ProviderCapability::ExactIntegration,
                ][..],
            ),
            (
                RecipeKind::FiniteDiscrete,
                InferenceSelection::GeneralizedBayesian,
                &[ProviderCapability::EstimatingEquations][..],
            ),
            (
                RecipeKind::Glm,
                InferenceSelection::Frequentist,
                &[
                    ProviderCapability::DifferentiableLogDensity,
                    ProviderCapability::EstimatingEquations,
                ][..],
            ),
            (
                RecipeKind::Glm,
                InferenceSelection::Bayesian,
                &[ProviderCapability::Density, ProviderCapability::Updating][..],
            ),
            (
                RecipeKind::Glm,
                InferenceSelection::GeneralizedBayesian,
                &[ProviderCapability::EstimatingEquations][..],
            ),
            (
                RecipeKind::GComputation,
                InferenceSelection::Frequentist,
                &[ProviderCapability::EstimatingEquations][..],
            ),
            (
                RecipeKind::GComputation,
                InferenceSelection::Bayesian,
                &[
                    ProviderCapability::Prediction,
                    ProviderCapability::Density,
                    ProviderCapability::Updating,
                ][..],
            ),
            (
                RecipeKind::OrthogonalScore,
                InferenceSelection::Frequentist,
                &[
                    ProviderCapability::EstimatingEquations,
                    ProviderCapability::InfluenceContributions,
                ][..],
            ),
        ];
        for (kind, inference, capabilities) in cases {
            let recipe = Recipe::new(kind, inference).unwrap();
            assert_eq!(recipe.kind(), kind);
            assert_eq!(recipe.inference(), inference);
            assert_eq!(recipe.required_capabilities(), capabilities);
            assert_eq!(recipe.uses_likelihood(), inference == InferenceSelection::Bayesian);
        }
    }

    #[test]
    fn unnamed_and_unsupported_inference_is_refused() {
        assert_eq!(RecipeKind::LinearGaussian.as_str(), "linear_gaussian");
        assert_eq!(RecipeKind::FiniteDiscrete.as_str(), "finite_discrete");
        assert_eq!(RecipeKind::Glm.as_str(), "glm");
        assert_eq!(RecipeKind::GComputation.as_str(), "g_computation");
        assert_eq!(RecipeKind::OrthogonalScore.as_str(), "orthogonal_score");
        for kind in RecipeKind::ALL {
            for inference in [
                InferenceSelection::Unspecified,
                InferenceSelection::Frequentist,
                InferenceSelection::Bayesian,
                InferenceSelection::GeneralizedBayesian,
            ] {
                assert_eq!(kind.supports(inference), Recipe::new(kind, inference).is_ok());
            }
        }
        assert_eq!(
            Recipe::new(RecipeKind::OrthogonalScore, InferenceSelection::Bayesian).unwrap_err(),
            RecipeError::UnsupportedRealization {
                kind: RecipeKind::OrthogonalScore,
                inference: InferenceSelection::Bayesian,
                supported: "frequentist",
            }
        );
        assert_eq!(
            Recipe::new(RecipeKind::GComputation, InferenceSelection::GeneralizedBayesian)
                .unwrap_err(),
            RecipeError::UnsupportedRealization {
                kind: RecipeKind::GComputation,
                inference: InferenceSelection::GeneralizedBayesian,
                supported: "frequentist or bayesian",
            }
        );
        assert_eq!(
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Unspecified).unwrap_err(),
            RecipeError::InferenceNotNamed {
                site: InferenceSite::Recipe,
                next_attempt: NAME_RECIPE_INFERENCE,
            }
        );
    }

    #[test]
    fn study_default_fills_only_an_omitted_component_inference() {
        let study = StudyDefault::new(InferenceSelection::Frequentist).unwrap();
        let bayesian_study = StudyDefault::new(InferenceSelection::Bayesian).unwrap();
        assert_eq!(
            StudyDefault::new(InferenceSelection::Unspecified).unwrap_err(),
            RecipeError::InferenceNotNamed {
                site: InferenceSite::StudyDefault,
                next_attempt: NAME_STUDY_DEFAULT,
            }
        );
        assert_eq!(
            CoherentComponent::new("y", Some(InferenceSelection::Unspecified), &[1]).unwrap_err(),
            RecipeError::InferenceNotNamed {
                site: InferenceSite::Component,
                next_attempt: NAME_COMPONENT_INFERENCE,
            }
        );

        let declared = component("outcome", Some(InferenceSelection::Bayesian), &[0, 1]);
        let inheriting = component("outcome", None, &[0, 1]);
        assert_eq!(declared.resolve(study), InferenceSelection::Bayesian);
        assert_eq!(declared.resolve(bayesian_study), InferenceSelection::Bayesian);
        assert_eq!(inheriting.resolve(study), InferenceSelection::Frequentist);
        assert_eq!(inheriting.resolve(bayesian_study), InferenceSelection::Bayesian);
        assert_eq!(declared.declared(), Some(InferenceSelection::Bayesian));
        assert_eq!(inheriting.declared(), None);

        let bayes = Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Bayesian).unwrap();
        let frequentist =
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Frequentist).unwrap();
        let likelihood = declare(
            1,
            "gaussian",
            &[
                ProviderCapability::Density,
                ProviderCapability::Updating,
                ProviderCapability::Sampling,
            ],
        )
        .answering([ProviderCapability::Density, ProviderCapability::Updating]);
        let equations = declare(2, "ols", &[ProviderCapability::EstimatingEquations]);

        let bound = bayes.bind(&declared, study, &likelihood).unwrap();
        assert_eq!(bound.inference(), InferenceSelection::Bayesian);
        assert_eq!(bound.provider(), ProviderId::from_raw(1));
        assert_eq!(
            frequentist.bind(&declared, study, &equations).unwrap_err(),
            RecipeError::InferenceMismatch {
                component: "outcome".to_owned(),
                declared: Some(InferenceSelection::Bayesian),
                study_default: InferenceSelection::Frequentist,
                resolved: InferenceSelection::Bayesian,
                kind: RecipeKind::LinearGaussian,
                recipe_inference: InferenceSelection::Frequentist,
                next_attempt: SELECT_DECLARED_INFERENCE,
            }
        );
        let inherited = frequentist.bind(&inheriting, study, &equations).unwrap();
        assert_eq!(inherited.inference(), InferenceSelection::Frequentist);
        assert_eq!(
            bayes.bind(&inheriting, study, &likelihood).unwrap_err(),
            RecipeError::InferenceMismatch {
                component: "outcome".to_owned(),
                declared: None,
                study_default: InferenceSelection::Frequentist,
                resolved: InferenceSelection::Frequentist,
                kind: RecipeKind::LinearGaussian,
                recipe_inference: InferenceSelection::Bayesian,
                next_attempt: SELECT_STUDY_DEFAULT,
            }
        );

        let score = component("nuisance", Some(InferenceSelection::Bayesian), &[4]);
        assert_eq!(
            RecipeKind::OrthogonalScore.for_component(&score, study).unwrap_err(),
            RecipeError::UnsupportedRealization {
                kind: RecipeKind::OrthogonalScore,
                inference: InferenceSelection::Bayesian,
                supported: "frequentist",
            }
        );
        let open = component("nuisance", None, &[4, 5]);
        let orthogonal = RecipeKind::OrthogonalScore.for_component(&open, study).unwrap();
        assert_eq!(orthogonal.inference(), InferenceSelection::Frequentist);
        assert_eq!(orthogonal.kind(), RecipeKind::OrthogonalScore);
    }

    #[test]
    fn prediction_only_provider_is_refused_as_a_likelihood() {
        let bayes = Recipe::new(RecipeKind::FiniteDiscrete, InferenceSelection::Bayesian).unwrap();
        let named_likelihood = declare(7, "likelihood", &[ProviderCapability::Prediction]);
        assert_eq!(
            bayes.accept_as_likelihood(&named_likelihood).unwrap_err(),
            RecipeError::PredictionOnlyAsLikelihood {
                provider: ProviderId::from_raw(7),
                provider_name: "likelihood".to_owned(),
                kind: RecipeKind::FiniteDiscrete,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }
        );
        assert_eq!(
            bayes.accept_provider(&named_likelihood).unwrap_err(),
            RecipeError::PredictionOnlyAsLikelihood {
                provider: ProviderId::from_raw(7),
                provider_name: "likelihood".to_owned(),
                kind: RecipeKind::FiniteDiscrete,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }
        );

        let smooth = declare(
            8,
            "prediction",
            &[
                ProviderCapability::Density,
                ProviderCapability::Updating,
                ProviderCapability::ExactIntegration,
            ],
        )
        .answering([
            ProviderCapability::Density,
            ProviderCapability::Updating,
            ProviderCapability::ExactIntegration,
        ]);
        let role = bayes.accept_as_likelihood(&smooth).unwrap();
        assert_eq!(role.provider(), ProviderId::from_raw(8));
        assert_eq!(role.recipe().kind(), RecipeKind::FiniteDiscrete);
        assert!(role.recipe().uses_likelihood());

        let gradient = declare(9, "gradient", &[ProviderCapability::DifferentiableLogDensity]);
        assert_eq!(
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Bayesian)
                .unwrap()
                .accept_as_likelihood(&gradient)
                .unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(9),
                provider_name: "gradient".to_owned(),
                capability: ProviderCapability::Density,
                legal_next: Some(ProviderCapability::DifferentiableLogDensity),
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );
    }

    #[test]
    fn prediction_boundary_stays_prediction_only() {
        let boundary = PredictionOnlyBoundary::new();
        assert_eq!(boundary.capability(), ProviderCapability::Prediction);
        assert_eq!(PredictionOnlyBoundary::default().capability(), ProviderCapability::Prediction);
        assert_eq!(
            boundary.as_likelihood().unwrap_err(),
            RecipeError::PredictionBoundaryAsLikelihood {
                capability: ProviderCapability::Prediction,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }
        );

        let rows = [3u64, 1, 9];
        let orthogonal =
            Recipe::new(RecipeKind::OrthogonalScore, InferenceSelection::Frequentist).unwrap();
        let nuisance = orthogonal.attach_prediction_boundary(&rows).unwrap();
        assert_eq!(nuisance.use_as(), BoundaryUse::Nuisance);
        assert_eq!(nuisance.role(), ProviderCapability::Prediction);
        assert!(std::ptr::eq(nuisance.rows(), rows.as_slice()));
        assert_eq!(
            nuisance.as_likelihood().unwrap_err(),
            RecipeError::PredictionBoundaryAsLikelihood {
                capability: ProviderCapability::Prediction,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }
        );

        let gcomp = Recipe::new(RecipeKind::GComputation, InferenceSelection::Frequentist).unwrap();
        let plug_in = gcomp.attach_prediction_boundary(&rows).unwrap();
        assert_eq!(plug_in.use_as(), BoundaryUse::PlugInMean);
        assert!(plug_in.as_likelihood().is_err());

        let bayesian_gcomp =
            Recipe::new(RecipeKind::GComputation, InferenceSelection::Bayesian).unwrap();
        assert_eq!(
            bayesian_gcomp.attach_prediction_boundary(&rows).unwrap_err(),
            RecipeError::PredictionBoundaryAsLikelihood {
                capability: ProviderCapability::Prediction,
                next_attempt: KEEP_PREDICTION_BOUNDARY,
            }
        );
        let linear =
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Frequentist).unwrap();
        assert_eq!(
            linear.attach_prediction_boundary(&rows).unwrap_err(),
            RecipeError::PredictionBoundaryRefused {
                kind: RecipeKind::LinearGaussian,
                inference: InferenceSelection::Frequentist,
                next_attempt: BIND_MODEL_PROVIDER,
            }
        );
    }

    #[test]
    fn declared_capabilities_are_required_and_names_are_not() {
        let glm = Recipe::new(RecipeKind::Glm, InferenceSelection::Frequentist).unwrap();
        let by_name = declare(4, "differentiable_log_density", &[]);
        assert_eq!(
            glm.accept_provider(&by_name).unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(4),
                provider_name: "differentiable_log_density".to_owned(),
                capability: ProviderCapability::DifferentiableLogDensity,
                legal_next: None,
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );
        let score_only = declare(5, "score", &[ProviderCapability::EstimatingEquations]);
        assert_eq!(
            glm.accept_provider(&score_only).unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(5),
                provider_name: "score".to_owned(),
                capability: ProviderCapability::DifferentiableLogDensity,
                legal_next: Some(ProviderCapability::EstimatingEquations),
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );
        let glm_provider = declare(
            6,
            "glm",
            &[
                ProviderCapability::Sampling,
                ProviderCapability::DifferentiableLogDensity,
                ProviderCapability::EstimatingEquations,
            ],
        )
        .answering([ProviderCapability::DifferentiableLogDensity]);
        assert_eq!(glm.accept_provider(&glm_provider).unwrap().provider(), ProviderId::from_raw(6));

        let discrete =
            Recipe::new(RecipeKind::FiniteDiscrete, InferenceSelection::Bayesian).unwrap();
        let no_integral =
            declare(10, "dirichlet", &[ProviderCapability::Density, ProviderCapability::Updating]);
        assert_eq!(
            discrete.accept_provider(&no_integral).unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(10),
                provider_name: "dirichlet".to_owned(),
                capability: ProviderCapability::ExactIntegration,
                legal_next: Some(ProviderCapability::Density),
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );

        let gcomp = Recipe::new(RecipeKind::GComputation, InferenceSelection::Bayesian).unwrap();
        let partial =
            declare(11, "partial", &[ProviderCapability::Prediction, ProviderCapability::Density]);
        assert_eq!(
            gcomp.accept_provider(&partial).unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(11),
                provider_name: "partial".to_owned(),
                capability: ProviderCapability::Updating,
                legal_next: Some(ProviderCapability::Prediction),
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );
    }

    #[test]
    fn row_indices_stay_u64() {
        let name = "panel";
        let rows = [2u64, 0, u64::MAX];
        let study = StudyDefault::new(InferenceSelection::Frequentist).unwrap();
        let component = component(name, None, &rows);
        assert_eq!(component.rows(), &rows);
        let recipe =
            Recipe::new(RecipeKind::GComputation, InferenceSelection::Frequentist).unwrap();
        let provider = declare(3, "equations", &[ProviderCapability::EstimatingEquations]);
        let bound = recipe.bind(&component, study, &provider).unwrap();
        assert!(std::ptr::eq(bound.component(), name));
        assert!(std::ptr::eq(bound.rows(), rows.as_slice()));
        assert_eq!(bound.rows(), &[2u64, 0, u64::MAX]);

        let repeated = 1u64 << 40;
        assert_eq!(
            CoherentComponent::new("y", None, &[repeated, 0, repeated]).unwrap_err(),
            RecipeError::DuplicateRow { label: "y".to_owned(), row: repeated }
        );
        assert_eq!(
            CoherentComponent::new("y", None, &[]).unwrap_err(),
            RecipeError::EmptyRows { label: "y".to_owned() }
        );
        assert_eq!(
            CoherentComponent::new(" \t", None, &[1]).unwrap_err(),
            RecipeError::BlankComponentName
        );
        let orthogonal =
            Recipe::new(RecipeKind::OrthogonalScore, InferenceSelection::Frequentist).unwrap();
        assert_eq!(
            orthogonal.attach_prediction_boundary(&[]).unwrap_err(),
            RecipeError::EmptyRows { label: "prediction boundary".to_owned() }
        );
        assert_eq!(
            orthogonal.attach_prediction_boundary(&[repeated, repeated]).unwrap_err(),
            RecipeError::DuplicateRow { label: "prediction boundary".to_owned(), row: repeated }
        );
    }

    #[test]
    fn generalized_bayes_is_not_an_ordinary_likelihood() {
        let recipe =
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::GeneralizedBayesian)
                .unwrap();
        assert!(!recipe.uses_likelihood());
        let prediction = declare(12, "learner", &[ProviderCapability::Prediction]);
        let density =
            declare(13, "density", &[ProviderCapability::Density, ProviderCapability::Updating]);
        assert_eq!(
            recipe.accept_as_likelihood(&prediction).unwrap_err(),
            RecipeError::NotALikelihood {
                kind: RecipeKind::LinearGaussian,
                inference: InferenceSelection::GeneralizedBayesian,
            }
        );
        assert_eq!(
            recipe.accept_as_likelihood(&density).unwrap_err(),
            RecipeError::NotALikelihood {
                kind: RecipeKind::LinearGaussian,
                inference: InferenceSelection::GeneralizedBayesian,
            }
        );
        assert_eq!(
            recipe.accept_provider(&prediction).unwrap_err(),
            RecipeError::MissingCapability {
                provider: ProviderId::from_raw(12),
                provider_name: "learner".to_owned(),
                capability: ProviderCapability::EstimatingEquations,
                legal_next: Some(ProviderCapability::Prediction),
                next_attempt: DECLARE_REQUIRED_CAPABILITIES,
            }
        );
        let loss = declare(14, "squared_error", &[ProviderCapability::EstimatingEquations]);
        assert_eq!(recipe.accept_provider(&loss).unwrap().provider(), ProviderId::from_raw(14));
        assert!(recipe.accept_as_likelihood(&loss).is_err());

        let frequentist =
            Recipe::new(RecipeKind::OrthogonalScore, InferenceSelection::Frequentist).unwrap();
        assert_eq!(
            frequentist.accept_as_likelihood(&density).unwrap_err(),
            RecipeError::NotALikelihood {
                kind: RecipeKind::OrthogonalScore,
                inference: InferenceSelection::Frequentist,
            }
        );
    }

    #[test]
    fn declared_log_density_without_the_call_is_refused() {
        let glm = Recipe::new(RecipeKind::Glm, InferenceSelection::Frequentist).unwrap();
        let declared = declare(
            21,
            "listed",
            &[
                ProviderCapability::DifferentiableLogDensity,
                ProviderCapability::EstimatingEquations,
            ],
        );
        let log_density = CapabilityRequest::new(ProviderCapability::DifferentiableLogDensity);
        assert_eq!(declared.compile(log_density).unwrap().capability(), log_density.capability());
        assert_eq!(
            glm.accept_provider(&declared).unwrap_err(),
            RecipeError::UnansweredCall {
                provider: ProviderId::from_raw(21),
                provider_name: "listed".to_owned(),
                capability: ProviderCapability::DifferentiableLogDensity,
                next_attempt: ANSWER_CAPABILITY_CALL,
            }
        );
        let answering = declared.answering([ProviderCapability::DifferentiableLogDensity]);
        assert_eq!(glm.accept_provider(&answering).unwrap().provider(), ProviderId::from_raw(21));

        let bayes = Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Bayesian).unwrap();
        let likelihood = declare(
            22,
            "likelihood",
            &[
                ProviderCapability::Prediction,
                ProviderCapability::Density,
                ProviderCapability::Updating,
            ],
        );
        assert_eq!(
            bayes.accept_as_likelihood(&likelihood).unwrap_err(),
            RecipeError::UnansweredCall {
                provider: ProviderId::from_raw(22),
                provider_name: "likelihood".to_owned(),
                capability: ProviderCapability::Density,
                next_attempt: ANSWER_CAPABILITY_CALL,
            }
        );
        let density_only = likelihood.clone().answering([ProviderCapability::Density]);
        assert_eq!(
            bayes.accept_provider(&density_only).unwrap_err(),
            RecipeError::UnansweredCall {
                provider: ProviderId::from_raw(22),
                provider_name: "likelihood".to_owned(),
                capability: ProviderCapability::Updating,
                next_attempt: ANSWER_CAPABILITY_CALL,
            }
        );
        let answered =
            likelihood.answering([ProviderCapability::Density, ProviderCapability::Updating]);
        assert_eq!(
            bayes.accept_as_likelihood(&answered).unwrap().provider(),
            ProviderId::from_raw(22)
        );
    }
}
