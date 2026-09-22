//! Identifier / estimator ids, defaults, and pair allowlists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::str::FromStr;
use std::sync::OnceLock;

use antecedent_core::{AssumptionRecord, IdentificationStatus};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_identify::IdentificationResult;

use crate::error::CausalError;

/// Wire names in [`IdentifierId::ALL`] order, derived from [`IdentifierId::as_str`].
fn identifier_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| IdentifierId::ALL.iter().map(IdentifierId::as_str).collect()).as_slice()
}

/// Wire names in [`EstimatorId::ALL`] order, derived from [`EstimatorId::as_str`].
fn estimator_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| EstimatorId::ALL.iter().map(EstimatorId::as_str).collect()).as_slice()
}

/// Parse a closed-set id by walking [`ALL`] / [`as_str`] so a new variant cannot
/// compile into `estimator_data` / `identifier_data` and still miss `FromStr`.
fn parse_closed<T: Copy + PartialEq>(
    s: &str,
    all: &'static [T],
    as_str: fn(&T) -> &'static str,
    kind: &'static str,
    expected: &'static [&'static str],
) -> Result<T, UnknownStrategy> {
    all.iter().copied().find(|id| as_str(id) == s).ok_or_else(|| UnknownStrategy {
        kind,
        got: s.to_string(),
        expected,
    })
}

/// Declare a closed-set id enum together with its `ALL` catalog, both
/// generated from one variant list.
///
/// `FromStr` walks `ALL`, so a variant added to the enum but missing from a
/// hand-maintained catalog would compile and then fail to parse. Generating
/// both from the same list makes that omission impossible.
macro_rules! closed_id_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident, )+
        }
        all_doc = $all_doc:literal;
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $( $(#[$vmeta])* $variant, )+
        }

        impl $name {
            #[doc = $all_doc]
            pub const ALL: &'static [$name] = &[ $( Self::$variant, )+ ];
        }
    };
}

closed_id_enum! {
/// Closed set of identification strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IdentifierId {
    /// Classic backdoor adjustment-set search.
    BackdoorAdjustment,
    /// Efficient (optimal) backdoor adjustment.
    BackdoorEfficient,
    /// Front-door identification.
    Frontdoor,
    /// Instrumental-variable identification.
    Iv,
    /// Sharp regression discontinuity.
    RdSharp,
    /// Temporal unfolded backdoor.
    TemporalBackdoorUnfolded,
    /// Class-aware / generalized adjustment (PAG-safe).
    GeneralizedAdjustment,
    /// Shpitser–Pearl general ID (semi-Markovian).
    GeneralId,
    /// Path-restricted natural effects.
    PathSpecificNatural,
    /// Pairwise backdoor identification for response functionals.
    ResponseBackdoor,
    /// Structural-mechanism identification for staged counterfactuals.
    GcmParametric,
    /// Conservative single-source sID (direct / S-admissible / singleton c-component).
    TransportSid,
    /// Design-based interference: assignment design identifies the exposure contrast.
    InterferenceDesign,
    /// `AutoIdentifier` — all applicable estimands, no silent estimator choice.
    Auto,
}
all_doc = "Every closed-set identifier, in declaration order (powers [`UnknownStrategy::expected`]).";
}

/// Per-identifier data-only facts backing [`IdentifierId::as_str`],
/// [`IdentifierId::is_dag_only`], and [`identify_provenance_step`].
///
/// Purely descriptive — the real behavioral dispatch lives in
/// [`identify_static_query_with_rd`] / [`identify_pag`] / [`identify_admg`], which stay
/// exhaustive `match`es and are not table-driven.
pub(super) struct IdentifierData {
    name: &'static str,
    is_dag_only: bool,
    pub(super) provenance: (&'static str, &'static str),
}

pub(super) const fn identifier_data(id: IdentifierId) -> IdentifierData {
    match id {
        IdentifierId::BackdoorAdjustment => IdentifierData {
            name: "backdoor.adjustment",
            is_dag_only: true,
            provenance: ("identify.backdoor", "identify.backdoor"),
        },
        IdentifierId::BackdoorEfficient => IdentifierData {
            name: "backdoor.efficient",
            is_dag_only: true,
            provenance: ("identify.efficient_backdoor", "identify.efficient_backdoor"),
        },
        IdentifierId::Frontdoor => IdentifierData {
            name: "frontdoor",
            is_dag_only: true,
            provenance: ("identify.frontdoor", "identify.frontdoor"),
        },
        IdentifierId::Iv => IdentifierData {
            name: "iv",
            is_dag_only: true,
            provenance: ("identify.iv", "identify.iv"),
        },
        IdentifierId::RdSharp => IdentifierData {
            name: "rd.sharp",
            is_dag_only: true,
            provenance: ("identify.rd_design", "identify.rd_sharp"),
        },
        IdentifierId::TemporalBackdoorUnfolded => IdentifierData {
            name: "temporal.backdoor.unfolded",
            is_dag_only: true,
            provenance: ("identify.temporal_backdoor", "identify.temporal_backdoor_unfolded"),
        },
        IdentifierId::GeneralizedAdjustment => IdentifierData {
            name: "generalized.adjustment",
            is_dag_only: false,
            provenance: ("identify.generalized_adjustment", "identify.generalized_adjustment"),
        },
        IdentifierId::GeneralId => IdentifierData {
            name: "general.id",
            is_dag_only: true,
            provenance: ("identify.general_id", "identify.general_id"),
        },
        IdentifierId::PathSpecificNatural => IdentifierData {
            name: "path_specific.natural",
            is_dag_only: true,
            provenance: ("identify.path_specific", "identify.path_specific"),
        },
        IdentifierId::GcmParametric => IdentifierData {
            name: "gcm.parametric",
            is_dag_only: true,
            provenance: ("counterfactual.aap", "counterfactual.aap"),
        },
        IdentifierId::TransportSid => IdentifierData {
            name: "transport.sid",
            is_dag_only: false,
            provenance: ("identify.transport.sid", "identify.transport.sid"),
        },
        IdentifierId::InterferenceDesign => IdentifierData {
            name: "interference.design",
            is_dag_only: true,
            provenance: ("identify.interference.design", "identify.interference.design"),
        },
        IdentifierId::ResponseBackdoor => IdentifierData {
            name: "response.backdoor",
            is_dag_only: true,
            provenance: ("identify.response", "identify.response_backdoor"),
        },
        IdentifierId::Auto => IdentifierData {
            name: "auto",
            is_dag_only: true,
            provenance: ("identify.auto", "identify.auto"),
        },
    }
}

impl IdentifierId {
    /// Canonical wire id.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        identifier_data(*self).name
    }

    /// Whether this identifier requires a DAG (not a raw PAG).
    #[must_use]
    pub const fn is_dag_only(&self) -> bool {
        identifier_data(*self).is_dag_only
    }
}

impl FromStr for IdentifierId {
    type Err = UnknownStrategy;

    /// Parse a wire / builder id string.
    ///
    /// # Errors
    ///
    /// [`UnknownStrategy`] when `s` does not match any closed-set identifier name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_closed(s, Self::ALL, Self::as_str, "identifier", identifier_names())
    }
}

impl std::fmt::Display for IdentifierId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

closed_id_enum! {
/// Closed set of estimators.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum EstimatorId {
    /// OLS g-computation / linear adjustment ATE.
    LinearAdjustmentAte,
    /// Inverse-probability weighting.
    PropensityWeighting,
    /// Propensity-score matching.
    PropensityMatching,
    /// Propensity stratification.
    PropensityStratification,
    /// Covariate distance matching.
    DistanceMatching,
    /// Augmented IPW.
    Aipw,
    /// GLM (logit) adjustment.
    GlmAdjustment,
    /// Linear front-door two-stage (product of coefficients).
    FrontDoorTwoStage,
    /// Plug-in of the nonparametric front-door functional.
    FrontDoorFunctional,
    /// Wald IV.
    IvWald,
    /// Two-stage least squares.
    Iv2Sls,
    /// Sharp local-linear RD.
    RdSharp,
    /// Bayesian g-computation.
    BayesianGcomp,
    /// Bayesian Gaussian interaction model, averaged over observed modifiers.
    BayesianConditional,
    /// Temporal linear adjustment.
    TemporalLinearAdjustment,
    /// Bayesian g-computation on the lag-aligned temporal adjustment design.
    BayesianTemporalGcomp,
    /// Sequential g-computation of a sustained treatment window.
    TemporalSequentialGcomp,
    /// Discrete plug-in evaluation of an identified interventional distribution.
    FunctionalDistribution,
    /// Discrete plug-in evaluation of an identified scalar functional (ATE / path NE).
    FunctionalEffect,
    /// Static linear structural mediation.
    StaticMediationLinear,
    /// Conditional linear adjustment (effect modifiers).
    ConditionalLinearAdjustment,
    /// Temporal linear mediation (path-product).
    TemporalMediation,
    /// Bayesian response.temporal.bayesian estimator.
    TemporalResponseBayesian,
    /// Bayesian response.bayesian estimator.
    ResponseBayesian,
    /// Bayesian temporal.mediation.bayesian estimator.
    BayesianTemporalMediation,
    /// Kennedy-style doubly robust continuous response curve / point derivative.
    ResponseKennedyDr,
    /// Riesz-representer average derivative estimator.
    ResponseRieszAde,
    /// Low-dimensional GAM plug-in Jacobian / directional derivative.
    ResponseGamDerivative,
    /// Additive-GAM g-computation for static, shifted, or stochastic interventions.
    ResponseInterventionGcomp,
    /// Cell-saturated AIPW for discrete joint `Set` interventions.
    CellAipw,
    /// Temporal dose-over-horizon / policy-path g-computation (ADR 0021).
    TemporalResponseGcomp,
    /// Fitted additive GCM mechanisms with abduction–action–prediction ITE.
    GcmFit,
    /// Dahabreh trial-to-target IPW (Direct / S-admissible standardize only).
    TransportTrialIpw,
    /// Horvitz–Thompson / Hájek exposure contrast under a known assignment design.
    InterferenceHtHajek,
    /// Cross-fitted DML / AIPW average treatment effect.
    Dml,
    /// Doubly robust CATE learner (DRLearner).
    DrLearner,
    /// Native honest causal forest CATE.
    CausalForest,
}
all_doc = "Every closed-set estimator, in declaration order (powers [`UnknownStrategy::expected`]).";
}

impl EstimatorId {
    /// Default response estimator for `functional` when the caller did not override it.
    #[must_use]
    pub const fn default_for_response(functional: &antecedent_core::ResponseFunctional) -> Self {
        use antecedent_core::ResponseFunctional;
        match functional {
            ResponseFunctional::MeanCurve { .. } | ResponseFunctional::PointDerivative { .. } => {
                Self::ResponseKennedyDr
            }
            ResponseFunctional::AverageDerivative { .. } => Self::ResponseRieszAde,
            ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => Self::ResponseGamDerivative,
            ResponseFunctional::InterventionResponse { .. } => Self::ResponseInterventionGcomp,
        }
    }
}

/// Per-estimator data-only facts backing [`EstimatorId::as_str`],
/// [`EstimatorId::parallel_task_dimension`], [`EstimatorId::kernel_label`], and
/// [`estimate_provenance_step`].
///
/// Purely descriptive — the real behavioral dispatch lives in
/// [`estimate_static_effect`] / [`estimate_static_effect_default`] /
/// [`estimand_compatible_with_estimator`], which stay exhaustive `match`es and are not
/// table-driven.
pub(super) struct EstimatorData {
    name: &'static str,
    parallel_task_dimension: &'static str,
    kernel_label: &'static str,
    pub(super) provenance: (&'static str, &'static str),
}

// One exhaustive match over every estimator, returning pure data. It is long because
// there are many estimators, not because it does several things. Keeping it as a `match`
// rather than an indexed table preserves the compile error when a new variant is added and
// this row is forgotten -- which is the whole point of centralising the data here.
#[allow(clippy::too_many_lines)]
pub(super) const fn estimator_data(id: EstimatorId) -> EstimatorData {
    match id {
        EstimatorId::LinearAdjustmentAte => EstimatorData {
            name: "linear.adjustment.ate",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer",
            provenance: ("estimate.linear_adjustment", "estimate.linear_adjustment_ate"),
        },
        EstimatorId::PropensityWeighting => EstimatorData {
            name: "propensity.weighting",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ipw",
            provenance: ("estimate.propensity", "estimate.propensity_weighting"),
        },
        EstimatorId::PropensityMatching => EstimatorData {
            name: "propensity.matching",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "matching",
            provenance: ("estimate.propensity", "estimate.propensity_matching"),
        },
        EstimatorId::PropensityStratification => EstimatorData {
            name: "propensity.stratification",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "propensity.stratification",
            provenance: ("estimate.propensity", "estimate.propensity_stratification"),
        },
        EstimatorId::DistanceMatching => EstimatorData {
            name: "distance.matching",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "matching",
            provenance: ("estimate.matching", "estimate.distance_matching"),
        },
        EstimatorId::Aipw => EstimatorData {
            name: "aipw",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "aipw",
            provenance: ("estimate.aipw", "estimate.aipw"),
        },
        EstimatorId::GlmAdjustment => EstimatorData {
            name: "glm.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "glm.logit",
            provenance: ("estimate.glm_adjustment", "estimate.glm_adjustment_ate"),
        },
        EstimatorId::FrontDoorTwoStage => EstimatorData {
            name: "frontdoor.linear_two_stage",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "frontdoor.linear_two_stage",
            provenance: ("estimate.frontdoor", "estimate.frontdoor_two_stage"),
        },
        EstimatorId::FrontDoorFunctional => EstimatorData {
            name: "frontdoor.functional",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "frontdoor.functional",
            provenance: ("estimate.frontdoor_functional", "estimate.frontdoor_functional"),
        },
        EstimatorId::IvWald => EstimatorData {
            name: "iv.wald",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "iv.wald",
            provenance: ("estimate.iv", "estimate.wald_iv"),
        },
        EstimatorId::Iv2Sls => EstimatorData {
            name: "iv.2sls",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "2sls",
            provenance: ("estimate.iv", "estimate.two_stage_least_squares"),
        },
        EstimatorId::RdSharp => EstimatorData {
            name: "rd.sharp",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "rd.local_linear",
            provenance: ("estimate.rd", "estimate.rd_sharp"),
        },
        EstimatorId::BayesianConditional => EstimatorData {
            name: "conditional.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.conditional",
            provenance: ("estimate.bayesian_conditional", "estimate.bayesian_conditional"),
        },
        EstimatorId::BayesianGcomp => EstimatorData {
            name: "bayesian.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer",
            provenance: ("estimate.bayesian_gcomp", "estimate.bayesian_gcomp"),
        },
        EstimatorId::TemporalSequentialGcomp => EstimatorData {
            name: "temporal.sequential.gcomp",
            parallel_task_dimension: "mechanism",
            kernel_label: "ols.faer.temporal.sequential",
            provenance: ("estimate.temporal_sequential", "estimate.temporal_sequential"),
        },
        EstimatorId::TemporalLinearAdjustment => EstimatorData {
            name: "temporal.linear.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.temporal_linear", "estimate.temporal_linear_adjustment"),
        },
        EstimatorId::BayesianTemporalGcomp => EstimatorData {
            name: "bayesian.temporal.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.bayesian_temporal_gcomp", "estimate.bayesian.temporal.gcomp"),
        },
        EstimatorId::FunctionalDistribution => EstimatorData {
            name: "functional.distribution",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "functional.distribution",
            provenance: ("estimate.functional_distribution", "estimate.functional_distribution"),
        },
        EstimatorId::StaticMediationLinear => EstimatorData {
            name: "mediation.linear",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.mediation",
            provenance: ("estimate.mediation.linear", "estimate.mediation.linear"),
        },
        EstimatorId::FunctionalEffect => EstimatorData {
            name: "functional.effect",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "functional.effect",
            provenance: ("estimate.functional_effect", "estimate.functional_effect"),
        },
        EstimatorId::ConditionalLinearAdjustment => EstimatorData {
            name: "conditional.linear.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.conditional",
            provenance: ("estimate.conditional_linear", "estimate.conditional_linear_adjustment"),
        },
        EstimatorId::BayesianTemporalMediation => EstimatorData {
            name: "temporal.mediation.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: (
                "estimate.bayesian_temporal_mediation",
                "estimate.bayesian_temporal_mediation",
            ),
        },
        EstimatorId::ResponseBayesian => EstimatorData {
            name: "response.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: ("estimate.response.bayesian", "estimate.response.bayesian"),
        },
        EstimatorId::TemporalResponseBayesian => EstimatorData {
            name: "response.temporal.bayesian",
            parallel_task_dimension: "analysis",
            kernel_label: "bayesian.gaussian",
            provenance: (
                "estimate.response.temporal.bayesian",
                "estimate.response.temporal.bayesian",
            ),
        },
        EstimatorId::TemporalMediation => EstimatorData {
            name: "temporal.mediation",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.temporal_mediation", "estimate.temporal_mediation"),
        },
        EstimatorId::ResponseKennedyDr => EstimatorData {
            name: "response.kennedy_dr",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "response.kennedy_dr.gam",
            provenance: ("estimate.response.kennedy_dr", "estimate.response.kennedy_dr"),
        },
        EstimatorId::ResponseRieszAde => EstimatorData {
            name: "response.riesz_ade",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "response.riesz_ade.gam",
            provenance: ("estimate.response.riesz_ade", "estimate.response.riesz_ade"),
        },
        EstimatorId::ResponseGamDerivative => EstimatorData {
            name: "response.gam_derivative",
            parallel_task_dimension: "outcome",
            kernel_label: "response.gam.derivative",
            provenance: ("estimate.response.gam_derivative", "estimate.response.gam_derivative"),
        },
        EstimatorId::ResponseInterventionGcomp => EstimatorData {
            name: "response.intervention_gcomp",
            parallel_task_dimension: "intervention",
            kernel_label: "response.gam.intervention_gcomp",
            provenance: (
                "estimate.response.intervention_gcomp",
                "estimate.response.intervention_gcomp",
            ),
        },
        EstimatorId::CellAipw => EstimatorData {
            name: "cell.aipw",
            parallel_task_dimension: "cell",
            kernel_label: "aipw.cell",
            provenance: ("estimate.cell.aipw", "estimate.cell.aipw"),
        },
        EstimatorId::TemporalResponseGcomp => EstimatorData {
            name: "temporal.response.gcomp",
            parallel_task_dimension: "horizon",
            kernel_label: "ols.faer.temporal.response",
            provenance: ("estimate.temporal_response.gcomp", "estimate.temporal_response.gcomp"),
        },
        EstimatorId::GcmFit => EstimatorData {
            name: "gcm.fit",
            parallel_task_dimension: "analysis",
            kernel_label: "gcm.aap",
            provenance: ("estimate.gcm.fit", "estimate.gcm.fit"),
        },
        EstimatorId::TransportTrialIpw => EstimatorData {
            name: "transport.trial_ipw",
            parallel_task_dimension: "analysis",
            kernel_label: "transport.trial_ipw",
            provenance: ("estimate.transport.trial_ipw", "estimate.transport.trial_ipw"),
        },
        EstimatorId::InterferenceHtHajek => EstimatorData {
            name: "interference.ht_hajek",
            parallel_task_dimension: "analysis",
            kernel_label: "interference.ht_hajek",
            provenance: ("estimate.interference.ht_hajek", "estimate.interference.ht_hajek"),
        },
        EstimatorId::Dml => EstimatorData {
            name: "dml",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "dml",
            provenance: ("estimate.dml", "estimate.dml"),
        },
        EstimatorId::DrLearner => EstimatorData {
            name: "dr.learner",
            parallel_task_dimension: "crossfit.fold",
            kernel_label: "dr.learner",
            provenance: ("estimate.dr.learner", "estimate.dr.learner"),
        },
        EstimatorId::CausalForest => EstimatorData {
            name: "causal.forest",
            parallel_task_dimension: "forest.tree",
            kernel_label: "causal.forest",
            provenance: ("estimate.causal.forest", "estimate.causal.forest"),
        },
    }
}

impl EstimatorId {
    /// Canonical wire id.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        estimator_data(*self).name
    }

    /// Parallel-task dimension label for physical planning.
    #[must_use]
    pub const fn parallel_task_dimension(&self) -> &'static str {
        estimator_data(*self).parallel_task_dimension
    }

    /// Dense-kernel label recorded on the physical plan.
    #[must_use]
    pub const fn kernel_label(&self) -> &'static str {
        estimator_data(*self).kernel_label
    }
}

impl FromStr for EstimatorId {
    type Err = UnknownStrategy;

    /// Parse a wire / builder id string.
    ///
    /// # Errors
    ///
    /// [`UnknownStrategy`] when `s` does not match any closed-set estimator name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_closed(s, Self::ALL, Self::as_str, "estimator", estimator_names())
    }
}

impl std::fmt::Display for EstimatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a strategy name does not match any known strategy.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("unknown {kind} `{got}`; expected one of: {}", .expected.join(", "))]
pub struct UnknownStrategy {
    /// Which strategy family failed to parse (`"identifier"` or `"estimator"`).
    pub kind: &'static str,
    /// The name that failed to parse.
    pub got: String,
    /// Every accepted name, for the error message.
    pub expected: &'static [&'static str],
}

impl From<UnknownStrategy> for CausalError {
    fn from(e: UnknownStrategy) -> Self {
        crate::compile_reason!("unknown_strategy", "{e}")
    }
}

/// Default identifier id when the builder omits one.
pub const DEFAULT_IDENTIFIER: &str = "backdoor.adjustment";

/// Default estimator id when the builder omits one.
pub const DEFAULT_ESTIMATOR: &str = "linear.adjustment.ate";

/// Default identifier as a closed enum.
pub const DEFAULT_IDENTIFIER_ID: IdentifierId = IdentifierId::BackdoorAdjustment;

/// Default estimator as a closed enum.
pub const DEFAULT_ESTIMATOR_ID: EstimatorId = EstimatorId::LinearAdjustmentAte;

/// Default distribution identifier.
pub const DEFAULT_DISTRIBUTION_IDENTIFIER: &str = "general.id";
/// Default distribution estimator.
pub const DEFAULT_DISTRIBUTION_ESTIMATOR: &str = "functional.distribution";
/// Default distribution identifier enum.
pub const DEFAULT_DISTRIBUTION_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralId;
/// Default distribution estimator enum.
pub const DEFAULT_DISTRIBUTION_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalDistribution;

/// Default response identifier.
pub const DEFAULT_RESPONSE_IDENTIFIER: &str = "response.backdoor";
/// Default response identifier enum.
pub const DEFAULT_RESPONSE_IDENTIFIER_ID: IdentifierId = IdentifierId::ResponseBackdoor;
/// Default response-curve estimator.
pub const DEFAULT_RESPONSE_ESTIMATOR: &str = "response.kennedy_dr";
/// Default response-curve estimator enum.
pub const DEFAULT_RESPONSE_ESTIMATOR_ID: EstimatorId = EstimatorId::ResponseKennedyDr;

/// Compile-time allowlist for continuous-response identify/estimate pairs.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_response_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    if identifier != IdentifierId::ResponseBackdoor
        || !matches!(
            estimator,
            EstimatorId::ResponseKennedyDr
                | EstimatorId::ResponseRieszAde
                | EstimatorId::ResponseGamDerivative
                | EstimatorId::ResponseInterventionGcomp
                | EstimatorId::CellAipw
                | EstimatorId::ResponseBayesian
        )
    {
        return Err(CausalError::Compile {
            message: format!(
                "Response requires identifier response.backdoor and a response.* estimator (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Compile-time allowlist for class-aware Cpdag/Pag response (same theory as ATE).
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_class_response_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    if identifier != IdentifierId::GeneralizedAdjustment
        || !matches!(
            estimator,
            EstimatorId::ResponseKennedyDr
                | EstimatorId::ResponseInterventionGcomp
                | EstimatorId::CellAipw
                | EstimatorId::ResponseBayesian
        )
    {
        return Err(CausalError::Compile {
            message: format!(
                "Cpdag/Pag response requires identifier generalized.adjustment and \
                 response.kennedy_dr, response.intervention_gcomp, or response.bayesian \
                 (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Compile-time allowlist of identifier/estimator pairs for the static ATE path.
///
/// # Errors
///
/// Unknown ids or incompatible pairs.
pub fn validate_static_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let backdoor_estimators = matches!(
        estimator,
        EstimatorId::LinearAdjustmentAte
            | EstimatorId::PropensityWeighting
            | EstimatorId::PropensityMatching
            | EstimatorId::PropensityStratification
            | EstimatorId::DistanceMatching
            | EstimatorId::Aipw
            | EstimatorId::GlmAdjustment
            | EstimatorId::Dml
            | EstimatorId::DrLearner
            | EstimatorId::CausalForest
            | EstimatorId::BayesianGcomp
            | EstimatorId::BayesianConditional
            | EstimatorId::ConditionalLinearAdjustment
    );
    let supported = match (&identifier, &estimator) {
        (IdentifierId::BackdoorAdjustment | IdentifierId::BackdoorEfficient, _)
            if backdoor_estimators =>
        {
            true
        }
        (
            IdentifierId::Frontdoor,
            EstimatorId::FrontDoorTwoStage | EstimatorId::FrontDoorFunctional,
        )
        | (IdentifierId::Iv, EstimatorId::IvWald | EstimatorId::Iv2Sls)
        | (IdentifierId::RdSharp, EstimatorId::RdSharp)
        | (
            IdentifierId::GeneralizedAdjustment,
            EstimatorId::LinearAdjustmentAte
            | EstimatorId::PropensityWeighting
            | EstimatorId::PropensityMatching
            | EstimatorId::PropensityStratification
            | EstimatorId::DistanceMatching
            | EstimatorId::Aipw
            | EstimatorId::GlmAdjustment
            | EstimatorId::Dml
            | EstimatorId::DrLearner
            | EstimatorId::CausalForest
            | EstimatorId::BayesianGcomp
            | EstimatorId::BayesianConditional
            | EstimatorId::ConditionalLinearAdjustment,
        )
        | (IdentifierId::GeneralId, EstimatorId::FunctionalEffect) => true,
        (IdentifierId::Auto, _)
            if backdoor_estimators
                || matches!(
                    estimator,
                    EstimatorId::FrontDoorTwoStage
                        | EstimatorId::FrontDoorFunctional
                        | EstimatorId::IvWald
                        | EstimatorId::Iv2Sls
                ) =>
        {
            true
        }
        _ => false,
    };
    if !supported {
        return Err(crate::compile_reason!(
            "strategy_incompatible",
            "identifier {:?} is not compatible with estimator {:?}",
            identifier.as_str(),
            estimator.as_str()
        ));
    }
    Ok(())
}

/// Default path-specific identifier.
pub const DEFAULT_PATH_IDENTIFIER: &str = "path_specific.natural";
/// Default path-specific estimator.
pub const DEFAULT_PATH_ESTIMATOR: &str = "functional.effect";
/// Default path-specific identifier enum.
pub const DEFAULT_PATH_IDENTIFIER_ID: IdentifierId = IdentifierId::PathSpecificNatural;
/// Default path-specific estimator enum.
pub const DEFAULT_PATH_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalEffect;

/// Default PAG / generalized-adjustment identifier.
pub const DEFAULT_PAG_IDENTIFIER: &str = "generalized.adjustment";
/// Default PAG estimator.
pub const DEFAULT_PAG_ESTIMATOR: &str = "linear.adjustment.ate";
/// Default PAG identifier enum.
pub const DEFAULT_PAG_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralizedAdjustment;
/// Default PAG estimator enum.
pub const DEFAULT_PAG_ESTIMATOR_ID: EstimatorId = EstimatorId::LinearAdjustmentAte;

/// Default ADMG identifier (general ID).
pub const DEFAULT_ADMG_IDENTIFIER: &str = "general.id";
/// Default ADMG estimator (functional plug-in).
pub const DEFAULT_ADMG_ESTIMATOR: &str = "functional.effect";
/// Default ADMG identifier enum.
pub const DEFAULT_ADMG_IDENTIFIER_ID: IdentifierId = IdentifierId::GeneralId;
/// Default ADMG estimator enum.
pub const DEFAULT_ADMG_ESTIMATOR_ID: EstimatorId = EstimatorId::FunctionalEffect;

/// Default conditional-effect identifier.
pub const DEFAULT_CONDITIONAL_IDENTIFIER: &str = "backdoor.adjustment";
/// Default conditional-effect estimator.
pub const DEFAULT_CONDITIONAL_ESTIMATOR: &str = "conditional.linear.adjustment";
/// Default conditional identifier enum.
pub const DEFAULT_CONDITIONAL_IDENTIFIER_ID: IdentifierId = IdentifierId::BackdoorAdjustment;
/// Default conditional estimator enum.
pub const DEFAULT_CONDITIONAL_ESTIMATOR_ID: EstimatorId = EstimatorId::ConditionalLinearAdjustment;

/// Default mediation identifier (static Total uses front-door).
pub const DEFAULT_MEDIATION_IDENTIFIER: &str = "frontdoor";
/// Default mediation estimator (temporal path).
pub const DEFAULT_MEDIATION_ESTIMATOR: &str = "temporal.mediation";
/// Default mediation identifier enum.
pub const DEFAULT_MEDIATION_IDENTIFIER_ID: IdentifierId = IdentifierId::Frontdoor;
/// Default mediation estimator enum.
pub const DEFAULT_MEDIATION_ESTIMATOR_ID: EstimatorId = EstimatorId::TemporalMediation;

/// Whether an identification status is acceptable when estimands are present.
#[must_use]
pub fn identification_status_acceptable(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::GraphDependent
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

/// Gate identification: reject `NotIdentified` and empty estimands.
///
/// # Errors
///
/// Effect not identified or no estimand returned.
pub fn require_identified(result: &IdentificationResult) -> Result<(), CausalError> {
    if matches!(result.status, IdentificationStatus::NotIdentified)
        || result.estimands.is_empty()
        || !identification_status_acceptable(result.status)
    {
        return Err(CausalError::not_identified(
            result.status,
            antecedent_identify::search_truncated(result),
            "effect not identified",
        ));
    }
    Ok(())
}

/// Whether an estimand method is compatible with an estimator.
#[must_use]
pub fn estimand_compatible_with_estimator(method: EstimandMethod, estimator: &EstimatorId) -> bool {
    match estimator {
        EstimatorId::LinearAdjustmentAte
        | EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification
        | EstimatorId::DistanceMatching
        | EstimatorId::Aipw
        | EstimatorId::GlmAdjustment
        | EstimatorId::BayesianGcomp
        | EstimatorId::BayesianConditional
        | EstimatorId::ConditionalLinearAdjustment
        | EstimatorId::ResponseKennedyDr
        | EstimatorId::ResponseRieszAde
        | EstimatorId::ResponseGamDerivative
        | EstimatorId::ResponseInterventionGcomp
        | EstimatorId::CellAipw
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
        | EstimatorId::ResponseBayesian => method.is_backdoor_family(),
        EstimatorId::FrontDoorTwoStage | EstimatorId::FrontDoorFunctional => {
            matches!(method, EstimandMethod::FrontDoor)
        }
        EstimatorId::IvWald | EstimatorId::Iv2Sls => matches!(method, EstimandMethod::Iv),
        EstimatorId::RdSharp => matches!(method, EstimandMethod::RdSharp),
        EstimatorId::TemporalLinearAdjustment
        | EstimatorId::BayesianTemporalGcomp
        | EstimatorId::TemporalSequentialGcomp => {
            matches!(method, EstimandMethod::TemporalBackdoorUnfolded)
        }
        EstimatorId::TemporalResponseGcomp | EstimatorId::TemporalResponseBayesian => {
            matches!(
                method,
                EstimandMethod::TemporalBackdoorUnfolded | EstimandMethod::BackdoorAdjustment
            )
        }
        EstimatorId::TemporalMediation | EstimatorId::BayesianTemporalMediation => {
            method.is_temporal_mediation() || matches!(method, EstimandMethod::FrontDoor)
        }
        EstimatorId::FunctionalDistribution => matches!(method, EstimandMethod::GeneralId),
        EstimatorId::StaticMediationLinear => matches!(method, EstimandMethod::PathSpecificNatural),
        EstimatorId::FunctionalEffect => {
            matches!(method, EstimandMethod::PathSpecificNatural | EstimandMethod::GeneralId)
        }
        EstimatorId::GcmFit | EstimatorId::TransportTrialIpw | EstimatorId::InterferenceHtHajek => {
            false
        }
    }
}

/// Select a single estimand matching the estimator (no silent Auto `.first()`).
///
/// # Errors
///
/// No estimand, or multiple estimands without a unique estimator-compatible match.
pub fn select_estimand(
    identification: &IdentificationResult,
    estimator: EstimatorId,
) -> Result<IdentifiedEstimand, CausalError> {
    select_estimand_index(identification, estimator).map(|i| identification.estimands[i].clone())
}

/// Select the estimand as [`select_estimand`] does and narrow the identification to it.
///
/// A multi-strategy identification lists alternatives whose status and assumptions
/// differ. The returned identification carries exactly the selected estimand's claim, so
/// what reaches the estimate and the result is the claim of the strategy actually used.
///
/// # Errors
///
/// No estimand, or multiple estimands without a unique estimator-compatible match.
pub fn select_claim(
    identification: IdentificationResult,
    estimator: EstimatorId,
) -> Result<(IdentificationResult, IdentifiedEstimand), CausalError> {
    let index = select_estimand_index(&identification, estimator)?;
    let estimand = identification.estimands[index].clone();
    let mut claim = if identification.estimand_claims.is_empty() {
        identification
    } else {
        identification.narrowed_to(index).unwrap_or(identification)
    };
    if let Some(restriction) = estimator_claim_restriction(estimator) {
        restrict_claim(&mut claim, &restriction);
    }
    Ok((claim, estimand))
}

/// Restriction under which `estimator`'s target equals the identified functional.
///
/// An estimator that evaluates the identified functional keeps the identifier's claim; how
/// it models a regression inside that functional is an estimation-scope record, because the
/// regression is a feature of the observed law that a richer model can fit without changing
/// the target. An estimator that computes a *different* functional of the observed law is
/// the queried effect only under a restriction on the structural model, and that
/// restriction is part of the identification claim: the constant-effect (or monotonicity)
/// restriction of a Wald ratio, recorded by the IV identifier, the restriction under
/// which a product of regression coefficients is the front-door effect, and the restriction
/// under which `total − direct` is the pure natural indirect effect the path-specific
/// identifier certifies, recorded here because those identifiers cannot know which estimator
/// will run.
fn estimator_claim_restriction(estimator: EstimatorId) -> Option<AssumptionRecord> {
    match estimator {
        EstimatorId::FrontDoorTwoStage => {
            Some(antecedent_estimate::linear_path_product_restriction())
        }
        EstimatorId::StaticMediationLinear => {
            Some(antecedent_estimate::linear_no_interaction_restriction())
        }
        _ => None,
    }
}

/// Add `restriction` to a claim and weaken a nonparametric status to a parametric one.
/// Statuses that are already weaker are kept.
fn restrict_claim(claim: &mut IdentificationResult, restriction: &AssumptionRecord) {
    let weaken = |status: &mut IdentificationStatus| {
        if *status == IdentificationStatus::NonparametricallyIdentified {
            *status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
        }
    };
    weaken(&mut claim.status);
    claim.required_assumptions.extend_unique([restriction]);
    for own in &mut claim.estimand_claims {
        weaken(&mut own.status);
        own.required_assumptions.extend_unique([restriction]);
    }
}

fn select_estimand_index(
    identification: &IdentificationResult,
    estimator: EstimatorId,
) -> Result<usize, CausalError> {
    let estimands = &identification.estimands;
    if estimands.is_empty() {
        return Err(CausalError::Compile { message: "no estimand returned".into() });
    }
    if estimands.len() == 1 {
        return Ok(0);
    }
    let matches: Vec<usize> = estimands
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            if e.is_adjustment_shaped() {
                return estimand_compatible_with_estimator(
                    EstimandMethod::BackdoorAdjustment,
                    &estimator,
                );
            }
            e.method_kind()
                .map(|m| estimand_compatible_with_estimator(m, &estimator))
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect();
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    // Auto lists every strategy that identified the query (criterion estimands and the general
    // ID functional); name them so a caller sees which methods the estimator matched.
    let methods: Vec<&str> = estimands.iter().map(|e| e.method.as_ref()).collect();
    Err(CausalError::Compile {
        message: format!(
            "identifier returned {} estimands ({}); select an explicit identifier or an estimator \
             that uniquely matches one method (got estimator {:?}, matching {})",
            estimands.len(),
            methods.join(", "),
            estimator.as_str(),
            matches.len()
        ),
    })
}

/// Allowlist for interventional-distribution identify+estimate.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_distribution_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::GeneralId | IdentifierId::Auto, EstimatorId::FunctionalDistribution)
    );
    if !supported {
        return Err(CausalError::Compile {
            message: format!(
                "Distribution requires identifier general.id|auto with estimator \
                 functional.distribution (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

/// Allowlist for path-specific natural-effect identify+estimate.
///
/// # Errors
///
/// Incompatible identifier/estimator pair.
pub fn validate_path_specific_pair(
    identifier: IdentifierId,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::PathSpecificNatural | IdentifierId::Auto, EstimatorId::FunctionalEffect)
    );
    if !supported {
        return Err(CausalError::Compile {
            message: format!(
                "PathSpecific requires identifier path_specific.natural|auto with estimator \
                 functional.effect (got {:?} / {:?})",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod names {
    use super::{EstimatorId, IdentifierId};

    #[test]
    fn identifier_from_str_roundtrips_all() {
        for id in IdentifierId::ALL {
            assert_eq!(id.as_str().parse::<IdentifierId>().unwrap(), *id);
        }
        assert!("not.an.identifier".parse::<IdentifierId>().is_err());
    }

    #[test]
    fn estimator_from_str_roundtrips_all() {
        for id in EstimatorId::ALL {
            assert_eq!(id.as_str().parse::<EstimatorId>().unwrap(), *id);
        }
        assert!("not.an.estimator".parse::<EstimatorId>().is_err());
    }

    /// `ALL` is generated from the enum's own variant list, so every variant
    /// is present exactly once; distinct wire names keep `FromStr` injective.
    #[test]
    fn catalogs_have_distinct_wire_names() {
        let identifiers: std::collections::HashSet<_> =
            IdentifierId::ALL.iter().map(IdentifierId::as_str).collect();
        assert_eq!(identifiers.len(), IdentifierId::ALL.len());
        let estimators: std::collections::HashSet<_> =
            EstimatorId::ALL.iter().map(EstimatorId::as_str).collect();
        assert_eq!(estimators.len(), EstimatorId::ALL.len());
    }
}
