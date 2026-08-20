//! Identifier / estimator ids, defaults, and pair allowlists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::str::FromStr;

use antecedent_core::IdentificationStatus;
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_identify::IdentificationResult;

use crate::error::CausalError;

/// Every accepted wire name for [`IdentifierId`], in [`IdentifierId::ALL`] order.
const IDENTIFIER_NAMES: &[&str] = &[
    "backdoor.adjustment",
    "backdoor.efficient",
    "frontdoor",
    "iv",
    "rd.sharp",
    "temporal.backdoor.unfolded",
    "generalized.adjustment",
    "general.id",
    "path_specific.natural",
    "response.backdoor",
    "auto",
];

/// Every accepted wire name for [`EstimatorId`], in [`EstimatorId::ALL`] order.
const ESTIMATOR_NAMES: &[&str] = &[
    "linear.adjustment.ate",
    "propensity.weighting",
    "propensity.matching",
    "propensity.stratification",
    "distance.matching",
    "aipw",
    "glm.adjustment",
    "frontdoor.two_stage",
    "iv.wald",
    "iv.2sls",
    "rd.sharp",
    "bayesian.gcomp",
    "temporal.linear.adjustment",
    "functional.distribution",
    "functional.effect",
    "conditional.linear.adjustment",
    "temporal.mediation",
    "response.kennedy_dr",
    "response.riesz_ade",
    "response.gam_derivative",
    "response.intervention_gcomp",
];

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
    /// `AutoIdentifier` — all applicable estimands, no silent estimator choice.
    Auto,
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
    /// Every closed-set identifier, in declaration order (powers [`UnknownStrategy::expected`]).
    pub const ALL: &'static [IdentifierId] = &[
        Self::BackdoorAdjustment,
        Self::BackdoorEfficient,
        Self::Frontdoor,
        Self::Iv,
        Self::RdSharp,
        Self::TemporalBackdoorUnfolded,
        Self::GeneralizedAdjustment,
        Self::GeneralId,
        Self::PathSpecificNatural,
        Self::ResponseBackdoor,
        Self::Auto,
    ];

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
        match s {
            "backdoor.adjustment" => Ok(Self::BackdoorAdjustment),
            "backdoor.efficient" => Ok(Self::BackdoorEfficient),
            "frontdoor" => Ok(Self::Frontdoor),
            "iv" => Ok(Self::Iv),
            "rd.sharp" => Ok(Self::RdSharp),
            "temporal.backdoor.unfolded" => Ok(Self::TemporalBackdoorUnfolded),
            "generalized.adjustment" => Ok(Self::GeneralizedAdjustment),
            "general.id" => Ok(Self::GeneralId),
            "path_specific.natural" => Ok(Self::PathSpecificNatural),
            "response.backdoor" => Ok(Self::ResponseBackdoor),
            "auto" => Ok(Self::Auto),
            other => Err(UnknownStrategy {
                kind: "identifier",
                got: other.to_string(),
                expected: IDENTIFIER_NAMES,
            }),
        }
    }
}

impl std::fmt::Display for IdentifierId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

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
    /// Front-door two-stage.
    FrontDoorTwoStage,
    /// Wald IV.
    IvWald,
    /// Two-stage least squares.
    Iv2Sls,
    /// Sharp local-linear RD.
    RdSharp,
    /// Bayesian g-computation.
    BayesianGcomp,
    /// Temporal linear adjustment.
    TemporalLinearAdjustment,
    /// Discrete plug-in evaluation of an identified interventional distribution.
    FunctionalDistribution,
    /// Discrete plug-in evaluation of an identified scalar functional (ATE / path NE).
    FunctionalEffect,
    /// Conditional linear adjustment (effect modifiers).
    ConditionalLinearAdjustment,
    /// Temporal linear mediation (path-product).
    TemporalMediation,
    /// Kennedy-style doubly robust continuous response curve / point derivative.
    ResponseKennedyDr,
    /// Riesz-representer average derivative estimator.
    ResponseRieszAde,
    /// Low-dimensional GAM plug-in Jacobian / directional derivative.
    ResponseGamDerivative,
    /// Additive-GAM g-computation for static, shifted, or stochastic interventions.
    ResponseInterventionGcomp,
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
            name: "frontdoor.two_stage",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "frontdoor.two_stage",
            provenance: ("estimate.frontdoor", "estimate.frontdoor_two_stage"),
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
        EstimatorId::BayesianGcomp => EstimatorData {
            name: "bayesian.gcomp",
            parallel_task_dimension: "analysis",
            kernel_label: "ols.faer",
            provenance: ("estimate.bayesian_gcomp", "estimate.bayesian_gcomp"),
        },
        EstimatorId::TemporalLinearAdjustment => EstimatorData {
            name: "temporal.linear.adjustment",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "ols.faer.temporal",
            provenance: ("estimate.temporal_linear", "estimate.temporal_linear_adjustment"),
        },
        EstimatorId::FunctionalDistribution => EstimatorData {
            name: "functional.distribution",
            parallel_task_dimension: "bootstrap.replicate",
            kernel_label: "functional.distribution",
            provenance: ("estimate.functional_distribution", "estimate.functional_distribution"),
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
    }
}

impl EstimatorId {
    /// Every closed-set estimator, in declaration order (powers [`UnknownStrategy::expected`]).
    pub const ALL: &'static [EstimatorId] = &[
        Self::LinearAdjustmentAte,
        Self::PropensityWeighting,
        Self::PropensityMatching,
        Self::PropensityStratification,
        Self::DistanceMatching,
        Self::Aipw,
        Self::GlmAdjustment,
        Self::FrontDoorTwoStage,
        Self::IvWald,
        Self::Iv2Sls,
        Self::RdSharp,
        Self::BayesianGcomp,
        Self::TemporalLinearAdjustment,
        Self::FunctionalDistribution,
        Self::FunctionalEffect,
        Self::ConditionalLinearAdjustment,
        Self::TemporalMediation,
        Self::ResponseKennedyDr,
        Self::ResponseRieszAde,
        Self::ResponseGamDerivative,
        Self::ResponseInterventionGcomp,
    ];

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
        match s {
            "linear.adjustment.ate" => Ok(Self::LinearAdjustmentAte),
            "propensity.weighting" => Ok(Self::PropensityWeighting),
            "propensity.matching" => Ok(Self::PropensityMatching),
            "propensity.stratification" => Ok(Self::PropensityStratification),
            "distance.matching" => Ok(Self::DistanceMatching),
            "aipw" => Ok(Self::Aipw),
            "glm.adjustment" => Ok(Self::GlmAdjustment),
            "frontdoor.two_stage" => Ok(Self::FrontDoorTwoStage),
            "iv.wald" => Ok(Self::IvWald),
            "iv.2sls" => Ok(Self::Iv2Sls),
            "rd.sharp" => Ok(Self::RdSharp),
            "bayesian.gcomp" => Ok(Self::BayesianGcomp),
            "temporal.linear.adjustment" => Ok(Self::TemporalLinearAdjustment),
            "functional.distribution" => Ok(Self::FunctionalDistribution),
            "functional.effect" => Ok(Self::FunctionalEffect),
            "conditional.linear.adjustment" => Ok(Self::ConditionalLinearAdjustment),
            "temporal.mediation" => Ok(Self::TemporalMediation),
            "response.kennedy_dr" => Ok(Self::ResponseKennedyDr),
            "response.riesz_ade" => Ok(Self::ResponseRieszAde),
            "response.gam_derivative" => Ok(Self::ResponseGamDerivative),
            "response.intervention_gcomp" => Ok(Self::ResponseInterventionGcomp),
            other => Err(UnknownStrategy {
                kind: "estimator",
                got: other.to_string(),
                expected: ESTIMATOR_NAMES,
            }),
        }
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
        CausalError::Compile { message: e.to_string() }
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
            | EstimatorId::BayesianGcomp
            | EstimatorId::ConditionalLinearAdjustment
    );
    let supported = match (&identifier, &estimator) {
        (IdentifierId::BackdoorAdjustment | IdentifierId::BackdoorEfficient, _)
            if backdoor_estimators =>
        {
            true
        }
        (IdentifierId::Frontdoor, EstimatorId::FrontDoorTwoStage)
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
            | EstimatorId::BayesianGcomp,
        )
        | (IdentifierId::GeneralId, EstimatorId::FunctionalEffect) => true,
        (IdentifierId::Auto, _)
            if backdoor_estimators
                || matches!(
                    estimator,
                    EstimatorId::FrontDoorTwoStage | EstimatorId::IvWald | EstimatorId::Iv2Sls
                ) =>
        {
            true
        }
        _ => false,
    };
    if !supported {
        return Err(CausalError::Compile {
            message: format!(
                "identifier {:?} is not compatible with estimator {:?}",
                identifier.as_str(),
                estimator.as_str()
            ),
        });
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
    if result.status == IdentificationStatus::NotIdentified || result.estimands.is_empty() {
        return Err(CausalError::Compile { message: "effect not identified".into() });
    }
    if !identification_status_acceptable(result.status) {
        return Err(CausalError::Compile {
            message: format!("effect not identified (status {:?})", result.status),
        });
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
        | EstimatorId::ConditionalLinearAdjustment
        | EstimatorId::ResponseKennedyDr
        | EstimatorId::ResponseRieszAde
        | EstimatorId::ResponseGamDerivative
        | EstimatorId::ResponseInterventionGcomp => method.is_backdoor_family(),
        EstimatorId::FrontDoorTwoStage => matches!(method, EstimandMethod::FrontDoor),
        EstimatorId::IvWald | EstimatorId::Iv2Sls => matches!(method, EstimandMethod::Iv),
        EstimatorId::RdSharp => matches!(method, EstimandMethod::RdSharp),
        EstimatorId::TemporalLinearAdjustment => {
            matches!(method, EstimandMethod::TemporalBackdoorUnfolded)
        }
        EstimatorId::TemporalMediation => {
            method.is_temporal_mediation() || matches!(method, EstimandMethod::FrontDoor)
        }
        EstimatorId::FunctionalDistribution => matches!(method, EstimandMethod::GeneralId),
        EstimatorId::FunctionalEffect => {
            matches!(method, EstimandMethod::PathSpecificNatural | EstimandMethod::GeneralId)
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
    let estimands = &identification.estimands;
    if estimands.is_empty() {
        return Err(CausalError::Compile { message: "no estimand returned".into() });
    }
    if estimands.len() == 1 {
        return Ok(estimands[0].clone());
    }
    let matches: Vec<&IdentifiedEstimand> = estimands
        .iter()
        .filter(|e| {
            e.method_kind()
                .map(|m| estimand_compatible_with_estimator(m, &estimator))
                .unwrap_or(false)
        })
        .collect();
    if matches.len() == 1 {
        return Ok(matches[0].clone());
    }
    Err(CausalError::Compile {
        message: format!(
            "identifier returned {} estimands; select an explicit identifier or an estimator \
             that uniquely matches one method (got estimator {:?})",
            estimands.len(),
            estimator.as_str()
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
