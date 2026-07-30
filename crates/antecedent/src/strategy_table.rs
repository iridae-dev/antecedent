//! Identifier / estimator strategy tables for plan compilation and static execution
//!. Incremental extraction from the analysis workflow — does not
//! replace [`crate::Study`] / plans / [`crate::StudyResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::str::FromStr;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, IdentificationStatus,
    PopulationRegistry,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AipwWorkspace, DistanceMatching, EffectEstimate, EstimationError, EstimationWorkspace,
    FrontDoorTwoStage, FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, TwoStageLeastSquares,
    TwoStageLeastSquaresWorkspace, WaldIv,
};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{Dag, Pag};
use antecedent_identify::{
    AutoIdentifier, BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier,
    GeneralizedAdjustmentIdentifier, IdIdentifier, IdentificationEnvelope, IdentificationError,
    IdentificationResult, IdentificationWorkspace, InstrumentalVariableIdentifier,
};

use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;

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
    /// `AutoIdentifier` — all applicable estimands, no silent estimator choice.
    Auto,
}

/// Per-identifier data-only facts backing [`IdentifierId::as_str`],
/// [`IdentifierId::is_dag_only`], and [`identify_provenance_step`].
///
/// Purely descriptive — the real behavioral dispatch lives in
/// [`identify_static_query_with_rd`] / [`identify_pag`] / [`identify_admg`], which stay
/// exhaustive `match`es and are not table-driven.
struct IdentifierData {
    name: &'static str,
    is_dag_only: bool,
    provenance: (&'static str, &'static str),
}

const fn identifier_data(id: IdentifierId) -> IdentifierData {
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
}

/// Per-estimator data-only facts backing [`EstimatorId::as_str`],
/// [`EstimatorId::parallel_task_dimension`], [`EstimatorId::kernel_label`], and
/// [`estimate_provenance_step`].
///
/// Purely descriptive — the real behavioral dispatch lives in
/// [`estimate_static_effect`] / [`estimate_static_effect_default`] /
/// [`estimand_compatible_with_estimator`], which stay exhaustive `match`es and are not
/// table-driven.
struct EstimatorData {
    name: &'static str,
    parallel_task_dimension: &'static str,
    kernel_label: &'static str,
    provenance: (&'static str, &'static str),
}

// One exhaustive match over every estimator, returning pure data. It is long because
// there are many estimators, not because it does several things. Keeping it as a `match`
// rather than an indexed table preserves the compile error when a new variant is added and
// this row is forgotten -- which is the whole point of centralising the data here.
#[allow(clippy::too_many_lines)]
const fn estimator_data(id: EstimatorId) -> EstimatorData {
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
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
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
        | EstimatorId::ConditionalLinearAdjustment => method.is_backdoor_family(),
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

/// Run the identifier named by `identifier` against `graph`/`query`.
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub fn identify_static(
    identifier: IdentifierId,
    graph: &Dag,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, CausalError> {
    identify_static_query(identifier, graph, &CausalQuery::AverageEffect(query.clone()))
}

/// Run a static identifier against an arbitrary [`CausalQuery`].
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub fn identify_static_query(
    identifier: IdentifierId,
    graph: &Dag,
    query: &CausalQuery,
) -> Result<IdentificationResult, CausalError> {
    identify_static_query_with_rd(identifier, graph, query, None)
}

/// Like [`identify_static_query`], optionally attaching sharp-RD design config for Auto.
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub fn identify_static_query_with_rd(
    identifier: IdentifierId,
    graph: &Dag,
    query: &CausalQuery,
    rd: Option<antecedent_identify::SharpRdConfig>,
) -> Result<IdentificationResult, CausalError> {
    let mut id_ws = IdentificationWorkspace::default();
    let result = match identifier {
        IdentifierId::BackdoorAdjustment => {
            let id = BackdoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::BackdoorEfficient => {
            let id = EfficientBackdoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::Frontdoor => {
            let id = FrontDoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::Iv => {
            let id = InstrumentalVariableIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::GeneralId => {
            let id = IdIdentifier::new();
            let prepared = id.prepare_dag(graph).map_err(identify_err)?;
            if matches!(query, CausalQuery::Distribution(q) if !q.conditioning.is_empty()) {
                let idc = antecedent_identify::IdcIdentifier::new();
                idc.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
            } else {
                id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
            }
        }
        IdentifierId::PathSpecificNatural => {
            let id = antecedent_identify::PathSpecificIdentifier::new();
            let prepared = id.prepare_dag(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::Auto => {
            let mut id = AutoIdentifier::new();
            if let Some(cfg) = rd {
                id = id.with_rd(cfg);
            }
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::GeneralizedAdjustment => {
            return Err(CausalError::Unsupported {
                message: "identifier \"generalized.adjustment\" requires a PAG \
                     (supply AcceptedGraph::pag(..) / FCI / RFCI output, not a static DAG)",
            });
        }
        IdentifierId::RdSharp => {
            return Err(CausalError::Unsupported {
                message: "identifier \"rd.sharp\" is not a graph-based static identifier; \
                     select estimator \"rd.sharp\" with builder.rd_config(...)",
            });
        }
        IdentifierId::TemporalBackdoorUnfolded => {
            return Err(CausalError::Unsupported {
                message: "identifier \"temporal.backdoor.unfolded\" requires a temporal graph \
                     and TemporalEffect query",
            });
        }
    };
    require_identified(&result)?;
    Ok(result)
}

/// Class-aware identification over a PAG (generalized adjustment envelope).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_pag(
    identifier: IdentifierId,
    pag: &Pag,
    query: &AverageEffectQuery,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            let id = GeneralizedAdjustmentIdentifier::new();
            id.identify_pag_envelope(pag, query).map_err(identify_err)
        }
        other if other.is_dag_only() => Err(CausalError::Compile {
            message: format!(
                "DAG-only identification {:?} cannot accept a PAG; use generalized.adjustment",
                other.as_str()
            ),
        }),
        _ => Err(CausalError::Unsupported { message: "unsupported PAG identifier" }),
    }
}

/// General ID over an ADMG for an average-effect query.
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_admg(
    identifier: IdentifierId,
    admg: &antecedent_graph::Admg,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, CausalError> {
    match identifier {
        IdentifierId::GeneralId => {
            let id = IdIdentifier::new();
            let prepared = id.prepare(admg).map_err(identify_err)?;
            let mut id_ws = IdentificationWorkspace::default();
            let result = id
                .identify(&prepared, &CausalQuery::AverageEffect(query.clone()), &mut id_ws)
                .map_err(identify_err)?;
            require_identified(&result)?;
            Ok(result)
        }
        other => Err(CausalError::Compile {
            message: format!(
                "ADMG ATE requires identifier \"general.id\"; got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// Provenance `(artifact_id, operation)` for an identifier id.
#[must_use]
pub fn identify_provenance_step(identifier: IdentifierId) -> (&'static str, &'static str) {
    identifier_data(identifier).provenance
}

/// Provenance `(artifact_id, operation)` for an estimator id.
#[must_use]
pub fn estimate_provenance_step(estimator: EstimatorId) -> (&'static str, &'static str) {
    estimator_data(estimator).provenance
}

/// Run a frequentist static estimator by strategy spec (excludes `rd.sharp` / `bayesian.gcomp`).
///
/// [`EstimatorSpec::Default`] builds a fresh estimator and applies
/// `bootstrap_replicates` / `overlap_policy` / `population_registry` exactly as the
/// closed-set id-only path always has. Every other [`EstimatorSpec`] variant carries a
/// caller-configured estimator that is used verbatim — `bootstrap_replicates`,
/// `overlap_policy`, and `population_registry` are ignored in that case (the builder
/// refuses the conflicting-configuration case before this is ever called).
///
/// # Errors
///
/// Unknown estimator or estimation failure.
pub fn estimate_static_effect(
    spec: &EstimatorSpec,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<&PopulationRegistry>,
    ctx: &ExecutionContext,
    workspaces: &mut StaticEstimateWorkspaces,
) -> Result<EffectEstimate, CausalError> {
    match spec {
        EstimatorSpec::Default(id) => estimate_static_effect_default(
            *id,
            data,
            estimand,
            query,
            assumptions,
            bootstrap_replicates,
            overlap_policy,
            population_registry,
            ctx,
            workspaces,
        ),
        EstimatorSpec::LinearAdjustmentAte(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.linear, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::PropensityWeighting(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::PropensityMatching(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::PropensityStratification(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::DistanceMatching(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::Aipw(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, &mut workspaces.aipw, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::GlmAdjustment(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = GlmAdjustmentWorkspace::default();
            cfg.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::FrontDoorTwoStage(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = FrontDoorWorkspace::default();
            cfg.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::IvWald(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            cfg.fit(&prep, ctx, assumptions).map_err(est_err)
        }
        EstimatorSpec::Iv2Sls(cfg) => {
            let prep = cfg.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = TwoStageLeastSquaresWorkspace::default();
            cfg.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
    }
}

/// [`EstimatorSpec::Default`] path: construct a fresh estimator by id and apply the
/// same `bootstrap_replicates` / `overlap_policy` / `population_registry` defaulting
/// [`estimate_static_effect`] has always applied (byte-identical to the pre-`EstimatorSpec`
/// behavior).
fn estimate_static_effect_default(
    estimator: EstimatorId,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<&PopulationRegistry>,
    ctx: &ExecutionContext,
    workspaces: &mut StaticEstimateWorkspaces,
) -> Result<EffectEstimate, CausalError> {
    match estimator {
        EstimatorId::LinearAdjustmentAte => {
            let mut est = LinearAdjustmentAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            est.overlap = OverlapPolicy::ExplicitOverride;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.linear, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityWeighting => {
            let mut est = PropensityWeighting::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityMatching => {
            let mut est = PropensityMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityStratification => {
            let mut est = PropensityStratification::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::DistanceMatching => {
            let mut est = DistanceMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.propensity, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::Aipw => {
            let mut est = AipwAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, &mut workspaces.aipw, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::GlmAdjustment => {
            let mut est = GlmAdjustmentAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = GlmAdjustmentWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::FrontDoorTwoStage => {
            let mut est = FrontDoorTwoStage::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = FrontDoorWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::IvWald => {
            let mut est = WaldIv::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::Iv2Sls => {
            let mut est = TwoStageLeastSquares::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = TwoStageLeastSquaresWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        _ => Err(CausalError::Unsupported { message: "unknown static estimator" }),
    }
}

/// Shared estimate→refute scratch for static ATE hot paths.
#[derive(Clone, Debug, Default)]
pub struct StaticEstimateWorkspaces {
    /// OLS linear-adjustment scratch.
    pub linear: EstimationWorkspace,
    /// Propensity / matching scratch.
    pub propensity: PropensityEstimationWorkspace,
    /// AIPW propensity + outcome scratch.
    pub aipw: AipwWorkspace,
}

fn est_err(e: EstimationError) -> CausalError {
    CausalError::from(e)
}

fn identify_err(e: IdentificationError) -> CausalError {
    CausalError::from(e)
}
