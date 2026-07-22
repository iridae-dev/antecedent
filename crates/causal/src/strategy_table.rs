//! Identifier / estimator strategy tables for plan compilation and static execution
//!. Incremental extraction from the analysis workflow — does not
//! replace [`crate::CausalAnalysis`] / plans / [`crate::CausalAnalysisResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, IdentificationStatus,
    PopulationRegistry,
};
use causal_data::TabularData;
use causal_estimate::{
    AipwAte, AipwWorkspace, DistanceMatching, EffectEstimate, EstimationError, EstimationWorkspace,
    FrontDoorTwoStage, FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, TwoStageLeastSquares,
    TwoStageLeastSquaresWorkspace, WaldIv,
};
use causal_expr::{EstimandMethod, IdentifiedEstimand};
use causal_graph::{Dag, Pag};
use causal_identify::{
    AutoIdentifier, BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier,
    GeneralizedAdjustmentIdentifier, IdIdentifier, IdentificationEnvelope, IdentificationError,
    IdentificationResult, IdentificationWorkspace, InstrumentalVariableIdentifier,
};

use crate::error::AnalysisError;

/// Closed set of identification strategies (plus [`IdentifierId::Other`] escape).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
    /// Unknown / extension id (not in the compile-time allowlist).
    Other(Arc<str>),
}

impl IdentifierId {
    /// Parse a wire / builder id string.
    #[must_use]
    pub fn parse(id: &str) -> Self {
        match id {
            "backdoor.adjustment" => Self::BackdoorAdjustment,
            "backdoor.efficient" => Self::BackdoorEfficient,
            "frontdoor" => Self::Frontdoor,
            "iv" => Self::Iv,
            "rd.sharp" => Self::RdSharp,
            "temporal.backdoor.unfolded" => Self::TemporalBackdoorUnfolded,
            "generalized.adjustment" => Self::GeneralizedAdjustment,
            "general.id" => Self::GeneralId,
            "path_specific.natural" => Self::PathSpecificNatural,
            "auto" => Self::Auto,
            other => Self::Other(Arc::from(other)),
        }
    }

    /// Canonical wire id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::BackdoorAdjustment => "backdoor.adjustment",
            Self::BackdoorEfficient => "backdoor.efficient",
            Self::Frontdoor => "frontdoor",
            Self::Iv => "iv",
            Self::RdSharp => "rd.sharp",
            Self::TemporalBackdoorUnfolded => "temporal.backdoor.unfolded",
            Self::GeneralizedAdjustment => "generalized.adjustment",
            Self::GeneralId => "general.id",
            Self::PathSpecificNatural => "path_specific.natural",
            Self::Auto => "auto",
            Self::Other(s) => s.as_ref(),
        }
    }

    /// Whether this identifier requires a DAG (not a raw PAG).
    #[must_use]
    pub const fn is_dag_only(&self) -> bool {
        matches!(
            self,
            Self::BackdoorAdjustment
                | Self::BackdoorEfficient
                | Self::Frontdoor
                | Self::Iv
                | Self::RdSharp
                | Self::TemporalBackdoorUnfolded
                | Self::GeneralId
                | Self::PathSpecificNatural
                | Self::Auto
        )
    }
}

impl From<&str> for IdentifierId {
    fn from(value: &str) -> Self {
        Self::parse(value)
    }
}

impl From<String> for IdentifierId {
    fn from(value: String) -> Self {
        Self::parse(&value)
    }
}

impl From<&Arc<str>> for IdentifierId {
    fn from(value: &Arc<str>) -> Self {
        Self::parse(value.as_ref())
    }
}

impl From<Arc<str>> for IdentifierId {
    fn from(value: Arc<str>) -> Self {
        Self::parse(value.as_ref())
    }
}

/// Closed set of estimators (plus [`EstimatorId::Other`] escape).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
    /// Unknown / extension id.
    Other(Arc<str>),
}

impl EstimatorId {
    /// Parse a wire / builder id string.
    #[must_use]
    pub fn parse(id: &str) -> Self {
        match id {
            "linear.adjustment.ate" => Self::LinearAdjustmentAte,
            "propensity.weighting" => Self::PropensityWeighting,
            "propensity.matching" => Self::PropensityMatching,
            "propensity.stratification" => Self::PropensityStratification,
            "distance.matching" => Self::DistanceMatching,
            "aipw" => Self::Aipw,
            "glm.adjustment" => Self::GlmAdjustment,
            "frontdoor.two_stage" => Self::FrontDoorTwoStage,
            "iv.wald" => Self::IvWald,
            "iv.2sls" => Self::Iv2Sls,
            "rd.sharp" => Self::RdSharp,
            "bayesian.gcomp" => Self::BayesianGcomp,
            "temporal.linear.adjustment" => Self::TemporalLinearAdjustment,
            "functional.distribution" => Self::FunctionalDistribution,
            "functional.effect" => Self::FunctionalEffect,
            "conditional.linear.adjustment" => Self::ConditionalLinearAdjustment,
            "temporal.mediation" => Self::TemporalMediation,
            other => Self::Other(Arc::from(other)),
        }
    }

    /// Canonical wire id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::LinearAdjustmentAte => "linear.adjustment.ate",
            Self::PropensityWeighting => "propensity.weighting",
            Self::PropensityMatching => "propensity.matching",
            Self::PropensityStratification => "propensity.stratification",
            Self::DistanceMatching => "distance.matching",
            Self::Aipw => "aipw",
            Self::GlmAdjustment => "glm.adjustment",
            Self::FrontDoorTwoStage => "frontdoor.two_stage",
            Self::IvWald => "iv.wald",
            Self::Iv2Sls => "iv.2sls",
            Self::RdSharp => "rd.sharp",
            Self::BayesianGcomp => "bayesian.gcomp",
            Self::TemporalLinearAdjustment => "temporal.linear.adjustment",
            Self::FunctionalDistribution => "functional.distribution",
            Self::FunctionalEffect => "functional.effect",
            Self::ConditionalLinearAdjustment => "conditional.linear.adjustment",
            Self::TemporalMediation => "temporal.mediation",
            Self::Other(s) => s.as_ref(),
        }
    }

    /// Parallel-task dimension label for physical planning.
    #[must_use]
    pub const fn parallel_task_dimension(&self) -> &'static str {
        match self {
            Self::TemporalLinearAdjustment
            | Self::TemporalMediation
            | Self::LinearAdjustmentAte
            | Self::ConditionalLinearAdjustment
            | Self::PropensityWeighting
            | Self::PropensityMatching
            | Self::PropensityStratification
            | Self::DistanceMatching
            | Self::Aipw
            | Self::GlmAdjustment
            | Self::FrontDoorTwoStage
            | Self::IvWald
            | Self::Iv2Sls
            | Self::RdSharp
            | Self::FunctionalDistribution
            | Self::FunctionalEffect => "bootstrap.replicate",
            Self::BayesianGcomp | Self::Other(_) => "analysis",
        }
    }

    /// Dense-kernel label recorded on the physical plan.
    #[must_use]
    pub const fn kernel_label(&self) -> &'static str {
        match self {
            Self::TemporalLinearAdjustment | Self::TemporalMediation => "ols.faer.temporal",
            Self::ConditionalLinearAdjustment => "ols.faer.conditional",
            Self::PropensityWeighting => "ipw",
            Self::PropensityMatching | Self::DistanceMatching => "matching",
            Self::PropensityStratification => "propensity.stratification",
            Self::Aipw => "aipw",
            Self::GlmAdjustment => "glm.logit",
            Self::FrontDoorTwoStage => "frontdoor.two_stage",
            Self::IvWald => "iv.wald",
            Self::Iv2Sls => "2sls",
            Self::RdSharp => "rd.local_linear",
            Self::FunctionalDistribution => "functional.distribution",
            Self::FunctionalEffect => "functional.effect",
            Self::LinearAdjustmentAte | Self::BayesianGcomp | Self::Other(_) => "ols.faer",
        }
    }
}

impl From<&str> for EstimatorId {
    fn from(value: &str) -> Self {
        Self::parse(value)
    }
}

impl From<String> for EstimatorId {
    fn from(value: String) -> Self {
        Self::parse(&value)
    }
}

impl From<&Arc<str>> for EstimatorId {
    fn from(value: &Arc<str>) -> Self {
        Self::parse(value.as_ref())
    }
}

impl From<Arc<str>> for EstimatorId {
    fn from(value: Arc<str>) -> Self {
        Self::parse(value.as_ref())
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
    identifier: impl Into<IdentifierId>,
    estimator: impl Into<EstimatorId>,
) -> Result<(), AnalysisError> {
    let identifier = identifier.into();
    let estimator = estimator.into();
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
        return Err(AnalysisError::Compile {
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
pub fn require_identified(result: &IdentificationResult) -> Result<(), AnalysisError> {
    if result.status == IdentificationStatus::NotIdentified || result.estimands.is_empty() {
        return Err(AnalysisError::Compile { message: "effect not identified".into() });
    }
    if !identification_status_acceptable(result.status) {
        return Err(AnalysisError::Compile {
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
        EstimatorId::Other(_) => true,
    }
}

/// Select a single estimand matching the estimator (no silent Auto `.first()`).
///
/// # Errors
///
/// No estimand, or multiple estimands without a unique estimator-compatible match.
pub fn select_estimand(
    identification: &IdentificationResult,
    estimator: impl Into<EstimatorId>,
) -> Result<IdentifiedEstimand, AnalysisError> {
    let estimator = estimator.into();
    let estimands = &identification.estimands;
    if estimands.is_empty() {
        return Err(AnalysisError::Compile { message: "no estimand returned".into() });
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
    Err(AnalysisError::Compile {
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
    identifier: impl Into<IdentifierId>,
    estimator: impl Into<EstimatorId>,
) -> Result<(), AnalysisError> {
    let identifier = identifier.into();
    let estimator = estimator.into();
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::GeneralId | IdentifierId::Auto, EstimatorId::FunctionalDistribution)
    );
    if !supported {
        return Err(AnalysisError::Compile {
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
    identifier: impl Into<IdentifierId>,
    estimator: impl Into<EstimatorId>,
) -> Result<(), AnalysisError> {
    let identifier = identifier.into();
    let estimator = estimator.into();
    let supported = matches!(
        (&identifier, &estimator),
        (IdentifierId::PathSpecificNatural | IdentifierId::Auto, EstimatorId::FunctionalEffect)
    );
    if !supported {
        return Err(AnalysisError::Compile {
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
    identifier: impl Into<IdentifierId>,
    graph: &Dag,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, AnalysisError> {
    identify_static_query(identifier, graph, &CausalQuery::AverageEffect(query.clone()))
}

/// Run a static identifier against an arbitrary [`CausalQuery`].
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub fn identify_static_query(
    identifier: impl Into<IdentifierId>,
    graph: &Dag,
    query: &CausalQuery,
) -> Result<IdentificationResult, AnalysisError> {
    identify_static_query_with_rd(identifier, graph, query, None)
}

/// Like [`identify_static_query`], optionally attaching sharp-RD design config for Auto.
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub fn identify_static_query_with_rd(
    identifier: impl Into<IdentifierId>,
    graph: &Dag,
    query: &CausalQuery,
    rd: Option<causal_identify::SharpRdConfig>,
) -> Result<IdentificationResult, AnalysisError> {
    let identifier = identifier.into();
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
                let idc = causal_identify::IdcIdentifier::new();
                idc.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
            } else {
                id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
            }
        }
        IdentifierId::PathSpecificNatural => {
            let id = causal_identify::PathSpecificIdentifier::new();
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
            return Err(AnalysisError::Unsupported {
                message: "identifier \"generalized.adjustment\" requires a PAG \
                     (use GraphInput::Pag / FCI / RFCI, not a static DAG)",
            });
        }
        IdentifierId::RdSharp => {
            return Err(AnalysisError::Unsupported {
                message: "identifier \"rd.sharp\" is not a graph-based static identifier; \
                     select estimator \"rd.sharp\" with builder.rd_config(...)",
            });
        }
        IdentifierId::TemporalBackdoorUnfolded => {
            return Err(AnalysisError::Unsupported {
                message: "identifier \"temporal.backdoor.unfolded\" requires a temporal graph \
                     and TemporalEffect query",
            });
        }
        IdentifierId::Other(_) => {
            return Err(AnalysisError::Unsupported { message: "unknown static identifier" });
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
    identifier: impl Into<IdentifierId>,
    pag: &Pag,
    query: &AverageEffectQuery,
) -> Result<IdentificationEnvelope<Pag>, AnalysisError> {
    let identifier = identifier.into();
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            let id = GeneralizedAdjustmentIdentifier::new();
            id.identify_pag_envelope(pag, query).map_err(identify_err)
        }
        other if other.is_dag_only() => Err(AnalysisError::Compile {
            message: format!(
                "DAG-only identification {:?} cannot accept a PAG; use generalized.adjustment",
                other.as_str()
            ),
        }),
        _ => Err(AnalysisError::Unsupported { message: "unsupported PAG identifier" }),
    }
}

/// General ID over an ADMG for an average-effect query.
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_admg(
    identifier: impl Into<IdentifierId>,
    admg: &causal_graph::Admg,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, AnalysisError> {
    let identifier = identifier.into();
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
        other => Err(AnalysisError::Compile {
            message: format!(
                "ADMG ATE requires identifier \"general.id\"; got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// Provenance `(artifact_id, operation)` for an identifier id.
#[must_use]
pub fn identify_provenance_step(
    identifier: impl Into<IdentifierId>,
) -> (&'static str, &'static str) {
    match identifier.into() {
        IdentifierId::BackdoorAdjustment => ("identify.backdoor", "identify.backdoor"),
        IdentifierId::BackdoorEfficient => {
            ("identify.efficient_backdoor", "identify.efficient_backdoor")
        }
        IdentifierId::Frontdoor => ("identify.frontdoor", "identify.frontdoor"),
        IdentifierId::Iv => ("identify.iv", "identify.iv"),
        IdentifierId::RdSharp => ("identify.rd_design", "identify.rd_sharp"),
        IdentifierId::TemporalBackdoorUnfolded => {
            ("identify.temporal_backdoor", "identify.temporal_backdoor_unfolded")
        }
        IdentifierId::GeneralizedAdjustment => {
            ("identify.generalized_adjustment", "identify.generalized_adjustment")
        }
        IdentifierId::GeneralId => ("identify.general_id", "identify.general_id"),
        IdentifierId::PathSpecificNatural => ("identify.path_specific", "identify.path_specific"),
        IdentifierId::Auto => ("identify.auto", "identify.auto"),
        IdentifierId::Other(_) => ("identify.unknown", "identify.unknown"),
    }
}

/// Provenance `(artifact_id, operation)` for an estimator id.
#[must_use]
pub fn estimate_provenance_step(estimator: impl Into<EstimatorId>) -> (&'static str, &'static str) {
    match estimator.into() {
        EstimatorId::LinearAdjustmentAte => {
            ("estimate.linear_adjustment", "estimate.linear_adjustment_ate")
        }
        EstimatorId::PropensityWeighting => {
            ("estimate.propensity", "estimate.propensity_weighting")
        }
        EstimatorId::PropensityMatching => ("estimate.propensity", "estimate.propensity_matching"),
        EstimatorId::PropensityStratification => {
            ("estimate.propensity", "estimate.propensity_stratification")
        }
        EstimatorId::DistanceMatching => ("estimate.matching", "estimate.distance_matching"),
        EstimatorId::Aipw => ("estimate.aipw", "estimate.aipw"),
        EstimatorId::GlmAdjustment => ("estimate.glm_adjustment", "estimate.glm_adjustment_ate"),
        EstimatorId::FrontDoorTwoStage => ("estimate.frontdoor", "estimate.frontdoor_two_stage"),
        EstimatorId::IvWald => ("estimate.iv", "estimate.wald_iv"),
        EstimatorId::Iv2Sls => ("estimate.iv", "estimate.two_stage_least_squares"),
        EstimatorId::RdSharp => ("estimate.rd", "estimate.rd_sharp"),
        EstimatorId::BayesianGcomp => ("estimate.bayesian_gcomp", "estimate.bayesian_gcomp"),
        EstimatorId::TemporalLinearAdjustment => {
            ("estimate.temporal_linear", "estimate.temporal_linear_adjustment")
        }
        EstimatorId::FunctionalDistribution => {
            ("estimate.functional_distribution", "estimate.functional_distribution")
        }
        EstimatorId::FunctionalEffect => {
            ("estimate.functional_effect", "estimate.functional_effect")
        }
        EstimatorId::ConditionalLinearAdjustment => {
            ("estimate.conditional_linear", "estimate.conditional_linear_adjustment")
        }
        EstimatorId::TemporalMediation => {
            ("estimate.temporal_mediation", "estimate.temporal_mediation")
        }
        EstimatorId::Other(_) => ("estimate.unknown", "estimate.unknown"),
    }
}

/// Run a frequentist static estimator by strategy id (excludes `rd.sharp` / `bayesian.gcomp`).
///
/// # Errors
///
/// Unknown estimator or estimation failure.
pub fn estimate_static_effect(
    estimator: impl Into<EstimatorId>,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<&PopulationRegistry>,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, AnalysisError> {
    match estimator.into() {
        EstimatorId::LinearAdjustmentAte => {
            let mut est = LinearAdjustmentAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            est.overlap = OverlapPolicy::ExplicitOverride;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = EstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityWeighting => {
            let mut est = PropensityWeighting::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityMatching => {
            let mut est = PropensityMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::PropensityStratification => {
            let mut est = PropensityStratification::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::DistanceMatching => {
            let mut est = DistanceMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        EstimatorId::Aipw => {
            let mut est = AipwAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = AipwWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
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
        _ => Err(AnalysisError::Unsupported { message: "unknown static estimator" }),
    }
}

fn est_err(e: EstimationError) -> AnalysisError {
    AnalysisError::from(e)
}

fn identify_err(e: IdentificationError) -> AnalysisError {
    AnalysisError::from(e)
}
