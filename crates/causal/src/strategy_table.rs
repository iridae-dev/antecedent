//! Identifier / estimator strategy tables for plan compilation and static execution
//! (DESIGN.md §21.2). Incremental extraction from the analysis workflow — does not
//! replace [`crate::CausalAnalysis`] / plans / [`crate::CausalAnalysisResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext};
use causal_data::TabularData;
use causal_estimate::{
    AipwAte, AipwWorkspace, DistanceMatching, EffectEstimate, EstimationError, EstimationWorkspace,
    FrontDoorTwoStage, FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, TwoStageLeastSquares,
    TwoStageLeastSquaresWorkspace, WaldIv,
};
use causal_expr::IdentifiedEstimand;
use causal_graph::Dag;
use causal_identify::{
    AutoIdentifier, BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier,
    IdIdentifier, IdentificationError, IdentificationResult, IdentificationStatus,
    IdentificationWorkspace, InstrumentalVariableIdentifier,
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
    /// AutoIdentifier — all applicable estimands, no silent estimator choice.
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
            Self::Other(s) => s.as_ref(),
        }
    }

    /// Parallel-task dimension label for physical planning.
    #[must_use]
    pub const fn parallel_task_dimension(&self) -> &'static str {
        match self {
            Self::TemporalLinearAdjustment
            | Self::LinearAdjustmentAte
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
            Self::TemporalLinearAdjustment => "ols.faer.temporal",
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
    let supported = matches!(
        (&identifier, &estimator),
        (
            IdentifierId::BackdoorAdjustment | IdentifierId::BackdoorEfficient,
            EstimatorId::LinearAdjustmentAte
                | EstimatorId::PropensityWeighting
                | EstimatorId::PropensityMatching
                | EstimatorId::PropensityStratification
                | EstimatorId::DistanceMatching
                | EstimatorId::Aipw
                | EstimatorId::GlmAdjustment
                | EstimatorId::BayesianGcomp
        ) | (IdentifierId::Frontdoor, EstimatorId::FrontDoorTwoStage)
            | (IdentifierId::Iv, EstimatorId::IvWald | EstimatorId::Iv2Sls)
            | (IdentifierId::RdSharp, EstimatorId::RdSharp)
    );
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
        (
            IdentifierId::GeneralId | IdentifierId::Auto,
            EstimatorId::FunctionalDistribution
        )
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
        (
            IdentifierId::PathSpecificNatural | IdentifierId::Auto,
            EstimatorId::FunctionalEffect
        )
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
            // Conditional Distribution → IDC.
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
            let id = AutoIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        _ => {
            return Err(AnalysisError::Unsupported { message: "unknown static identifier" });
        }
    };
    if result.status != IdentificationStatus::NonparametricallyIdentified {
        return Err(AnalysisError::Compile { message: "effect not identified".into() });
    }
    Ok(result)
}

/// Provenance `(artifact_id, operation)` for an identifier id.
#[must_use]
pub fn identify_provenance_step(identifier: impl Into<IdentifierId>) -> (&'static str, &'static str) {
    match identifier.into() {
        IdentifierId::BackdoorAdjustment => ("identify.backdoor", "identify.backdoor"),
        IdentifierId::BackdoorEfficient => {
            ("identify.efficient_backdoor", "identify.efficient_backdoor")
        }
        IdentifierId::Frontdoor => ("identify.frontdoor", "identify.frontdoor"),
        IdentifierId::Iv => ("identify.iv", "identify.iv"),
        IdentifierId::GeneralId => ("identify.general_id", "identify.general_id"),
        IdentifierId::PathSpecificNatural => ("identify.path_specific", "identify.path_specific"),
        IdentifierId::Auto => ("identify.auto", "identify.auto"),
        _ => ("identify.unknown", "identify.unknown"),
    }
}

/// Provenance `(artifact_id, operation)` for an estimator id.
#[must_use]
pub fn estimate_provenance_step(estimator: impl Into<EstimatorId>) -> (&'static str, &'static str) {
    match estimator.into() {
        EstimatorId::LinearAdjustmentAte => {
            ("estimate.linear_adjustment", "estimate.linear_adjustment_ate")
        }
        EstimatorId::PropensityWeighting => ("estimate.propensity", "estimate.propensity_weighting"),
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
        EstimatorId::BayesianGcomp => ("estimate.bayesian_gcomp", "estimate.bayesian_gcomp"),
        EstimatorId::FunctionalDistribution => {
            ("estimate.functional_distribution", "estimate.functional_distribution")
        }
        EstimatorId::FunctionalEffect => ("estimate.functional_effect", "estimate.functional_effect"),
        _ => ("estimate.unknown", "estimate.unknown"),
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
