//! Identifier / estimator strategy tables for plan compilation and static execution
//! (DESIGN.md §21.2). Incremental extraction from the analysis workflow — does not
//! replace [`crate::CausalAnalysis`] / plans / [`crate::CausalAnalysisResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

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
    BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier, IdentificationError,
    IdentificationResult, IdentificationStatus, InstrumentalVariableIdentifier,
};

use crate::error::AnalysisError;

/// Default identifier id when the builder omits one.
pub const DEFAULT_IDENTIFIER: &str = "backdoor.adjustment";

/// Default estimator id when the builder omits one.
pub const DEFAULT_ESTIMATOR: &str = "linear.adjustment.ate";

/// Compile-time allowlist of identifier/estimator pairs for the static ATE path.
///
/// # Errors
///
/// Unknown ids or incompatible pairs.
pub fn validate_static_pair(identifier: &str, estimator: &str) -> Result<(), AnalysisError> {
    let supported = matches!(
        (identifier, estimator),
        (
            "backdoor.adjustment" | "backdoor.efficient",
            "linear.adjustment.ate"
                | "propensity.weighting"
                | "propensity.matching"
                | "propensity.stratification"
                | "distance.matching"
                | "aipw"
                | "glm.adjustment"
                | "bayesian.gcomp"
        ) | ("frontdoor", "frontdoor.two_stage")
            | ("iv", "iv.wald" | "iv.2sls")
            | ("rd.sharp", "rd.sharp")
    );
    if !supported {
        return Err(AnalysisError::Compile {
            message: format!(
                "identifier {identifier:?} is not compatible with estimator {estimator:?}"
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
    identifier: &str,
    graph: &Dag,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, AnalysisError> {
    let q = CausalQuery::AverageEffect(query.clone());
    let result = match identifier {
        "backdoor.adjustment" => {
            let id = BackdoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, &q).map_err(identify_err)?
        }
        "backdoor.efficient" => {
            let id = EfficientBackdoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, &q).map_err(identify_err)?
        }
        "frontdoor" => {
            let id = FrontDoorIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, &q).map_err(identify_err)?
        }
        "iv" => {
            let id = InstrumentalVariableIdentifier::new();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, &q).map_err(identify_err)?
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
pub fn identify_provenance_step(identifier: &str) -> (&'static str, &'static str) {
    match identifier {
        "backdoor.adjustment" => ("identify.backdoor", "identify.backdoor"),
        "backdoor.efficient" => ("identify.efficient_backdoor", "identify.efficient_backdoor"),
        "frontdoor" => ("identify.frontdoor", "identify.frontdoor"),
        "iv" => ("identify.iv", "identify.iv"),
        _ => ("identify.unknown", "identify.unknown"),
    }
}

/// Provenance `(artifact_id, operation)` for an estimator id.
#[must_use]
pub fn estimate_provenance_step(estimator: &str) -> (&'static str, &'static str) {
    match estimator {
        "linear.adjustment.ate" => ("estimate.linear_adjustment", "estimate.linear_adjustment_ate"),
        "propensity.weighting" => ("estimate.propensity", "estimate.propensity_weighting"),
        "propensity.matching" => ("estimate.propensity", "estimate.propensity_matching"),
        "propensity.stratification" => {
            ("estimate.propensity", "estimate.propensity_stratification")
        }
        "distance.matching" => ("estimate.matching", "estimate.distance_matching"),
        "aipw" => ("estimate.aipw", "estimate.aipw"),
        "glm.adjustment" => ("estimate.glm_adjustment", "estimate.glm_adjustment_ate"),
        "frontdoor.two_stage" => ("estimate.frontdoor", "estimate.frontdoor_two_stage"),
        "iv.wald" => ("estimate.iv", "estimate.wald_iv"),
        "iv.2sls" => ("estimate.iv", "estimate.two_stage_least_squares"),
        "bayesian.gcomp" => ("estimate.bayesian_gcomp", "estimate.bayesian_gcomp"),
        _ => ("estimate.unknown", "estimate.unknown"),
    }
}

/// Run a frequentist static estimator by strategy id (excludes `rd.sharp` / `bayesian.gcomp`).
///
/// # Errors
///
/// Unknown estimator or estimation failure.
pub fn estimate_static_effect(
    estimator: &str,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, AnalysisError> {
    match estimator {
        "linear.adjustment.ate" => {
            let mut est = LinearAdjustmentAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            est.overlap = OverlapPolicy::ExplicitOverride;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = EstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "propensity.weighting" => {
            let mut est = PropensityWeighting::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "propensity.matching" => {
            let mut est = PropensityMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "propensity.stratification" => {
            let mut est = PropensityStratification::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "distance.matching" => {
            let mut est = DistanceMatching::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = PropensityEstimationWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "aipw" => {
            let mut est = AipwAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = AipwWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "glm.adjustment" => {
            let mut est = GlmAdjustmentAte::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = GlmAdjustmentWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "frontdoor.two_stage" => {
            let mut est = FrontDoorTwoStage::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = FrontDoorWorkspace::default();
            est.fit(&prep, &mut ws, ctx, assumptions).map_err(est_err)
        }
        "iv.wald" => {
            let mut est = WaldIv::new();
            est.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.fit(&prep, ctx, assumptions).map_err(est_err)
        }
        "iv.2sls" => {
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
