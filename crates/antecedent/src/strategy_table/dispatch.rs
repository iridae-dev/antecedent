//! Identify / estimate dispatch for closed-set strategy ids.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, PopulationRegistry,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AipwWorkspace, DistanceMatching, EffectEstimate, EstimationError, EstimationWorkspace,
    FrontDoorTwoStage, FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, TwoStageLeastSquares,
    TwoStageLeastSquaresWorkspace, WaldIv,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Dag, Pag};
use antecedent_identify::{
    AutoIdentifier, BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier,
    GeneralizedAdjustmentIdentifier, IdIdentifier, IdentificationEnvelope, IdentificationError,
    IdentificationResult, IdentificationWorkspace, InstrumentalVariableIdentifier,
    ResponseIdentifier,
};

use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;

use super::{EstimatorId, IdentifierId, estimator_data, identifier_data, require_identified};

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
        IdentifierId::ResponseBackdoor => {
            let id = ResponseIdentifier::new();
            let prepared =
                id.prepare_with_assumptions(graph, AssumptionSet::new()).map_err(identify_err)?;
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
