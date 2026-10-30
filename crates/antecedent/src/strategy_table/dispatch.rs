//! Identify / estimate dispatch for closed-set strategy ids.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, PopulationRegistry,
    TemporalEffectQuery,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AipwWorkspace, CausalForest, DistanceMatching, DmlAte, DrLearner, EffectEstimate,
    EstimationError, EstimationWorkspace, FrontDoorFunctional, FrontDoorTwoStage,
    FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace, LinearAdjustmentAte,
    OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching, PropensityStratification,
    PropensityWeighting, TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Dag, Pag};
use antecedent_identify::{
    AutoIdentifier, BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier,
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier, IdIdentifier,
    IdentificationEnvelope, IdentificationError, IdentificationResult, IdentificationWorkspace,
    InstrumentalVariableIdentifier, ResponseIdentifier, TemporalClassEnvelope,
};

use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;

use super::{EstimatorId, IdentifierId, estimator_data, identifier_data, require_identified};

/// Run the identifier named by `identifier` against `graph`/`query`.
///
/// # Errors
///
/// Unknown identifier, identification failure, or non-identified status.
pub(crate) fn identify_static(
    identifier: IdentifierId,
    graph: &Dag,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, CausalError> {
    identify_static_query(identifier, graph, &CausalQuery::AverageEffect(query.clone()))
}

/// Obtain a modifier-aware DAG backdoor certificate for a conditional query.
/// Keep the query variant until this boundary so marginal-only strategies cannot
/// be mistaken for conditional certificates. The returned AverageEffect result
/// is the backdoor certificate used by the conditional estimator, not an IDC expression.
pub(crate) fn identify_conditional_static_in_context(
    identifier: IdentifierId,
    graph: &Dag,
    query: &antecedent_core::ConditionalEffectQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationResult, CausalError> {
    query.validate()?;
    if !matches!(
        identifier,
        IdentifierId::BackdoorAdjustment | IdentifierId::BackdoorEfficient | IdentifierId::Auto
    ) {
        return Err(CausalError::Unsupported {
            message: "conditional effects require a modifier-aware backdoor identifier",
        });
    }
    identify_static_query_in_context(
        identifier,
        graph,
        &CausalQuery::AverageEffect(query.inner.clone()),
        None,
        context,
    )
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

const fn static_identifier_refusal(identifier: IdentifierId) -> &'static str {
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            "identifier \"generalized.adjustment\" requires a PAG \
             (supply AcceptedGraph::pag(..) / FCI / RFCI output, not a static DAG)"
        }
        IdentifierId::RdSharp => {
            "identifier \"rd.sharp\" is not a graph-based static identifier; \
             select estimator \"rd.sharp\" with builder.rd_config(...)"
        }
        IdentifierId::TemporalBackdoorUnfolded => {
            "identifier \"temporal.backdoor.unfolded\" requires a temporal graph \
             and TemporalEffect query"
        }
        IdentifierId::TemporalIvUnfolded => {
            "identifier \"temporal.iv.unfolded\" requires a temporal graph \
             and TemporalEffect query"
        }
        IdentifierId::TransportSid => {
            "identifier \"transport.sid\" requires a TransportQuery, selection \
             diagram, and the staged transport path"
        }
        IdentifierId::InterferenceDesign => {
            "identifier \"interference.design\" requires an InterferenceQuery and \
             the staged interference path"
        }
        _ => "identifier is not a static DAG identifier",
    }
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
    identify_static_query_in_context(identifier, graph, query, rd, None)
}

/// [`identify_static_query_with_rd`] under the caller's execution context: the exact
/// adjustment searches (back-door, efficient back-door, and the automatic route's back-door)
/// lease their storage from the context's admission pool before allocating it and stop at its
/// cancellation token.
///
/// # Errors
///
/// As [`identify_static_query_with_rd`]; a pool that cannot fund the search's graph copies is a
/// budget refusal, never a non-identification.
pub fn identify_static_query_in_context(
    identifier: IdentifierId,
    graph: &Dag,
    query: &CausalQuery,
    rd: Option<antecedent_identify::SharpRdConfig>,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationResult, CausalError> {
    let mut id_ws = IdentificationWorkspace::default();
    let result = match identifier {
        IdentifierId::BackdoorAdjustment => {
            let mut id = BackdoorIdentifier::new();
            id.config.context = context.cloned();
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::BackdoorEfficient => {
            let mut id = EfficientBackdoorIdentifier::new();
            id.config.context = context.cloned();
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
        IdentifierId::GcmParametric => {
            let CausalQuery::Counterfactual(q) = query else {
                return Err(CausalError::Unsupported {
                    message: "gcm.parametric requires Counterfactual",
                });
            };
            q.validate()?;
            let single = !q.allow_nested
                && q.outcomes.len() == 1
                && q.interventions.len() == 1
                && q.joint_control.is_empty();
            if single {
                let antecedent_core::Intervention::Set { variable, value } = &q.interventions[0]
                else {
                    return Err(CausalError::Unsupported {
                        message: "staged Counterfactual requires Set",
                    });
                };
                if value.as_f64().is_none_or(|v| !v.is_finite())
                    || variable.as_usize() >= graph.node_count()
                    || q.outcomes[0].as_usize() >= graph.node_count()
                {
                    return Err(CausalError::Unsupported {
                        message: "invalid counterfactual variables or level",
                    });
                }
                crate::analysis::parametric_scm_identification(
                    query.clone(),
                    *variable,
                    q.outcomes[0],
                )
                .0
            } else {
                // Several outcomes, simultaneous interventions or nested interventions: the
                // checked operation owns the worlds' levels; identification names every
                // intervened variable and outcome the graph holds.
                let treatments = q.active_variables();
                if treatments.is_empty()
                    || treatments
                        .iter()
                        .chain(q.outcomes.iter())
                        .any(|v| v.as_usize() >= graph.node_count())
                {
                    return Err(CausalError::Unsupported {
                        message: "invalid counterfactual variables or level",
                    });
                }
                crate::analysis::parametric_scm_identification_over(
                    query.clone(),
                    &treatments,
                    &q.outcomes,
                )
                .0
            }
        }
        IdentifierId::ResponseBackdoor => {
            let mut id = ResponseIdentifier::new();
            id.backdoor.config.context = context.cloned();
            let prepared =
                id.prepare_with_assumptions(graph, AssumptionSet::new()).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::Auto => {
            let mut id = AutoIdentifier::new();
            id.backdoor.config.context = context.cloned();
            id.efficient.config.context = context.cloned();
            if let Some(cfg) = rd {
                id = id.with_rd(cfg);
            }
            let prepared = id.prepare(graph).map_err(identify_err)?;
            id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?
        }
        IdentifierId::GeneralizedAdjustment
        | IdentifierId::RdSharp
        | IdentifierId::TemporalBackdoorUnfolded
        | IdentifierId::TemporalIvUnfolded
        | IdentifierId::TransportSid
        | IdentifierId::InterferenceDesign
        | IdentifierId::RandomizedDesign
        | IdentifierId::ContinuousDoseExchangeability => {
            return Err(CausalError::Unsupported {
                message: static_identifier_refusal(identifier),
            });
        }
    };
    require_identified(&result)?;
    Ok(result)
}

/// Class-aware identification over a CPDAG (MEC completion envelope).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_cpdag(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::Cpdag,
    query: &AverageEffectQuery,
) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, CausalError> {
    identify_cpdag_in_context(identifier, cpdag, query, None)
}

/// [`identify_cpdag`] under the caller's execution context: the completion enumeration and the
/// per-completion exact searches lease their storage from its admission pool before allocating
/// and stop at its cancellation token.
///
/// # Errors
///
/// As [`identify_cpdag`].
pub fn identify_cpdag_in_context(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::Cpdag,
    query: &AverageEffectQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, CausalError> {
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            let mut id = GeneralizedAdjustmentIdentifier::new();
            id.config.context = context.cloned();
            id.identify_cpdag_envelope(cpdag, query).map_err(identify_err)
        }
        other => reject_non_generalized_identifier(other, "CPDAG", "unsupported CPDAG identifier"),
    }
}

/// Class-aware identification over a PAG (generalized adjustment envelope).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub(crate) fn identify_pag(
    identifier: IdentifierId,
    pag: &Pag,
    query: &AverageEffectQuery,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    identify_pag_in_context(identifier, pag, query, None)
}

/// [`identify_pag`] under the caller's execution context (see [`identify_cpdag_in_context`]).
///
/// # Errors
///
/// As [`identify_pag`].
pub(crate) fn identify_pag_in_context(
    identifier: IdentifierId,
    pag: &Pag,
    query: &AverageEffectQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            let mut id = GeneralizedAdjustmentIdentifier::new();
            id.config.context = context.cloned();
            id.identify_pag_envelope(pag, query).map_err(identify_err)
        }
        other => reject_non_generalized_identifier(other, "PAG", "unsupported PAG identifier"),
    }
}

/// Class-aware identification over a `TemporalCpdag` (MEC `TemporalDag` envelope).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_temporal_cpdag(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::TemporalCpdag,
    query: &TemporalEffectQuery,
) -> Result<TemporalClassEnvelope, CausalError> {
    identify_temporal_cpdag_configured(
        identifier,
        cpdag,
        query,
        GeneralizedAdjustmentConfig::default(),
    )
}

/// Class-aware `TemporalCpdag` identification with an explicit completion cap.
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_temporal_cpdag_configured(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::TemporalCpdag,
    query: &TemporalEffectQuery,
    config: GeneralizedAdjustmentConfig,
) -> Result<TemporalClassEnvelope, CausalError> {
    identify_generalized_temporal_class(
        identifier,
        "TemporalCpdag",
        "unsupported TemporalCpdag identifier",
        config,
        cpdag,
        |id, graph| id.identify_temporal_cpdag_envelope(graph, query),
    )
}

/// Class-aware identification over a `TemporalPag` (`TemporalDag` completion envelope).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_temporal_pag(
    identifier: IdentifierId,
    pag: &antecedent_graph::TemporalPag,
    query: &TemporalEffectQuery,
) -> Result<TemporalClassEnvelope, CausalError> {
    identify_temporal_pag_configured(identifier, pag, query, GeneralizedAdjustmentConfig::default())
}

/// Class-aware `TemporalPag` identification with an explicit completion cap.
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_temporal_pag_configured(
    identifier: IdentifierId,
    pag: &antecedent_graph::TemporalPag,
    query: &TemporalEffectQuery,
    config: GeneralizedAdjustmentConfig,
) -> Result<TemporalClassEnvelope, CausalError> {
    identify_generalized_temporal_class(
        identifier,
        "TemporalPag",
        "unsupported TemporalPag identifier",
        config,
        pag,
        |id, graph| id.identify_temporal_pag_envelope(graph, query),
    )
}

/// General ID over an ADMG for an average-effect query.
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub(crate) fn identify_admg(
    identifier: IdentifierId,
    admg: &antecedent_graph::Admg,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, CausalError> {
    identify_admg_query(identifier, admg, &CausalQuery::AverageEffect(query.clone()))
}

/// The general-ID product over an ADMG, whatever its status: the refusal is raised by the
/// caller that needs an identified product, never by the algorithm that produced it.
fn general_id_result(
    admg: &antecedent_graph::Admg,
    query: &CausalQuery,
) -> Result<IdentificationResult, CausalError> {
    let id = IdIdentifier::new();
    let prepared = id.prepare(admg).map_err(identify_err)?;
    let mut id_ws = IdentificationWorkspace::default();
    if matches!(query, CausalQuery::Distribution(q) if !q.conditioning.is_empty()) {
        let idc = antecedent_identify::IdcIdentifier::new();
        idc.identify(&prepared, query, &mut id_ws).map_err(identify_err)
    } else {
        id.identify(&prepared, query, &mut id_ws).map_err(identify_err)
    }
}

/// The not-identified general-ID product of `query` over `admg`, with its status, its
/// derivation and the query it answers; `None` when the algorithm identifies the query or
/// cannot read it. This is the product [`identify_admg_query`] refuses to return.
pub(crate) fn not_identified_admg(
    admg: &antecedent_graph::Admg,
    query: &CausalQuery,
) -> Option<IdentificationResult> {
    general_id_result(admg, query)
        .ok()
        .filter(|result| result.status == antecedent_core::IdentificationStatus::NotIdentified)
}

/// General ID over an ADMG for a static query (ATE or interventional distribution).
///
/// # Errors
///
/// Unsupported identifier or identification failure.
pub fn identify_admg_query(
    identifier: IdentifierId,
    admg: &antecedent_graph::Admg,
    query: &CausalQuery,
) -> Result<IdentificationResult, CausalError> {
    match identifier {
        IdentifierId::GeneralId => {
            let result = general_id_result(admg, query)?;
            require_identified(&result)?;
            Ok(result)
        }
        IdentifierId::PathSpecificNatural => {
            // A path-restricted natural effect (or a natural direct effect, which reads the
            // mediator as the path node) on a mixed graph: Avin-Shpitser-Pearl witness check,
            // then the surgical total effect, the edge g-formula or Shpitser's district
            // formula, whichever the graph's latent confounding calls for.
            let id = antecedent_identify::PathSpecificIdentifier::new();
            let prepared = id.prepare(admg).map_err(identify_err)?;
            let mut id_ws = IdentificationWorkspace::default();
            let result = id.identify(&prepared, query, &mut id_ws).map_err(identify_err)?;
            require_identified(&result)?;
            Ok(result)
        }
        other => Err(CausalError::Compile {
            message: format!(
                "ADMG identification requires identifier \"general.id\" or \
                 \"path_specific.natural\"; got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// Provenance `(artifact_id, operation)` for an identifier id.
#[must_use]
pub(crate) fn identify_provenance_step(identifier: IdentifierId) -> (&'static str, &'static str) {
    identifier_data(identifier).provenance
}

/// Provenance `(artifact_id, operation)` for an estimator id.
#[must_use]
pub(crate) fn estimate_provenance_step(estimator: EstimatorId) -> (&'static str, &'static str) {
    estimator_data(estimator).provenance
}

/// What every static estimator prepares against and fits under.
#[derive(Clone, Copy)]
struct FitTarget<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a AverageEffectQuery,
    ctx: &'a ExecutionContext,
}

/// `prepare` then `fit` one estimator, lifting both errors; the optional last argument is the
/// workspace the estimator's `fit` takes.
macro_rules! prepare_fit {
    ($est:expr, $target:expr, $assumptions:expr) => {{
        let prep = $est.prepare($target.data, $target.estimand, $target.query).map_err(est_err)?;
        $est.fit(&prep, $target.ctx, $assumptions).map_err(est_err)
    }};
    ($est:expr, $target:expr, $assumptions:expr, $workspace:expr) => {{
        let prep = $est.prepare($target.data, $target.estimand, $target.query).map_err(est_err)?;
        $est.fit(&prep, $workspace, $target.ctx, $assumptions).map_err(est_err)
    }};
}

/// Apply the facade knobs a default-built estimator accepts. Each estimator names the knobs
/// it takes, so which estimators bootstrap, and which take an overlap policy or a population
/// registry, is stated once per estimator in [`estimate_static_effect_default`]:
///
/// | knob           | effect                                                          |
/// |----------------|-----------------------------------------------------------------|
/// | `inference_sampling` | the study's declared sampling design                      |
/// | `inference_bootstrap` | the requested bootstrap replicates                       |
/// | `no_bootstrap` | zero replicates (matching and IV do not take the facade bootstrap) |
/// | `overlap`      | the overlap policy, when one is set                             |
/// | `registry`     | the population registry                                         |
macro_rules! apply_knobs {
    ($est:ident, $input:ident; $($knob:ident),* $(,)?) => {
        $( apply_knobs!(@one $est, $input, $knob); )*
    };
    (@one $est:ident, $input:ident, inference_sampling) => {
        $est.inference.sampling_design =
            crate::estimator_spec::declared_sampling($input.sampling_design);
    };
    (@one $est:ident, $input:ident, inference_bootstrap) => {
        $est.inference.bootstrap_replicates = $input.bootstrap_replicates;
    };
    (@one $est:ident, $input:ident, no_bootstrap) => {
        $est.inference.bootstrap_replicates = 0;
    };
    (@one $est:ident, $input:ident, overlap) => {
        if let Some(policy) = $input.overlap_policy {
            $est.overlap = policy;
        }
    };
    (@one $est:ident, $input:ident, registry) => {
        $est.population_registry = $input.population_registry.cloned();
    };
}

/// The data, estimand, query and estimator settings a static estimate runs against.
///
/// `bootstrap_replicates`, `overlap_policy` and `population_registry` configure a fresh
/// estimator built from an [`EstimatorId`]; a caller-configured [`EstimatorSpec`] variant
/// carries its own and ignores them.
#[derive(Clone, Copy)]
pub struct StaticEstimateInput<'a> {
    /// Estimation data.
    pub data: &'a TabularData,
    /// Identified estimand to estimate.
    pub estimand: &'a IdentifiedEstimand,
    /// Average-effect query the estimand answers.
    pub query: &'a AverageEffectQuery,
    /// Bootstrap replicates applied to a default-built estimator.
    pub bootstrap_replicates: u32,
    /// Overlap policy applied to a default-built estimator, when set.
    pub overlap_policy: Option<OverlapPolicy>,
    /// Population registry attached to a default-built estimator.
    pub population_registry: Option<&'a PopulationRegistry>,
    /// Sampling design the study declared, applied to a default-built estimator that
    /// bootstraps; a caller-configured estimator carries its own, bound at study build.
    pub sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
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
pub(crate) fn estimate_static_effect(
    spec: &EstimatorSpec,
    input: StaticEstimateInput<'_>,
    assumptions: AssumptionSet,
    ctx: &ExecutionContext,
    workspaces: &mut StaticEstimateWorkspaces,
) -> Result<EffectEstimate, CausalError> {
    let StaticEstimateInput { data, estimand, query, .. } = input;
    let target = FitTarget { data, estimand, query, ctx };
    match spec {
        EstimatorSpec::Default(id) => {
            estimate_static_effect_default(*id, input, assumptions, ctx, workspaces)
        }
        EstimatorSpec::LinearAdjustmentAte(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut workspaces.linear)
        }
        EstimatorSpec::PropensityWeighting(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorSpec::PropensityMatching(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorSpec::PropensityStratification(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorSpec::DistanceMatching(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorSpec::Aipw(cfg) => prepare_fit!(cfg, target, assumptions, &mut workspaces.aipw),
        EstimatorSpec::GlmAdjustment(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut GlmAdjustmentWorkspace::default())
        }
        EstimatorSpec::FrontDoorTwoStage(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut FrontDoorWorkspace::default())
        }
        EstimatorSpec::IvWald(cfg) => prepare_fit!(cfg, target, assumptions),
        EstimatorSpec::Iv2Sls(cfg) => {
            prepare_fit!(cfg, target, assumptions, &mut TwoStageLeastSquaresWorkspace::default())
        }
        EstimatorSpec::Dml(cfg) => prepare_fit!(cfg, target, assumptions),
        EstimatorSpec::DrLearner(cfg) => prepare_fit!(cfg, target, assumptions),
        EstimatorSpec::CausalForest(cfg) => prepare_fit!(cfg, target, assumptions),
    }
}

/// The two-stage least-squares estimator `spec` selects, when it selects one.
fn streamed_iv_estimator(
    spec: &EstimatorSpec,
    estimator: EstimatorId,
) -> Option<TwoStageLeastSquares> {
    match (spec, estimator) {
        (EstimatorSpec::Iv2Sls(cfg), EstimatorId::Iv2Sls) => Some((**cfg).clone()),
        (EstimatorSpec::Default(EstimatorId::Iv2Sls), EstimatorId::Iv2Sls) => {
            Some(TwoStageLeastSquares::new())
        }
        _ => None,
    }
}

/// The inverse-probability-weighting estimator `spec` selects, built as the point stage
/// builds it, when it selects one.
fn streamed_weighting_estimator(
    spec: &EstimatorSpec,
    estimator: EstimatorId,
    input: &StaticEstimateInput<'_>,
) -> Option<PropensityWeighting> {
    match (spec, estimator) {
        (EstimatorSpec::PropensityWeighting(cfg), EstimatorId::PropensityWeighting) => {
            Some((**cfg).clone())
        }
        (
            EstimatorSpec::Default(EstimatorId::PropensityWeighting),
            EstimatorId::PropensityWeighting,
        ) => {
            let mut est = PropensityWeighting::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(input.sampling_design);
            est.inference.bootstrap_replicates = input.bootstrap_replicates;
            if let Some(policy) = input.overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = input.population_registry.cloned();
            Some(est)
        }
        _ => None,
    }
}

/// The cross-fitted AIPW estimator `spec` selects, built as the point stage builds it, when it
/// selects one.
fn streamed_aipw_estimator(
    spec: &EstimatorSpec,
    estimator: EstimatorId,
    input: &StaticEstimateInput<'_>,
) -> Option<AipwAte> {
    match (spec, estimator) {
        (EstimatorSpec::Aipw(cfg), EstimatorId::Aipw) => Some((**cfg).clone()),
        (EstimatorSpec::Default(EstimatorId::Aipw), EstimatorId::Aipw) => {
            let mut est = AipwAte::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(input.sampling_design);
            est.inference.bootstrap_replicates = input.bootstrap_replicates;
            if let Some(policy) = input.overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = input.population_registry.cloned();
            Some(est)
        }
        _ => None,
    }
}

/// Whether [`estimate_static_effect_streamed`] runs `spec` as `estimator` under `input`'s
/// defaults. Two-stage least squares and inverse-probability weighting have a streamed
/// reduction; a configuration with no closed form (a bootstrap, a trim, sampling weights,
/// clustering) does not.
pub(crate) fn streams_static_effect(
    spec: &EstimatorSpec,
    estimator: EstimatorId,
    input: &StaticEstimateInput<'_>,
) -> bool {
    streamed_iv_estimator(spec, estimator).is_some_and(|est| est.supports_streamed_reduction())
        || streamed_weighting_estimator(spec, estimator, input)
            .is_some_and(|est| est.supports_streamed_reduction())
        || streamed_aipw_estimator(spec, estimator, input)
            .is_some_and(|est| est.supports_streamed_reduction())
}

/// Estimate through the streamed reduction of an estimator that has one, or `None` when the
/// estimator (or its configuration) has no closed-form reduction.
///
/// Two-stage least squares, inverse-probability weighting and cross-fitted AIPW stream: the rows are read in
/// bounded batches into the weighted-Gaussian capacity engine and the design is never built.
/// A default-built weighting estimator takes the point-stage bootstrap count in `input`, so it
/// streams exactly when that count is zero.
///
/// # Errors
///
/// The reduction refuses (a rank-deficient block, a separated propensity fit, a ceiling below
/// the batch window) or fails.
pub(crate) fn estimate_static_effect_streamed(
    spec: &EstimatorSpec,
    estimator: EstimatorId,
    input: StaticEstimateInput<'_>,
    assumptions: AssumptionSet,
    fold_seed: u64,
    batch_rows: usize,
    ceiling_bytes: u64,
) -> Result<Option<EffectEstimate>, CausalError> {
    let StaticEstimateInput { data, estimand, query, .. } = input;
    if let Some(est) = streamed_iv_estimator(spec, estimator) {
        if !est.supports_streamed_reduction() {
            return Ok(None);
        }
        let streamed = est
            .estimate_streamed(data, estimand, query, batch_rows, ceiling_bytes)
            .map_err(est_err)?;
        return Ok(Some(streamed.into_effect(assumptions, est.overlap, est.inference.se_kind)));
    }
    if let Some(est) = streamed_weighting_estimator(spec, estimator, &input) {
        if !est.supports_streamed_reduction() {
            return Ok(None);
        }
        let streamed = est
            .estimate_streamed(data, estimand, query, batch_rows, ceiling_bytes)
            .map_err(est_err)?;
        return Ok(Some(streamed.into_effect(assumptions, est.overlap)));
    }
    if let Some(est) = streamed_aipw_estimator(spec, estimator, &input) {
        if !est.supports_streamed_reduction() {
            return Ok(None);
        }
        let streamed = est
            .estimate_streamed(data, estimand, query, fold_seed, batch_rows, ceiling_bytes)
            .map_err(est_err)?;
        return Ok(Some(streamed.into_effect(assumptions, est.overlap)));
    }
    Ok(None)
}

/// [`EstimatorSpec::Default`] path: construct a fresh estimator by id and apply the
/// same `bootstrap_replicates` / `overlap_policy` / `population_registry` defaulting
/// [`estimate_static_effect`] has always applied (byte-identical to the pre-`EstimatorSpec`
/// behavior).
#[allow(
    clippy::too_many_lines,
    reason = "one arm per EstimatorId constructing a fresh estimator under the facade's per-estimator bootstrap and overlap defaults (matching and IV deliberately zero the bootstrap) and fitting it in that estimator's own workspace slot"
)]
fn estimate_static_effect_default(
    estimator: EstimatorId,
    input: StaticEstimateInput<'_>,
    assumptions: AssumptionSet,
    ctx: &ExecutionContext,
    workspaces: &mut StaticEstimateWorkspaces,
) -> Result<EffectEstimate, CausalError> {
    let StaticEstimateInput { data, estimand, query, .. } = input;
    let target = FitTarget { data, estimand, query, ctx };
    match estimator {
        EstimatorId::LinearAdjustmentAte => {
            let mut est = LinearAdjustmentAte::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap);
            est.overlap = OverlapPolicy::ExplicitOverride;
            prepare_fit!(est, target, assumptions, &mut workspaces.linear)
        }
        EstimatorId::PropensityWeighting => {
            let mut est = PropensityWeighting::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap, overlap, registry);
            prepare_fit!(est, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorId::PropensityMatching => {
            let mut est = PropensityMatching::new();
            // NN matching bootstrap is not a valid CI (Abadie–Imbens 2008).
            apply_knobs!(est, input; no_bootstrap, overlap, registry);
            prepare_fit!(est, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorId::PropensityStratification => {
            let mut est = PropensityStratification::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap, overlap, registry);
            prepare_fit!(est, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorId::DistanceMatching => {
            let mut est = DistanceMatching::new();
            // NN matching bootstrap is not a valid CI (Abadie–Imbens 2008).
            apply_knobs!(est, input; no_bootstrap, overlap, registry);
            prepare_fit!(est, target, assumptions, &mut workspaces.propensity)
        }
        EstimatorId::Aipw => {
            let mut est = AipwAte::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap, overlap, registry);
            prepare_fit!(est, target, assumptions, &mut workspaces.aipw)
        }
        EstimatorId::GlmAdjustment => {
            let mut est = GlmAdjustmentAte::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap);
            prepare_fit!(est, target, assumptions, &mut GlmAdjustmentWorkspace::default())
        }
        EstimatorId::FrontDoorTwoStage => {
            let mut est = FrontDoorTwoStage::new();
            apply_knobs!(est, input; inference_sampling, inference_bootstrap);
            prepare_fit!(est, target, assumptions, &mut FrontDoorWorkspace::default())
        }
        EstimatorId::FrontDoorFunctional => {
            let est =
                FrontDoorFunctional::new().with_bootstrap_replicates(input.bootstrap_replicates);
            prepare_fit!(est, target, assumptions)
        }
        EstimatorId::IvWald => {
            let mut est = WaldIv::new();
            // Do not arm IV with the facade bootstrap; that bypasses the weak-instrument gate.
            apply_knobs!(est, input; no_bootstrap);
            prepare_fit!(est, target, assumptions)
        }
        EstimatorId::Iv2Sls => {
            let mut est = TwoStageLeastSquares::new();
            // Do not arm IV with the facade bootstrap; that bypasses the weak-instrument gate.
            apply_knobs!(est, input; no_bootstrap);
            prepare_fit!(est, target, assumptions, &mut TwoStageLeastSquaresWorkspace::default())
        }
        EstimatorId::GcmFit | EstimatorId::GcmFitBayesian | EstimatorId::GcmAttributionBayesian => {
            Err(CausalError::Unsupported { message: "gcm.fit is not a static ATE estimator" })
        }
        EstimatorId::Dml => {
            let mut est = DmlAte::new();
            apply_knobs!(est, input; overlap);
            prepare_fit!(est, target, assumptions)
        }
        EstimatorId::DrLearner => {
            let mut est = DrLearner::new();
            apply_knobs!(est, input; overlap);
            prepare_fit!(est, target, assumptions)
        }
        EstimatorId::CausalForest => {
            let est = CausalForest::new();
            prepare_fit!(est, target, assumptions)
        }
        _ => Err(CausalError::Unsupported { message: "unknown static estimator" }),
    }
}

/// Attach the facade bootstrap SE onto the point estimate `estimate_static_effect` returned
/// for an [`EstimatorSpec::Default`] id with `bootstrap_replicates == 0`, without refitting the
/// point model. The published estimate equals a one-shot `estimate_static_effect` with
/// `bootstrap_replicates` (the estimator's own `fit` is `fit point` + `attach_bootstrap`).
///
/// Only ids whose bootstrap the facade owns are supported; every other id (matching, IV,
/// double-ML, forests, linear adjustment with its own workspace) is refused.
///
/// # Errors
///
/// An id with no attachable bootstrap, preparation failure, or bootstrap failure.
pub fn attach_static_bootstrap(
    estimator: EstimatorId,
    input: StaticEstimateInput<'_>,
    point: EffectEstimate,
    ctx: &ExecutionContext,
    workspaces: &mut StaticEstimateWorkspaces,
) -> Result<EffectEstimate, CausalError> {
    let StaticEstimateInput {
        data,
        estimand,
        query,
        bootstrap_replicates,
        overlap_policy,
        population_registry,
        sampling_design,
    } = input;
    match estimator {
        EstimatorId::PropensityWeighting => {
            let mut est = PropensityWeighting::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(sampling_design);
            est.inference.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.attach_bootstrap(&prep, &mut workspaces.propensity, ctx, point).map_err(est_err)
        }
        EstimatorId::PropensityStratification => {
            let mut est = PropensityStratification::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(sampling_design);
            est.inference.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.attach_bootstrap(&prep, &mut workspaces.propensity, ctx, point).map_err(est_err)
        }
        EstimatorId::Aipw => {
            let mut est = AipwAte::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(sampling_design);
            est.inference.bootstrap_replicates = bootstrap_replicates;
            if let Some(policy) = overlap_policy {
                est.overlap = policy;
            }
            est.population_registry = population_registry.cloned();
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.attach_bootstrap(&prep, &mut workspaces.aipw, ctx, point).map_err(est_err)
        }
        EstimatorId::GlmAdjustment => {
            let mut est = GlmAdjustmentAte::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(sampling_design);
            est.inference.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = GlmAdjustmentWorkspace::default();
            est.attach_bootstrap(&prep, &mut ws, ctx, point).map_err(est_err)
        }
        EstimatorId::FrontDoorTwoStage => {
            let mut est = FrontDoorTwoStage::new();
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(sampling_design);
            est.inference.bootstrap_replicates = bootstrap_replicates;
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            let mut ws = FrontDoorWorkspace::default();
            est.attach_bootstrap(&prep, &mut ws, ctx, point).map_err(est_err)
        }
        EstimatorId::FrontDoorFunctional => {
            let est = FrontDoorFunctional::new().with_bootstrap_replicates(bootstrap_replicates);
            let prep = est.prepare(data, estimand, query).map_err(est_err)?;
            est.attach_bootstrap(&prep, ctx, point).map_err(est_err)
        }
        _ => Err(CausalError::Unsupported {
            message: "the facade bootstrap cannot be attached to this estimator",
        }),
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

fn identify_generalized_temporal_class<G>(
    identifier: IdentifierId,
    class_name: &'static str,
    unsupported: &'static str,
    config: GeneralizedAdjustmentConfig,
    graph: &G,
    run: impl FnOnce(
        &GeneralizedAdjustmentIdentifier,
        &G,
    ) -> Result<TemporalClassEnvelope, IdentificationError>,
) -> Result<TemporalClassEnvelope, CausalError> {
    match identifier {
        IdentifierId::GeneralizedAdjustment => {
            let id = GeneralizedAdjustmentIdentifier { config };
            run(&id, graph).map_err(identify_err)
        }
        other => reject_non_generalized_identifier(other, class_name, unsupported),
    }
}

fn identify_err(e: IdentificationError) -> CausalError {
    CausalError::from(e)
}

/// Shared refusal for the two non-`GeneralizedAdjustment` arms of the class-aware
/// identify dispatchers ([`identify_cpdag`], [`identify_pag`],
/// [`identify_temporal_cpdag_configured`], [`identify_temporal_pag_configured`]): a
/// DAG-only identifier gets a [`CausalError::Compile`] naming the graph class it cannot
/// accept; every other identifier is [`CausalError::Unsupported`].
fn reject_non_generalized_identifier<T>(
    identifier: IdentifierId,
    label: &str,
    unsupported: &'static str,
) -> Result<T, CausalError> {
    if identifier.is_dag_only() {
        Err(CausalError::Compile {
            message: format!(
                "DAG-only identification {:?} cannot accept a {label}; use generalized.adjustment",
                identifier.as_str()
            ),
        })
    } else {
        Err(CausalError::Unsupported { message: unsupported })
    }
}

/// Class-aware response identification covering every intervention target.
///
/// # Errors
/// Unsupported strategy/query or graph errors.
pub fn identify_cpdag_response(
    identifier: IdentifierId,
    graph: &antecedent_graph::Cpdag,
    query: &antecedent_core::ResponseQuery,
) -> Result<IdentificationEnvelope<Dag>, CausalError> {
    identify_cpdag_response_in_context(identifier, graph, query, None)
}

/// Whether a derivative functional names more than one treatment/outcome pair (a Jacobian or
/// directional derivative over several coordinates), so each pair must be identified by the
/// one adjustment set the regression conditions on.
fn names_several_pairs(functional: &antecedent_core::ResponseFunctional) -> bool {
    use antecedent_core::ResponseFunctional as F;
    match functional {
        F::DirectionalDerivative { outcomes, treatments, .. }
        | F::Jacobian { outcomes, treatments, .. } => outcomes.len() * treatments.len() > 1,
        _ => false,
    }
}

/// Whether a response functional is one of the derivative family.
fn is_derivative_functional(functional: &antecedent_core::ResponseFunctional) -> bool {
    use antecedent_core::ResponseFunctional as F;
    matches!(
        functional,
        F::AverageDerivative { .. }
            | F::PointDerivative { .. }
            | F::DirectionalDerivative { .. }
            | F::Jacobian { .. }
    )
}

/// A multi-pair derivative is identified by adjustment on each completion.
fn require_adjustment_identifier(
    identifier: IdentifierId,
    class_name: &'static str,
) -> Result<(), CausalError> {
    if identifier == IdentifierId::GeneralizedAdjustment {
        return Ok(());
    }
    Err(CausalError::Compile {
        message: format!(
            "a derivative over several treatment/outcome pairs on a {class_name} requires \
             generalized.adjustment; got {:?}",
            identifier.as_str()
        ),
    })
}

/// Identify a derivative, elasticity, directional derivative or Jacobian on a latent-confounded
/// ADMG by one common adjustment set per pair.
///
/// When no adjustment set exists the response is reported through the general ID algorithm
/// on each pair: a pair with a hedge is not identified by any estimator (Shpitser and Pearl
/// 2006), and that typed refusal carries the hedge certificate. A pair that ID identifies
/// through a functional that is not an adjustment has no continuous-treatment estimator
/// here and is refused as such.
///
/// # Errors
///
/// Identification failure, a hedge on a pair, or an ID-only functional.
pub(crate) fn identify_admg_derivative(
    admg: &antecedent_graph::Admg,
    query: &antecedent_core::ResponseQuery,
) -> Result<IdentificationResult, CausalError> {
    use antecedent_core::{DerivativeScale, ResponseFunctional};
    let adjusted = antecedent_identify::identify_admg_response_adjustment(admg, query)
        .map_err(identify_err)?;
    if !matches!(adjusted.status, antecedent_core::IdentificationStatus::NotIdentified) {
        return Ok(adjusted);
    }
    // No adjustment set: the front-door criterion identifies the mean through the mediators.
    if let Some(front_door) = antecedent_identify::identify_admg_front_door_derivative(admg, query)
        .map_err(identify_err)?
    {
        return Ok(front_door);
    }
    let pairs: Vec<(antecedent_core::VariableId, antecedent_core::VariableId)> =
        match &query.functional {
            ResponseFunctional::AverageDerivative { outcome, treatment, .. }
            | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
                vec![(*treatment, *outcome)]
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
            | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
                treatments.iter().flat_map(|t| outcomes.iter().map(move |y| (*t, *y))).collect()
            }
            _ => Vec::new(),
        };
    for (treatment, outcome) in pairs {
        let mut pair_query = query.clone();
        pair_query.functional = ResponseFunctional::PointDerivative {
            outcome,
            treatment,
            at: 0.0,
            order: 1,
            scale: DerivativeScale::Identity,
        };
        let witness = general_id_result(admg, &CausalQuery::Response(pair_query))?;
        // A hedge: the pair's interventional law is not identified by any estimator.
        require_identified(&witness)?;
    }
    Err(CausalError::Unsupported {
        message: "the derivative is identified by the general ID algorithm but by neither an \
                  adjustment set nor a front-door set: it is a general ID functional of the \
                  observed law, and no continuous-treatment estimator evaluates such a functional",
    })
}

/// Class-aware MAG response identification covering every intervention target.
///
/// # Errors
/// Unsupported strategy/query or graph errors.
pub fn identify_pag_response(
    identifier: IdentifierId,
    graph: &Pag,
    query: &antecedent_core::ResponseQuery,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    identify_pag_response_in_context(identifier, graph, query, None)
}

/// Preserve declared work/depth truncation, but never publish a resource interruption as mass.
pub(crate) fn check_class_search_stop(
    stop: Option<antecedent_core::SearchStop>,
) -> Result<(), CausalError> {
    match stop {
        Some(antecedent_core::SearchStop::Cancelled) => {
            Err(CausalError::Cancelled { stage: "identify" })
        }
        Some(antecedent_core::SearchStop::Memory) => Err(CausalError::Resource {
            message: "class completion search cannot be funded by the caller's admission pool"
                .into(),
        }),
        _ => Ok(()),
    }
}

/// One configured class owner shared by static facade routes and posterior workers.
pub(crate) fn class_identifier_in_context(
    context: Option<&ExecutionContext>,
) -> GeneralizedAdjustmentIdentifier {
    let mut identifier = GeneralizedAdjustmentIdentifier::new();
    identifier.config.context = context.cloned();
    identifier
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_cpdag_path_specific_in_context(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::Cpdag,
    query: &CausalQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, CausalError> {
    match identifier {
        IdentifierId::PathSpecificNatural => class_identifier_in_context(context)
            .identify_cpdag_path_specific_envelope(cpdag, query)
            .map_err(identify_err),
        other => Err(CausalError::Compile {
            message: format!(
                "CPDAG path-specific identification requires identifier \"path_specific.natural\"; \
                 got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_pag_path_specific_in_context(
    identifier: IdentifierId,
    pag: &Pag,
    query: &CausalQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    match identifier {
        IdentifierId::PathSpecificNatural => class_identifier_in_context(context)
            .identify_pag_path_specific_envelope(pag, query)
            .map_err(identify_err),
        other => Err(CausalError::Compile {
            message: format!(
                "PAG path-specific identification requires identifier \"path_specific.natural\"; \
                 got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_cpdag_distribution_in_context(
    identifier: IdentifierId,
    cpdag: &antecedent_graph::Cpdag,
    query: &CausalQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, CausalError> {
    match identifier {
        IdentifierId::GeneralId => class_identifier_in_context(context)
            .identify_cpdag_distribution_envelope(cpdag, query)
            .map_err(identify_err),
        other => Err(CausalError::Compile {
            message: format!(
                "CPDAG distribution identification requires identifier \"general.id\"; got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_pag_distribution_in_context(
    identifier: IdentifierId,
    pag: &Pag,
    query: &CausalQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    match identifier {
        IdentifierId::GeneralId => class_identifier_in_context(context)
            .identify_pag_distribution_envelope(pag, query)
            .map_err(identify_err),
        other => Err(CausalError::Compile {
            message: format!(
                "PAG distribution identification requires identifier \"general.id\"; got {:?}",
                other.as_str()
            ),
        }),
    }
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_cpdag_response_in_context(
    identifier: IdentifierId,
    graph: &antecedent_graph::Cpdag,
    query: &antecedent_core::ResponseQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<Dag>, CausalError> {
    let witness = crate::analysis::response_witness_ate(query)?;
    if names_several_pairs(&query.functional) {
        return require_adjustment_identifier(identifier, "CPDAG").and_then(|()| {
            class_identifier_in_context(context)
                .identify_cpdag_derivative_pairs(graph, query)
                .map_err(identify_err)
        });
    }
    if query.functional.treatment_ids().len() == 1 {
        if identifier != IdentifierId::GeneralizedAdjustment
            && identifier != IdentifierId::GeneralId
        {
            return identify_cpdag_in_context(identifier, graph, &witness, context);
        }
        return class_identifier_in_context(context)
            .identify_cpdag_response_general(graph, query)
            .map_err(identify_err);
    }
    if identifier != IdentifierId::GeneralizedAdjustment {
        return Err(CausalError::Unsupported {
            message: "joint CPDAG response requires generalized.adjustment",
        });
    }
    class_identifier_in_context(context)
        .identify_cpdag_response_envelope(graph, query)
        .map_err(identify_err)
}

/// The same scientific dispatch under the actual caller context.
pub(crate) fn identify_pag_response_in_context(
    identifier: IdentifierId,
    graph: &Pag,
    query: &antecedent_core::ResponseQuery,
    context: Option<&ExecutionContext>,
) -> Result<IdentificationEnvelope<Pag>, CausalError> {
    let witness = crate::analysis::response_witness_ate(query)?;
    if names_several_pairs(&query.functional) {
        return require_adjustment_identifier(identifier, "PAG").and_then(|()| {
            class_identifier_in_context(context)
                .identify_pag_derivative_pairs(graph, query)
                .map_err(identify_err)
        });
    }
    if query.functional.treatment_ids().len() == 1 {
        let _ = witness;
        if identifier != IdentifierId::GeneralizedAdjustment
            && identifier != IdentifierId::GeneralId
        {
            return identify_pag_in_context(identifier, graph, &witness, context);
        }
        let mut envelope = class_identifier_in_context(context)
            .identify_pag_response_general(graph, query)
            .map_err(identify_err)?;
        if is_derivative_functional(&query.functional) {
            // A completion general ID identifies but no adjustment reaches: front door.
            antecedent_identify::upgrade_id_only_derivative_cases(&mut envelope, query)
                .map_err(identify_err)?;
        }
        return Ok(envelope);
    }
    if identifier != IdentifierId::GeneralizedAdjustment {
        return Err(CausalError::Unsupported {
            message: "joint PAG response requires generalized.adjustment",
        });
    }
    class_identifier_in_context(context)
        .identify_pag_response_envelope(graph, query)
        .map_err(identify_err)
}

#[cfg(test)]
mod class_context_tests {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_graph::{Cpdag, DenseNodeId, Pag};

    use super::{identify_cpdag_in_context, identify_pag_in_context};
    use crate::strategy_table::IdentifierId;

    fn query(treatment: u32, outcome: u32) -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
        )
    }

    /// A context whose pool has `free` of `capacity` bytes left; the returned lease holds the rest.
    fn constrained(
        capacity: u64,
        free: u64,
    ) -> (ExecutionContext, antecedent_core::execution::AdmissionLease, AdmissionPool) {
        let pool = AdmissionPool::fixed(1, capacity);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let held = ctx.acquire_admission(0, capacity - free).unwrap();
        (ctx, held, pool)
    }

    #[test]
    fn a_cpdag_envelope_leases_its_completion_search_from_the_callers_pool() {
        let mut cpdag = Cpdag::with_variables(3);
        let (a, b, c) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        cpdag.insert_undirected(a, b).unwrap();
        cpdag.insert_undirected(b, c).unwrap();
        let q = query(0, 2);
        let (ctx, held, pool) = constrained(1 << 20, 1);
        let starved =
            identify_cpdag_in_context(IdentifierId::GeneralizedAdjustment, &cpdag, &q, Some(&ctx))
                .unwrap_err();
        assert!(matches!(starved, crate::CausalError::Resource { .. }));
        assert_eq!(pool.available().bytes(), 1);
        drop(held);
        let funded = identify_cpdag_in_context(
            IdentifierId::GeneralizedAdjustment,
            &cpdag,
            &q,
            Some(&ExecutionContext::for_tests(1)),
        )
        .unwrap();
        assert_eq!(funded.cases.len(), 3);
    }

    #[test]
    fn a_pag_envelope_leases_its_completions_from_the_callers_pool() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let q = query(0, 1);
        let (ctx, held, pool) = constrained(1 << 20, 1);
        let starved =
            identify_pag_in_context(IdentifierId::GeneralizedAdjustment, &pag, &q, Some(&ctx))
                .unwrap_err();
        assert!(matches!(starved, crate::CausalError::Resource { .. }));
        assert_eq!(pool.available().bytes(), 1);
        drop(held);
        let funded = identify_pag_in_context(
            IdentifierId::GeneralizedAdjustment,
            &pag,
            &q,
            Some(&ExecutionContext::for_tests(1)),
        )
        .unwrap();
        assert!(!funded.cases.is_empty());
    }
}
