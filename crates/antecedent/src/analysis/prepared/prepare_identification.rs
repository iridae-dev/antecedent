//! Prepare-time identification caches and horizon schedules.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedCpdagIdentification, CachedPagIdentification, CachedStaticIdentification,
    CachedTemporalClassIdentification, CachedTemporalHorizonIdentification,
    CachedTemporalIdentification, discrete_set_treatments,
};
use crate::accepted::{GraphClass, GraphRef};
use crate::analysis::builder::DataInput;
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::{PhysicalExecutionPlan, RouteDefaults};
use crate::strategy_table::EstimatorId;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, MediationQuery,
    TargetPopulation, TemporalEffectQuery, TemporalResponseSpec, Value,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    AipwAte, CellSaturatedAipw, OverlapPolicy, ScoreTable, crossfit_binary_scores,
};
use antecedent_graph::TemporalDag;
use antecedent_identify::{TemporalBackdoorIdentifier, TemporalMediationIdentifier};
use std::sync::Arc;

/// Identification schedule of a sequence plan: `(variable, lag, optional level)` per step.
type IdentificationSchedule = Vec<(antecedent_core::VariableId, i32, Option<f64>)>;

fn sequence_identification_schedule(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<IdentificationSchedule>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(plan)) if plan.mechanism_overlays().is_some() => {
            let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "sequence schedule requires TemporalResponseSpec".into(),
            })?;
            Ok(Some(plan.identification_schedule(temporal)))
        }
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// Requested hard-set dose of a single-step temporal response, when it names one.
pub(crate) fn single_step_dose(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<f64>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(antecedent_estimate::TemporalInterventionPlan::Single { level, .. })) => Ok(level),
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// The response-query facts a per-horizon temporal identification reads.
///
/// `dose` is the single-step active level (control stays at 0); `None` keeps the unit contrast.
pub(crate) struct TemporalResponseIdentificationRequest<'a> {
    pub(crate) treatment: antecedent_core::VariableId,
    pub(crate) outcome: antecedent_core::VariableId,
    pub(crate) temporal: &'a TemporalResponseSpec,
    pub(crate) target_population: &'a TargetPopulation,
    pub(crate) schedule: Option<&'a [(antecedent_core::VariableId, i32, Option<f64>)]>,
    pub(crate) dose: Option<f64>,
}

pub(crate) fn identify_temporal_response_horizons(
    graph: &TemporalDag,
    estimator_id: crate::strategy_table::EstimatorId,
    request: &TemporalResponseIdentificationRequest<'_>,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    let TemporalResponseIdentificationRequest {
        treatment,
        outcome,
        temporal,
        target_population,
        schedule,
        dose,
    } = *request;
    if temporal.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon".into(),
        });
    }
    let origin = temporal.treatment_offset()?;
    let sequential = schedule
        .filter(|nodes| nodes.len() != 1 || (nodes[0].0, nodes[0].1) != (treatment, origin));
    let mut by_horizon = Vec::with_capacity(temporal.horizons.len());
    for &horizon in temporal.horizons.iter() {
        let id_res = if let Some(nodes) = sequential {
            let outcome_at = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
            TemporalBackdoorIdentifier::new()
                .identify_temporal_schedule(
                    graph,
                    outcome,
                    outcome_at,
                    nodes,
                    temporal.max_history_lag,
                    target_population.clone(),
                )
                .map_err(CausalError::from)?
        } else {
            let id_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(dose.unwrap_or(1.0))),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: target_population.clone(),
                effect_modifiers: std::sync::Arc::from([]),
            };
            TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, &id_query)
                .map_err(CausalError::from)?
        };
        let estimand = select_estimand(&id_res.result, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification: id_res.result,
            estimand,
            indexer: id_res.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

pub(crate) fn identify_temporal_mediation_horizons(
    graph: &TemporalDag,
    query: &MediationQuery,
    estimator_id: crate::strategy_table::EstimatorId,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    query.validate()?;
    if query.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    let ider = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    };
    let mut by_horizon = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let (identification, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        let estimand = select_estimand(&identification, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification,
            estimand,
            indexer: temporal.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

impl Study {
    /// The sharp-RD design identification, identified once here and retained with the
    /// program. `None` when the study declares no design, the structure is not a DAG, or the
    /// design does not identify: execution then identifies and reports the refusal.
    fn prepare_sharp_rd_identification(
        &self,
        estimator_id: EstimatorId,
    ) -> Option<CachedStaticIdentification> {
        let rd = self.rd?;
        let CausalQuery::AverageEffect(_) = &self.query else {
            return None;
        };
        let graph = self.graph.as_dag()?;
        let identification = antecedent_identify::SharpRdIdentifier::new(
            antecedent_identify::SharpRdConfig::new(rd.running_variable, rd.cutoff, rd.bandwidth),
        )
        .identify_on(graph, self.query.clone())
        .ok()?;
        crate::strategy_table::require_identified(&identification).ok()?;
        let estimand =
            crate::strategy_table::select_estimand(&identification, estimator_id).ok()?;
        Some(CachedStaticIdentification { identification, estimand })
    }

    /// The not-identified product behind `error`, for a latent-confounded ADMG average-effect
    /// query the complete algorithm does not identify.
    ///
    /// `error` is the refusal static preparation raised; this reads the product the identifier
    /// computed before that refusal was raised (its status, derivation and query) so compile
    /// can retain it. `None` for any other refusal, graph class or query, and for a search a
    /// budget stopped: such a refusal is not a proof of non-identification.
    pub(crate) fn not_identified_product(
        &self,
        error: &CausalError,
    ) -> Option<antecedent_identify::IdentificationResult> {
        if !error.is_identification_refusal() {
            return None;
        }
        let CausalQuery::AverageEffect(_) = &self.query else {
            return None;
        };
        if self.declared.tiered.is_some()
            || !matches!(self.graph.class(), GraphClass::Admg | GraphClass::TemporalAdmg)
        {
            return None;
        }
        let admg = self.graph.graph_ref().projected_admg().ok().flatten()?;
        if !crate::analysis::execute::admg_has_bidirected(&admg) {
            return None;
        }
        crate::strategy_table::not_identified_admg(&admg, &self.query)
    }

    /// Compute the static-path identification once at prepare time.
    ///
    /// Mirrors `execute_static`'s (and, for `bayesian.gcomp`, `execute_bayesian`'s)
    /// stage-1 inputs exactly. Sharp RD and PAG envelopes dispatch to separate
    /// preparation paths. Graph posteriors use their per-atom caches.
    pub(crate) fn prepare_static_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        self.prepare_static_identification_in(plan, None)
    }

    /// [`Self::prepare_static_identification`] under the caller's execution context: the exact
    /// adjustment searches lease their storage from its admission pool and stop at its
    /// cancellation token.
    pub(crate) fn prepare_static_identification_in(
        &self,
        plan: &PhysicalExecutionPlan,
        context: Option<&ExecutionContext>,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{
            EstimatorId, IdentifierId, identify_static_query, select_estimand,
        };
        if let Some(cached) = self.prepare_static_query_only_identification()? {
            return Ok(Some(cached));
        }
        let (identifier_id, estimator_id) = plan.binding().resolve(RouteDefaults::STATIC_ATE)?;
        if matches!(estimator_id, EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear) {
            return Ok(self.prepare_sharp_rd_identification(estimator_id));
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => self.prepare_static_average_effect_identification(
                plan,
                query,
                identifier_id,
                estimator_id,
                context,
            ),
            CausalQuery::Response(query) => self.prepare_static_response_identification(
                plan,
                query,
                identifier_id,
                estimator_id,
                context,
            ),
            CausalQuery::Mediation(query) => {
                let GraphRef::Dag(graph) = self.graph.graph_ref() else {
                    return Ok(None);
                };
                let identification = identify_static_query(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                )?;
                let (identification, estimand) = crate::strategy_table::select_claim(
                    identification,
                    EstimatorId::StaticMediationLinear,
                )?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::ConditionalEffect(query) => {
                self.prepare_conditional_identification(plan, query)
            }
            CausalQuery::PathSpecific(query) => {
                // Mirrors `execute_path_specific`'s identify+select-estimand step exactly.
                let (identifier_id, estimator_id) = plan.binding().resolve(RouteDefaults::PATH)?;
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = match self.graph.graph_ref() {
                    GraphRef::Dag(graph) => identify_static_query(identifier_id, graph, &cq)?,
                    GraphRef::Admg(admg) => {
                        crate::strategy_table::identify_admg_query(identifier_id, admg, &cq)?
                    }
                    GraphRef::TemporalAdmg(temporal) => {
                        let admg = crate::analysis::execute::temporal_admg_projection(temporal)?;
                        crate::strategy_table::identify_admg_query(identifier_id, &admg, &cq)?
                    }
                    _ => return Ok(None),
                };
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::NestedCounterfactual(query) => {
                // The natural direct effect on a mixed graph, identified as its mediation
                // reading and retained under the nested query it answers. A DAG runs the
                // shared-exogenous operation, which keeps no identification cache.
                let admg = match self.graph.graph_ref() {
                    GraphRef::Admg(admg) => admg.clone(),
                    GraphRef::TemporalAdmg(temporal) => {
                        crate::analysis::execute::temporal_admg_projection(temporal)?
                    }
                    _ => return Ok(None),
                };
                let (identifier_id, estimator_id) = plan.binding().resolve(RouteDefaults::PATH)?;
                let mut identification = crate::strategy_table::identify_admg_query(
                    identifier_id,
                    &admg,
                    &CausalQuery::Mediation(query.as_mediation_query()),
                )?;
                identification.query = self.query.clone();
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Distribution(query) => {
                // Mirrors `execute_distribution`.
                use crate::analysis::execute::DistributionGraph;
                let (identifier_id, estimator_id) =
                    plan.binding().resolve(RouteDefaults::DISTRIBUTION)?;
                let projected;
                let graph = match self.graph.graph_ref() {
                    GraphRef::Admg(admg) => DistributionGraph::Admg(admg),
                    GraphRef::TemporalAdmg(temporal) => {
                        projected = crate::analysis::execute::temporal_admg_projection(temporal)?;
                        DistributionGraph::Admg(&projected)
                    }
                    GraphRef::Dag(dag) => DistributionGraph::Dag(dag),
                    _ => return Ok(None),
                };
                let identification = graph.identify(identifier_id, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            _ => Ok(None),
        }
    }

    fn prepare_conditional_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        query: &antecedent_core::ConditionalEffectQuery,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{EstimatorId, identify_static, select_claim};
        // A latent-confounded ADMG identifies the contrast by IDC at the two levels;
        // `execute_admg_conditional` reads this cache. A latent-free ADMG runs the
        // Dag route, which keeps no prepare-time cache.
        let projected;
        let mixed = match self.graph.graph_ref() {
            GraphRef::Admg(admg) => Some(admg),
            GraphRef::TemporalAdmg(temporal) => {
                projected = crate::analysis::execute::temporal_admg_projection(temporal)?;
                Some(&projected)
            }
            _ => None,
        };
        if let Some(admg) = mixed {
            if crate::analysis::execute::admg_has_bidirected(admg) {
                let identification =
                    crate::analysis::conditional_admg::admg_conditional_identification(
                        admg, query,
                    )?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "ADMG conditional identification returned no estimand".into(),
                    }
                })?;
                return Ok(Some(CachedStaticIdentification { identification, estimand }));
            }
        }
        // Mirrors execute_conditional / execute_bayesian: identifier is
        // builder-selected or defaults to backdoor; estimator follows inference.
        let Some(graph) = self.graph.as_dag().cloned() else {
            return Ok(None);
        };
        let identifier_id = plan.binding().identifier_or(RouteDefaults::CONDITIONAL)?;
        let identification = identify_static(identifier_id, &graph, &query.inner)?;
        let estimator_id = if matches!(self.settings.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        let (identification, estimand) = select_claim(identification, estimator_id)?;
        Ok(Some(CachedStaticIdentification { identification, estimand }))
    }

    /// Identification of a pulse, sustained or conditional contrast on a temporal ADMG: the
    /// complete ID algorithm (IDC with modifiers) on the unrolled window of the template,
    /// retained at the query's horizon with the indexer that names the window's coordinates.
    ///
    /// # Errors
    ///
    /// The contrast is not identified on the window (the refusal names the hedge by variable
    /// and period), or the query is outside what the window route evaluates.
    pub(crate) fn prepare_temporal_admg_identification(
        &self,
        query: &TemporalEffectQuery,
    ) -> Result<CachedTemporalIdentification, CausalError> {
        let (
            GraphRef::TemporalAdmg(template),
            DataInput::Temporal(series) | DataInput::Event(series),
        ) = (self.graph.graph_ref(), &self.data)
        else {
            return Err(CausalError::Compile {
                message: "temporal ADMG identification requires a supplied TemporalAdmg over \
                          series data"
                    .into(),
            });
        };
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        let contrast = crate::analysis::temporal_admg_contrast::identify_window_contrast(
            template,
            query,
            variable_count,
        )?;
        Ok(crate::analysis::temporal_admg_contrast::retained_identification(contrast, query))
    }

    /// Identification of a derivative at one horizon on a temporal ADMG: one adjustment set of
    /// the unrolled window of the template, retained with the indexer that names the window's
    /// coordinates.
    ///
    /// # Errors
    ///
    /// The derivative has no adjustment set on the window (a hedge names the variables and
    /// periods that carry it), or the query is outside what the slope regression evaluates.
    pub(crate) fn prepare_temporal_admg_response_identification(
        &self,
        query: &antecedent_core::ResponseQuery,
    ) -> Result<CachedTemporalIdentification, CausalError> {
        let (
            GraphRef::TemporalAdmg(template),
            DataInput::Temporal(series) | DataInput::Event(series),
        ) = (self.graph.graph_ref(), &self.data)
        else {
            return Err(CausalError::Compile {
                message: "temporal ADMG derivative identification requires a supplied \
                          TemporalAdmg over series data"
                    .into(),
            });
        };
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        let by_horizon = crate::analysis::temporal_admg_contrast::identify_window_derivatives(
            template,
            query,
            variable_count,
        )?
        .into_iter()
        .map(|(horizon, derivative)| CachedTemporalHorizonIdentification {
            horizon,
            identification: derivative.identification,
            estimand: derivative.estimand,
            indexer: derivative.indexer,
        })
        .collect::<Vec<_>>();
        Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
    }

    /// Static claims identified from the query alone, before any route is resolved.
    fn prepare_static_query_only_identification(
        &self,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{IdentifierId, identify_static_query};
        if let CausalQuery::RandomizedEffect(query) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::randomized_identification(query);
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if let CausalQuery::PanelDid(query) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::panel_did_identification(query);
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if let CausalQuery::SyntheticControl(query) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::synthetic_control_identification(query);
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if let CausalQuery::LocalPolynomialRatio(query) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::local_polynomial_ratio_identification(query);
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if let CausalQuery::Survival(query) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::survival_identification(query);
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if matches!(self.query, CausalQuery::Counterfactual(_)) {
            // A structure latent confounding or an equivalence class leaves open identifies at
            // execution (ID* per ADMG, or the per-unit route per DAG completion).
            if matches!(
                self.graph.class(),
                GraphClass::Admg | GraphClass::TemporalAdmg | GraphClass::Pag | GraphClass::Cpdag
            ) {
                return Ok(None);
            }
            let graph = self
                .graph
                .as_dag()
                .ok_or(CausalError::Unsupported { message: "counterfactual requires Dag" })?;
            let identification =
                identify_static_query(IdentifierId::GcmParametric, graph, &self.query)?;
            let estimand = identification.estimands[0].clone();
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if let CausalQuery::Discover(search) = &self.query {
            let (identification, estimand) =
                crate::analysis::execute::structure_search_identification(
                    self.query.clone(),
                    search,
                );
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if matches!(
            self.query,
            CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_)
                | CausalQuery::Attribute(_)
                | CausalQuery::Simulate(_)
        ) {
            let (treatment, outcome) = crate::analysis::execute::gcm_query_vars(&self.query)?;
            let (identification, estimand) =
                crate::analysis::execute::parametric_scm_identification(
                    self.query.clone(),
                    treatment,
                    outcome,
                );
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        Ok(None)
    }

    /// Static identification of an average-effect query on its resolved route.
    fn prepare_static_average_effect_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        query: &AverageEffectQuery,
        identifier_id: crate::strategy_table::IdentifierId,
        estimator_id: EstimatorId,
        context: Option<&ExecutionContext>,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{
            identify_static_query_in_context, select_claim, select_estimand,
        };
        if let Some(background) = &self.declared.tiered {
            let identification = antecedent_identify::identify_tiered(background, query)?;
            let estimand =
                identification.estimands.first().cloned().ok_or_else(|| CausalError::Compile {
                    message: "tiered identification returned no estimand".into(),
                })?;
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if matches!(self.graph.class(), GraphClass::Admg | GraphClass::TemporalAdmg) {
            use crate::strategy_table::identify_admg;

            let admg = self.graph.graph_ref().projected_admg()?;
            let Some(admg) = admg else {
                return Err(CausalError::Compile {
                    message: "ADMG prepare missing supplied graph".into(),
                });
            };
            if crate::analysis::execute::admg_has_bidirected(&admg) {
                let identification = identify_admg(identifier_id, &admg, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                return Ok(Some(CachedStaticIdentification { identification, estimand }));
            }
        }
        let graph = match self.graph.graph_ref() {
            GraphRef::Dag(dag) => Some(dag.clone()),
            GraphRef::Admg(_) | GraphRef::TemporalAdmg(_) => plan.static_graph().cloned(),
            GraphRef::Cpdag(_) => return Ok(None),
            _ => None,
        };
        let Some(graph) = graph else {
            return Ok(None);
        };
        // `execute_bayesian` never consults `self.rd` (it calls
        // `identify_static`, which is `identify_static_query_with_rd`
        // with `rd: None`); mirror that exactly so the cached
        // identification matches what an uncached Bayesian run would
        // compute, even if a caller set `.rd_config(..)` alongside
        // `bayesian.gcomp`.
        let rd = if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            None
        } else {
            self.rd.map(|c| {
                antecedent_identify::SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth)
            })
        };
        let identification = identify_static_query_in_context(
            identifier_id,
            &graph,
            &CausalQuery::AverageEffect(query.clone()),
            rd,
            context,
        )?;
        let (identification, estimand) = select_claim(identification, estimator_id)?;
        Ok(Some(CachedStaticIdentification { identification, estimand }))
    }

    /// Static identification of a response query on its resolved route.
    fn prepare_static_response_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        query: &antecedent_core::ResponseQuery,
        identifier_id: crate::strategy_table::IdentifierId,
        estimator_id: EstimatorId,
        context: Option<&ExecutionContext>,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{identify_static_query_in_context, select_estimand};
        if let Some(background) = &self.declared.tiered {
            let schema = match &self.data {
                DataInput::Tabular(data) => data.schema(),
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "CoDetermined joint prepare requires tabular data",
                    });
                }
            };
            let identification = match self.graph.as_admg() {
                Some(admg) => {
                    antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                }
                None => antecedent_identify::identify_tiered_joint(background, schema, query)?,
            };
            let estimand =
                identification.estimands.first().cloned().ok_or(CausalError::Unsupported {
                    message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                })?;
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        let response_admg = self.graph.graph_ref().projected_admg()?;
        if let Some(admg) = response_admg
            .as_ref()
            .filter(|admg| crate::analysis::execute::admg_has_bidirected(admg))
        {
            use crate::strategy_table::identify_admg_query;
            if crate::analysis::execute::response_functional_is_derivative(&query.functional) {
                // A derivative of a continuous treatment is evaluated on a regression over an
                // adjustment set of the ADMG, not by the discrete general-ID plug-in.
                let identification = crate::strategy_table::identify_admg_derivative(admg, query)?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "response identifier returned no estimand".into(),
                    }
                })?;
                return Ok(Some(CachedStaticIdentification { identification, estimand }));
            }
            // `execute_admg_response` reuses this cache verbatim as the *first grid
            // level's* claim for a MeanCurve (it only re-identifies per level from the
            // second level on, mirroring the graph-posterior ADMG response cache). A
            // MeanCurve identified whole produces one general.id estimand per grid
            // level, and `select_estimand` then has no unique estimator match to pick
            // among them (they all report the same method). Cache the first level's
            // InterventionResponse claim instead, which is what downstream code
            // actually consumes and — being a single intervention level — is exactly
            // what `select_estimand` can disambiguate.
            let causal_query = match &query.functional {
                antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
                    let first_level =
                        treatment.grid.values()?.into_iter().next().ok_or_else(|| {
                            CausalError::Compile {
                                message: "MeanCurve response requires a non-empty evaluation \
                                      grid"
                                    .into(),
                            }
                        })?;
                    let mut level_query = query.clone();
                    level_query.functional =
                        antecedent_core::ResponseFunctional::InterventionResponse {
                            outcome: *outcome,
                            interventions: Arc::from([Intervention::set(
                                treatment.variable,
                                Value::f64(first_level),
                            )]),
                        };
                    CausalQuery::Response(level_query)
                }
                _ => CausalQuery::Response(query.clone()),
            };
            let identification = identify_admg_query(identifier_id, admg, &causal_query)?;
            let estimand = select_estimand(&identification, estimator_id)?;
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        let graph = match (self.graph.graph_ref(), plan.static_graph()) {
            (GraphRef::Dag(dag), _)
            | (GraphRef::Admg(_) | GraphRef::TemporalAdmg(_), Some(dag)) => Some(dag.clone()),
            _ => None,
        };
        let Some(graph) = graph else {
            return Ok(None);
        };
        let identification = identify_static_query_in_context(
            identifier_id,
            &graph,
            &CausalQuery::Response(query.clone()),
            None,
            context,
        )?;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            CausalError::Compile { message: "response identifier returned no estimand".into() }
        })?;
        Ok(Some(CachedStaticIdentification { identification, estimand }))
    }

    /// Compute the generalized-adjustment envelope once for a supplied PAG.
    pub(super) fn prepare_pag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        context: Option<&ExecutionContext>,
    ) -> Result<Option<CachedPagIdentification>, CausalError> {
        use crate::strategy_table::identify_pag_in_context;

        let Some(query) = self.envelope_witness_ate()? else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::Pag {
            return Ok(None);
        }
        let pag = plan.static_pag().ok_or_else(|| CausalError::Compile {
            message: "PAG prepare missing resolved static PAG".into(),
        })?;
        let identifier_id = plan.binding().identifier_or(RouteDefaults::PAG)?;
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_pag_response_in_context(
                identifier_id,
                pag,
                response,
                context,
            )?
        } else {
            identify_pag_in_context(identifier_id, pag, &query, context)?
        };
        let identification = crate::analysis::execute::envelope_to_identification_result_for(
            &envelope,
            self.query.clone(),
        );
        Ok(Some(CachedPagIdentification {
            envelope: Arc::new(envelope),
            identification,
            transport: None,
        }))
    }

    /// Compute the MEC envelope once for a supplied CPDAG.
    pub(super) fn prepare_cpdag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        context: Option<&ExecutionContext>,
    ) -> Result<Option<CachedCpdagIdentification>, CausalError> {
        use crate::strategy_table::identify_cpdag_in_context;

        let Some(query) = self.envelope_witness_ate()? else {
            return Ok(None);
        };
        let GraphRef::Cpdag(cpdag) = self.graph.graph_ref() else {
            return Ok(None);
        };
        let identifier_id = plan.binding().identifier_or(RouteDefaults::PAG)?;
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_cpdag_response_in_context(
                identifier_id,
                cpdag,
                response,
                context,
            )?
        } else {
            identify_cpdag_in_context(identifier_id, cpdag, &query, context)?
        };
        let identification = crate::analysis::execute::envelope_to_identification_result_for(
            &envelope,
            self.query.clone(),
        );
        Ok(Some(CachedCpdagIdentification {
            envelope: Arc::new(envelope),
            identification,
            transport: None,
        }))
    }

    fn envelope_witness_ate(&self) -> Result<Option<AverageEffectQuery>, CausalError> {
        match &self.query {
            CausalQuery::AverageEffect(query) => Ok(Some(query.clone())),
            CausalQuery::Response(query)
                if crate::analysis::execute::class_aware_response_supported(query) =>
            {
                Ok(Some(crate::analysis::execute::response_witness_ate(query)?))
            }
            CausalQuery::ConditionalEffect(query) => Ok(Some(query.inner.clone())),
            _ => Ok(None),
        }
    }

    /// Identify once per requested horizon at prepare for temporal response or `TemporalEffect`.
    pub(super) fn prepare_temporal_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::{EstimatorId, select_estimand};
        let GraphRef::TemporalDag(graph) = self.graph.graph_ref() else {
            return Ok(None);
        };
        match &self.query {
            CausalQuery::Response(query) => {
                let Some(temporal) = query.temporal.as_ref() else {
                    return Ok(None);
                };
                let (treatment, outcome) =
                    crate::analysis::proof_support::require_primary_pair(&query.functional)?;
                let schedule = sequence_identification_schedule(query)?;
                Ok(Some(identify_temporal_response_horizons(
                    graph,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                    &TemporalResponseIdentificationRequest {
                        treatment,
                        outcome,
                        temporal,
                        target_population: &query.target_population,
                        schedule: schedule.as_deref(),
                        dose: single_step_dose(query)?,
                    },
                )?))
            }
            CausalQuery::TemporalEffect(query)
                if self.declared.identifier
                    == Some(crate::strategy_table::IdentifierId::TemporalIvUnfolded) =>
            {
                let found =
                    crate::analysis::temporal_iv_identification::identify_temporal_instrument(
                        graph, query,
                    )?;
                Ok(Some(CachedTemporalIdentification {
                    by_horizon: Arc::from([CachedTemporalHorizonIdentification {
                        horizon: query.horizon_steps,
                        identification: found.identification,
                        estimand: found.estimand,
                        indexer: found.indexer,
                    }]),
                }))
            }
            CausalQuery::TemporalEffect(query) => {
                let id_res = TemporalBackdoorIdentifier::new()
                    .with_parent_adjustment_fallback()
                    .identify_temporal(graph, query)
                    .map_err(CausalError::from)?;
                let estimand = select_estimand(
                    &id_res.result,
                    EstimatorId::temporal_effect_procedure(query, false),
                )?;
                Ok(Some(CachedTemporalIdentification {
                    by_horizon: Arc::from([CachedTemporalHorizonIdentification {
                        horizon: query.horizon_steps,
                        identification: id_res.result,
                        estimand,
                        indexer: id_res.indexer,
                    }]),
                }))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn prepare_temporal_class_identification(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<CachedTemporalClassIdentification>, CausalError> {
        if !matches!(self.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag) {
            return Ok(None);
        }
        let query = match &self.query {
            CausalQuery::TemporalEffect(query) => query.clone(),
            CausalQuery::Response(response) => {
                let temporal = response.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                    message: "temporal class response prepare requires TemporalResponseSpec".into(),
                })?;
                let (treatment, outcome) =
                    crate::analysis::proof_support::require_primary_pair(&response.functional)?;
                TemporalEffectQuery {
                    treatment,
                    outcome,
                    policy: temporal.policy.clone(),
                    control: Intervention::set(treatment, Value::f64(0.0)),
                    active: Intervention::set(treatment, Value::f64(1.0)),
                    horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
                    max_history_lag: temporal.max_history_lag,
                    target_population: response.target_population.clone(),
                    effect_modifiers: std::sync::Arc::from([]),
                }
            }
            CausalQuery::Mediation(mediation)
                if self.graph.class() == GraphClass::TemporalCpdag =>
            {
                let mut witness =
                    TemporalEffectQuery::pulse(mediation.treatment, mediation.outcome, 1.0);
                witness.horizon_steps = mediation.horizons.first().copied().unwrap_or(1);
                witness
            }
            _ => return Ok(None),
        };
        let identifier_id = self.declared.identifier.unwrap_or(RouteDefaults::PAG.identifier);
        let mut bundle = std::sync::Arc::unwrap_or_clone(self.identify_temporal_class(
            identifier_id,
            &query,
            ctx,
        )?);
        let horizons: &[u32] = match &self.query {
            CausalQuery::Response(response) => {
                response.temporal.as_ref().map_or(&[][..], |temporal| &temporal.horizons[..])
            }
            CausalQuery::Mediation(mediation) => &mediation.horizons,
            _ => &[],
        };
        for &horizon in horizons {
            let mut qh = query.clone();
            qh.horizon_steps = horizon;
            let identified = if horizon == query.horizon_steps {
                bundle.envelope.clone()
            } else {
                self.identify_temporal_class(identifier_id, &qh, ctx)?.envelope.clone()
            };
            bundle.by_horizon.push((horizon, identified));
        }
        Ok(Some(bundle))
    }

    /// One `I(h)` per requested mediation horizon (path-product + that horizon's backdoor `Z`).
    pub(super) fn prepare_temporal_mediation_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::EstimatorId;
        let CausalQuery::Mediation(query) = &self.query else {
            return Ok(None);
        };
        let GraphRef::TemporalDag(graph) = self.graph.graph_ref() else {
            return Ok(None);
        };
        Ok(Some(identify_temporal_mediation_horizons(
            graph,
            query,
            if matches!(self.settings.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::BayesianTemporalMediation
            } else {
                EstimatorId::TemporalMediation
            },
        )?))
    }

    /// Cross-fitted AIPW scores of the marginal ATE a DML, DR-Learner or `CausalForest` plan
    /// reports (the estimator's own out-of-fold nuisances and fold plan, so a retarget under
    /// uniform weights returns the route's point). `None` where the route keeps no such
    /// table: a partially linear score, trimming, a non-mean functional or a non-`AllObserved`
    /// target. A CATE prediction contributes no score.
    fn prepare_learner_score_table(
        &self,
        data: &TabularData,
        query: &antecedent_core::AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
        use crate::estimator_spec::EstimatorSpec;
        if !query.outcome_functional.is_mean()
            || !matches!(query.target_population, TargetPopulation::AllObserved)
        {
            return Ok(None);
        }
        let Some(cache) = self.static_cache() else {
            return Ok(None);
        };
        let method = cache.estimand.method.as_ref();
        if !(method.contains("adjustment")
            || method.contains("backdoor")
            || method.starts_with("tiered."))
        {
            return Ok(None);
        }
        let estimand = &cache.estimand;
        let spec = self.declared.estimator_spec.as_deref();
        let table = match self.declared.estimator {
            Some(EstimatorId::Dml) => {
                let mut est = match spec {
                    Some(EstimatorSpec::Dml(config)) => (**config).clone(),
                    _ => antecedent_estimate::DmlAte::new(),
                };
                if let Some(overlap) = self.declared.overlap_policy {
                    est.overlap = overlap;
                }
                let problem = est.prepare(data, estimand, query)?;
                est.score_table(&problem, ctx)?
            }
            Some(EstimatorId::DrLearner) => {
                let mut est = match spec {
                    Some(EstimatorSpec::DrLearner(config)) => (**config).clone(),
                    _ => antecedent_estimate::DrLearner::new(),
                };
                if let Some(overlap) = self.declared.overlap_policy {
                    est.overlap = overlap;
                }
                let problem = est.prepare(data, estimand, query)?;
                est.score_table(&problem, ctx)?
            }
            _ => {
                // The forest's marginal ATE is the DML AIPW score; no forest option enters it.
                let est = antecedent_estimate::CausalForest::new();
                let problem = est.prepare(data, estimand, query)?;
                est.score_table(&problem, ctx)?
            }
        };
        Ok(table)
    }

    /// Cross-fitted AIPW scores for retarget / exceedance / joint cells.
    pub(crate) fn prepare_score_table(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
        // A stats-layer refusal of a fit learns here which treatment or cell it was for and,
        // for a rank refusal, which columns are dependent.
        self.prepare_score_table_unnamed(ctx)
            .map_err(|error| crate::analysis::preflight::name_stats_refusal(self, None, error, ctx))
    }

    fn prepare_score_table_unnamed(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
        let fold_seed = ctx.rng.master_seed();
        let DataInput::Tabular(data) = &self.data else {
            return Ok(None);
        };
        // Unknown is two canonical sets / GraphDependent / no single Z.
        // One score table would collapse the envelope; retarget refuses.
        if self
            .declared
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::Unknown)
            || !matches!(self.settings.inference, InferenceMode::Frequentist)
        {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query)
                if matches!(
                    self.declared.estimator,
                    Some(EstimatorId::Dml | EstimatorId::DrLearner | EstimatorId::CausalForest)
                ) =>
            {
                self.prepare_learner_score_table(data, query, ctx)
            }
            CausalQuery::AverageEffect(query) => {
                self.prepare_aipw_score_table(data, query, fold_seed)
            }
            CausalQuery::Response(query) => {
                self.prepare_cell_response_score_table(data, query, fold_seed)
            }
            _ => Ok(None),
        }
    }

    /// Cross-fitted AIPW scores of an `Aipw` average-effect plan, refusing what trimming or a non-iid SE would change.
    fn prepare_aipw_score_table(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        fold_seed: u64,
    ) -> Result<Option<ScoreTable>, CausalError> {
        // Score artifacts are an explicit AIPW execution contract; do not
        // silently fit a second estimator for linear/IV/matching plans.
        if self.declared.estimator != Some(crate::strategy_table::EstimatorId::Aipw)
            || !matches!(query.target_population, TargetPopulation::AllObserved)
        {
            return Ok(None);
        }
        let Some(cache) = self.static_cache() else {
            return Ok(None);
        };
        let method = cache.estimand.method.as_ref();
        if !(method.contains("adjustment")
            || method.contains("backdoor")
            || method.starts_with("tiered."))
        {
            return Ok(None);
        }
        let mut est = match self.declared.estimator_spec.as_deref() {
            Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => *config.clone(),
            _ => AipwAte::new(),
        };
        if let Some(design) = self.declared.sampling_design {
            est.inference.sampling_design =
                antecedent_estimate::util::SamplingDeclaration::Declared(design);
        }
        if let Some(overlap) = self.declared.overlap_policy {
            est.overlap = overlap;
        }
        if matches!(
            est.overlap.canonical(),
            OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
        ) || est.inference.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
        {
            // Frozen cross-fitted scores are untrimmed iid influence
            // values. A mean estimate under trimming or a non-iid SE is
            // still estimated by the configured AIPW; it keeps no score
            // table, so a retarget refuses with `score_table_unavailable`
            // instead of reweighting scores of a different construction.
            // A quantile or exceedance functional is estimated from the
            // scores and has no such fallback.
            if query.outcome_functional.is_mean() {
                return Ok(None);
            }
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "prepared AIPW functional scores require iid inference without \
                 propensity trimming"
            ));
        }
        let estimand = cache.estimand.clone();
        let mut problem = est.prepare(data, &estimand, query)?;
        problem.fold_seed = fold_seed;
        if let Some(shared) = self.shared_batch_design.as_ref() {
            shared.apply_to_propensity(&mut problem)?;
        }
        let table = if query.outcome_functional.quantile_level().is_some() {
            let grid = antecedent_estimate::empirical_threshold_grid(&problem.outcome, 19)?;
            antecedent_estimate::build_binary_scores(
                &problem,
                query.treatment,
                &grid.into_iter().map(Some).collect::<Vec<_>>(),
                antecedent_estimate::DEFAULT_AIPW_FOLDS,
                &est.glm_options,
                est.backend,
            )?
        } else {
            crossfit_binary_scores(
                &problem,
                query,
                antecedent_estimate::DEFAULT_AIPW_FOLDS,
                &est.glm_options,
                est.backend,
            )?
        };
        Ok(Some(table))
    }

    /// Cell-saturated AIPW scores of a discrete-set intervention response.
    fn prepare_cell_response_score_table(
        &self,
        data: &TabularData,
        query: &antecedent_core::ResponseQuery,
        fold_seed: u64,
    ) -> Result<Option<ScoreTable>, CausalError> {
        if self.declared.estimator != Some(crate::strategy_table::EstimatorId::CellAipw) {
            return Ok(None);
        }
        let Some(cache) = self.static_cache() else {
            return Ok(None);
        };
        let antecedent_core::ResponseFunctional::InterventionResponse { outcome, interventions } =
            &query.functional
        else {
            return Ok(None);
        };
        let treatments = discrete_set_treatments(interventions);
        if treatments.len() < 2 {
            return Ok(None);
        }
        let est = CellSaturatedAipw::new().with_fold_seed(fold_seed);
        let continuous = self.continuous_cell.as_ref().map(|(variable, grid)| {
            antecedent_estimate::ContinuousCellSpec { variable: *variable, grid }
        });
        // Folds are deliberately not shared here: `fit_scores_with_assignment`
        // leaves them unset so the cell path draws its own cell-stratified
        // `crossfit_fold_plan` (keyed by `est.fold_seed`), reproducing the solo
        // fold plan for this query's own cells bit-for-bit instead of a private,
        // unstratified batch shuffle. See `SharedBatchDesign` docs.
        let fold_ids: Option<Vec<u32>> = None;
        let design = match self.shared_batch_design.as_ref() {
            Some(shared) => {
                let mut ids: Vec<_> = treatments
                    .iter()
                    .copied()
                    .chain(std::iter::once(*outcome))
                    .chain(cache.estimand.adjustment_set.iter().copied())
                    .collect();
                if let Some((variable, _)) = self.continuous_cell.as_ref() {
                    ids.push(*variable);
                }
                let ids = data.complete_case_mask(&ids);
                let row_index = match ids {
                    Ok(mask) => mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| {
                            keep.then_some(antecedent_data::GlobalId::new(i as u64))
                        })
                        .collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                if row_index.is_empty() {
                    None
                } else {
                    shared.design_for(&cache.estimand.adjustment_set, &row_index)?
                }
            }
            None => None,
        };
        let spec = antecedent_estimate::CellScoreSpec {
            treatments: &treatments,
            outcome: *outcome,
            adjustment: &cache.estimand.adjustment_set,
            functional: &query.outcome_functional,
            continuous,
        };
        let shared = antecedent_estimate::SharedCellInputs {
            fold_ids: fold_ids.as_deref(),
            design: design.as_deref(),
        };
        if data.storage().weights().is_some() {
            // A batch shares one unweighted design and summarizes every cell as independent
            // unweighted rows, so it refuses the weights. A single prepared study estimates
            // them at run time from fresh scores; it keeps no prepare-time table to retarget,
            // because a retarget would drop the weights the scores are summarized under.
            if self.shared_batch_design.is_some() {
                return Err(CausalError::Estimate(
                    antecedent_estimate::sampling_weights::unsupported_route(
                        "the table carries observation weights, which a batch of joint cells \
                         shares one unweighted design over and would not read, so the cells \
                         would describe the unweighted rows. Next attempt: estimate each cell \
                         as its own study, which reads the weights, or drop them from the table",
                    ),
                ));
            }
            return Ok(None);
        }
        Ok(Some(est.fit_scores_with_assignment(data, &spec, shared)?))
    }
}
