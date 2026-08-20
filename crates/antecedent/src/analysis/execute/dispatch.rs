// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn validation_suite_id(&self) -> Option<Arc<str>> {
        self.refute.validation_suite_id().map(Arc::from)
    }

    pub(super) fn ensure_supported_combination(&self) -> Result<(), CausalError> {
        let class = self.graph.class();
        match (&self.data, &self.query, class) {
            (_, CausalQuery::Response(_), class)
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Response requires tabular data and a static DAG",
                });
            }
            (_, CausalQuery::Distribution(_), class)
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static DAG",
                });
            }
            (_, CausalQuery::PathSpecific(_), class)
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static DAG",
                });
            }
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                CausalQuery::AverageEffect(_),
                _,
            ) => {
                return Err(CausalError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::Panel(_), CausalQuery::AverageEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "static ATE on panel data is unsupported; use TemporalEffect".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphClass::Pag | GraphClass::Cpdag | GraphClass::Admg,
            ) => {
                return Err(CausalError::Compile {
                    message: "static Pag/Cpdag/Admg requires tabular data and an average-effect \
                              query"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "multi-environment data currently supports only a supplied \
                              TemporalDag"
                        .into(),
                });
            }
            (DataInput::Panel(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "panel data supports only a supplied TemporalDag (pooled units)"
                        .into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphClass::Pag) => {
                let (identifier, _) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, identifier.parse::<IdentifierId>()?)?;
            }
            _ => {}
        }
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            if let Some(id) = &self.identifier {
                if *id != IdentifierId::TemporalBackdoorUnfolded {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports identifier \"temporal.backdoor.unfolded\"; got {id:?}"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                if *est != EstimatorId::TemporalLinearAdjustment {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports estimator \"temporal.linear.adjustment\"; got {est:?}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve builder-selected identifier/estimator ids against modality defaults.
    pub(super) fn resolve_id_est_pair(
        &self,
        default_identifier: IdentifierId,
        default_estimator: EstimatorId,
    ) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.unwrap_or(default_identifier);
        let estimator = self.estimator.unwrap_or(default_estimator);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    pub(super) fn resolve_static_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(DEFAULT_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    pub(super) fn resolve_pag_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for ADMG ATE (general ID + functional effect).
    pub(super) fn resolve_admg_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_ADMG_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for ConditionalEffect.
    pub(super) fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            DEFAULT_CONDITIONAL_IDENTIFIER_ID,
            DEFAULT_CONDITIONAL_ESTIMATOR_ID,
        )
    }

    /// Resolve identifier/estimator for Distribution queries.
    pub(super) fn resolve_distribution_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            DEFAULT_DISTRIBUTION_IDENTIFIER_ID,
            DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
        )
    }

    /// Resolve the response estimator from the functional when the caller did not override it.
    pub(super) fn resolve_response_pair(&self, query: &ResponseQuery) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            DEFAULT_RESPONSE_IDENTIFIER_ID,
            EstimatorId::default_for_response(&query.functional),
        )
    }

    /// Resolve identifier/estimator for PathSpecific queries.
    pub(super) fn resolve_path_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID)
    }

    pub(super) fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), CausalError> {
        if matches!(estimator.parse::<EstimatorId>()?, EstimatorId::RdSharp) && self.rd.is_none() {
            return Err(CausalError::Compile {
                message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
            });
        }
        Ok(())
    }

    /// Execute a compiled physical plan.
    ///
    /// # Errors
    ///
    /// Identification / estimation / validation failures.
    ///
    /// # Panics
    ///
    /// Never in practice; see [`Study::compile_logical`]'s panics note — the same
    /// `AcceptedGraph::class`/accessor invariant applies here.
    pub fn execute(
        &self,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_on(&self.data, physical, ctx)
    }

    /// Execute against an explicit data slot (prepare/estimate must not clone `Study`).
    pub(crate) fn execute_on(
        &self,
        data: &DataInput,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return match (data, &self.query) {
                (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                    self.execute_graph_posterior_bayesian(data, gp, q, physical, ctx)
                }
                (
                    DataInput::Temporal(data) | DataInput::Event(data),
                    CausalQuery::TemporalEffect(q),
                ) => self.execute_dbn_posterior_bayesian(data, gp, q, physical, ctx),
                _ => Err(CausalError::Unsupported {
                    message: "graph-posterior analysis supports tabular average-effect or \
                              temporal-effect queries only",
                }),
            };
        }
        match classify_analysis_route(data, &self.query) {
            Some(route) if matches!(data_modality(data), DataModality::Tabular) => {
                let DataInput::Tabular(data) = data else { unreachable!() };
                self.execute_tabular_route(route, data, physical, ctx)
            }
            Some(AnalysisRoute::TemporalMediation) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    unreachable!()
                };
                let CausalQuery::Mediation(q) = &self.query else { unreachable!() };
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::TemporalEffect) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    unreachable!()
                };
                let CausalQuery::TemporalEffect(q) = &self.query else { unreachable!() };
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph".into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::PanelTemporalEffect) => {
                let DataInput::Panel(panel) = data else { unreachable!() };
                let CausalQuery::TemporalEffect(q) = &self.query else { unreachable!() };
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready panel plan missing resolved graph".into(),
                })?;
                self.execute_panel(panel, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::MultiEnvTemporalEffect) | None | Some(_) => {
                Err(CausalError::Unsupported {
                    message: "execute path unsupported for this configuration",
                })
            }
        }
    }

    /// Prepared estimate click: tabular data without wrapping a `DataInput`.
    pub(crate) fn execute_tabular(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return match &self.query {
                CausalQuery::AverageEffect(q) => {
                    self.execute_graph_posterior_bayesian(data, gp, q, physical, ctx)
                }
                _ => Err(CausalError::Unsupported {
                    message: "graph-posterior analysis supports tabular average-effect or \
                              temporal-effect queries only",
                }),
            };
        }
        let route = classify_route(DataModality::Tabular, &self.query).ok_or(
            CausalError::Unsupported { message: "execute path unsupported for this configuration" },
        )?;
        self.execute_tabular_route(route, data, physical, ctx)
    }

    fn execute_tabular_route(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match route {
            AnalysisRoute::Response => {
                let CausalQuery::Response(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("Response execute requires a supplied static DAG")?;
                self.execute_response(data, graph, q, physical, ctx)
            }
            AnalysisRoute::StaticAte => {
                let CausalQuery::AverageEffect(q) = &self.query else { unreachable!() };
                match self.graph.class() {
                    GraphClass::Dag => {
                        let graph =
                            self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
                        self.execute_static(data, graph, q, physical, ctx)
                    }
                    GraphClass::Cpdag => {
                        let graph = physical.static_graph().ok_or(CausalError::Compile {
                            message: "Ready CPDAG plan missing resolved static DAG".into(),
                        })?;
                        self.execute_static(data, graph, q, physical, ctx)
                    }
                    GraphClass::Admg => {
                        let admg =
                            self.graph.as_admg().expect("class() == Admg implies as_admg() is Some");
                        if admg_has_bidirected(admg) {
                            self.execute_admg(data, admg, q, physical, ctx)
                        } else {
                            let graph = physical.static_graph().ok_or(CausalError::Compile {
                                message: "Ready ADMG (DAG-coerced) plan missing resolved static DAG"
                                    .into(),
                            })?;
                            self.execute_static(data, graph, q, physical, ctx)
                        }
                    }
                    GraphClass::Pag => {
                        let pag = physical.static_pag().ok_or(CausalError::Compile {
                            message: "Ready PAG plan missing resolved static PAG".into(),
                        })?;
                        self.execute_pag(data, pag, q, physical, ctx)
                    }
                    GraphClass::TemporalDag
                    | GraphClass::TemporalCpdag
                    | GraphClass::TemporalPag => Err(CausalError::Unsupported {
                        message: "static ATE execute requires a static graph class",
                    }),
                }
            }
            AnalysisRoute::Distribution => {
                let CausalQuery::Distribution(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("Distribution execute requires a supplied static DAG")?;
                self.execute_distribution(data, graph, q, physical, ctx)
            }
            AnalysisRoute::PathSpecific => {
                let CausalQuery::PathSpecific(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("PathSpecific execute requires a supplied static DAG")?;
                self.execute_path_specific(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Conditional => {
                let CausalQuery::ConditionalEffect(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("ConditionalEffect execute requires a supplied static DAG")?;
                self.execute_conditional(data, graph, q, physical, ctx)
            }
            AnalysisRoute::StaticMediation => {
                let CausalQuery::Mediation(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("static Mediation execute requires a supplied static DAG")?;
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Counterfactual => {
                let CausalQuery::Counterfactual(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("Counterfactual execute requires a supplied static DAG")?;
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Anomaly => {
                let CausalQuery::AnomalyAttribution(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("AnomalyAttribution execute requires a supplied static DAG")?;
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            AnalysisRoute::ChangeAttribution => {
                let CausalQuery::ChangeAttribution(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("ChangeAttribution execute requires a supplied static DAG")?;
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            AnalysisRoute::MechanismChange => {
                let CausalQuery::MechanismChange(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("MechanismChange execute requires a supplied static DAG")?;
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            AnalysisRoute::UnitChange => {
                let CausalQuery::UnitChange(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag("UnitChange execute requires a supplied static DAG")?;
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            AnalysisRoute::TemporalMediation
            | AnalysisRoute::TemporalEffect
            | AnalysisRoute::PanelTemporalEffect
            | AnalysisRoute::MultiEnvTemporalEffect => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Compile and run.
    ///
    /// # Errors
    ///
    /// Compile / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Identify only (no estimation). Supports static DAG and ADMG average-effect
    /// / related queries.
    ///
    /// The ADMG path matters for correctness, not just coverage: a DAG cannot
    /// express "this variable is unobservable", so a latent common cause
    /// flattened into one is identified by adjusting on a variable no study can
    /// measure. Routing an ADMG through general ID lets that case report
    /// `NotIdentified` instead. Mirrors `execute()`: an ADMG with no bidirected
    /// edges is just a DAG, and is coerced rather than forced down the general
    /// ID path.
    ///
    /// # Errors
    ///
    /// Missing graph structure, unsupported graph class, or identification failure.
    pub fn identify_only(&self) -> Result<IdentificationResult, CausalError> {
        use crate::strategy_table::{
            DEFAULT_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID, identify_admg,
            identify_static_query,
        };

        if self.graph_posterior.is_some() {
            // `self.graph` is only the placeholder shape here (see `stub_accepted_graph_for`);
            // identification runs per-graph, against the real posterior atoms, inside `execute()`.
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "identify_only is not a graph-posterior cell; identification \
                          runs per-graph inside execute.",
            });
        }
        let default_id = if matches!(&self.query, CausalQuery::Response(_)) {
            DEFAULT_RESPONSE_IDENTIFIER_ID
        } else {
            DEFAULT_IDENTIFIER_ID
        };
        let id = self.identifier.unwrap_or(default_id);

        if let Some(admg) = self.graph.as_admg() {
            if admg_has_bidirected(admg) {
                let CausalQuery::AverageEffect(query) = &self.query else {
                    return Err(CausalError::Support {
                        id: crate::support::SupportRefusal::Refused,
                        message: "identify_only on an ADMG supports AverageEffect only.",
                    });
                };
                // Only general ID handles bidirected structure; the default
                // identifier is a backdoor strategy that would ignore it.
                let identifier =
                    if self.identifier.is_some() { id } else { IdentifierId::GeneralId };
                return identify_admg(identifier, admg, query);
            }
            // No bidirected edges: this ADMG *is* a DAG. Coercing keeps the
            // caller's identifier choice meaningful instead of forcing general
            // ID on a graph with no latent structure to reason about.
            let coerced = admg_to_dag(admg)?;
            return identify_static_query(id, &coerced, &self.query);
        }

        let graph = self.graph.as_dag().ok_or(CausalError::Support {
            id: crate::support::SupportRefusal::Refused,
            message: "identify_only supports static DAG and ADMG graphs only.",
        })?;
        identify_static_query(id, graph, &self.query)
    }

    /// Inspectable compile result (logical + physical).
    ///
    /// Prefer this over `compile` when documenting plan inspection in user code.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile`].
    pub fn plan(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile(ctx)
    }

    /// Mark physical plan when custom validators are present.
    pub(super) fn apply_callback_plan_marks(
        &self,
        mut record: antecedent_core::PhysicalExecutionPlanRecord,
        diagnostics: &mut Vec<antecedent_core::Diagnostic>,
    ) -> antecedent_core::PhysicalExecutionPlanRecord {
        if !self.custom_validators.is_empty() {
            let (r, d) = mark_python_callback_plan(record, "validator");
            record = r;
            diagnostics.push(d);
        }
        record
    }
}
