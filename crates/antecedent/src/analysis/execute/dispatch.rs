// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::CausalAnalysis {
    pub(super) fn validation_suite_id(&self) -> Option<Arc<str>> {
        match self.refute {
            RefuteSuite::None => None,
            RefuteSuite::Cheap => Some(Arc::from("overlap+evalue")),
            RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
            RefuteSuite::Full => Some(Arc::from("validation.full")),
        }
    }

    pub(super) fn ensure_supported_combination(&self) -> Result<(), CausalError> {
        match (&self.data, &self.query, &self.graph) {
            (_, CausalQuery::Distribution(_), graph)
                if !matches!(
                    (&self.data, graph),
                    (DataInput::Tabular(_), GraphInput::Static(_))
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static DAG",
                });
            }
            (_, CausalQuery::PathSpecific(_), graph)
                if !matches!(
                    (&self.data, graph),
                    (DataInput::Tabular(_), GraphInput::Static(_))
                ) =>
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
                DataInput::Tabular(_),
                _,
                GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverJpcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                return Err(CausalError::Compile {
                    message:
                        "PCMCI-family / temporal PAG discovery requires temporal data and a temporal effect query"
                            .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphInput::DiscoverPc { .. }
                | GraphInput::DiscoverGes { .. }
                | GraphInput::DiscoverLingam { .. }
                | GraphInput::DiscoverNotears { .. }
                | GraphInput::DiscoverFci { .. }
                | GraphInput::DiscoverRfci { .. },
            ) => {
                return Err(CausalError::Compile {
                    message: "static PC/GES/LiNGAM/NOTEARS/FCI/RFCI discovery requires tabular data and AverageEffect"
                        .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                _,
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                return Err(CausalError::Compile {
                    message:
                        "J-PCMCI+ discovery requires series_multi (MultiEnvironmentData) or panel"
                            .into(),
                });
            }
            (DataInput::MultiEnv(_), _, graph)
                if !matches!(graph, GraphInput::DiscoverJpcmciPlus { .. }) =>
            {
                return Err(CausalError::Compile {
                    message: "multi-environment data currently supports only DiscoverJpcmciPlus"
                        .into(),
                });
            }
            (DataInput::Panel(_), _, graph)
                if !matches!(
                    graph,
                    GraphInput::DiscoverJpcmciPlus { .. }
                        | GraphInput::DiscoverPcmci { .. }
                        | GraphInput::DiscoverPcmciPlus { .. }
                        | GraphInput::DiscoverLpcmci { .. }
                        | GraphInput::Temporal(_)
                ) =>
            {
                return Err(CausalError::Compile {
                    message: "panel data supports DiscoverJpcmciPlus, DiscoverPcmci/Plus/Lpcmci \
                              (pooled units), or a supplied TemporalDag"
                        .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphInput::Pag(_),
            ) => {
                return Err(CausalError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, IdentifierId::parse(&identifier))?;
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
        default_identifier: &IdentifierId,
        default_estimator: &EstimatorId,
    ) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(default_identifier);
        let estimator = self.estimator.as_ref().unwrap_or(default_estimator);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    pub(super) fn resolve_static_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(&DEFAULT_IDENTIFIER_ID, &DEFAULT_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    pub(super) fn resolve_pag_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(&DEFAULT_PAG_IDENTIFIER_ID, &DEFAULT_PAG_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for ADMG ATE (general ID + functional effect).
    pub(super) fn resolve_admg_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(&DEFAULT_ADMG_IDENTIFIER_ID, &DEFAULT_ADMG_ESTIMATOR_ID)
    }

    /// Resolve identifier/estimator for ConditionalEffect.
    pub(super) fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            &DEFAULT_CONDITIONAL_IDENTIFIER_ID,
            &DEFAULT_CONDITIONAL_ESTIMATOR_ID,
        )
    }

    /// Resolve identifier/estimator for Distribution queries.
    pub(super) fn resolve_distribution_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            &DEFAULT_DISTRIBUTION_IDENTIFIER_ID,
            &DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
        )
    }

    /// Resolve identifier/estimator for PathSpecific queries.
    pub(super) fn resolve_path_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(&DEFAULT_PATH_IDENTIFIER_ID, &DEFAULT_PATH_ESTIMATOR_ID)
    }

    pub(super) fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), CausalError> {
        if matches!(EstimatorId::parse(estimator), EstimatorId::RdSharp) && self.rd.is_none() {
            return Err(CausalError::Compile {
                message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
            });
        }
        Ok(())
    }

    /// Execute a Ready physical plan.
    ///
    /// # Errors
    ///
    /// Identification / estimation / validation failures.
    pub fn execute(
        &self,
        plan: &CompiledAnalysis,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let CompiledAnalysis::Ready(physical) = plan else {
            let (kind, algorithm, pending, hint) = match plan {
                CompiledAnalysis::ReviewRequired(r) => (
                    "temporal_dag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len(),
                    "call finish_review_and_run after accepting pending edges",
                ),
                CompiledAnalysis::ReviewRequiredCpdag(r) => (
                    "temporal_cpdag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len() + r.pending_undirected.len(),
                    "orient undirected edges then finish_cpdag_review_and_run",
                ),
                CompiledAnalysis::ReviewRequiredStaticCpdag(r) => (
                    "static_cpdag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len() + r.pending_undirected.len(),
                    "orient undirected CPDAG edges or supply a Dag",
                ),
                CompiledAnalysis::ReviewRequiredStaticDag(r) => (
                    "static_dag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len(),
                    "accept pending directed edges or supply a fully oriented Dag",
                ),
                CompiledAnalysis::ReviewRequiredStaticPag(r) => (
                    "static_pag",
                    Some(r.algorithm.to_string()),
                    r.pending_circles.len(),
                    "finish_static_pag_review_and_run or supply a completed Pag/Dag",
                ),
                CompiledAnalysis::ReviewRequiredPag(r) => (
                    "temporal_pag",
                    Some(r.algorithm.to_string()),
                    r.pending_circles.len(),
                    "complete TemporalPag to a TemporalDag (no circle/ambiguous marks), \
                     or finish PAG review; temporal backdoor does not run on PAG class ID",
                ),
                CompiledAnalysis::Ready(_) => unreachable!(),
            };
            return Err(CausalError::review_required(
                kind,
                algorithm,
                pending,
                "cannot execute while graph review is required",
                hint,
            ));
        };
        ensure_review_complete(&physical.logical)?;
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => match &self.graph {
                GraphInput::Static(graph) => self.execute_static(data, graph, q, physical, ctx),
                GraphInput::DiscoverPc { .. }
                | GraphInput::DiscoverGes { .. }
                | GraphInput::DiscoverLingam { .. }
                | GraphInput::DiscoverNotears { .. }
                | GraphInput::Cpdag(_) => {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                            message:
                                "Ready PC/GES/LiNGAM/NOTEARS/CPDAG plan missing resolved static DAG (complete review first)"
                                    .into(),
                        })?;
                    self.execute_static(data, graph, q, physical, ctx)
                }
                GraphInput::Admg(admg) => {
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
                GraphInput::Pag(_)
                | GraphInput::DiscoverFci { .. }
                | GraphInput::DiscoverRfci { .. } => {
                    let pag = physical.static_pag().ok_or(CausalError::Compile {
                        message:
                            "Ready PAG plan missing resolved static PAG (complete review first)"
                                .into(),
                    })?;
                    self.execute_pag(data, pag, q, physical, ctx)
                }
                GraphInput::DiscoverExactDagPosterior
                | GraphInput::DiscoverOrderMcmc { .. }
                | GraphInput::DiscoverStructureMcmc { .. }
                | GraphInput::DiscoverCiScreenedPosterior { .. } => {
                    self.execute_graph_posterior_bayesian(data, q, physical, ctx)
                }
                _ => Err(CausalError::Unsupported {
                    message: "static ATE execute requires a supplied static DAG/PAG/CPDAG/ADMG or DiscoverPc/Ges/Lingam/Notears/Fci/Rfci/graph-posterior",
                }),
            },
            (DataInput::Tabular(data), CausalQuery::Distribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "Distribution execute requires a supplied static DAG",
                    });
                };
                self.execute_distribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "PathSpecific execute requires a supplied static DAG",
                    });
                };
                self.execute_path_specific(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "ConditionalEffect execute requires a supplied static DAG",
                    });
                };
                self.execute_conditional(data, graph, q, physical, ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation execute requires a supplied static DAG",
                    });
                };
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Counterfactual(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "Counterfactual execute requires a supplied static DAG",
                    });
                };
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::AnomalyAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "AnomalyAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ChangeAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "ChangeAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::MechanismChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "MechanismChange execute requires a supplied static DAG",
                    });
                };
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::UnitChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "UnitChange execute requires a supplied static DAG",
                    });
                };
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
            ) => {
                if matches!(self.graph, GraphInput::DiscoverDbnPosterior { .. }) {
                    return self.execute_dbn_posterior_bayesian(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            (DataInput::Panel(panel), CausalQuery::TemporalEffect(q)) => {
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready panel plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_panel(panel, graph, q, physical, ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Compile and run when Ready; error if review is required.
    ///
    /// # Errors
    ///
    /// Compile / review / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<CausalAnalysisResult, CausalError> {
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Identify only (no estimation). Supports static DAG average-effect / related queries.
    ///
    /// # Errors
    ///
    /// Missing DAG graph, unsupported graph class, or identification failure.
    pub fn identify_only(&self) -> Result<IdentificationResult, CausalError> {
        use crate::strategy_table::{DEFAULT_IDENTIFIER_ID, identify_static_query};

        let GraphInput::Static(graph) = &self.graph else {
            return Err(CausalError::Unsupported {
                message: "identify_only currently supports static DAG graphs only",
            });
        };
        let id = self.identifier.clone().unwrap_or(DEFAULT_IDENTIFIER_ID);
        identify_static_query(id, graph, &self.query)
    }

    /// Inspectable compile result (logical + physical when Ready).
    ///
    /// Prefer this over `compile` when documenting plan inspection in user code.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile`].
    pub fn plan(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, CausalError> {
        self.compile(ctx)
    }

    /// Continue after DAG review: accept all pending edges then execute.
    ///
    /// # Errors
    ///
    /// Review / execute failures.
    pub fn finish_review_and_run(
        &self,
        review: TemporalGraphReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_review_and_run requires temporal effect query".into(),
            });
        };
        let compiled = PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
            .accept_all()
            .finish(data, ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Continue after PCMCI+ CPDAG review once undirected marks are oriented and directed accepted.
    ///
    /// # Errors
    ///
    /// Incomplete CPDAG review (undirected remain) or execute failures.
    pub fn finish_cpdag_review_and_run(
        &self,
        review: TemporalCpdagReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_cpdag_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_cpdag_review_and_run requires temporal effect query".into(),
            });
        };
        let compiled = PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
            .finish(data, ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Dispatch identify → estimate for the static ATE path, routing on the plan's resolved
    /// identifier/estimator .
    /// Continue after static PAG review once circle marks are resolved.
    ///
    /// # Errors
    ///
    /// Incomplete review or execute failures.
    pub fn finish_static_pag_review_and_run(
        &self,
        review: PagReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let DataInput::Tabular(data) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_static_pag_review_and_run requires tabular data".into(),
            });
        };
        let CausalQuery::AverageEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_static_pag_review_and_run requires AverageEffect".into(),
            });
        };
        // Circles are fine: generalized adjustment samples MAG completions.
        let (identifier, estimator) = self.resolve_pag_pair();
        let logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
            data,
            pag: &review.graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        let physical =
            logical.compile_physical_with_all_graphs(ctx, None, None, Some(review.graph))?;
        self.execute(&CompiledAnalysis::Ready(physical), ctx)
    }

    /// Builder entry point.
    #[must_use]
    pub fn builder() -> CausalAnalysisBuilder {
        CausalAnalysisBuilder::new()
    }

    /// Mark physical plan when discovery CI override or custom validators are present.
    pub(super) fn apply_callback_plan_marks(
        &self,
        mut record: antecedent_core::PhysicalExecutionPlanRecord,
        diagnostics: &mut Vec<antecedent_core::Diagnostic>,
    ) -> antecedent_core::PhysicalExecutionPlanRecord {
        if self.discovery_ci.is_some() {
            let (r, d) = mark_python_callback_plan(record, "ci");
            record = r;
            diagnostics.push(d);
        }
        if !self.custom_validators.is_empty() {
            let (r, d) = mark_python_callback_plan(record, "validator");
            record = r;
            diagnostics.push(d);
        }
        record
    }
}
