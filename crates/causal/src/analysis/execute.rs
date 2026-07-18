//! Unified `CausalAnalysis` facade (DESIGN.md §21).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Analysis execution.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use std::time::Instant;

use causal_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext,
    TemporalEffectQuery,
};
use causal_data::{
    DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData,
};
use causal_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte,
    EstimationWorkspace, OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment,
};
use causal_graph::{Dag, TemporalCpdagReview, TemporalDag, TemporalGraphReview};
use causal_identify::{
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
};
use causal_validate::{
    BayesianSuiteContext, ValidationSuite,
};

use crate::callback_plan::mark_python_callback_plan;
use crate::error::AnalysisError;
use crate::inference::{BayesianConfig, InferenceMode};
use crate::planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, compile_logical_static_ate, compile_logical_temporal_effect,
    reject_dag_only_on_pag,
};
use crate::result::CausalAnalysisResult;
use crate::review::{
    PendingCpdagReview, PendingGraphReview, compile_review_required, compile_review_required_cpdag,
    compile_review_required_pag, compile_review_required_static_cpdag, ensure_review_complete,
};
use crate::strategy_table::{
    DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, EstimatorId,
    IdentifierId, estimate_provenance_step, estimate_static_effect, identify_provenance_step,
    identify_static,
};

use super::builder::{CausalAnalysisBuilder, DataInput, RdConfig, RefuteSuite};
use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, overlap_diagnostic,
    provenance_pair, resolve_analysis_ci, run_jpcmci_plus_review, run_lpcmci_review,
    run_pcmci_plus_review, run_pcmci_review, run_pc_review, run_refuters, run_rpcmci_discovery,
};

/// Prepared analysis (static or temporal).
#[derive(Clone)]
pub struct CausalAnalysis {
    pub(crate) data: DataInput,
    pub(crate) graph: GraphInput,
    pub(crate) query: CausalQuery,
    pub(crate) refute: RefuteSuite,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) rd: Option<RdConfig>,
    pub(crate) inference: InferenceMode,
    pub(crate) overlap_policy: Option<OverlapPolicy>,
    pub(crate) discovery_ci: Option<Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>>,
    pub(crate) custom_validators: Vec<Arc<dyn causal_validate::CustomEffectValidator>>,
}

impl std::fmt::Debug for CausalAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalAnalysis")
            .field("graph", &self.graph)
            .field("refute", &self.refute)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("discovery_ci", &self.discovery_ci.as_ref().map(|_| "<dyn CI>"))
            .field("custom_validators", &self.custom_validators.len())
            .finish()
    }
}

impl CausalAnalysis {
    /// Builder entry point.
    ///
    /// # Examples
    ///
    /// ```
    /// use causal::CausalAnalysis;
    ///
    /// let builder = CausalAnalysis::builder().bootstrap_replicates(50);
    /// let _ = builder;
    /// ```
    #[must_use]
    pub fn builder() -> CausalAnalysisBuilder {
        CausalAnalysisBuilder::new()
    }

    /// Mark physical plan when discovery CI override or custom validators are present.
    fn apply_callback_plan_marks(
        &self,
        mut record: causal_core::PhysicalExecutionPlanRecord,
        diagnostics: &mut Vec<causal_core::Diagnostic>,
    ) -> causal_core::PhysicalExecutionPlanRecord {
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

    /// Compile logical plan only (inspectable semantics).
    ///
    /// # Errors
    ///
    /// Modality / query validation failures. Does not run discovery.
    pub fn compile_logical(&self) -> Result<LogicalAnalysisPlan, AnalysisError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => compile_logical_temporal_effect(data, graph, q, self.split, false),
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                // Review usually required; logical metadata still inspectable.
                compile_logical_temporal_effect(data, &TemporalDag::empty(), q, self.split, true)
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                let data = multi.environment(0).map_err(|e| AnalysisError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                compile_logical_temporal_effect(data, &TemporalDag::empty(), q, self.split, true)
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), graph @ GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_static_pair();
                reject_dag_only_on_pag(graph, &identifier)?;
                Err(AnalysisError::Compile {
                    message: "static Pag requires class-aware identification \
                     (generalized.adjustment envelope); CausalAnalysis execute is not wired for PAG"
                        .into(),
                })
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("pc"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (_, CausalQuery::Distribution(_), _) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::Distribution is not wired through CausalAnalysis; \
                 use sample_interventional_distribution (identify/estimate deferred — IDC)",
            }),
            (_, CausalQuery::PathSpecific(_), _) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::PathSpecific is not wired through CausalAnalysis; \
                 use attribute_path_specific for path contribution (identify/estimate deferred)",
            }),
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    /// Compile logical → physical plan (or review-required).
    ///
    /// # Errors
    ///
    /// Modality / resource / discovery failures.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, AnalysisError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                let logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let logical = compile_logical_temporal_effect(data, graph, q, self.split, false)?;
                ensure_review_complete(&logical)?;
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_pcmci_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                if *accept_discovered {
                    PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required(review))
                }
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmciPlus { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_pcmci_plus_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus {
                    max_lag,
                    alpha,
                    fdr,
                    accept_discovered,
                    multi_dataset,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_jpcmci_plus_review(
                    multi,
                    *max_lag,
                    *alpha,
                    *fdr,
                    multi_dataset,
                    ci,
                    ctx,
                )?;
                let data = multi.environment(0).map_err(|e| AnalysisError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(_q),
                GraphInput::DiscoverRpcmci {
                    max_lag,
                    alpha,
                    fdr,
                    accept_discovered,
                    regime_assignment,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let result =
                    run_rpcmci_discovery(data, *max_lag, *alpha, *fdr, regime_assignment, ci, ctx)?;
                // Multi-regime estimation is not auto-wired; surface the first regime's CPDAG
                // for review. Auto-accept only when a single fully-oriented regime exists.
                let Some(first) = result.per_regime.first() else {
                    return Err(AnalysisError::Compile {
                        message: "RPCMCI returned no regime graphs".into(),
                    });
                };
                let review = first.review.clone();
                if *accept_discovered
                    && result.per_regime.len() == 1
                    && review.pending_undirected.is_empty()
                {
                    let q = match &self.query {
                        CausalQuery::TemporalEffect(q) => q.clone(),
                        _ => unreachable!(),
                    };
                    PendingCpdagReview::new(review, data.row_count(), q, self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(_),
                GraphInput::DiscoverLpcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_lpcmci_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                // Temporal backdoor is DAG-only; never auto-finish a PAG into Ready.
                let _ = accept_discovered;
                Ok(compile_review_required_pag(review))
            }
            (
                DataInput::Temporal(_),
                CausalQuery::TemporalEffect(_),
                GraphInput::TemporalPag(pag),
            ) => {
                // Never silently estimate on a PAG with temporal.backdoor (DAG-only).
                Ok(compile_review_required_pag(causal_graph::TemporalPagReview::from_pag(
                    pag.clone(),
                    "supplied.temporal_pag",
                )))
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc {
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_pc_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted.try_into_dag().map_err(|e| AnalysisError::ReviewRequired {
                        message: e.to_string(),
                    })?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("pc"));
                    let physical =
                        logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), graph @ GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_static_pair();
                reject_dag_only_on_pag(graph, &identifier)?;
                Err(AnalysisError::Compile {
                    message: "static Pag requires class-aware identification \
                     (generalized.adjustment envelope); CausalAnalysis execute is not wired for PAG"
                        .into(),
                })
            }
            (_, CausalQuery::Distribution(_), _) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::Distribution is not wired through CausalAnalysis; \
                 use sample_interventional_distribution (identify/estimate deferred — IDC)",
            }),
            (_, CausalQuery::PathSpecific(_), _) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::PathSpecific is not wired through CausalAnalysis; \
                 use attribute_path_specific for path contribution (identify/estimate deferred)",
            }),
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    fn validation_suite_id(&self) -> Option<Arc<str>> {
        match self.refute {
            RefuteSuite::None => None,
            RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
            RefuteSuite::Full => Some(Arc::from("validation.full")),
        }
    }

    fn ensure_supported_combination(&self) -> Result<(), AnalysisError> {
        match (&self.data, &self.query, &self.graph) {
            (_, CausalQuery::Distribution(_), _) => {
                return Err(AnalysisError::Unsupported {
                    message: "CausalQuery::Distribution is not wired through CausalAnalysis; \
                     use sample_interventional_distribution (identify/estimate deferred — IDC)",
                });
            }
            (_, CausalQuery::PathSpecific(_), _) => {
                return Err(AnalysisError::Unsupported {
                    message: "CausalQuery::PathSpecific is not wired through CausalAnalysis; \
                     use attribute_path_specific for path contribution (identify/estimate deferred)",
                });
            }
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (DataInput::Temporal(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
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
                return Err(AnalysisError::Compile {
                    message:
                        "PCMCI-family / temporal PAG discovery requires temporal data and a temporal effect query"
                            .into(),
                });
            }
            (DataInput::Temporal(_), _, GraphInput::DiscoverPc { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverPc { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "static PC discovery requires tabular data and AverageEffect".into(),
                });
            }
            (DataInput::Temporal(_), _, GraphInput::DiscoverJpcmciPlus { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "J-PCMCI+ discovery requires series_multi (MultiEnvironmentData)"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), _, graph) if !matches!(graph, GraphInput::DiscoverJpcmciPlus { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "multi-environment data currently supports only DiscoverJpcmciPlus"
                        .into(),
                });
            }
            (DataInput::Temporal(_), _, GraphInput::Pag(_)) => {
                return Err(AnalysisError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::MultiEnv(_), _, GraphInput::Pag(_)) => {
                return Err(AnalysisError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_static_pair();
                reject_dag_only_on_pag(&self.graph, &identifier)?;
            }
            _ => {}
        }
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            if let Some(id) = &self.identifier {
                if *id != IdentifierId::TemporalBackdoorUnfolded {
                    return Err(AnalysisError::Compile {
                        message: format!(
                            "temporal path only supports identifier \"temporal.backdoor.unfolded\"; got {id:?}"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                if *est != EstimatorId::TemporalLinearAdjustment {
                    return Err(AnalysisError::Compile {
                        message: format!(
                            "temporal path only supports estimator \"temporal.linear.adjustment\"; got {est:?}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    fn resolve_static_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), AnalysisError> {
        if matches!(EstimatorId::parse(estimator), EstimatorId::RdSharp) && self.rd.is_none() {
            return Err(AnalysisError::Compile {
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let CompiledAnalysis::Ready(physical) = plan else {
            return Err(AnalysisError::ReviewRequired {
                message: "cannot execute while graph review is required".into(),
            });
        };
        ensure_review_complete(&physical.logical)?;
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                let graph = match &self.graph {
                    GraphInput::Static(graph) => graph,
                    GraphInput::DiscoverPc { .. } => physical.static_graph().ok_or(
                        AnalysisError::Compile {
                            message: "Ready PC plan missing resolved static DAG (complete review first)"
                                .into(),
                        },
                    )?,
                    _ => {
                        return Err(AnalysisError::Unsupported {
                            message: "static ATE execute requires a supplied static DAG or DiscoverPc",
                        });
                    }
                };
                self.execute_static(data, graph, q, physical, ctx)
            }
            (DataInput::Temporal(data), CausalQuery::TemporalEffect(q)) => {
                let graph = physical.temporal_graph().ok_or(AnalysisError::Compile {
                    message: "Ready temporal plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            (_, CausalQuery::Distribution(_)) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::Distribution is not wired through CausalAnalysis; \
                 use sample_interventional_distribution (identify/estimate deferred — IDC)",
            }),
            (_, CausalQuery::PathSpecific(_)) => Err(AnalysisError::Unsupported {
                message: "CausalQuery::PathSpecific is not wired through CausalAnalysis; \
                 use attribute_path_specific for path contribution (identify/estimate deferred)",
            }),
            _ => Err(AnalysisError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Compile and run when Ready; error if review is required.
    ///
    /// # Errors
    ///
    /// Compile / review / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<CausalAnalysisResult, AnalysisError> {
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let DataInput::Temporal(data) = &self.data else {
            return Err(AnalysisError::Compile {
                message: "finish_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(AnalysisError::Compile {
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let DataInput::Temporal(data) = &self.data else {
            return Err(AnalysisError::Compile {
                message: "finish_cpdag_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(AnalysisError::Compile {
                message: "finish_cpdag_review_and_run requires temporal effect query".into(),
            });
        };
        let compiled = PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
            .finish(data, ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Dispatch identify → estimate for the static ATE path, routing on the plan's resolved
    /// identifier/estimator .
    fn execute_static(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let estimator_id = EstimatorId::parse(estimator);

        // rd.sharp has no graph-based identification step (DESIGN.md §21.2); dispatch to its
        // own path before touching `graph`.
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return self.execute_rd(data, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_bayesian(data, graph, query, physical, ctx);
        }

        let identification = identify_static(identifier, graph, query)?;
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Compile { message: "no estimand returned".into() })?;
        let assumptions = identification.required_assumptions.clone();

        let estimate = estimate_static_effect(
            estimator,
            data,
            &estimand,
            query,
            assumptions,
            self.bootstrap_replicates,
            self.overlap_policy,
            ctx,
        )?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        // ValidationSuite skips incompatible validators with NotApplicable rather than failing.
        let refutations = {
            let mut refute_ws = EstimationWorkspace::default();
            run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
            )?
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            posterior: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }))
    }

    /// Bayesian g-computation execute path.
    fn execute_bayesian(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let identification = identify_static(identifier, graph, query)?;
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Compile { message: "no estimand returned".into() })?;

        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
        };
        let prep = est.prepare(data, &estimand, query).map_err(AnalysisError::from)?;
        let mut ws = BayesianGCompWorkspace::default();
        let posterior =
            est.fit(&prep, identification.status, &mut ws, ctx).map_err(AnalysisError::from)?;
        let estimate = effect_from_posterior(&posterior)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let suite = ValidationSuite::bayesian_diagnostics();
                let mut bayes_ctx = BayesianSuiteContext::new(
                    &est,
                    &prep,
                    &posterior,
                    identification.status,
                    &mut ws,
                    estimate.ate,
                );
                let outcomes =
                    suite.run_bayesian(&mut bayes_ctx, ctx).map_err(AnalysisError::from)?;
                ValidationSuite::reports_only(&outcomes)
            }
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (
                "estimate.bayesian_gcomp",
                "estimate.bayesian_gcomp",
                &[id_artifact],
                &estimate.assumptions,
            ),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            posterior: Some(posterior),
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }))
    }

    /// `rd.sharp` execute path: identify via [`SharpRdIdentifier`], then estimate.
    fn execute_rd(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let rd = self.rd.ok_or_else(|| AnalysisError::Compile {
            message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
        })?;
        let identification = SharpRdIdentifier::new(SharpRdConfig {
            running_variable: rd.running_variable,
            cutoff: rd.cutoff,
            bandwidth: rd.bandwidth,
        })
        .identify(CausalQuery::AverageEffect(query.clone()))
        .map_err(AnalysisError::from)?;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            AnalysisError::Compile { message: "rd.sharp returned no estimand".into() }
        })?;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        let prep = est.prepare(data, &estimand, query).map_err(AnalysisError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est
            .fit(&prep, &mut ws, ctx, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;

        let mut diagnostics = vec![overlap_diagnostic(estimate.overlap)];

        let provenance = provenance_pair(
            ("identify.rd_design", "identify.rd_sharp", &[], &identification.required_assumptions),
            ("estimate.rd", "estimate.rd_sharp", &["identify.rd_design"], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            posterior: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }))
    }

    fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(AnalysisError::from)?;
        let identification = id_res.result;
        if identification.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(AnalysisError::Compile {
                message: "temporal effect not identified".into(),
            });
        }
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Compile { message: "no estimand returned".into() })?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(
                data,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(AnalysisError::from)?;
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.temporal_linear_adjustment",
                "estimate.temporal.linear.adjustment",
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            posterior: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }))
    }
}
