//! Unified `CausalAnalysis` facade (DESIGN.md §21).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    AssumptionSet, AverageEffectQuery, BufferMaterialization, CausalQuery, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord,
    LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode,
    TemporalEffectQuery, VERSION, VariableId,
};
use causal_data::{
    DiscoveryEstimationSplit, MultiEnvironmentData, TableView, TabularData, TimeSeriesData,
};
use causal_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, EffectEstimate,
    EstimationWorkspace, OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment,
};
use causal_expr::IdentifiedEstimand;
use causal_graph::{Dag, Pag, TemporalCpdagReview, TemporalDag, TemporalGraphReview, TemporalPag};
use causal_identify::{
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
};
use causal_validate::{
    BayesianSuiteContext, RefutationProblem, RefutationReport, ValidationSuite,
};

use crate::discovery::{
    DiscoverParams, discover_jpcmci_plus, discover_lpcmci, discover_pcmci, discover_pcmci_plus,
    discover_rpcmci,
};
use crate::discovery_defaults::resolve_ci;
use causal_discovery::{MultiDatasetConstraints, two_regime_half_split};
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
    compile_review_required_pag, ensure_review_complete,
};
use crate::strategy_table::{
    DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, EstimatorId,
    IdentifierId, estimate_provenance_step, estimate_static_effect, identify_provenance_step,
    identify_static,
};

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Placebo + random common cause (linear backdoor only).
    PlaceboAndRcc,
    /// Full validation suite (applicable validators only; others NotApplicable).
    Full,
}

#[derive(Clone, Debug)]
enum DataInput {
    Tabular(TabularData),
    Temporal(TimeSeriesData),
}

/// Running-variable configuration for the `rd.sharp` estimator; required when `rd.sharp` is
/// selected as the estimator (see [`CausalAnalysisBuilder::rd_config`]).
#[derive(Clone, Copy, Debug)]
pub struct RdConfig {
    /// Running (assignment) variable.
    pub running_variable: VariableId,
    /// Discontinuity cutoff.
    pub cutoff: f64,
    /// Symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is retained).
    pub bandwidth: f64,
}

/// Builder for static or temporal analysis.
#[derive(Clone, Debug)]
pub struct CausalAnalysisBuilder {
    data: Option<DataInput>,
    graph: Option<GraphInput>,
    query: Option<CausalQuery>,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
    split: Option<DiscoveryEstimationSplit>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
    /// Optional override for propensity / AIPW overlap (clip/trim). `None` keeps estimator defaults.
    overlap_policy: Option<OverlapPolicy>,
}

impl Default for CausalAnalysisBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalAnalysisBuilder {
    /// Start a builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            graph: None,
            query: None,
            refute: RefuteSuite::PlaceboAndRcc,
            bootstrap_replicates: 100,
            split: None,
            identifier: None,
            estimator: None,
            rd: None,
            inference: InferenceMode::Frequentist,
            overlap_policy: None,
        }
    }

    /// Supply tabular data.
    #[must_use]
    pub fn data(mut self, data: TabularData) -> Self {
        self.data = Some(DataInput::Tabular(data));
        self
    }

    /// Supply temporal series data.
    #[must_use]
    pub fn series(mut self, data: TimeSeriesData) -> Self {
        self.data = Some(DataInput::Temporal(data));
        self
    }

    /// Supply a validated static DAG.
    #[must_use]
    pub fn graph(mut self, graph: Dag) -> Self {
        self.graph = Some(GraphInput::Static(graph));
        self
    }

    /// Supply a temporal DAG template.
    #[must_use]
    pub fn temporal_graph(mut self, graph: TemporalDag) -> Self {
        self.graph = Some(GraphInput::Temporal(graph));
        self
    }

    /// Discover with PCMCI (typically yields [`CompiledAnalysis::ReviewRequired`]).
    #[must_use]
    pub fn discover_pcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverPcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with PCMCI+ (typically yields [`CompiledAnalysis::ReviewRequiredCpdag`]).
    ///
    /// `accept` only auto-completes when the oriented CPDAG has no undirected marks;
    /// otherwise compile still returns review-required (no silent coercion).
    #[must_use]
    pub fn discover_pcmci_plus(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverPcmciPlus {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with J-PCMCI+ (multi-environment; typically review-required via Python path).
    #[must_use]
    pub fn discover_jpcmci_plus(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverJpcmciPlus {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with RPCMCI (regime graphs; typically review-required via Python path).
    #[must_use]
    pub fn discover_rpcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverRpcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with LPCMCI (temporal PAG; typically [`CompiledAnalysis::ReviewRequiredPag`]).
    #[must_use]
    pub fn discover_lpcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverLpcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Supply a static PAG (class-aware identification required; DAG-only IDs are refused).
    #[must_use]
    pub fn pag(mut self, graph: Pag) -> Self {
        self.graph = Some(GraphInput::Pag(graph));
        self
    }

    /// Supply a temporal PAG (review / class-aware identification required).
    #[must_use]
    pub fn temporal_pag(mut self, graph: TemporalPag) -> Self {
        self.graph = Some(GraphInput::TemporalPag(graph));
        self
    }

    /// Average-effect query (static).
    #[must_use]
    pub fn query(mut self, query: AverageEffectQuery) -> Self {
        self.query = Some(CausalQuery::AverageEffect(query));
        self
    }

    /// Generic causal query (static or temporal).
    #[must_use]
    pub fn causal_query(mut self, query: CausalQuery) -> Self {
        self.query = Some(query);
        self
    }

    /// Temporal effect query.
    #[must_use]
    pub fn temporal_query(mut self, query: TemporalEffectQuery) -> Self {
        self.query = Some(CausalQuery::TemporalEffect(query));
        self
    }

    /// Discovery / estimation temporal-gap split.
    #[must_use]
    pub fn split(mut self, split: DiscoveryEstimationSplit) -> Self {
        self.split = Some(split);
        self
    }

    /// Configure refutation suite (static path).
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
        self
    }

    /// Bootstrap replicates for the primary estimate.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self
    }

    /// Select the identification strategy for the static ATE path.
    ///
    /// Defaults to [`IdentifierId::BackdoorAdjustment`] when unset. Wire strings such as
    /// `"backdoor.adjustment"` are accepted via [`From<&str>`]. `compile` refuses any
    /// identifier/estimator pair outside the allowlist. Ignored on the temporal path (which
    /// always uses [`IdentifierId::TemporalBackdoorUnfolded`]).
    #[must_use]
    pub fn identifier(mut self, id: impl Into<IdentifierId>) -> Self {
        self.identifier = Some(id.into());
        self
    }

    /// Select the estimator for the static ATE path.
    ///
    /// Defaults to [`EstimatorId::LinearAdjustmentAte`] when unset. Wire strings such as
    /// `"linear.adjustment.ate"` are accepted via [`From<&str>`]. `compile` refuses any
    /// identifier/estimator pair outside the allowlist. Ignored on the temporal path (which
    /// always uses [`EstimatorId::TemporalLinearAdjustment`]).
    #[must_use]
    pub fn estimator(mut self, id: impl Into<EstimatorId>) -> Self {
        self.estimator = Some(id.into());
        self
    }

    /// Configure frequentist vs Bayesian inference (DESIGN.md §34.1).
    ///
    /// [`InferenceMode::Bayesian`] selects estimator [`EstimatorId::BayesianGcomp`].
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_)) {
            self.estimator = Some(EstimatorId::BayesianGcomp);
        }
        self.inference = mode;
        self
    }

    /// Overlap / positivity policy for propensity and AIPW estimators (DESIGN.md §14.3).
    ///
    /// When unset, those estimators keep their built-in defaults (clip = 0.01, no trim).
    /// Ignored by estimators that require [`OverlapPolicy::ExplicitOverride`] (linear, GLM, IV,
    /// front-door, RD).
    #[must_use]
    pub fn overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.overlap_policy = Some(policy);
        self
    }

    /// Configure the running variable / cutoff / bandwidth required by the `rd.sharp`
    /// estimator. `compile` refuses `rd.sharp` without this.
    #[must_use]
    pub fn rd_config(mut self, running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        self.rd = Some(RdConfig { running_variable, cutoff, bandwidth });
        self
    }

    /// Build the analysis object.
    ///
    /// # Errors
    ///
    /// Missing required fields.
    pub fn build(self) -> Result<CausalAnalysis, AnalysisError> {
        Ok(CausalAnalysis {
            data: self.data.ok_or(AnalysisError::Missing { field: "data" })?,
            graph: self.graph.ok_or(AnalysisError::Missing { field: "graph" })?,
            query: self.query.ok_or(AnalysisError::Missing { field: "query" })?,
            refute: self.refute,
            bootstrap_replicates: self.bootstrap_replicates,
            split: self.split,
            identifier: self.identifier,
            estimator: self.estimator,
            rd: self.rd,
            inference: self.inference,
            overlap_policy: self.overlap_policy,
        })
    }
}

/// Prepared analysis (static or temporal).
#[derive(Clone, Debug)]
pub struct CausalAnalysis {
    data: DataInput,
    graph: GraphInput,
    query: CausalQuery,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
    split: Option<DiscoveryEstimationSplit>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
    overlap_policy: Option<OverlapPolicy>,
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
                | GraphInput::DiscoverJpcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                // Review usually required; logical metadata still inspectable.
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
                let review = run_pcmci_review(data, *max_lag, *alpha, *fdr, ctx)?;
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
                let review = run_pcmci_plus_review(data, *max_lag, *alpha, *fdr, ctx)?;
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
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let review = run_jpcmci_plus_review(data, *max_lag, *alpha, *fdr, ctx)?;
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
                GraphInput::DiscoverRpcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let result = run_rpcmci_discovery(data, *max_lag, *alpha, *fdr, ctx)?;
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
                let review = run_lpcmci_review(data, *max_lag, *alpha, *fdr, ctx)?;
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
            (DataInput::Temporal(_), _, GraphInput::Pag(_)) => {
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
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "static ATE execute requires a supplied static DAG",
                    });
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
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
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
                )?
            }
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
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

        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
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

        let diagnostics = vec![overlap_diagnostic(estimate.overlap)];

        let provenance = provenance_pair(
            ("identify.rd_design", "identify.rd_sharp", &[], &identification.required_assumptions),
            ("estimate.rd", "estimate.rd_sharp", &["identify.rd_design"], &estimate.assumptions),
        );

        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
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

        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
            identification,
            estimand,
            estimate,
            posterior: None,
            refutations: Vec::new(),
            diagnostics: Vec::new(),
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }))
    }
}

struct AssembleArgs<'a> {
    logical: &'a LogicalAnalysisPlanRecord,
    physical: &'a PhysicalExecutionPlanRecord,
    identification: causal_identify::IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<causal_estimate::CausalPosterior>,
    refutations: Vec<RefutationReport>,
    diagnostics: Vec<Diagnostic>,
    provenance: ProvenanceGraph,
    treatment: VariableId,
    outcome: VariableId,
    /// Wall-clock nanoseconds for identify→estimate→refute.
    wall_time_ns: u64,
}

fn assemble_result(args: AssembleArgs<'_>) -> CausalAnalysisResult {
    let copy_count = args
        .physical
        .materializations
        .iter()
        .filter(|(_, m)| !matches!(m, BufferMaterialization::Borrowed))
        .count() as u64;
    CausalAnalysisResult {
        logical_plan: args.logical.clone(),
        physical_plan: args.physical.clone(),
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        posterior: args.posterior,
        refutations: args.refutations,
        diagnostics: args.diagnostics,
        provenance: args.provenance,
        performance: ExecutionPerformanceRecord {
            wall_time_ns: Some(args.wall_time_ns),
            peak_rss_bytes: None,
            copy_count,
            scalar_fallback_count: 0,
        },
        treatment: args.treatment,
        outcome: args.outcome,
    }
}

type ProvStep<'a> = (&'a str, &'a str, &'a [&'a str], &'a AssumptionSet);

fn provenance_pair(first: ProvStep<'_>, second: ProvStep<'_>) -> ProvenanceGraph {
    let mut provenance = ProvenanceGraph::new();
    for (artifact_id, operation, parents, assumptions) in [first, second] {
        let parent_arcs: Arc<[Arc<str>]> =
            parents.iter().map(|p| Arc::<str>::from(*p)).collect::<Vec<_>>().into();
        provenance.push(ProvenanceNode {
            artifact_id: Arc::from(artifact_id),
            operation: Arc::from(operation),
            parents: parent_arcs,
            assumptions: assumptions.clone(),
            library_version: Arc::from(VERSION),
            config_digest: Some(Arc::from("temporal")),
        });
    }
    provenance
}

fn run_pcmci_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ctx: &ExecutionContext,
) -> Result<TemporalGraphReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_pcmci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

fn run_pcmci_plus_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_pcmci_plus(data, &vars, &params, ctx)?;
    Ok(result.review)
}

fn run_jpcmci_plus_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let multi = MultiEnvironmentData::try_new(Arc::from([data.clone()])).map_err(|e| {
        AnalysisError::Compile { message: format!("jpcmci+ multi-env wrap failed: {e}") }
    })?;
    let result = discover_jpcmci_plus(&multi, &vars, &params, ctx)?;
    Ok(result.review)
}

fn run_rpcmci_discovery(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ctx: &ExecutionContext,
) -> Result<causal_discovery::RpcmciDiscoveryResult, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let assign = two_regime_half_split(data.row_count());
    discover_rpcmci(data, &vars, &assign, &params, None, ctx)
}

fn run_lpcmci_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ctx: &ExecutionContext,
) -> Result<causal_graph::TemporalPagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_lpcmci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
) -> Result<Vec<RefutationReport>, AnalysisError> {
    let problem =
        RefutationProblem { data, estimand, query, original: estimate, estimator: Some(estimator) };
    let validation = match suite {
        RefuteSuite::None => return Ok(Vec::new()),
        RefuteSuite::PlaceboAndRcc => ValidationSuite::placebo_and_rcc(),
        RefuteSuite::Full => ValidationSuite::full_effect(),
    };
    let outcomes = validation.run(&problem, workspace, ctx).map_err(AnalysisError::from)?;
    Ok(ValidationSuite::reports_only(&outcomes))
}

fn effect_from_posterior(posterior: &CausalPosterior) -> Result<EffectEstimate, AnalysisError> {
    let eq = posterior.effect_column().ok_or_else(|| AnalysisError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let ate = posterior.summaries.mean[eq];
    // Report posterior SD of the effect (sampling uncertainty), not MCSE of the mean.
    let se = posterior.summaries.sd[eq];
    Ok(EffectEstimate {
        ate,
        se_analytic: se,
        se_bootstrap: None,
        bootstrap_replicates_ok: None,
        bootstrap_replicates_failed: None,
        assumptions: posterior.assumptions.clone(),
        overlap: OverlapPolicy::ExplicitOverride,
        overlap_report: None,
        retained_memory_bytes: None,
    })
}

/// Diagnostic recording which overlap policy an estimator applied.
fn overlap_diagnostic(overlap: OverlapPolicy) -> Diagnostic {
    match overlap {
        OverlapPolicy::ExplicitOverride => Diagnostic::new(
            "estimate.overlap.explicit_override",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "estimator used ExplicitOverride for positivity (not a propensity-based method)",
        ),
        OverlapPolicy::RequireDiagnostics { .. } => Diagnostic::new(
            "estimate.overlap.require_diagnostics",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "estimator used RequireDiagnostics for mandatory positivity diagnostics",
        ),
    }
}
