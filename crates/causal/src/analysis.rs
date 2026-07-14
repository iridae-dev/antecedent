//! Unified `CausalAnalysis` facade (DESIGN.md §21 Phase 3).
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
use causal_data::{DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData};
use causal_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Pcmci, PcmciPlus, TemporalConstraints,
};
use causal_estimate::{
    AipwAte, AipwWorkspace, BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior,
    DistanceMatching, EffectEstimate, EstimationError, EstimationWorkspace, FrontDoorTwoStage,
    FrontDoorWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace, LinearAdjustmentAte,
    OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching, PropensityStratification,
    PropensityWeighting, RdWorkspace, SharpRegressionDiscontinuity, TemporalLinearAdjustment,
    TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv,
};
use causal_expr::IdentifiedEstimand;
use causal_graph::{Dag, TemporalCpdagReview, TemporalDag, TemporalGraphReview};
use causal_identify::{
    BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier, IdentificationError,
    IdentificationResult, IdentificationStatus, InstrumentalVariableIdentifier, SharpRdConfig,
    SharpRdIdentifier, TemporalBackdoorIdentifier,
};
use causal_validate::{RefutationProblem, RefutationReport, ValidationSuite};

use crate::error::AnalysisError;
use crate::inference::{BayesianConfig, InferenceMode};
use crate::planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, compile_logical_static_ate, compile_logical_temporal_effect,
};
use crate::result::CausalAnalysisResult;
use crate::review::{
    PendingCpdagReview, PendingGraphReview, compile_review_required, compile_review_required_cpdag,
    ensure_review_complete,
};

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Placebo + random common cause (linear backdoor only).
    PlaceboAndRcc,
    /// Full Phase 4 validation suite (applicable validators only; others NotApplicable).
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
    identifier: Option<Arc<str>>,
    estimator: Option<Arc<str>>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
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
    pub fn discover_pcmci(mut self, max_lag: u32, alpha: f64, fdr: bool, accept: bool) -> Self {
        self.graph =
            Some(GraphInput::DiscoverPcmci { max_lag, alpha, fdr, accept_discovered: accept });
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
        fdr: bool,
        accept: bool,
    ) -> Self {
        self.graph =
            Some(GraphInput::DiscoverPcmciPlus { max_lag, alpha, fdr, accept_discovered: accept });
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

    /// Select the identification strategy for the static ATE path (Phase 4; DESIGN.md §21.2).
    ///
    /// Defaults to `backdoor.adjustment` when unset. Supported ids: `backdoor.adjustment`,
    /// `backdoor.efficient`, `frontdoor`, `iv`, `rd.sharp`. `compile` refuses any
    /// identifier/estimator pair outside the allowlist. Ignored on the temporal path (which
    /// always uses `temporal.backdoor.unfolded`).
    #[must_use]
    pub fn identifier(mut self, id: impl Into<Arc<str>>) -> Self {
        self.identifier = Some(id.into());
        self
    }

    /// Select the estimator for the static ATE path (Phase 4; DESIGN.md §21.2).
    ///
    /// Defaults to `linear.adjustment.ate` when unset. Supported ids: `linear.adjustment.ate`,
    /// `propensity.weighting`, `propensity.matching`, `propensity.stratification`,
    /// `distance.matching`, `aipw`, `glm.adjustment`, `frontdoor.two_stage`, `iv.wald`,
    /// `iv.2sls`, `rd.sharp`. `compile` refuses any identifier/estimator pair outside the
    /// allowlist. Ignored on the temporal path (which always uses
    /// `temporal.linear.adjustment`).
    #[must_use]
    pub fn estimator(mut self, id: impl Into<Arc<str>>) -> Self {
        self.estimator = Some(id.into());
        self
    }

    /// Configure frequentist vs Bayesian inference (DESIGN.md §34.1).
    ///
    /// [`InferenceMode::Bayesian`] selects estimator `bayesian.gcomp`.
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_)) {
            self.estimator = Some(Arc::from("bayesian.gcomp"));
        }
        self.inference = mode;
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
    identifier: Option<Arc<str>>,
    estimator: Option<Arc<str>>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
}

impl CausalAnalysis {
    /// Builder entry point.
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
                GraphInput::DiscoverPcmci { .. } | GraphInput::DiscoverPcmciPlus { .. },
            ) => {
                // Review usually required; logical metadata still inspectable.
                compile_logical_temporal_effect(data, &TemporalDag::empty(), q, self.split, true)
            }
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination in Phase 3",
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
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination in Phase 3",
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
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (DataInput::Temporal(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message:
                        "static ATE on temporal data is not a Phase 3 path; use TemporalEffect"
                            .into(),
                });
            }
            (
                DataInput::Tabular(_),
                _,
                GraphInput::DiscoverPcmci { .. } | GraphInput::DiscoverPcmciPlus { .. },
            ) => {
                return Err(AnalysisError::Compile {
                    message:
                        "PCMCI / PCMCI+ discovery requires temporal data and a temporal effect query"
                            .into(),
                });
            }
            _ => {}
        }
        // The temporal path is Phase 3 linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            if let Some(id) = &self.identifier {
                if &**id != "temporal.backdoor.unfolded" {
                    return Err(AnalysisError::Compile {
                        message: format!(
                            "temporal path only supports identifier \"temporal.backdoor.unfolded\"; got {id:?}"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                if &**est != "temporal.linear.adjustment" {
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
        let identifier =
            self.identifier.clone().unwrap_or_else(|| Arc::from("backdoor.adjustment"));
        let estimator =
            self.estimator.clone().unwrap_or_else(|| Arc::from("linear.adjustment.ate"));
        (identifier, estimator)
    }

    fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), AnalysisError> {
        if estimator == "rd.sharp" && self.rd.is_none() {
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
    /// identifier/estimator (Phase 4; DESIGN.md §21.2).
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
            physical.logical.record.identifier.as_deref().unwrap_or("backdoor.adjustment");
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or("linear.adjustment.ate");

        // rd.sharp has no graph-based identification step (DESIGN.md §21.2); dispatch to its
        // own path before touching `graph`.
        if estimator == "rd.sharp" {
            return self.execute_rd(data, query, physical, ctx);
        }
        if estimator == "bayesian.gcomp" {
            return self.execute_bayesian(data, graph, query, physical, ctx);
        }

        let identification = identify_static(identifier, graph, query)?;
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Identify("no estimand returned".into()))?;
        let assumptions = identification.required_assumptions.clone();

        let estimate: EffectEstimate = match estimator {
            "linear.adjustment.ate" => {
                let mut est = LinearAdjustmentAte::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                est.overlap = OverlapPolicy::ExplicitOverride;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = EstimationWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "propensity.weighting" => {
                let mut est = PropensityWeighting::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = PropensityEstimationWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "propensity.matching" => {
                let mut est = PropensityMatching::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = PropensityEstimationWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "propensity.stratification" => {
                let mut est = PropensityStratification::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = PropensityEstimationWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "distance.matching" => {
                let mut est = DistanceMatching::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = PropensityEstimationWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "aipw" => {
                let mut est = AipwAte::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = AipwWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "glm.adjustment" => {
                let mut est = GlmAdjustmentAte::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = GlmAdjustmentWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "frontdoor.two_stage" => {
                let mut est = FrontDoorTwoStage::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = FrontDoorWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            "iv.wald" => {
                let mut est = WaldIv::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                est.fit(&prep, ctx, assumptions.clone()).map_err(est_err)?
            }
            "iv.2sls" => {
                let mut est = TwoStageLeastSquares::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
                let mut ws = TwoStageLeastSquaresWorkspace::default();
                est.fit(&prep, &mut ws, ctx, assumptions.clone()).map_err(est_err)?
            }
            _ => {
                return Err(AnalysisError::Unsupported { message: "unknown static estimator" });
            }
        };

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
            physical.logical.record.identifier.as_deref().unwrap_or("backdoor.adjustment");
        let identification = identify_static(identifier, graph, query)?;
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Identify("no estimand returned".into()))?;

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
        let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
        let mut ws = BayesianGCompWorkspace::default();
        let posterior = est.fit(&prep, identification.status, &mut ws, ctx).map_err(est_err)?;
        let estimate = effect_from_posterior(&posterior)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

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
            refutations: Vec::new(),
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
        .map_err(identify_err)?;
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Identify("rd.sharp returned no estimand".into()))?;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        let prep = est.prepare(data, &estimand, query).map_err(est_err)?;
        let mut ws = RdWorkspace::default();
        let estimate = est
            .fit(&prep, &mut ws, ctx, identification.required_assumptions.clone())
            .map_err(est_err)?;

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
            .map_err(|e| AnalysisError::Identify(e.to_string()))?;
        let identification = id_res.result;
        if identification.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(AnalysisError::Identify("temporal effect not identified".into()));
        }
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Identify("no estimand returned".into()))?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(data, &estimand, query, &id_res.indexer, self.split.as_ref())
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;

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
            config_digest: Some(Arc::from("phase3")),
        });
    }
    provenance
}

fn run_pcmci_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ctx: &ExecutionContext,
) -> Result<TemporalGraphReview, AnalysisError> {
    let schema = data.schema();
    let vars: Vec<VariableId> = schema.variables().iter().map(|v| v.id).collect();
    let pcmci = Pcmci::new().with_fdr(fdr).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: causal_core::Lag::from_raw(max_lag),
            min_lag: causal_core::Lag::from_raw(1),
        },
        alpha,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = pcmci
        .run(data, &vars, &mut ws, ctx)
        .map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
    Ok(result.review)
}

fn run_pcmci_plus_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let schema = data.schema();
    let vars: Vec<VariableId> = schema.variables().iter().map(|v| v.id).collect();
    let plus = PcmciPlus::new().with_fdr(fdr).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: causal_core::Lag::from_raw(max_lag),
            min_lag: causal_core::Lag::CONTEMPORANEOUS,
        },
        alpha,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = plus
        .run(data, &vars, &mut ws, ctx)
        .map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
    let outcomes = validation
        .run(&problem, workspace, ctx)
        .map_err(|e| AnalysisError::Validate(e.to_string()))?;
    Ok(ValidationSuite::reports_only(&outcomes))
}

// Owned-value signature keeps every `.map_err(est_err)` / `.map_err(identify_err)` call site
// terse (`fn(E) -> AnalysisError` matches `map_err`'s closure signature directly).
#[allow(clippy::needless_pass_by_value)]
fn est_err(e: EstimationError) -> AnalysisError {
    AnalysisError::Estimate(e.to_string())
}

fn effect_from_posterior(posterior: &CausalPosterior) -> Result<EffectEstimate, AnalysisError> {
    let eq = posterior.effect_column().ok_or_else(|| {
        AnalysisError::Estimate("Bayesian posterior missing effect column".into())
    })?;
    let ate = posterior.summaries.mean[eq];
    let se = posterior.summaries.sd[eq] / (posterior.draws.n_draws as f64).sqrt().max(1.0);
    Ok(EffectEstimate {
        ate,
        se_analytic: se,
        se_bootstrap: None,
        assumptions: posterior.assumptions.clone(),
        overlap: OverlapPolicy::ExplicitOverride,
        overlap_report: None,
        retained_memory_bytes: None,
    })
}

#[allow(clippy::needless_pass_by_value)]
fn identify_err(e: IdentificationError) -> AnalysisError {
    AnalysisError::Identify(e.to_string())
}

/// Run the identifier named by `identifier` against `graph`/`query`, returning its
/// [`IdentificationResult`] (Phase 4 static dispatch table; DESIGN.md §21.2).
fn identify_static(
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
        return Err(AnalysisError::Identify("effect not identified".into()));
    }
    Ok(result)
}

/// Provenance `(artifact_id, operation)` for an identifier id.
fn identify_provenance_step(identifier: &str) -> (&'static str, &'static str) {
    match identifier {
        "backdoor.adjustment" => ("identify.backdoor", "identify.backdoor"),
        "backdoor.efficient" => ("identify.efficient_backdoor", "identify.efficient_backdoor"),
        "frontdoor" => ("identify.frontdoor", "identify.frontdoor"),
        "iv" => ("identify.iv", "identify.iv"),
        _ => ("identify.unknown", "identify.unknown"),
    }
}

/// Provenance `(artifact_id, operation)` for an estimator id.
fn estimate_provenance_step(estimator: &str) -> (&'static str, &'static str) {
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
