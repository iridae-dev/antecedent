//! Unified `CausalAnalysis` facade (DESIGN.md §21 Phase 3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names, clippy::too_many_lines, clippy::doc_markdown)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord, LogicalAnalysisPlanRecord,
    PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode, TemporalEffectQuery, VERSION,
    VariableId,
};
use causal_data::{DiscoveryEstimationSplit, TabularData, TableView, TimeSeriesData};
use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Pcmci, TemporalConstraints};
use causal_estimate::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy,
    TemporalLinearAdjustment,
};
use causal_graph::{Dag, TemporalDag, TemporalGraphReview};
use causal_identify::{
    BackdoorIdentifier, IdentificationStatus, IdentifiedEstimand, TemporalBackdoorIdentifier,
};
use causal_validate::{PlaceboTreatment, RandomCommonCause, RefutationProblem, RefutationReport};

use crate::error::AnalysisError;
use crate::planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, compile_logical_static_ate, compile_logical_temporal_effect,
};
use crate::result::CausalAnalysisResult;
use crate::review::{PendingGraphReview, compile_review_required, ensure_review_complete};

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Placebo + random common cause.
    PlaceboAndRcc,
}

#[derive(Clone, Debug)]
enum DataInput {
    Tabular(TabularData),
    Temporal(TimeSeriesData),
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
        self.graph = Some(GraphInput::DiscoverPcmci {
            max_lag,
            alpha,
            fdr,
            accept_discovered: accept,
        });
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
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Static(graph)) => {
                compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
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
                GraphInput::DiscoverPcmci { .. },
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
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Static(graph)) => {
                let logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let logical =
                    compile_logical_temporal_effect(data, graph, q, self.split, false)?;
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
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination in Phase 3",
            }),
        }
    }

    fn validation_suite_id(&self) -> Option<Arc<str>> {
        match self.refute {
            RefuteSuite::None => None,
            RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
        }
    }

    fn ensure_supported_combination(&self) -> Result<(), AnalysisError> {
        match (&self.data, &self.query, &self.graph) {
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                Err(AnalysisError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                })
            }
            (DataInput::Temporal(_), CausalQuery::AverageEffect(_), _) => {
                Err(AnalysisError::Compile {
                    message: "static ATE on temporal data is not a Phase 3 path; use TemporalEffect"
                        .into(),
                })
            }
            (DataInput::Tabular(_), _, GraphInput::DiscoverPcmci { .. }) => {
                Err(AnalysisError::Compile {
                    message: "PCMCI discovery requires temporal data and a temporal effect query"
                        .into(),
                })
            }
            _ => Ok(()),
        }
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

    /// Continue after review: accept all pending edges then execute.
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

    fn execute_static(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let identifier = BackdoorIdentifier::new();
        let prepared =
            identifier.prepare(graph).map_err(|e| AnalysisError::Identify(e.to_string()))?;
        let identification = identifier
            .identify(&prepared, &CausalQuery::AverageEffect(query.clone()))
            .map_err(|e| AnalysisError::Identify(e.to_string()))?;

        if identification.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(AnalysisError::Identify("effect not identified".into()));
        }
        let estimand = identification
            .estimands
            .first()
            .cloned()
            .ok_or_else(|| AnalysisError::Identify("no estimand returned".into()))?;

        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = self.bootstrap_replicates;
        estimator.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(data, &estimand, query)
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "estimate.overlap.explicit_override",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "OLS path used ExplicitOverride for positivity",
        ));

        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc => {
                run_refuters(data, &estimand, query, &estimate, &mut workspace, ctx)?
            }
        };

        let provenance = provenance_pair(
            (
                "identify.backdoor",
                "identify.backdoor",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.linear_adjustment",
                "estimate.linear_adjustment_ate",
                &["identify.backdoor"],
                &estimate.assumptions,
            ),
        );

        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
            identification,
            estimand,
            estimate,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
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
            refutations: Vec::new(),
            diagnostics: Vec::new(),
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
        }))
    }
}

struct AssembleArgs<'a> {
    logical: &'a LogicalAnalysisPlanRecord,
    physical: &'a PhysicalExecutionPlanRecord,
    identification: causal_identify::IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    refutations: Vec<RefutationReport>,
    diagnostics: Vec<Diagnostic>,
    provenance: ProvenanceGraph,
    treatment: VariableId,
    outcome: VariableId,
}

fn assemble_result(args: AssembleArgs<'_>) -> CausalAnalysisResult {
    CausalAnalysisResult {
        logical_plan: args.logical.clone(),
        physical_plan: args.physical.clone(),
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        refutations: args.refutations,
        diagnostics: args.diagnostics,
        provenance: args.provenance,
        performance: ExecutionPerformanceRecord::default(),
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

fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, AnalysisError> {
    let problem = RefutationProblem { data, estimand, query, original: estimate };
    let placebo = PlaceboTreatment::new()
        .refute(&problem, workspace, ctx)
        .map_err(|e| AnalysisError::Validate(e.to_string()))?;
    let rcc = RandomCommonCause::new()
        .refute(&problem, workspace, ctx)
        .map_err(|e| AnalysisError::Validate(e.to_string()))?;
    Ok(vec![placebo, rcc])
}
