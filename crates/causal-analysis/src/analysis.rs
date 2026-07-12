//! Static ATE analysis facade (DESIGN.md §21 Phase 1 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

use std::sync::Arc;

use causal_core::{
    AverageEffectQuery, BufferMaterialization, CausalQuery, DataClassification, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord,
    KernelSelection, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph,
    ProvenanceNode, VERSION, VariableId,
};
use causal_data::TabularData;
use causal_estimate::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy,
};
use causal_graph::Dag;
use causal_identify::{
    BackdoorIdentifier, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
};
use causal_validate::{
    PlaceboTreatment, RandomCommonCause, RefutationProblem, RefutationReport,
};

use crate::error::AnalysisError;

/// Which Phase 1 refuters to run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Placebo + random common cause.
    PlaceboAndRcc,
}

/// Builder for a static ATE analysis.
#[derive(Clone, Debug)]
pub struct CausalAnalysisBuilder {
    data: Option<TabularData>,
    graph: Option<Dag>,
    query: Option<AverageEffectQuery>,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
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
        }
    }

    /// Supply tabular data.
    #[must_use]
    pub fn data(mut self, data: TabularData) -> Self {
        self.data = Some(data);
        self
    }

    /// Supply a validated DAG.
    #[must_use]
    pub fn graph(mut self, graph: Dag) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Supply an average-effect query.
    #[must_use]
    pub fn query(mut self, query: AverageEffectQuery) -> Self {
        self.query = Some(query);
        self
    }

    /// Configure refutation suite.
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
        })
    }
}

/// Prepared static ATE analysis.
#[derive(Clone, Debug)]
pub struct CausalAnalysis {
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
}

impl CausalAnalysis {
    /// Builder entry point.
    #[must_use]
    pub fn builder() -> CausalAnalysisBuilder {
        CausalAnalysisBuilder::new()
    }

    /// Run identify → estimate → optional refute.
    ///
    /// # Errors
    ///
    /// Identification, estimation, or validation failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<CausalAnalysisResult, AnalysisError> {
        let logical = LogicalAnalysisPlanRecord {
            plan_id: Arc::from("phase1.static_ate"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(Arc::from("backdoor.adjustment")),
            estimator: Some(Arc::from("linear.adjustment.ate")),
            validation_suite: match self.refute {
                RefuteSuite::None => None,
                RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
            },
            query_variables: Arc::from([self.query.treatment, self.query.outcome]),
        };
        let physical = PhysicalExecutionPlanRecord {
            plan_id: Arc::clone(&logical.plan_id),
            materializations: Arc::from([(
                Arc::from("design.matrix"),
                BufferMaterialization::CopiedContiguous,
            )]),
            kernels: Arc::from([(Arc::from("ols.faer"), KernelSelection::DenseBackend)]),
            batch_size: None,
            workspace_bytes: None,
            estimated_peak_memory_bytes: None,
            worker_threads: 0,
            deterministic_reductions: true,
            expected_python_crossings: 1,
        };

        let identifier = BackdoorIdentifier::new();
        let prepared = identifier
            .prepare(&self.graph)
            .map_err(|e| AnalysisError::Identify(e.to_string()))?;
        let identification = identifier
            .identify(&prepared, &CausalQuery::AverageEffect(self.query.clone()))
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
            .prepare(&self.data, &estimand, self.query.treatment, self.query.outcome)
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(
                &prep,
                &mut workspace,
                ctx,
                identification.required_assumptions.clone(),
            )
            .map_err(|e| AnalysisError::Estimate(e.to_string()))?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "estimate.overlap.explicit_override",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Phase 1 OLS path used ExplicitOverride for positivity",
        ));

        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc => {
                run_refuters(&self.data, &estimand, &self.query, &estimate, &mut workspace, ctx)?
            }
        };

        let mut provenance = ProvenanceGraph::new();
        provenance.push(ProvenanceNode {
            artifact_id: Arc::from("identify.backdoor"),
            operation: Arc::from("identify.backdoor"),
            parents: Arc::from([]),
            assumptions: identification.required_assumptions.clone(),
            library_version: Arc::from(VERSION),
            config_digest: Some(Arc::from("phase1")),
        });
        provenance.push(ProvenanceNode {
            artifact_id: Arc::from("estimate.linear_adjustment"),
            operation: Arc::from("estimate.linear_adjustment_ate"),
            parents: Arc::from([Arc::from("identify.backdoor")]),
            assumptions: estimate.assumptions.clone(),
            library_version: Arc::from(VERSION),
            config_digest: Some(Arc::from("phase1")),
        });

        Ok(CausalAnalysisResult {
            logical_plan: logical,
            physical_plan: physical,
            identification,
            estimand,
            estimate,
            refutations,
            diagnostics,
            provenance,
            performance: ExecutionPerformanceRecord::default(),
            treatment: self.query.treatment,
            outcome: self.query.outcome,
        })
    }
}

fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, AnalysisError> {
    let problem = RefutationProblem {
        data,
        estimand,
        treatment: query.treatment,
        outcome: query.outcome,
        original: estimate,
    };
    let placebo = PlaceboTreatment::new()
        .refute(&problem, workspace, ctx)
        .map_err(|e| AnalysisError::Validate(e.to_string()))?;
    let rcc = RandomCommonCause::new()
        .refute(&problem, workspace, ctx)
        .map_err(|e| AnalysisError::Validate(e.to_string()))?;
    Ok(vec![placebo, rcc])
}

/// End-to-end analysis result.
#[derive(Clone, Debug)]
pub struct CausalAnalysisResult {
    /// Logical plan record.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Full identification artifact.
    pub identification: IdentificationResult,
    /// Primary estimand used for estimation.
    pub estimand: IdentifiedEstimand,
    /// Point estimate + uncertainty.
    pub estimate: EffectEstimate,
    /// Refutation reports (may be empty).
    pub refutations: Vec<RefutationReport>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Provenance.
    pub provenance: ProvenanceGraph,
    /// Performance record.
    pub performance: ExecutionPerformanceRecord,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
}
