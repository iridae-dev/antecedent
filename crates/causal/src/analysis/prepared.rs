//! Compile-once / re-estimate-many prepared analysis handle.
//!
//! Rediscover policy: structure is frozen at prepare time. Changing bootstrap,
//! prior scale, treatment levels, or latency never re-runs discovery — only an
//! explicit new discover / review → prepare cycle may replace the graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use causal_core::{CausalQuery, CausalSchema, ExecutionContext};
use causal_data::{TableView, TabularData};
use causal_estimate::EstimationWorkspace;

use crate::error::AnalysisError;
use crate::planner::{CompiledAnalysis, GraphInput, PhysicalExecutionPlan};
use crate::result::CausalAnalysisResult;
use crate::strategy_table::DEFAULT_ESTIMATOR;

use super::builder::{DataInput, RefuteSuite};
use super::execute::CausalAnalysis;
use super::helpers::{project_for_ate_estimate, run_refuters};
use super::stage::{STAGE_VALIDATE, StageClock};

/// Durable handle: fixed schema, graph, query, and estimator; swap data and re-estimate.
///
/// Created via [`CausalAnalysis::prepare`]. Discovery / review-required graphs are refused —
/// prepare is for the interactive estimate click path on an already-accepted artifact.
#[derive(Clone, Debug)]
pub struct PreparedAnalysis {
    /// Frozen analysis config (data slot replaced on each estimate).
    analysis: CausalAnalysis,
    /// Ready physical plan from the prepare-time compile (never recompiled on refresh).
    plan: PhysicalExecutionPlan,
    /// Schema fingerprint from prepare-time tabular data.
    schema: CausalSchema,
}

impl PreparedAnalysis {
    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Re-estimate on `data` without recompiling the physical plan.
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        self.ensure_schema_compatible(data)?;
        let mut analysis = self.analysis.clone();
        analysis.data = DataInput::Tabular(data.clone());
        analysis.execute(&CompiledAnalysis::Ready(self.plan.clone()), ctx)
    }

    /// Replace retained data and re-estimate (same semantics as [`Self::estimate`]).
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn refresh(
        &mut self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        self.ensure_schema_compatible(&data)?;
        self.analysis.data = DataInput::Tabular(data);
        self.analysis.execute(&CompiledAnalysis::Ready(self.plan.clone()), ctx)
    }

    /// Second-click / background refute: replace validation on a prior estimate.
    ///
    /// Leaves ATE / identification / estimand unchanged. Records `validate` stage timing.
    /// Prefer `suite=PlaceboAndRcc` or `Full` after an interactive first click with
    /// Cheap / None.
    ///
    /// # Errors
    ///
    /// Schema mismatch, missing AverageEffect query, cancel, or validator failures.
    pub fn refute(
        &self,
        prior: &CausalAnalysisResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        self.ensure_schema_compatible(data)?;
        let CausalQuery::AverageEffect(query) = &self.analysis.query else {
            return Err(AnalysisError::Unsupported {
                message: "PreparedAnalysis::refute requires AverageEffect",
            });
        };
        if prior.treatment != query.treatment || prior.outcome != query.outcome {
            return Err(AnalysisError::Compile {
                message: "refute prior result treatment/outcome does not match prepared query"
                    .into(),
            });
        }
        let estimator = self
            .analysis
            .estimator
            .as_ref()
            .map_or(DEFAULT_ESTIMATOR, |e| e.as_str());

        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &prior.estimand)?;

        let mut clock = StageClock::new();
        clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        if ctx.cancellation.is_cancelled() {
            return Err(AnalysisError::Cancelled { stage: STAGE_VALIDATE });
        }
        let mut workspace = EstimationWorkspace::default();
        let started = Instant::now();
        let reports = run_refuters(
            &data_est,
            &estimand_est,
            &query_est,
            &prior.estimate,
            &mut workspace,
            None,
            ctx,
            suite,
            estimator,
            &self.analysis.custom_validators,
            None,
        )?;
        clock.finish(STAGE_VALIDATE);
        let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let mut out = prior.clone();
        out.refutations = reports;
        out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
        out.performance.wall_time_ns = Some(
            out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns),
        );
        let suite_label: Arc<str> = match suite {
            RefuteSuite::None => Arc::from("none"),
            RefuteSuite::Cheap => Arc::from("overlap+evalue"),
            RefuteSuite::PlaceboAndRcc => Arc::from("placebo+rcc"),
            RefuteSuite::Full => Arc::from("validation.full"),
        };
        out.diagnostics.push(causal_core::Diagnostic::new(
            "exec.refute.second_click",
            causal_core::DiagnosticKind::Execution,
            causal_core::DiagnosticSeverity::Info,
            format!("second-click refute suite={suite_label}"),
        ));
        let _ = clock.wall_time_ns();
        Ok(out)
    }

    fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), AnalysisError> {
        if data.schema() != &self.schema {
            return Err(AnalysisError::Compile {
                message: "prepared analysis refresh requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        Ok(())
    }
}

impl CausalAnalysis {
    /// Compile once into a durable [`PreparedAnalysis`] for re-estimate-many.
    ///
    /// Requires tabular data, an average-effect query, and a **supplied** static graph
    /// (`Dag` / `Cpdag` / `Pag` / `Admg`). Discovery inputs and review-required compiles
    /// are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    pub fn prepare(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreparedAnalysis, AnalysisError> {
        ensure_prepared_supported(self)?;
        let compiled = self.compile(ctx)?;
        let CompiledAnalysis::Ready(plan) = compiled else {
            return Err(AnalysisError::Compile {
                message: "prepare requires a Ready plan; complete graph review first \
                    (discovery / incomplete CPDAG/PAG are not session-refreshable)"
                    .into(),
            });
        };
        let schema = match &self.data {
            DataInput::Tabular(data) => data.schema().clone(),
            _ => {
                return Err(AnalysisError::Unsupported {
                    message: "PreparedAnalysis requires tabular data",
                });
            }
        };
        Ok(PreparedAnalysis { analysis: self.clone(), plan, schema })
    }
}

fn ensure_prepared_supported(analysis: &CausalAnalysis) -> Result<(), AnalysisError> {
    let DataInput::Tabular(_) = &analysis.data else {
        return Err(AnalysisError::Unsupported {
            message: "PreparedAnalysis requires tabular data and AverageEffect",
        });
    };
    if !matches!(analysis.query, CausalQuery::AverageEffect(_)) {
        return Err(AnalysisError::Unsupported {
            message: "PreparedAnalysis currently supports AverageEffect only",
        });
    }
    if !is_supplied_static_graph(&analysis.graph) {
        return Err(AnalysisError::Unsupported {
            message: "PreparedAnalysis requires a supplied static Dag/Cpdag/Pag/Admg \
                (discovery graphs stay on one-shot analyze / review)",
        });
    }
    Ok(())
}

fn is_supplied_static_graph(graph: &GraphInput) -> bool {
    matches!(
        graph,
        GraphInput::Static(_) | GraphInput::Cpdag(_) | GraphInput::Pag(_) | GraphInput::Admg(_)
    )
}

#[cfg(test)]
mod tests {
    use super::is_supplied_static_graph;
    use crate::planner::GraphInput;
    use causal_graph::{Admg, Cpdag, Dag, Pag};

    #[test]
    fn supplied_static_graphs_only() {
        assert!(is_supplied_static_graph(&GraphInput::Static(Dag::with_variables(1))));
        assert!(is_supplied_static_graph(&GraphInput::Cpdag(Cpdag::with_variables(1))));
        assert!(is_supplied_static_graph(&GraphInput::Pag(Pag::with_variables(1))));
        assert!(is_supplied_static_graph(&GraphInput::Admg(Admg::with_variables(1))));
        assert!(!is_supplied_static_graph(&GraphInput::DiscoverPc {
            alpha: 0.05,
            max_cond_size: 3,
            fdr: None,
            accept_discovered: true,
        }));
    }
}
