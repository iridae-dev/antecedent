//! Compile-once / re-estimate-many prepared analysis handle.
//!
//! Rediscover policy: structure is frozen at prepare time. Changing bootstrap,
//! prior scale, treatment levels, or latency never re-runs discovery — only an
//! explicit new discover / review → prepare cycle may replace the graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use antecedent_core::{CausalQuery, CausalSchema, ExecutionContext};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::EstimationWorkspace;

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use crate::strategy_table::DEFAULT_ESTIMATOR;

use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;

use super::builder::{DataInput, RefuteSuite};
use super::execute::Study;
use super::helpers::{project_for_ate_estimate, run_refuters};
use super::stage::{STAGE_VALIDATE, StageClock};

/// Prepare-time identification products for the static ATE path.
///
/// Everything identification reads — identifier, graph, query, RD config — is
/// frozen when the handle is built, and identification is deterministic, so an
/// estimate click reuses these instead of re-running identification. Results
/// carry an `exec.identify.cached` diagnostic so reuse is observable.
#[derive(Clone, Debug)]
pub struct CachedStaticIdentification {
    /// Identification result computed at prepare time.
    pub identification: IdentificationResult,
    /// Estimand selected for the prepared estimator.
    pub estimand: IdentifiedEstimand,
}

/// Durable handle: fixed schema, graph, query, and estimator; swap data and re-estimate.
///
/// Created via [`Study::prepare`]. Discovery / review-required graphs are refused —
/// prepare is for the interactive estimate click path on an already-accepted artifact.
#[derive(Clone, Debug)]
pub struct PreparedStudy {
    /// Frozen analysis config (data slot replaced on each estimate).
    analysis: Study,
    /// Ready physical plan from the prepare-time compile (never recompiled on refresh).
    plan: PhysicalExecutionPlan,
    /// Schema fingerprint from prepare-time tabular data.
    schema: CausalSchema,
}

impl PreparedStudy {
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
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        let mut analysis = self.analysis.clone();
        analysis.data = DataInput::Tabular(data.clone());
        analysis.execute(&self.plan, ctx)
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
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(&data)?;
        self.analysis.data = DataInput::Tabular(data);
        self.analysis.execute(&self.plan, ctx)
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
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        let CausalQuery::AverageEffect(query) = &self.analysis.query else {
            return Err(CausalError::Unsupported {
                message: "PreparedStudy::refute requires AverageEffect",
            });
        };
        if prior.treatment != query.treatment || prior.outcome != query.outcome {
            return Err(CausalError::Compile {
                message: "refute prior result treatment/outcome does not match prepared query"
                    .into(),
            });
        }
        let estimator = self.analysis.estimator.as_ref().map_or(DEFAULT_ESTIMATOR, |e| e.as_str());

        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &prior.estimand)?;

        let mut clock = StageClock::new();
        clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_VALIDATE });
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
        out.performance.wall_time_ns =
            Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
        let suite_label: Arc<str> = match suite {
            RefuteSuite::None => Arc::from("none"),
            RefuteSuite::Cheap => Arc::from("overlap+evalue"),
            RefuteSuite::PlaceboAndRcc => Arc::from("placebo+rcc"),
            RefuteSuite::Full => Arc::from("validation.full"),
        };
        out.diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.refute.second_click",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            format!("second-click refute suite={suite_label}"),
        ));
        let _ = clock.wall_time_ns();
        Ok(out)
    }

    fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), CausalError> {
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared analysis refresh requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        Ok(())
    }
}

impl Study {
    /// Compile once into a durable [`PreparedStudy`] for re-estimate-many.
    ///
    /// Requires tabular data, an average-effect query, and a **supplied** static graph
    /// (`Dag` / `Cpdag` / `Pag` / `Admg`). Discovery inputs and review-required compiles
    /// are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        ensure_prepared_supported(self)?;
        let plan = self.compile(ctx)?;
        let schema = match &self.data {
            DataInput::Tabular(data) => data.schema().clone(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy requires tabular data",
                });
            }
        };
        let mut analysis = self.clone();
        analysis.identification_cache = self.prepare_static_identification(&plan)?.map(Arc::new);
        Ok(PreparedStudy { analysis, plan, schema })
    }

    /// Compute the static-path identification once at prepare time.
    ///
    /// Mirrors `execute_static`'s stage-1 inputs exactly. Configurations that
    /// dispatch elsewhere (sharp RD, Bayesian g-comp, bidirected ADMGs, PAG
    /// envelopes) return `None` and keep their identify-per-run behavior.
    fn prepare_static_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{
            DEFAULT_IDENTIFIER, EstimatorId, IdentifierId, identify_static_query_with_rd,
            select_estimand,
        };
        let identifier = plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if matches!(estimator_id, EstimatorId::RdSharp | EstimatorId::BayesianGcomp) {
            return Ok(None);
        }
        let CausalQuery::AverageEffect(query) = &self.query else {
            return Ok(None);
        };
        // A posterior-backed study holds only a placeholder `graph` shape;
        // identification against it would be meaningless.
        if self.graph_posterior.is_some() {
            return Ok(None);
        }
        // Resolve the same static DAG `execute` would hand to `execute_static`.
        let graph = match self.graph.class() {
            GraphClass::Dag => self.graph.as_dag().cloned(),
            GraphClass::Cpdag | GraphClass::Admg => plan.static_graph().cloned(),
            _ => None,
        };
        let Some(graph) = graph else {
            return Ok(None);
        };
        if self.graph.class() == GraphClass::Admg
            && self.graph.as_admg().is_some_and(super::execute::admg_has_bidirected)
        {
            return Ok(None);
        }
        let rd = self.rd.map(|c| {
            antecedent_identify::SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth)
        });
        let identification = identify_static_query_with_rd(
            identifier_id,
            &graph,
            &CausalQuery::AverageEffect(query.clone()),
            rd,
        )?;
        let estimand = select_estimand(&identification, estimator_id)?;
        Ok(Some(CachedStaticIdentification { identification, estimand }))
    }
}

fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    let DataInput::Tabular(_) = &analysis.data else {
        return Err(CausalError::Unsupported {
            message: "PreparedStudy requires tabular data and AverageEffect",
        });
    };
    if !matches!(analysis.query, CausalQuery::AverageEffect(_)) {
        return Err(CausalError::Unsupported {
            message: "PreparedStudy currently supports AverageEffect only",
        });
    }
    if !is_supplied_static_graph(analysis.graph.class()) {
        return Err(CausalError::Unsupported {
            message: "PreparedStudy requires a static Dag/Cpdag/Pag/Admg structure \
                (temporal classes are not session-refreshable here)",
        });
    }
    Ok(())
}

fn is_supplied_static_graph(class: GraphClass) -> bool {
    matches!(class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag | GraphClass::Admg)
}

#[cfg(test)]
mod tests {
    use super::is_supplied_static_graph;
    use crate::accepted::GraphClass;

    #[test]
    fn supplied_static_graphs_only() {
        assert!(is_supplied_static_graph(GraphClass::Dag));
        assert!(is_supplied_static_graph(GraphClass::Cpdag));
        assert!(is_supplied_static_graph(GraphClass::Pag));
        assert!(is_supplied_static_graph(GraphClass::Admg));
        assert!(!is_supplied_static_graph(GraphClass::TemporalDag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalCpdag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalPag));
    }
}
