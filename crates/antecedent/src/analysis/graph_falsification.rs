//! Graph falsification as a compiled validation operation on a prepared study.
//!
//! A prepared study holds the DAG it was declared on and the table it retained.
//! [`PreparedStudy::falsify_graph`] tests that DAG's local Markov implications on that table and
//! compares the violation count with that of random relabellings of the graph's nodes, the
//! computation [`crate::gcm::evaluate_graph`] runs, through a lowered program of its own (program
//! family `GraphFalsification`, support unit `gcm.model.graph_falsification`): the run is gated
//! on the unit, executes inside the program's lowered dispatch and returns the kernel's report
//! with the program that produced it. Nothing is fitted.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_model::{GraphFalsificationConfig, GraphFalsificationReport};

use crate::analysis::builder::DataInput;
use crate::analysis::prepared::PreparedStudy;
use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;
use crate::planner::{PlanStage, PlanView};

/// The lowered program of a falsification, before it has run.
struct FalsificationProgram(Arc<LoweredProgram>);

impl CompiledProgramHandle for FalsificationProgram {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.0
    }
}

/// A graph falsification run through its compiled program.
#[derive(Clone, Debug)]
pub struct CompiledGraphFalsification {
    report: GraphFalsificationReport,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledGraphFalsification {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledGraphFalsification {
    /// The falsification report: the verdict, every implication's test evidence, the seeds,
    /// the test budget consumed and the stated limitations.
    #[must_use]
    pub const fn report(&self) -> &GraphFalsificationReport {
        &self.report
    }

    /// The report, taking it out of the run.
    #[must_use]
    pub fn into_report(self) -> GraphFalsificationReport {
        self.report
    }

    /// The program the falsification ran.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Every layer of the program, with the support decision of the unit its route runs under.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }
}

fn refuse(message: &str) -> CausalError {
    crate::unsupported_reason!("option_not_applicable", "graph falsification {message}")
}

impl PreparedStudy {
    /// Falsify the study's own DAG against the table it retained, by graph-node permutation.
    ///
    /// The graph's local Markov implications are tested with `config.ci_test` and the number
    /// rejected is compared with the number for random relabellings of the graph's nodes on the
    /// same rows. The result is `Rejected`, `NotRejected`, `Uninformative` or `Incomplete`; a
    /// budget or cancellation stop is an `Incomplete` report, not an error. The numbers are those
    /// of [`crate::gcm::evaluate_graph`] on the same graph, rows, configuration and context.
    ///
    /// # Errors
    ///
    /// A study that is not a supplied or accepted static DAG over a table (a graph posterior,
    /// a class, a temporal graph, a series), an unsupported unit in a release build, an invalid
    /// configuration, an unknown or non-frequentist CI test, or a CI-test failure.
    pub fn falsify_graph(
        &self,
        config: &GraphFalsificationConfig,
        ctx: &ExecutionContext,
    ) -> Result<CompiledGraphFalsification, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        let study = self.study();
        if study.graph_posterior.is_some() {
            return Err(refuse("tests one declared DAG; a graph posterior holds many"));
        }
        let DataInput::Tabular(data) = &study.data else {
            return Err(refuse("tests a DAG against a table; this study holds another modality"));
        };
        let Some(graph) = study.graph.as_dag() else {
            return Err(refuse(
                "tests a fully oriented static DAG; this study holds another class",
            ));
        };
        let lowered =
            Arc::new(crate::analysis::lowering::graph_falsification::lower(graph, data, config)?);
        gate_program(ctx, &FalsificationProgram(Arc::clone(&lowered)))?;
        let scoped = ctx.enter_lowered_dispatch(lowered.identity());
        let report = antecedent_model::falsify_graph(graph, data, config, &scoped)
            .map_err(CausalError::from)?;
        Ok(CompiledGraphFalsification { report, lowered })
    }
}
