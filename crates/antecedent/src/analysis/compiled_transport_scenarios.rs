//! Compiled finite transport scenario set.
//!
//! [`crate::StudyBuilder::transport_scenarios`] (and its conditional and empirical forms)
//! decides every scenario of a finite graph/selection scenario set once and compiles each
//! identified one against supplied or fitted laws. [`Study::compile_transport_scenarios`]
//! binds that decided set into a lowered program: the decisions, the shared question and the
//! retained plans are the program's question, the law identities its source.
//! [`CompiledTransportScenarios::execute`] evaluates the retained plans through
//! `antecedent_estimate::transport_scenarios::PreparedScenarioSet::evaluate`, the kernel the
//! prepared set runs, so the report (every scenario whatever its status, the structural
//! envelope over the identified ones and, for declared weights, a report that never
//! renormalizes) is the one the prepared set returns. Refreshing with new laws or samples
//! keeps the decisions and recompiles the program over the new source.
//!
//! The report is per-scenario points; no inferential statement across scenarios is made
//! ([`crate::PreparedTransportScenarios::aggregate_interval`] refuses).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_estimate::StatisticalTransportInput;
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::ExactTransportData;

use super::compiled_point_program::PointProgram;
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::transport_scenarios::PreparedTransportScenarios;
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "transport_scenario_set.v1";
const EVALUATOR: &str = "antecedent_estimate::transport_scenarios::PreparedScenarioSet::evaluate";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A decided scenario set bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledTransportScenarios {
    prepared: PreparedTransportScenarios,
    lowered: Arc<LoweredProgram>,
}

/// The report the retained plans evaluate to, bound to the program that evaluated them.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TransportScenariosRun {
    /// Every scenario, the structural envelope and the masses, as the kernel reported them.
    pub report: ScenarioSetReport,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `per_scenario_points`: no inference across scenarios.
    pub claim: &'static str,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

fn lower(prepared: &PreparedTransportScenarios) -> Result<Arc<LoweredProgram>, CausalError> {
    let set = prepared.prepared();
    let mut question = CanonicalWriter::new("antecedent.transport_scenario_set.question.v1");
    question.text(&format!("{:?}", set.decision())).text(&format!("{:?}", set.request()));
    question.text(&format!("{:?}", set.limits())).text(set.provider());
    for (scenario, plan) in set.plan_summary() {
        question.text(&scenario).text(&plan);
    }
    let question = question.finish();
    let data = set.data();
    let mut source = CanonicalWriter::new("antecedent.transport_scenario_set.laws.v1");
    source.word(data.law_identities().len() as u64);
    for identity in data.law_identities() {
        source.bytes(identity.as_bytes());
    }
    if let Some(input) = set.empirical_input() {
        source.text(&format!("{input:?}"));
    }
    let source = source.finish();
    let scenarios = set.plan_summary().len() as u64;
    PointProgram {
        cell: ProgramCell::TransportScenarioSet,
        plan_id: PLAN_ID,
        kind: FunctionalKind::IdentifiedSet,
        outcome: None,
        graph_class: "finite_selection_scenario_set",
        query_variables: Arc::from([]),
        question,
        source,
        source_role: EvidenceRole::JointScenarios,
        evaluator: EVALUATOR,
        supplied: Vec::new(),
        rows: data.laws().len() as u64,
        columns: scenarios,
        input_bytes: (data.laws().len() as u64).saturating_mul(32),
        claim: antecedent_contract::ClaimKind::Mixture,
    }
    .lower()
}

impl Study {
    /// Bind a decided scenario set into a compiled program.
    ///
    /// # Errors
    ///
    /// A physical operator that cannot be declared.
    pub fn compile_transport_scenarios(
        prepared: PreparedTransportScenarios,
    ) -> Result<CompiledTransportScenarios, CausalError> {
        let lowered = lower(&prepared)?;
        Ok(CompiledTransportScenarios { prepared, lowered })
    }
}

impl CompiledProgramHandle for CompiledTransportScenarios {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTransportScenarios {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The decided set the program binds.
    #[must_use]
    pub const fn prepared(&self) -> &PreparedTransportScenarios {
        &self.prepared
    }

    /// Evaluate every retained plan and report every scenario.
    ///
    /// # Errors
    ///
    /// A numerical failure, an invalid outcome, cancellation, or the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TransportScenariosRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let report = self.prepared.prepared().evaluate(&scoped)?;
        Ok(TransportScenariosRun {
            report,
            lowered_program_identity: self.lowered.identity(),
            claim: "per_scenario_points",
            calibration: "not_measured",
        })
    }

    /// Recompile the identified scenarios against new supplied laws; the decisions are kept and
    /// the program is lowered again over the new source.
    ///
    /// # Errors
    ///
    /// As [`PreparedTransportScenarios::refresh`].
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Study::compile_transport_scenarios(self.prepared.refresh(data, ctx)?)
    }

    /// Refit the empirical plug-in from new samples and recompile; the decisions are kept and
    /// the program is lowered again over the new source.
    ///
    /// # Errors
    ///
    /// As [`PreparedTransportScenarios::refresh_empirical`].
    pub fn refresh_empirical(
        &self,
        input: StatisticalTransportInput,
        max_joint_cells: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Study::compile_transport_scenarios(self.prepared.refresh_empirical(
            input,
            max_joint_cells,
            ctx,
        )?)
    }

    /// Export a report with every premise needed for independent replay
    /// (`consume_transport_scenarios_artifact`).
    ///
    /// # Errors
    ///
    /// As [`PreparedTransportScenarios::export`].
    pub fn export(&self, run: &TransportScenariosRun) -> Result<Vec<u8>, CausalError> {
        Ok(self.prepared.export(&run.report)?)
    }
}
