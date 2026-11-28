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
//! [`CompiledTransportScenarios::estimate`] is that evaluation with the prepared set's own
//! refusals ([`PreparedTransportScenarios::estimate`] under the program's scope), so a prepared
//! stage that estimates through the program keeps its callers' error classes.
//!
//! The report is per-scenario points; no inferential statement across scenarios is made
//! ([`crate::PreparedTransportScenarios::aggregate_interval`] refuses).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CanonicalEncode, CanonicalWriter, ExecutionContext, SemanticDigest, VariableId,
};
use antecedent_estimate::StatisticalTransportInput;
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::sid::scenarios::{
    ScenarioDecisionLimits, ScenarioQuestion, TransportScenarioSet,
};
use antecedent_io::IoError;

use super::compiled_point_program::PointProgram;
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, SetOf, encoded};
use super::transport_scenarios::PreparedTransportScenarios;
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};

const PLAN_ID: &str = "transport_scenario_set.v1";
const EVALUATOR: &str = "antecedent_estimate::transport_scenarios::PreparedScenarioSet::evaluate";
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
    /// Calibration state of the route's evidence unit.
    pub calibration: &'static str,
}

/// Everything the scenario set's question declares, in canonical form: every scenario (the
/// set keeps them in name order) with its graph's canonical identity, its selection targets
/// and its weight; the shared coordinate schema; the question; the decision limits; the target
/// request (sorted by variable); the evaluation limits; the provider; and every retained plan.
/// The per-scenario derivations follow from these and are not written.
#[allow(
    clippy::too_many_arguments,
    reason = "one question: the set, the question, both limits, the request, the provider and \
              the retained plans"
)]
fn question_digest(
    set: &TransportScenarioSet,
    question: &ScenarioQuestion,
    decision_limits: &ScenarioDecisionLimits,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    provider: &str,
    plans: &[(Arc<str>, String)],
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.transport_scenario_set.question.v2");
    writer.word(set.scenarios().len() as u64);
    for scenario in set.scenarios() {
        writer.text(&scenario.name);
        Canonical::encode(scenario.diagram.causal_graph(), &mut writer);
        // A set: the identification reads selection targets as a set of nodes, so their order
        // and repetitions do not enter the identity.
        Canonical::encode(&SetOf(scenario.diagram.selection_targets()), &mut writer);
        writer.opt_real(scenario.weight);
    }
    writer.word(set.schema().len() as u64);
    for coordinate in set.schema() {
        CanonicalEncode::encode(&coordinate.variable, &mut writer);
        writer.text(&coordinate.name);
        CanonicalEncode::encode(&coordinate.domain, &mut writer);
        writer.opt_text(coordinate.unit.as_deref());
    }
    // Ordered: outcomes, treatments and conditioned coordinates are kept in declared order.
    let ids = |writer: &mut CanonicalWriter, list: &[VariableId]| {
        writer.u32s(&list.iter().map(|id| id.raw()).collect::<Vec<_>>());
    };
    let base = question.base();
    match question {
        ScenarioQuestion::Classical(_) => {
            writer.text("classical");
        }
        ScenarioQuestion::Conditional(_) => {
            writer.text("conditional");
        }
    }
    ids(&mut writer, &base.outcomes[..]);
    ids(&mut writer, &base.treatments[..]);
    writer.text(&base.source).text(&base.target);
    ids(&mut writer, question.conditioned_on());
    writer
        .word(decision_limits.budget.operations as u64)
        .word(decision_limits.budget.depth as u64)
        .opt_word(decision_limits.memory_limit_bytes);
    write_assignment(&mut writer, request);
    writer.word(limits.operations as u64).word(limits.depth as u64).text(provider);
    writer.word(plans.len() as u64);
    for (scenario, plan) in plans {
        writer.text(scenario).text(plan);
    }
    writer.finish()
}

/// An assignment's entries in the order it keeps them (sorted by variable), each value by its
/// canonical encoding.
fn write_assignment(writer: &mut CanonicalWriter, assignment: &Assignment) {
    writer.word(assignment.entries().len() as u64);
    for (variable, value) in assignment.entries() {
        CanonicalEncode::encode(variable, writer);
        CanonicalEncode::encode(value, writer);
    }
}

/// The laws the plans evaluate, by their content identities in retained order, and the samples
/// an empirical plug-in fitted them from: each sample's population, regime, snapshot,
/// intervention world (a set of assignments, sorted by variable) and every column (sorted by
/// variable) cell by cell.
fn source_digest(
    data: &ExactTransportData,
    empirical: Option<&StatisticalTransportInput>,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.transport_scenario_set.laws.v3");
    writer.word(data.law_identities().len() as u64);
    for identity in data.law_identities() {
        writer.bytes(identity.as_bytes());
    }
    let Some(input) = empirical else {
        writer.word(0);
        return writer.finish();
    };
    writer.word(1).word(input.supplied.len() as u64);
    for law in &input.supplied {
        writer.bytes(law.content_identity().as_bytes());
    }
    writer.word(input.samples.len() as u64);
    for sample in &input.samples {
        writer
            .text(&sample.population)
            .word(u64::from(sample.regime.raw()))
            .text(&sample.snapshot_identity)
            .word(sample.interventions.len() as u64);
        // A set: the fit compares the world to its regime's interventions by variable, so the
        // order of the assignments does not enter (a repeated variable disagrees with the regime
        // and is refused when the program runs).
        let mut interventions: Vec<_> = sample.interventions.iter().collect();
        interventions.sort_by_key(|intervention| intervention.variable);
        for intervention in interventions {
            CanonicalEncode::encode(&intervention.variable, &mut writer);
            CanonicalEncode::encode(&intervention.value, &mut writer);
        }
        writer.word(sample.columns.len() as u64);
        for (variable, column) in &sample.columns {
            CanonicalEncode::encode(variable, &mut writer);
            writer.word(column.len() as u64);
            for cell in column {
                writer.opt_real(*cell);
            }
        }
    }
    writer.finish()
}

fn lower(prepared: &PreparedTransportScenarios) -> Result<Arc<LoweredProgram>, CausalError> {
    let set = prepared.prepared();
    let decision = set.decision();
    // A graph without a canonical identity refuses the compile (`encoded`).
    let question = encoded(PLAN_ID, || {
        question_digest(
            &decision.set,
            &decision.question,
            &decision.limits,
            set.request(),
            set.limits(),
            set.provider(),
            &set.plan_summary(),
        )
    })?;
    let data = set.data();
    let source = source_digest(data, set.empirical_input());
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
        CompiledProgramHandle::plan_view(self)
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
    /// A numerical failure, an invalid outcome, cancellation, or a support refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TransportScenariosRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self)?;
        let report = self.prepared.prepared().evaluate(&scoped)?;
        publication.finish(Ok(TransportScenariosRun {
            report,
            lowered_program_identity: self.lowered.identity(),
            claim: "per_scenario_points",
            calibration: self.unit_calibration(),
        }))
    }

    /// [`Self::execute`] with the refusals of [`PreparedTransportScenarios::estimate`]: the
    /// prepared set's estimate under the program's scope, bound to the program.
    ///
    /// # Errors
    ///
    /// As [`PreparedTransportScenarios::estimate`], and a support refusal.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<TransportScenariosRun, IoError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self).map_err(|refusal| {
            IoError::from(antecedent_estimate::EstimationError::from(refusal))
        })?;
        let report = self.prepared.estimate(&scoped)?;
        publication.finish(Ok(TransportScenariosRun {
            report,
            lowered_program_identity: self.lowered.identity(),
            claim: "per_scenario_points",
            calibration: self.unit_calibration(),
        }))
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

#[cfg(test)]
mod identity_tests {
    use antecedent_core::{RegimeId, SearchLimits, Value, VariableDomain};
    use antecedent_estimate::RegimeSample;
    use antecedent_expr::InterventionAssignment;
    use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
    use antecedent_identify::ClassicalTransportQuery;
    use antecedent_identify::sid::scenarios::{ScenarioCoordinate, TransportScenario};

    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn scenario(name: &str, selections: &[u32]) -> TransportScenario {
        try_scenario(name, selections).expect("a valid scenario")
    }

    fn try_scenario(name: &str, selections: &[u32]) -> Option<TransportScenario> {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
        Some(TransportScenario {
            name: Arc::from(name),
            diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections))
                .ok()?,
            weight: None,
            coordinates: [(0, "z"), (1, "x"), (2, "y")]
                .into_iter()
                .map(|(id, name)| ScenarioCoordinate {
                    variable: v(id),
                    name: Arc::from(name),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect(),
        })
    }

    fn digest(scenarios: Vec<TransportScenario>, outcomes: &[u32], level: f64) -> [u8; 32] {
        let request = Assignment::from_pairs([(v(1), Value::f64(level))]);
        try_digest(scenarios, outcomes, &request).expect("a valid scenario set")
    }

    fn try_digest(
        scenarios: Vec<TransportScenario>,
        outcomes: &[u32],
        request: &Assignment,
    ) -> Option<[u8; 32]> {
        let set = TransportScenarioSet::try_new(scenarios).ok()?;
        let question = ScenarioQuestion::Classical(ClassicalTransportQuery {
            outcomes: outcomes.iter().copied().map(v).collect(),
            treatments: Arc::from([v(1)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        });
        let limits = ScenarioDecisionLimits {
            budget: SearchLimits { operations: 1000, depth: 16 },
            memory_limit_bytes: None,
        };
        Some(question_digest(
            &set,
            &question,
            &limits,
            request,
            ExactEvaluationLimits::default(),
            "provider",
            &[(Arc::from("direct"), "compiled".to_owned())],
        ))
    }

    fn request(pairs: &[(u32, f64)]) -> Assignment {
        Assignment::from_pairs(pairs.iter().map(|&(id, value)| (v(id), Value::f64(value))))
    }

    fn scenarios(specs: &[(&str, &[u32])]) -> Option<Vec<TransportScenario>> {
        specs.iter().map(|&(name, selections)| try_scenario(name, selections)).collect()
    }

    #[test]
    fn site_compiled_transport_scenarios_question_reads_scenarios_and_selections_as_sets() {
        let specs: [(&str, &[u32]); 3] = [("direct", &[]), ("shift", &[0, 2]), ("narrow", &[0])];
        set_laws(&specs, |chosen| try_digest(scenarios(chosen)?, &[2, 0], &Assignment::default()));
        set_laws(&[0_u32, 2], |selections| {
            let shifted = try_scenario("shift", selections)?;
            try_digest(vec![scenario("direct", &[]), shifted], &[2, 0], &Assignment::default())
        });
        set_laws(&[(1_u32, 1.0), (0, 2.0)], |pairs| try_digest(pair(), &[2, 0], &request(pairs)));
        bit_exact(|value| try_digest(pair(), &[2, 0], &request(&[(1, value)])));
    }

    #[test]
    fn site_compiled_transport_scenarios_question_orders_the_outcomes() {
        ordered_laws(&[2_u32, 0, 1], |outcomes| {
            try_digest(pair(), outcomes, &Assignment::default())
        });
    }

    fn sample(population: &str, interventions: &[(u32, f64)]) -> RegimeSample {
        RegimeSample {
            population: Arc::from(population),
            regime: RegimeId::from_raw(1),
            snapshot_identity: Arc::from("snapshot"),
            interventions: interventions
                .iter()
                .map(|&(id, value)| InterventionAssignment {
                    variable: v(id),
                    value: Value::f64(value),
                })
                .collect(),
            columns: [(v(2), vec![Some(1.0), None])].into_iter().collect(),
        }
    }

    fn sampled(samples: Vec<RegimeSample>) -> [u8; 32] {
        let empty = ExactTransportData::try_new(Vec::new(), 1).unwrap();
        let input = StatisticalTransportInput { supplied: Vec::new(), samples };
        source_digest(&empty, Some(&input))
    }

    #[test]
    fn site_compiled_transport_scenarios_source_orders_samples_and_sets_their_worlds() {
        ordered_laws(&["north", "south"], |populations| {
            Some(sampled(populations.iter().map(|population| sample(population, &[])).collect()))
        });
        // A repeated variable disagrees with the regime: the fit refuses it when the program runs.
        set_laws(&[(0_u32, 1.0), (1, 0.0)], |world| {
            let mut variables: Vec<u32> = world.iter().map(|&(id, _)| id).collect();
            variables.sort_unstable();
            variables.dedup();
            (variables.len() == world.len()).then(|| sampled(vec![sample("north", world)]))
        });
        let empty = ExactTransportData::try_new(Vec::new(), 1).unwrap();
        all_distinct(&[
            source_digest(&empty, None),
            sampled(Vec::new()),
            sampled(vec![sample("north", &[])]),
            sampled(vec![sample("north", &[(0, 1.0)])]),
            sampled(vec![sample("north", &[(0, 2.0)])]),
        ]);
    }

    fn pair() -> Vec<TransportScenario> {
        vec![scenario("direct", &[]), scenario("shift", &[0, 2])]
    }

    #[test]
    fn the_scenario_question_identity_is_canonical() {
        let base = digest(pair(), &[2, 0], 1.0);
        assert_eq!(base, digest(pair(), &[2, 0], 1.0), "repeatable");
        assert_ne!(base, digest(pair(), &[2, 0], 2.0), "a changed request");
        assert_ne!(base, digest(pair(), &[0, 2], 1.0), "outcomes are ordered");
        assert_ne!(digest(pair(), &[2, 0], 0.0), digest(pair(), &[2, 0], -0.0), "reals by bits");
        let reordered = vec![scenario("shift", &[2, 0]), scenario("direct", &[])];
        assert_eq!(base, digest(reordered, &[2, 0], 1.0), "selections and scenarios are sets");
    }

    #[test]
    fn the_scenario_source_identity_binds_the_samples() {
        let empty = ExactTransportData::try_new(Vec::new(), 1).unwrap();
        assert_eq!(source_digest(&empty, None), source_digest(&empty, None));
        let input = StatisticalTransportInput::default();
        assert_ne!(source_digest(&empty, None), source_digest(&empty, Some(&input)));
    }
}
