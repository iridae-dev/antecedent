//! Compiled posterior interval of a count-fitted observation probability.
//!
//! [`Study::compile_count_fitted_interval`] binds a retained [`CountFittedObservation`] (its
//! nodes with their counts or declared tables, the row count, the target and evidence, the
//! Dirichlet prior, the draw count and the seed) and a nominal level into a lowered program:
//! the question and construction are the program's question, the counts its source.
//! [`CompiledCountFittedInterval::execute`] runs the measured consumer the deprecated
//! `CountFittedObservation::interval` runs: the retained source at that level is encoded and
//! replayed by `CountFittedIntervalArtifactWire::measure`, which draws the independent
//! Dirichlet posteriors of the fitted tables, conditions on the evidence and reports the
//! equal-tailed posterior interval as a measured scalar with the calibration slot of its
//! construction. Calibration annotates the interval and never withholds it.
//!
//! The coverage of this interval is measured for the route `count_fitted_interval` (the
//! coverage records cited by the `observation_query.calibrated_probability` capability record);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::compiled_measured_interval::{MeasuredIntervalProgram, MeasuredIntervalRun, bayesian};
use super::count_fitted_observation::{
    CountFittedIntervalArtifactWire, CountFittedObservation, ROUTE, refusal_of,
};
use super::observation_query::ObservationQueryRefusal;
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const PLAN_ID: &str = "count_fitted_interval.v1";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A count-fitted observation and a nominal level bound to their lowered program.
#[derive(Clone, Debug)]
pub struct CompiledCountFittedInterval {
    observation: CountFittedObservation,
    level: f64,
    lowered: Arc<LoweredProgram>,
}

/// The question: the model's nodes in topological order (each node's levels in table column
/// order and its parents in table order, since parent indices and configurations are read by
/// position), the target, the evidence, the level and the posterior settings. The evidence is
/// a set: the posterior conditions on the conjunction of its `(variable, level)` observations,
/// so their order and an exact repeat do not change the question (two levels of one variable
/// have probability zero and are refused when the program runs).
fn question_digest(source: &CountFittedIntervalArtifactWire, level: f64) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.count_fitted_interval.question.v2");
    writer.word(u64::from(source.version)).word(source.nodes.len() as u64);
    for node in &source.nodes {
        writer.text(&node.name).texts(&node.levels).u32s(&node.parents);
        match &node.declared_bits {
            Some(bits) => writer.word(1).words(bits),
            None => writer.word(0),
        };
    }
    let mut evidence: Vec<(&str, &str)> = source
        .evidence
        .iter()
        .map(|(variable, value)| (variable.as_str(), value.as_str()))
        .collect();
    evidence.sort_unstable();
    evidence.dedup();
    writer.text(&source.target).text(&source.target_level).word(evidence.len() as u64);
    for (variable, value) in evidence {
        writer.text(variable).text(value);
    }
    writer
        .real(level)
        .word(source.prior_alpha_bits)
        .word(u64::from(source.draws))
        .word(source.seed);
    writer.finish()
}

fn source_digest(source: &CountFittedIntervalArtifactWire) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.count_fitted_interval.counts.v1");
    writer.word(source.rows).word(source.nodes.len() as u64);
    for node in &source.nodes {
        match &node.counts {
            Some(counts) => writer.word(1).words(counts),
            None => writer.word(0),
        };
    }
    writer.finish()
}

fn lower(
    observation: &CountFittedObservation,
    level: f64,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let source = observation.source();
    MeasuredIntervalProgram {
        cell: ProgramCell::CountFittedInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::ConditionalObservation,
        graph_class: "declared_finite_dag_with_count_fitted_tables",
        question: question_digest(source, level),
        source: source_digest(source),
        source_role: EvidenceRole::Observations,
        realization: bayesian(
            "categorical_multinomial_per_parent_configuration",
            format!("independent_dirichlet:{:016x}", source.prior_alpha_bits),
            PLAN_ID,
        )?,
        uncertainty: UncertaintyReading::Posterior,
        rows: source.rows,
        columns: source.nodes.len() as u64,
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind a count-fitted observation and a nominal level into a compiled program.
    ///
    /// # Errors
    ///
    /// A likelihood, prior or physical operator that cannot be declared.
    pub fn compile_count_fitted_interval(
        observation: CountFittedObservation,
        level: f64,
    ) -> Result<CompiledCountFittedInterval, CausalError> {
        let lowered = lower(&observation, level)?;
        Ok(CompiledCountFittedInterval { observation, level, lowered })
    }
}

impl CompiledProgramHandle for CompiledCountFittedInterval {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledCountFittedInterval {
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

    /// The bound observation.
    #[must_use]
    pub const fn observation(&self) -> &CountFittedObservation {
        &self.observation
    }

    /// The bound nominal level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.level
    }

    /// Replay the retained source at the bound level and report the measured interval.
    ///
    /// # Errors
    ///
    /// Every refusal of the measured consumer (a non-canonical source, a draw count out of
    /// range, endpoints whose Monte Carlo error is large against the width, evidence of
    /// probability zero), cancellation, or the support gate once it is on.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredIntervalRun, ObservationQueryRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| {
                refusal_of(&antecedent_io::IoError::Refused {
                    code: refusal.code(),
                    message: format!("{}: {refusal}", refusal.detail),
                })
            })?;
        }
        let measured = self.observation.measured_interval_at(self.level, &scoped)?;
        Ok(MeasuredIntervalRun {
            measured,
            lowered_program_identity: self.lowered.identity(),
            calibration_route: ROUTE,
        })
    }
}

#[cfg(test)]
mod site_tests {
    use super::{CountFittedIntervalArtifactWire, question_digest, source_digest};
    use crate::analysis::count_fitted_observation::CountFittedNodeWire;
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws, set_laws,
    };

    fn node(name: &str, levels: &[&str], parents: &[u32], counts: Vec<u64>) -> CountFittedNodeWire {
        CountFittedNodeWire {
            name: name.to_owned(),
            levels: levels.iter().map(|level| (*level).to_owned()).collect(),
            parents: parents.to_vec(),
            counts: Some(counts),
            declared_bits: None,
        }
    }

    fn wire(
        nodes: Vec<CountFittedNodeWire>,
        evidence: &[(&str, &str)],
    ) -> CountFittedIntervalArtifactWire {
        CountFittedIntervalArtifactWire {
            version: 1,
            nodes,
            rows: 10,
            target: "Y".to_owned(),
            target_level: "1".to_owned(),
            evidence: evidence
                .iter()
                .map(|(variable, value)| ((*variable).to_owned(), (*value).to_owned()))
                .collect(),
            level_bits: 0.95_f64.to_bits(),
            prior_alpha_bits: 1.0_f64.to_bits(),
            draws: 2000,
            seed: 7,
        }
    }

    fn model() -> Vec<CountFittedNodeWire> {
        vec![
            node("Z", &["0", "1"], &[], vec![4, 6]),
            node("X", &["0", "1"], &[0], vec![1, 3, 2, 4]),
            node("Y", &["0", "1"], &[0, 1], vec![1, 1, 2, 2, 3, 3, 4, 4]),
        ]
    }

    #[test]
    fn site_compiled_count_fitted_interval_question_reads_the_evidence_as_a_set() {
        set_laws(&[("Z", "1"), ("X", "0"), ("X", "1")], |evidence| {
            Some(question_digest(&wire(model(), evidence), 0.95))
        });
        bit_exact(|level| Some(question_digest(&wire(model(), &[]), level)));
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(variable, value)| Some(question_digest(&wire(model(), &[(variable, value)]), 0.95)),
        );
    }

    #[test]
    fn site_compiled_count_fitted_interval_question_orders_nodes_levels_and_parents() {
        ordered_laws(&model(), |nodes| Some(question_digest(&wire(nodes.to_vec(), &[]), 0.95)));
        ordered_laws(&["0", "1", "2"], |levels| {
            Some(question_digest(&wire(vec![node("Y", levels, &[], vec![1, 2, 3])], &[]), 0.95))
        });
        ordered_laws(&[0_u32, 1], |parents| {
            let mut nodes = model();
            nodes[2].parents = parents.to_vec();
            Some(question_digest(&wire(nodes, &[]), 0.95))
        });
    }

    #[test]
    fn site_compiled_count_fitted_interval_source_orders_the_counts() {
        ordered_laws(&[1_u64, 3, 2, 4], |counts| {
            let mut nodes = model();
            nodes[1].counts = Some(counts.to_vec());
            Some(source_digest(&wire(nodes, &[])))
        });
        let mut declared = model();
        declared[0].counts = None;
        declared[0].declared_bits = Some(vec![0.4_f64.to_bits(), 0.6_f64.to_bits()]);
        all_distinct(&[source_digest(&wire(model(), &[])), source_digest(&wire(declared, &[]))]);
    }
}
