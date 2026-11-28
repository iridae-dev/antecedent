//! Compiled measured whole-unit interval of a finite two-step temporal response.
//!
//! [`Study::compile_temporal_dependent_interval`] binds a [`TemporalDependentIntervalRequest`]
//! (the panel snapshot and its repeated units with their complete histories, the two-step
//! sequence, the estimand and the resampling design) into a lowered program: the sequence,
//! estimand and design are the program's question, the unit histories its source.
//! [`CompiledTemporalDependentInterval::execute`] runs what the deprecated
//! `temporal_dependent_interval` runs: the original whole-unit candidate is built (panel and
//! design validation, the studentized whole-unit resampling seeded from the design's seed) and
//! its artifact replayed by `TemporalIntervalArtifactWire::measure`, which reports the response
//! interval as a measured scalar with the calibration slot of its construction.
//!
//! The direct entry runs under the production context of the design's seed; the compiled route
//! runs under the caller's context. The resampling stream is seeded from the design's seed in
//! both, so the context changes cancellation and admission, never a number.
//!
//! The coverage of this interval is measured for the route `temporal_interval` (the coverage
//! records cited by the `temporal.dependent_interval` capability record);
//! [`MeasuredIntervalRun::calibration_route`] names it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, IntervalMethod, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{InitialStatePopulation, InitialStateSpec};
use antecedent_io::IoError;

use super::compiled_measured_interval::{
    MeasuredIntervalProgram, MeasuredIntervalRun, frequentist,
};
use super::program_handle::{CompiledProgramHandle, gate_program_when};
use super::supplied_identity::variant;
use super::temporal_extensions::{IntervalEstimand, measure_dependent_interval};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const PLAN_ID: &str = "temporal_dependent_interval.v1";
/// The measured route the coverage of this interval is recorded under.
const CALIBRATION_ROUTE: &str = "temporal_interval";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A whole-unit dependent interval request: the arguments `temporal_dependent_interval` takes,
/// owned.
#[derive(Clone, Debug)]
pub struct TemporalDependentIntervalRequest {
    snapshot_id: String,
    units: Option<Vec<UnitHistories>>,
    sequence: [u32; 2],
    estimand: IntervalEstimand,
    config: DependentIntervalConfig,
}

impl TemporalDependentIntervalRequest {
    /// Retain the panel snapshot, its units, the two-step sequence, the estimand and the
    /// resampling design. Nothing is validated here: the measured consumer validates the panel
    /// and design at execution, with the refusals of the direct entry.
    #[must_use]
    pub fn new(
        snapshot_id: &str,
        units: Option<Vec<UnitHistories>>,
        sequence: [u32; 2],
        estimand: IntervalEstimand,
        config: DependentIntervalConfig,
    ) -> Self {
        Self { snapshot_id: snapshot_id.to_owned(), units, sequence, estimand, config }
    }

    /// The panel snapshot id.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// The two-step sequence.
    #[must_use]
    pub const fn sequence(&self) -> [u32; 2] {
        self.sequence
    }

    /// The estimand.
    #[must_use]
    pub const fn estimand(&self) -> &IntervalEstimand {
        &self.estimand
    }

    /// The resampling design.
    #[must_use]
    pub const fn config(&self) -> &DependentIntervalConfig {
        &self.config
    }
}

/// A whole-unit dependent interval request bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledTemporalDependentInterval {
    request: TemporalDependentIntervalRequest,
    lowered: Arc<LoweredProgram>,
}

/// The estimand: its variant, then the fixed state or the law's population, snapshot and
/// `(state, mass)` support points in declared order (the marginalization sums them in that
/// order), masses by their bits.
fn write_estimand(writer: &mut CanonicalWriter, estimand: &IntervalEstimand) {
    match estimand {
        IntervalEstimand::Observed => {
            variant(writer, "observed");
        }
        IntervalEstimand::FixedState(state) => {
            variant(writer, "fixed_state").word(u64::from(*state));
        }
        IntervalEstimand::Marginalized(InitialStateSpec::Point(state)) => {
            variant(writer, "marginalized_point").word(u64::from(*state));
        }
        IntervalEstimand::Marginalized(InitialStateSpec::Law(law)) => {
            variant(writer, "marginalized_law")
                .text(match law.population() {
                    InitialStatePopulation::Source => "source",
                    InitialStatePopulation::Target => "target",
                })
                .text(law.snapshot_id())
                .word(law.states().len() as u64);
            for &(state, mass) in law.states() {
                writer.word(u64::from(state)).real(mass);
            }
        }
    }
}

fn question_digest(request: &TemporalDependentIntervalRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.temporal_dependent_interval.question.v2");
    let config = &request.config;
    writer.text(&request.snapshot_id).u32s(&request.sequence);
    write_estimand(&mut writer, &request.estimand);
    writer
        .word(config.replicates as u64)
        .word(config.seed)
        .real(config.level)
        .text(match config.method {
            IntervalMethod::Percentile => "percentile",
            IntervalMethod::Basic => "basic",
            IntervalMethod::Studentized => "studentized",
        })
        .word(config.min_units as u64)
        .real(config.max_failed_fraction);
    writer.finish()
}

fn source_digest(request: &TemporalDependentIntervalRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.temporal_dependent_interval.units.v1");
    let Some(units) = &request.units else {
        writer.word(0);
        return writer.finish();
    };
    writer.word(1).word(units.len() as u64);
    for unit in units {
        writer.word(unit.unit_id).word(unit.histories.len() as u64);
        for history in &unit.histories {
            writer
                .word(history.time_id)
                .word(u64::from(history.s0))
                .u32s(&history.actions)
                .u32s(&history.covariates)
                .real(history.y);
        }
    }
    writer.finish()
}

fn lower(request: &TemporalDependentIntervalRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let histories = request
        .units
        .as_ref()
        .map_or(0, |units| units.iter().map(|unit| unit.histories.len() as u64).sum());
    MeasuredIntervalProgram {
        cell: ProgramCell::TemporalDependentInterval,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        graph_class: "selected_two_step_binary_temporal_admg",
        question: question_digest(request),
        source: source_digest(request),
        source_role: EvidenceRole::Observations,
        realization: frequentist("whole_unit_studentized_bootstrap", PLAN_ID)?,
        uncertainty: UncertaintyReading::Sampling,
        rows: histories,
        columns: 6,
        claim: ClaimKind::Response,
    }
    .lower()
}

impl Study {
    /// Bind a whole-unit dependent interval request into a compiled program.
    ///
    /// # Errors
    ///
    /// A score or physical operator that cannot be declared.
    pub fn compile_temporal_dependent_interval(
        request: TemporalDependentIntervalRequest,
    ) -> Result<CompiledTemporalDependentInterval, CausalError> {
        let lowered = lower(&request)?;
        Ok(CompiledTemporalDependentInterval { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledTemporalDependentInterval {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTemporalDependentInterval {
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

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &TemporalDependentIntervalRequest {
        &self.request
    }

    /// Build the whole-unit candidate, replay it and report the measured interval.
    ///
    /// # Errors
    ///
    /// Every refusal of the direct entry (an invalid panel, design or estimand, an unsupported
    /// history, a retired method, absent unit dependence), cancellation, admission, or the
    /// support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MeasuredIntervalRun, IoError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        gate_program_when(GATE_ON_UNIT_SUPPORT, &scoped, self).map_err(|refusal| {
            IoError::Refused {
                code: refusal.code(),
                message: format!("{}: {refusal}", refusal.detail),
            }
        })?;
        let request = &self.request;
        let measured = measure_dependent_interval(
            &request.snapshot_id,
            request.units.clone(),
            request.sequence,
            request.estimand.clone(),
            &request.config,
            &scoped,
        )?;
        Ok(MeasuredIntervalRun {
            measured,
            lowered_program_identity: self.lowered.identity(),
            calibration_route: CALIBRATION_ROUTE,
        })
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_estimate::temporal_dependent_interval::DependentIntervalConfig;
    use antecedent_estimate::temporal_initial_state::{
        InitialStateLaw, InitialStatePopulation, InitialStateSpec,
    };

    use antecedent_estimate::temporal_dependent_interval::{SequenceHistory, UnitHistories};

    use super::{
        IntervalEstimand, TemporalDependentIntervalRequest, question_digest, source_digest,
    };
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};

    fn law(states: &[(u32, f64)]) -> Option<IntervalEstimand> {
        InitialStateLaw::new(InitialStatePopulation::Target, "law", states.to_vec())
            .ok()
            .map(|law| IntervalEstimand::Marginalized(InitialStateSpec::Law(law)))
    }

    fn panel(units: Option<Vec<UnitHistories>>) -> TemporalDependentIntervalRequest {
        TemporalDependentIntervalRequest::new(
            "panel",
            units,
            [0, 1],
            IntervalEstimand::Observed,
            DependentIntervalConfig::default(),
        )
    }

    fn unit(unit_id: u64, histories: Vec<SequenceHistory>) -> UnitHistories {
        UnitHistories { unit_id, histories }
    }

    #[test]
    fn site_compiled_temporal_interval_question_orders_the_sequence_and_the_law() {
        ordered_laws(&[0_u32, 1], |sequence| {
            Some(question_digest(&request(
                [sequence[0], sequence[1]],
                IntervalEstimand::Observed,
                0.1,
            )))
        });
        ordered_laws(&[(0_u32, 0.25), (1, 0.75)], |states| {
            law(states).map(|estimand| question_digest(&request([0, 1], estimand, 0.1)))
        });
        bit_exact(|fraction| {
            Some(question_digest(&request([0, 1], IntervalEstimand::Observed, fraction)))
        });
        all_distinct(&[
            question_digest(&request([0, 1], IntervalEstimand::Observed, 0.1)),
            question_digest(&request([0, 1], IntervalEstimand::FixedState(0), 0.1)),
            question_digest(&request(
                [0, 1],
                IntervalEstimand::Marginalized(InitialStateSpec::Point(0)),
                0.1,
            )),
            question_digest(&request([0, 1], law(&[(0, 1.0)]).expect("a law"), 0.1)),
        ]);
    }

    #[test]
    fn site_compiled_temporal_interval_source_orders_units_and_histories() {
        let first = unit(1, vec![SequenceHistory::two_step(0, 0, 1, 0, 1, 2.0)]);
        let second = unit(2, vec![SequenceHistory::two_step(0, 1, 0, 1, 0, -1.0)]);
        ordered_laws(&[first.clone(), second.clone()], |units| {
            Some(source_digest(&panel(Some(units.to_vec()))))
        });
        let histories = [
            SequenceHistory::two_step(0, 0, 1, 0, 1, 2.0),
            SequenceHistory::two_step(1, 1, 1, 0, 0, 3.0),
        ];
        ordered_laws(&histories, |ordered| {
            Some(source_digest(&panel(Some(vec![unit(1, ordered.to_vec())]))))
        });
        bit_exact(|y| {
            let history = SequenceHistory::two_step(0, 0, 1, 0, 1, y);
            Some(source_digest(&panel(Some(vec![unit(1, vec![history])]))))
        });
        all_distinct(&[
            source_digest(&panel(None)),
            source_digest(&panel(Some(Vec::new()))),
            source_digest(&panel(Some(vec![first]))),
            source_digest(&panel(Some(vec![second]))),
        ]);
    }

    fn request(
        sequence: [u32; 2],
        estimand: IntervalEstimand,
        max_failed_fraction: f64,
    ) -> TemporalDependentIntervalRequest {
        let config = DependentIntervalConfig { max_failed_fraction, ..Default::default() };
        TemporalDependentIntervalRequest::new("panel", None, sequence, estimand, config)
    }

    fn marginal(states: Vec<(u32, f64)>) -> IntervalEstimand {
        let law = InitialStateLaw::new(InitialStatePopulation::Target, "law", states).unwrap();
        IntervalEstimand::Marginalized(InitialStateSpec::Law(law))
    }

    #[test]
    fn the_same_request_has_the_same_identity() {
        let states = vec![(0, 0.25), (1, 0.75)];
        assert_eq!(
            question_digest(&request([0, 1], marginal(states.clone()), 0.1)),
            question_digest(&request([0, 1], marginal(states), 0.1))
        );
    }

    #[test]
    fn a_changed_field_changes_the_identity() {
        let base = question_digest(&request([0, 1], IntervalEstimand::FixedState(0), 0.1));
        assert_ne!(base, question_digest(&request([0, 1], IntervalEstimand::FixedState(1), 0.1)));
        assert_ne!(base, question_digest(&request([0, 1], IntervalEstimand::Observed, 0.1)));
        assert_ne!(
            base,
            question_digest(&request(
                [0, 1],
                IntervalEstimand::Marginalized(InitialStateSpec::Point(0)),
                0.1
            ))
        );
        assert_ne!(base, question_digest(&request([0, 1], IntervalEstimand::FixedState(0), 0.2)));
    }

    #[test]
    fn reordering_an_ordered_input_changes_the_identity() {
        assert_ne!(
            question_digest(&request([0, 1], IntervalEstimand::Observed, 0.1)),
            question_digest(&request([1, 0], IntervalEstimand::Observed, 0.1))
        );
        assert_ne!(
            question_digest(&request([0, 1], marginal(vec![(0, 0.5), (1, 0.5)]), 0.1)),
            question_digest(&request([0, 1], marginal(vec![(1, 0.5), (0, 0.5)]), 0.1))
        );
    }

    #[test]
    fn negative_zero_is_not_zero() {
        assert_ne!(
            question_digest(&request([0, 1], IntervalEstimand::Observed, 0.0)),
            question_digest(&request([0, 1], IntervalEstimand::Observed, -0.0))
        );
    }
}
