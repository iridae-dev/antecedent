//! The participant-level adaptive family at the facade: declare the procedure
//! (arms, binary responses per participant, historical priors, allocation,
//! decisions, confidence sequences), simulate the whole of it, read a retained
//! history look by look, and build the retained analysis contract it is
//! sealed with.
//!
//! The numerics, the premise gates and every refusal belong to
//! `antecedent_stats::adaptive_inference::participant_procedure`; this module
//! routes a request to them and adds no rule of its own. Every result is
//! labelled `derived_unmeasured`, and the posterior claim and the
//! anytime-valid frequentist claim of every look are separate fields.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism, InterimAnalysis,
    MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_stats::adaptive_inference::{
    ArmBorrowing, BorrowingPlan, CellLabel, DERIVED_UNMEASURED, LookDecision, LookReading,
    ParticipantDeclaration, ParticipantHistory, ParticipantRun, ParticipantSimulation,
    ParticipantSimulationRequest, ParticipantSimulationSummary, ParticipantSpec, ResolvedArmPrior,
    ResourceBudget, run_participant_procedure_within, simulate_participant_history,
    simulate_participant_procedure_within,
};

use super::adaptive_inference::AdaptiveInferenceRefusal;
use super::adaptive_procedure_declaration::gap_refusal;
use super::adaptive_procedure_report::refusal_of;

const POLICY_STATE_DOMAIN: &str = "antecedent.adaptive_procedure.participant_policy_state";

/// A declared participant-level adaptive procedure.
#[derive(Clone, Debug)]
pub struct ParticipantProcedure {
    spec: ParticipantSpec,
    identity: String,
}

/// A simulation of the whole procedure with the identity of what ran.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticipantSimulationReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The replicates and the receipt of the work done.
    pub simulation: ParticipantSimulation,
    /// Measured properties of this simulation under the truth: not a
    /// coverage statement.
    pub summary: ParticipantSimulationSummary,
}

/// One run of the procedure with the identity of what ran, complete or ended
/// early by cancellation.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticipantRunReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The run: its retained history, the reading of every completed look,
    /// its receipt and whether it reached its stopping look.
    pub run: ParticipantRun,
}

/// Every look of a retained history, recomputed under the declared procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticipantHistoryReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The prior of each arm with its sources, discounts and effective
    /// sample size.
    pub priors: Vec<ResolvedArmPrior>,
    /// Per arm, every source with its discount, the effective historical
    /// sample size and the prior-data conflict diagnostic against the
    /// participants whose responses had completed.
    pub borrowing: Vec<ArmBorrowing>,
    /// The reading at every performed look. The posterior claim and the
    /// frequentist claim of each are separate fields.
    pub readings: Vec<LookReading>,
    /// What the last look decided.
    pub decision: LookDecision,
}

impl ParticipantProcedure {
    /// Declare a procedure.
    ///
    /// # Errors
    /// One refusal per missing element (`adaptive_contract_incomplete`, its
    /// own `adaptive_inference.*` detail) and per invalid element
    /// (`invalid_argument`); nothing is defaulted.
    pub fn declare(
        declaration: ParticipantDeclaration,
    ) -> Result<Self, Vec<AdaptiveInferenceRefusal>> {
        ParticipantSpec::new(declaration)
            .map(|spec| {
                let identity = spec.identity();
                Self { spec, identity }
            })
            .map_err(|error| error.gaps.into_iter().map(gap_refusal).collect())
    }

    /// The validated procedure.
    #[must_use]
    pub const fn spec(&self) -> &ParticipantSpec {
        &self.spec
    }

    /// Canonical identity (hex) of the declared procedure, borrowing
    /// included.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The matrix cell of the procedure.
    #[must_use]
    pub fn cell(&self) -> CellLabel {
        self.spec.cell()
    }

    /// Re-declare the procedure with another borrowing plan while borrowing
    /// is still open.
    ///
    /// # Errors
    /// `adaptive_assumption_violated` with `adaptive_inference.borrowing_frozen`
    /// when `looks_performed` is above zero and the plan differs: borrowing
    /// is part of the declaration identity and frozen before recruitment.
    pub fn amend_borrowing(
        &self,
        proposed: &BorrowingPlan,
        looks_performed: usize,
    ) -> Result<Self, AdaptiveInferenceRefusal> {
        let spec =
            self.spec.amend_borrowing(proposed, looks_performed).map_err(|e| refusal_of(&e))?;
        let identity = spec.identity();
        Ok(Self { spec, identity })
    }

    /// Run the whole procedure `request.replications` times under the arm
    /// means `truth`.
    ///
    /// The request is admitted against the library work bound and the memory
    /// budget of `ctx` before any replicate runs, and the result carries a
    /// receipt of the work done.
    ///
    /// # Errors
    /// A truth of the wrong shape, a capacity refusal (replications, work or
    /// memory), a numerical failure or cancellation.
    pub fn simulate(
        &self,
        truth: &[f64],
        request: &ParticipantSimulationRequest,
        ctx: &ExecutionContext,
    ) -> Result<ParticipantSimulationReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let budget = ResourceBudget::from_context(ctx);
        let simulation =
            simulate_participant_procedure_within(&self.spec, truth, request, &budget, &cancelled)
                .map_err(|e| refusal_of(&e))?;
        let summary = simulation.summarize(self.spec.arms()).map_err(|e| refusal_of(&e))?;
        Ok(ParticipantSimulationReport {
            procedure_identity: self.identity.clone(),
            calibration: DERIVED_UNMEASURED,
            simulation,
            summary,
        })
    }

    /// Run the procedure once under the arm means `truth` and retain it, within
    /// the library work bound and the memory budget of `ctx`.
    ///
    /// Cancellation of `ctx` is honoured within one batch and is not a
    /// refusal: the report holds the batches that completed, its receipt
    /// counts what was done, and `run.complete` is `false`.
    ///
    /// # Errors
    /// A truth of the wrong shape, a capacity refusal (work or memory) or a
    /// numerical failure.
    pub fn run(
        &self,
        truth: &[f64],
        true_correlation: Option<f64>,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<ParticipantRunReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let budget = ResourceBudget::from_context(ctx);
        let run = run_participant_procedure_within(
            &self.spec,
            truth,
            true_correlation,
            seed,
            &budget,
            &cancelled,
        )
        .map_err(|e| refusal_of(&e))?;
        Ok(ParticipantRunReport {
            procedure_identity: self.identity.clone(),
            calibration: DERIVED_UNMEASURED,
            run,
        })
    }

    /// Run the procedure once under the arm means `truth`, and the true
    /// intra-participant correlation `true_correlation` (the declared one when
    /// absent; a procedure that estimates the correlation needs it), and
    /// retain what happened: the batches, each participant's arm and success count and
    /// the allocation probabilities they were randomized under. The history
    /// replays under this declaration ([`Self::read_history`]).
    ///
    /// # Errors
    /// A truth of the wrong shape, a numerical failure or cancellation.
    pub fn simulate_history(
        &self,
        truth: &[f64],
        true_correlation: Option<f64>,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<ParticipantHistory, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        simulate_participant_history(&self.spec, truth, true_correlation, seed, &cancelled)
            .map_err(|e| refusal_of(&e))
    }

    /// Recompute every look of a retained history under the declared
    /// procedure.
    ///
    /// # Errors
    /// A history that disagrees with the declaration, a retained assignment
    /// probability that is not the declared policy's
    /// (`adaptive_inference.assignment_not_replayed`), a procedure that
    /// continued past or stopped without its declared decisions
    /// (`adaptive_inference.stopping_not_replayed`), or cancellation.
    pub fn read_history(
        &self,
        history: &ParticipantHistory,
        ctx: &ExecutionContext,
    ) -> Result<ParticipantHistoryReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let readings = self.spec.read_history(history, &cancelled).map_err(|e| refusal_of(&e))?;
        let borrowing =
            self.spec.borrowing_receipt(history, &cancelled).map_err(|e| refusal_of(&e))?;
        let decision = readings.last().map_or(LookDecision::Continue, |r| r.decision);
        Ok(ParticipantHistoryReport {
            procedure_identity: self.identity.clone(),
            calibration: DERIVED_UNMEASURED,
            priors: self.spec.priors().to_vec(),
            borrowing,
            readings,
            decision,
        })
    }

    /// The retained analysis contract of a history: the realized allocation
    /// probabilities, the arm and outcome of every participant, when each
    /// outcome became available, the looks and the stopping rule.
    ///
    /// Each participant's responses are retained as their success count, as
    /// that many ones followed by zeros; their order within a participant is
    /// not part of the model. The history is read under the declared
    /// procedure first, so only a history that replays can be retained.
    ///
    /// # Errors
    /// As [`Self::read_history`]. A procedure that stopped at its first look
    /// has a single analysis and cannot be retained under interim-look
    /// monitoring; the contract refuses it when it is prepared.
    pub fn retained_declaration(
        &self,
        label: &str,
        arm_labels: Vec<String>,
        history: &ParticipantHistory,
        ctx: &ExecutionContext,
    ) -> Result<AdaptiveDesignDeclaration, AdaptiveInferenceRefusal> {
        let report = self.read_history(history, ctx)?;
        let lag = self.spec.lag_looks();
        let responses = self.spec.responses() as usize;
        let total: u32 = history.batches.iter().sum();
        let mut available_at_look = Vec::with_capacity(history.arms.len());
        let mut policy_history = Vec::with_capacity(history.batches.len());
        let mut interim_analyses = Vec::with_capacity(history.batches.len());
        let mut first_unit = 0_u32;
        for (look, (&size, reading)) in history.batches.iter().zip(&report.readings).enumerate() {
            let mut state = CanonicalWriter::new(POLICY_STATE_DOMAIN);
            state
                .text(&self.identity)
                .word(look as u64)
                .reals(&reading.assignment_probabilities)
                .reals(&reading.posterior.mean)
                .reals(&reading.posterior.sd)
                .real(reading.posterior.correlation_mean);
            policy_history.push(PolicyEpoch {
                first_unit,
                policy_identity: SemanticDigest::from_bytes(state.finish()),
            });
            first_unit += size;
            interim_analyses.push(InterimAnalysis {
                units_enrolled: first_unit,
                information_fraction: f64::from(first_unit) / f64::from(total),
            });
            for unit in first_unit - size..first_unit {
                available_at_look.push(
                    history.successes[unit as usize]
                        .map(|_| u32::try_from(look + lag).unwrap_or(u32::MAX)),
                );
            }
        }
        let outcomes = history
            .successes
            .iter()
            .map(|count| match count {
                Some(s) => {
                    let ones = *s as usize;
                    let mut values = vec![1.0; ones];
                    values.resize(responses, 0.0);
                    values
                }
                None => Vec::new(),
            })
            .collect();
        let ended_by_rule = matches!(
            report.decision,
            LookDecision::Stop { reason, .. } if reason.ended_by_rule()
        );
        Ok(AdaptiveDesignDeclaration {
            label: label.to_owned(),
            cell: ProcedureCell {
                assignment: AssignmentFamily::ThompsonBetaBernoulli,
                outcome: OutcomeStructure::RepeatedMeasures,
                monitoring: MonitoringStrategy::FixedInterimLooks,
                stopping: StoppingRuleKind::DecisionTheoretic,
            },
            arm_labels,
            assigned_arm: history
                .arms
                .iter()
                .map(|&arm| u32::try_from(arm).unwrap_or(u32::MAX))
                .collect(),
            policy_history: Some(policy_history),
            assignment_probabilities: Some(history.assignment.clone()),
            exposure: Some(
                history.arms.iter().map(|&arm| u32::try_from(arm).unwrap_or(u32::MAX)).collect(),
            ),
            availability: Some(OutcomeAvailability {
                available_at_look,
                delay: if lag == 0 {
                    DelayMechanism::Immediate
                } else {
                    DelayMechanism::IgnorableAdministrative
                },
            }),
            outcomes,
            interim_analyses: Some(interim_analyses),
            stopping_rule: Some(StoppingRuleDeclaration {
                kind: StoppingRuleKind::DecisionTheoretic,
                spec: StoppingSpec::PosteriorExpectedLoss {
                    loss_threshold: self.spec.decisions().loss_threshold,
                },
                ended_by_rule,
            }),
            probability_floor: self.spec.min_exploration(),
        })
    }
}
