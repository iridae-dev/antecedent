//! The declared whole procedure of an adaptive experiment: from a stated
//! declaration, or from the retained analysis contract plus the elements the
//! contract does not hold.
//!
//! The retained analysis contract
//! ([`antecedent_contract::adaptive_design::AdaptiveDesignContract`]) keeps
//! what the experiment *did*: realized assignment probabilities, the exposure
//! history, when each outcome became available, the interim looks and the
//! stopping rule's parameters. The uncertainty of the *procedure* is derived
//! from what the experiment *would do under other data*, so it also needs the
//! assignment rule itself, how lost outcomes relate to their values, the
//! exposure-failure model, how an arm is selected at the end and the priors a
//! posterior stopping rule used. [`ProcedureElements`] carries exactly those;
//! the adapter ([`AdaptiveProcedure::from_contract`]) takes everything else
//! from the contract, checks that the two agree, and refuses a missing element
//! with its own detail. Nothing is defaulted, and an element taken from the
//! retained data is listed by [`AdaptiveProcedure::taken_from_contract`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AssignmentFamily, DelayMechanism as ContractDelay, MonitoringStrategy,
    SpendingKind, StoppingSpec,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_stats::adaptive_inference::group_sequential::{
    Sidedness, SpendingFunction, boundaries,
};
use antecedent_stats::adaptive_inference::{
    AssignmentRule, Attribution, BetaPrior, ContractGap, ExposureModel, FinalSelection,
    OutcomeObservation, ProcedureDeclaration, ProcedureSpec, StoppingRule,
};

use super::adaptive_inference::AdaptiveInferenceRefusal;

/// Version word of the canonical procedure identity. Bump when the encoding
/// in [`AdaptiveProcedure::identity`] changes.
pub const ADAPTIVE_PROCEDURE_VERSION: u64 = 1;

const IDENTITY_DOMAIN: &str = "antecedent.adaptive_procedure.declaration";
const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Probabilities with which an outcome is observed once it is revealed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Retention {
    /// Probability that a success is observed.
    pub success: f64,
    /// Probability that a failure is observed.
    pub failure: f64,
}

/// The elements of the procedure the retained analysis contract does not
/// hold. Each is optional so that a missing one is refused with its own
/// detail.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProcedureElements {
    /// The assignment rule, as a function of the observed history (the
    /// contract keeps only the realized probabilities).
    pub assignment: Option<AssignmentRule>,
    /// Probabilities of the discrete contexts; `None` is a single context.
    pub context_probabilities: Option<Vec<f64>>,
    /// How outcomes are retained once revealed. Needed when outcomes are lost
    /// (an ignorable loss rate, or an outcome-dependent one).
    pub retention: Option<Retention>,
    /// The exposure-failure model. Needed when the retained exposure history
    /// differs from the assignments.
    pub exposure: Option<ExposureModel>,
    /// How the procedure selects an arm when no stopping rule decided.
    pub selection: Option<FinalSelection>,
    /// The prior of a posterior stopping rule (probability best, expected
    /// loss); the contract keeps only the threshold.
    pub stopping_prior: Option<BetaPrior>,
}

/// A validated, identified procedure.
#[derive(Clone, Debug)]
pub struct AdaptiveProcedure {
    pub(super) spec: ProcedureSpec,
    identity: String,
    contract_identity: Option<String>,
    from_contract: Vec<&'static str>,
}

fn gap_refusal(gap: ContractGap) -> AdaptiveInferenceRefusal {
    AdaptiveInferenceRefusal {
        code: gap.reason_code(),
        detail: gap.detail().to_owned(),
        message: gap.remedy().to_owned(),
    }
}

fn refusal(
    code: &'static str,
    detail: &str,
    message: impl Into<String>,
) -> AdaptiveInferenceRefusal {
    AdaptiveInferenceRefusal { code, detail: detail.to_owned(), message: message.into() }
}

fn invalid(detail: &str, message: impl Into<String>) -> AdaptiveInferenceRefusal {
    refusal(antecedent_core::reason_code!("invalid_argument"), detail, message)
}

fn incomplete(detail: &str, message: impl Into<String>) -> AdaptiveInferenceRefusal {
    refusal(antecedent_core::reason_code!("adaptive_contract_incomplete"), detail, message)
}

fn rule_family(rule: &AssignmentRule) -> Option<AssignmentFamily> {
    match rule {
        AssignmentRule::Fixed { .. } => Some(AssignmentFamily::FixedRandomization),
        AssignmentRule::EpsilonGreedy { .. } => Some(AssignmentFamily::EpsilonGreedy),
        AssignmentRule::ThompsonBeta { .. } => Some(AssignmentFamily::ThompsonBetaBernoulli),
        AssignmentRule::UpperConfidenceBound { .. } => Some(AssignmentFamily::UpperConfidenceBound),
        _ => None,
    }
}

fn encode_prior(writer: &mut CanonicalWriter, prior: &BetaPrior) {
    writer.real(prior.alpha).real(prior.beta);
}

fn identity_of(spec: &ProcedureSpec) -> String {
    let mut writer = CanonicalWriter::new(IDENTITY_DOMAIN);
    writer.word(ADAPTIVE_PROCEDURE_VERSION);
    writer.word(spec.arms() as u64);
    writer.reals(spec.contexts());
    match spec.assignment() {
        AssignmentRule::Fixed { probabilities } => {
            writer.text("fixed").reals(probabilities);
        }
        AssignmentRule::EpsilonGreedy { epsilon, prior } => {
            writer.text("epsilon_greedy").real(*epsilon);
            encode_prior(&mut writer, prior);
        }
        AssignmentRule::ThompsonBeta { prior } => {
            writer.text("thompson_beta");
            encode_prior(&mut writer, prior);
        }
        AssignmentRule::UpperConfidenceBound { exploration } => {
            writer.text("upper_confidence_bound").real(*exploration);
        }
        _ => {
            writer.text("unrecognized_rule");
        }
    }
    writer.word(spec.enrolment().len() as u64);
    for &batch in spec.enrolment() {
        writer.word(u64::from(batch));
    }
    let observation = spec.observation();
    writer
        .word(observation.lag_looks as u64)
        .real(observation.retain_success)
        .real(observation.retain_failure);
    let exposure = spec.exposure();
    writer.reals(&exposure.failure_probability).word(exposure.fallback_arm as u64).text(
        match exposure.attribution {
            Attribution::Assigned => "assigned",
            Attribution::Delivered => "delivered",
        },
    );
    match spec.stopping() {
        StoppingRule::NoStopping => {
            writer.text("no_stopping");
        }
        StoppingRule::PosteriorProbabilityBest { threshold, prior } => {
            writer.text("posterior_probability_best").real(*threshold);
            encode_prior(&mut writer, prior);
        }
        StoppingRule::PosteriorExpectedLoss { loss_threshold, prior } => {
            writer.text("posterior_expected_loss").real(*loss_threshold);
            encode_prior(&mut writer, prior);
        }
        StoppingRule::WaldBoundaries { critical, sidedness } => {
            writer.text("wald_boundaries").reals(critical).text(sidedness.as_str());
        }
        StoppingRule::ResourceCap { max_units } => {
            writer.text("resource_cap").word(u64::from(*max_units));
        }
        _ => {
            writer.text("unrecognized_stopping");
        }
    }
    match spec.selection() {
        FinalSelection::NoSelection => {
            writer.text("no_selection");
        }
        FinalSelection::HighestObservedMean => {
            writer.text("highest_observed_mean");
        }
        FinalSelection::HighestPosteriorMean(prior) => {
            writer.text("highest_posterior_mean");
            encode_prior(&mut writer, prior);
        }
        _ => {
            writer.text("unrecognized_selection");
        }
    }
    writer.finish_hex()
}

impl AdaptiveProcedure {
    fn sealed(
        spec: ProcedureSpec,
        contract_identity: Option<String>,
        from_contract: Vec<&'static str>,
    ) -> Self {
        let identity = identity_of(&spec);
        Self { spec, identity, contract_identity, from_contract }
    }

    /// Declare a procedure directly.
    ///
    /// # Errors
    /// One refusal per missing element (`adaptive_contract_incomplete`, its
    /// own `adaptive_inference.*` detail) and per invalid element
    /// (`invalid_argument`); nothing is defaulted.
    pub fn declare(
        declaration: ProcedureDeclaration,
    ) -> Result<Self, Vec<AdaptiveInferenceRefusal>> {
        ProcedureSpec::new(declaration)
            .map(|spec| Self::sealed(spec, None, Vec::new()))
            .map_err(|error| error.gaps.into_iter().map(gap_refusal).collect())
    }

    /// Build the procedure of a retained analysis contract.
    ///
    /// The enrolment schedule, the delay (in looks), the stopping rule's
    /// parameters (boundaries are recomputed from the spending function) and
    /// the number of arms come from the contract; the assignment rule, loss
    /// retention, exposure-failure model, selection and the posterior rules'
    /// priors come from `elements`. The two must agree: the cell's assignment
    /// family, stopping kind and monitoring strategy against the declared
    /// rule, and a fixed rule's probabilities against the realized rows.
    ///
    /// # Errors
    /// One refusal per missing element and per disagreement.
    #[allow(
        clippy::too_many_lines,
        reason = "one flat pass over the elements keeps each derivation next to its refusal"
    )]
    pub fn from_contract(
        contract: &AdaptiveDesignContract,
        elements: &ProcedureElements,
        ctx: &ExecutionContext,
    ) -> Result<Self, Vec<AdaptiveInferenceRefusal>> {
        let declaration = contract.declaration();
        let cell = declaration.cell;
        let arms = contract.arms();
        let mut problems: Vec<AdaptiveInferenceRefusal> = Vec::new();
        let mut taken: Vec<&'static str> = vec!["arms"];
        let (Some(looks), Some(availability), Some(rows), Some(exposure_history), Some(rule)) = (
            declaration.interim_analyses.as_deref(),
            declaration.availability.as_ref(),
            declaration.assignment_probabilities.as_deref(),
            declaration.exposure.as_deref(),
            declaration.stopping_rule,
        ) else {
            return Err(vec![incomplete(
                "adaptive_inference.interim_analyses_missing",
                "the contract is not complete; prepare the session before deriving the procedure",
            )]);
        };

        // Enrolment schedule: units between consecutive looks.
        let mut previous = 0_u32;
        let enrolment: Vec<u32> = looks
            .iter()
            .map(|look| {
                let batch = look.units_enrolled.saturating_sub(previous);
                previous = look.units_enrolled;
                batch
            })
            .collect();
        taken.push("enrolment_schedule");
        let enrolment_look = |unit: usize| -> usize {
            looks
                .iter()
                .position(|l| (unit as u64) < u64::from(l.units_enrolled))
                .unwrap_or(looks.len().saturating_sub(1))
        };

        // Delay in looks, from when each outcome became available.
        let lags: Vec<usize> = availability
            .available_at_look
            .iter()
            .enumerate()
            .filter_map(|(unit, available)| {
                available.map(|look| (look as usize).saturating_sub(enrolment_look(unit)))
            })
            .collect();
        let lag = lags.first().copied().unwrap_or(0);
        if lags.iter().any(|&l| l != lag) {
            problems.push(invalid(
                "adaptive_inference.outcome_availability_invalid",
                "outcomes became available after different delays in looks; the procedure model has one delay: declare a constant delay or analyse a route that does not model the procedure",
            ));
        }
        let looks_count = looks.len();
        let lost = availability.available_at_look.iter().enumerate().any(|(unit, available)| {
            available.is_none() && enrolment_look(unit) + lag < looks_count
        });
        taken.push("outcome_delay");
        let observation = match (availability.delay, elements.retention) {
            (_, Some(r)) => Some(OutcomeObservation {
                lag_looks: lag,
                retain_success: r.success,
                retain_failure: r.failure,
            }),
            (ContractDelay::Immediate | ContractDelay::IgnorableAdministrative, None) if !lost => {
                Some(OutcomeObservation::delayed(lag))
            }
            _ => {
                problems.push(incomplete(
                    "adaptive_inference.outcome_availability_missing",
                    "outcomes are lost or their availability depends on their values: declare the probability that a success and that a failure is observed once revealed",
                ));
                None
            }
        };

        // Exposure failures.
        let failures_observed = declaration
            .assigned_arm
            .iter()
            .zip(exposure_history)
            .any(|(assigned, delivered)| assigned != delivered);
        let exposure = match &elements.exposure {
            Some(model) => {
                let declared_failures = model.failure_probability.iter().any(|&p| p > 0.0);
                if failures_observed && !declared_failures {
                    problems.push(invalid(
                        "adaptive_inference.exposure_invalid",
                        "the retained exposure history differs from the assignments but the declared exposure model has no failures",
                    ));
                }
                Some(model.clone())
            }
            None if !failures_observed => {
                taken.push("exposure_model");
                Some(ExposureModel::none(arms))
            }
            None => {
                problems.push(incomplete(
                    "adaptive_inference.exposure_history_missing",
                    "the retained exposure differs from the assignments: declare the exposure-failure model (the probability each assigned arm is not delivered, the fallback arm, and the attribution)",
                ));
                None
            }
        };

        // Stopping rule parameters.
        let needs_prior = |detail: &str| {
            incomplete(
                "adaptive_inference.stopping_rule_missing",
                format!(
                    "a posterior stopping rule's prior is not retained ({detail}): declare the prior the rule used"
                ),
            )
        };
        let stopping = match rule.spec {
            StoppingSpec::NoStopping => Some(StoppingRule::NoStopping),
            StoppingSpec::ResourceCap { max_units } => {
                Some(StoppingRule::ResourceCap { max_units })
            }
            StoppingSpec::PosteriorProbabilityBest { threshold } => {
                if let Some(prior) = elements.stopping_prior {
                    Some(StoppingRule::PosteriorProbabilityBest { threshold, prior })
                } else {
                    problems.push(needs_prior("probability best"));
                    None
                }
            }
            StoppingSpec::PosteriorExpectedLoss { loss_threshold } => {
                if let Some(prior) = elements.stopping_prior {
                    Some(StoppingRule::PosteriorExpectedLoss { loss_threshold, prior })
                } else {
                    problems.push(needs_prior("expected loss"));
                    None
                }
            }
            StoppingSpec::AlphaSpending { spending, total_alpha, sidedness } => {
                let fractions: Vec<f64> = looks.iter().map(|l| l.information_fraction).collect();
                let spending = match spending {
                    SpendingKind::OBrienFlemingType => SpendingFunction::OBrienFlemingType,
                    SpendingKind::PocockType => SpendingFunction::PocockType,
                    _ => SpendingFunction::OBrienFlemingType,
                };
                let sidedness = match sidedness.as_str() {
                    "two_sided_symmetric" => Sidedness::TwoSidedSymmetric,
                    _ => Sidedness::OneSidedUpper,
                };
                let cancelled = || ctx.cancellation.is_cancelled();
                match boundaries(&fractions, total_alpha, spending, sidedness, &cancelled) {
                    Ok(b) => Some(StoppingRule::WaldBoundaries { critical: b.critical, sidedness }),
                    Err(error) => {
                        problems.push(invalid(
                            "adaptive_inference.rule_mismatch",
                            format!("the alpha-spending boundaries could not be computed: {error}"),
                        ));
                        None
                    }
                }
            }
            _ => {
                problems.push(invalid(
                    "adaptive_inference.rule_mismatch",
                    "the stopping rule has no procedure model",
                ));
                None
            }
        };
        taken.push("stopping_rule_parameters");

        if elements.assignment.is_none() {
            problems.push(incomplete(
                "adaptive_inference.policy_history_missing",
                "the contract keeps realized probabilities only: declare the assignment rule as a function of the observed history",
            ));
        }
        if elements.selection.is_none() {
            problems.push(gap_refusal(ContractGap::FinalSelectionMissing));
        }

        let declared = ProcedureDeclaration {
            arms,
            context_probabilities: elements.context_probabilities.clone(),
            assignment: elements.assignment.clone(),
            enrolment: Some(enrolment),
            observation,
            exposure,
            stopping,
            selection: elements.selection,
        };
        let spec = match ProcedureSpec::new(declared) {
            Ok(spec) => spec,
            Err(error) => {
                problems.extend(error.gaps.into_iter().map(gap_refusal));
                return Err(dedupe(problems));
            }
        };
        if !problems.is_empty() {
            return Err(dedupe(problems));
        }

        // Agreement with the contract's cell and realized rows.
        let family_ok = match (rule_family(spec.assignment()), cell.assignment) {
            (Some(rule), family) if rule == family => true,
            (Some(_), AssignmentFamily::ContextualBandit) => spec.contexts().len() > 1,
            (Some(rule), AssignmentFamily::BatchedAdaptive) => {
                rule != AssignmentFamily::FixedRandomization
            }
            _ => false,
        };
        if !family_ok {
            problems.push(invalid(
                "adaptive_inference.rule_mismatch",
                format!(
                    "the declared assignment rule is a {} procedure but the contract's cell is {}",
                    spec.family_name(),
                    cell.assignment.as_str()
                ),
            ));
        }
        if spec.stopping_name() != cell.stopping.as_str() {
            problems.push(invalid(
                "adaptive_inference.rule_mismatch",
                format!(
                    "the declared stopping rule is {} but the contract's cell is {}",
                    spec.stopping_name(),
                    cell.stopping.as_str()
                ),
            ));
        }
        let monitoring_ok = match cell.monitoring {
            MonitoringStrategy::FinalAnalysisOnly => spec.enrolment().len() == 1,
            MonitoringStrategy::GroupSequential => {
                matches!(spec.stopping(), StoppingRule::WaldBoundaries { .. })
            }
            _ => true,
        };
        if !monitoring_ok {
            problems.push(invalid(
                "adaptive_inference.rule_mismatch",
                format!(
                    "the contract's monitoring strategy {} disagrees with the declared looks and stopping rule",
                    cell.monitoring.as_str()
                ),
            ));
        }
        if spec.observation().mechanism().as_str() != availability.delay.as_str() {
            problems.push(invalid(
                "adaptive_inference.outcome_availability_invalid",
                format!(
                    "the declared retention makes outcome availability {} but the contract records {}",
                    spec.observation().mechanism().as_str(),
                    availability.delay.as_str()
                ),
            ));
        }
        if let AssignmentRule::Fixed { probabilities } = spec.assignment() {
            let replays = rows.iter().all(|row| {
                row.len() == probabilities.len()
                    && row
                        .iter()
                        .zip(probabilities)
                        .all(|(a, b)| (a - b).abs() <= PROBABILITY_TOLERANCE)
            });
            if !replays {
                problems.push(invalid(
                    "adaptive_inference.rule_mismatch",
                    "the realized assignment probabilities do not replay the declared fixed probabilities",
                ));
            }
        }
        if problems.is_empty() {
            Ok(Self::sealed(spec, Some(contract.identity().to_hex()), taken))
        } else {
            Err(dedupe(problems))
        }
    }

    /// The validated procedure.
    #[must_use]
    pub const fn spec(&self) -> &ProcedureSpec {
        &self.spec
    }

    /// Canonical identity (hex) of the declared procedure.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Identity (hex) of the retained contract the procedure was built from.
    #[must_use]
    pub fn contract_identity(&self) -> Option<&str> {
        self.contract_identity.as_deref()
    }

    /// The elements taken from the retained contract rather than declared.
    #[must_use]
    pub fn taken_from_contract(&self) -> &[&'static str] {
        &self.from_contract
    }
}

fn dedupe(refusals: Vec<AdaptiveInferenceRefusal>) -> Vec<AdaptiveInferenceRefusal> {
    let mut out: Vec<AdaptiveInferenceRefusal> = Vec::with_capacity(refusals.len());
    for r in refusals {
        if !out.iter().any(|seen| seen.detail == r.detail && seen.code == r.code) {
            out.push(r);
        }
    }
    out
}
