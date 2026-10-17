//! Derived uncertainty of an adaptive procedure, one report per paradigm.
//!
//! The uncertainty of an adaptively collected result is not the uncertainty
//! of a fixed-sample analysis of the final data: the stopping time, the
//! outcome delay and loss, the exposure failures and the policy's selection
//! are part of the data-generating process. This module routes a declared
//! [`AdaptiveProcedure`] to the derivations in
//! `antecedent_stats::adaptive_inference` and adds no rule of its own:
//!
//! * [`AdaptiveProcedure::law`]: the exact law of the terminal state under a
//!   hypothesized truth, by dynamic programming over the observed state.
//! * [`AdaptiveProcedure::simulate`]: the whole procedure run end to end, for
//!   the order-dependent weighted estimator.
//! * [`AdaptiveProcedure::report`]: one paradigm's statement. Each paradigm
//!   has a validity premise the declared procedure must meet; when it does
//!   not the request is refused with `cell_not_licensed` and the premise's
//!   `adaptive_inference.*` detail, and nothing is computed.
//!
//! Every result is labelled `derived_unmeasured`: the numbers are exact (or
//! simulated with a reported Monte Carlo error) under the stated model, and no
//! interval or coverage licence follows until the whole-procedure coverage is
//! measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError, AnalysisEstimator, AnytimeReport, AssignmentRule, BayesianReport,
    BetaPrior, CellLabel, DERIVED_UNMEASURED, DecisionReport, Estimand, EstimandStatement,
    EstimatorLaw, Functional, GaussianMixtureLaw, InversionGrid, InversionInterval,
    ObservedTerminal, OrderingPValues, OutcomeKernel, PosteriorMeanGap, ProcedureLaw,
    RepeatedMeasuresDeclaration, RepeatedMeasuresDerivation, RepeatedMeasuresRequest,
    RepeatedMeasuresSpec, ResourceBudget, ResourceReceipt, SimulatedLaw, SimulationRequest,
    SimulationSummary, StagewiseInference, StagewiseLook, StoppingRule, UncertaintyParadigm,
    UnitAssignment, admit, admit_repeated_measures, anytime_exclusion_probability,
    anytime_valid_report, bayesian_report, decision_report, derive_repeated_measures_within,
    enumerate_law, exact_fixed_contrast_law, inversion_interval, ordering_p_values,
    posterior_mean_gaps, simulate_procedure_within, stagewise_inference,
};

use super::adaptive_inference::AdaptiveInferenceRefusal;
use super::adaptive_procedure_declaration::{AdaptiveProcedure, gap_refusal};

/// A paradigm's request: what to compute and the hypothesized truth or
/// observed terminal state it needs.
///
/// `truth` is one success probability per arm-by-context cell, context-major
/// (`context * arms + arm`).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ProcedureReportRequest {
    /// The posterior at observed counts (when given) and the exact gap of the
    /// terminal posterior mean from a martingale of the whole procedure.
    BayesianPosterior {
        /// Prior of every cell.
        prior: BetaPrior,
        /// Success and failure counts per cell at the terminal state.
        observed: Option<Vec<u16>>,
    },
    /// Stopping-time law, expected sample size, decision law and regret of an
    /// expected-loss rule under a truth.
    DecisionTheoretic {
        /// Hypothesized success probability of each arm.
        truth: Vec<f64>,
    },
    /// The exact sampling law of the fixed design's contrast and, with an
    /// observed estimate, its ordering p-values.
    FixedSample {
        /// Hypothesized truth.
        truth: Vec<f64>,
        /// Cell with a plus sign.
        plus: usize,
        /// Cell with a minus sign.
        minus: usize,
        /// The observed terminal state, if any.
        observed: Option<ObservedTerminal>,
    },
    /// The actual crossing probabilities of the declared boundaries under a
    /// truth and, with an observed stopped statistic, the stagewise-ordered
    /// p-value, median-unbiased estimate and limits.
    GroupSequential {
        /// Hypothesized truth.
        truth: Vec<f64>,
        /// Two-sided confidence level of the stagewise limits.
        level: f64,
        /// The observed stopped statistic, if any.
        observed: Option<StagewiseLook>,
    },
    /// Distribution of the adaptively weighted estimate under the whole
    /// procedure, by simulation.
    AdaptiveWeighting {
        /// Hypothesized truth.
        truth: Vec<f64>,
        /// Replications, seed and estimator options.
        simulation: SimulationRequest,
        /// Wald half-width multiplier for the measured containment.
        z: f64,
    },
    /// Simultaneous time-uniform bounds at observed counts (when given) and
    /// the exact probability, under a truth (when given), that the bounds at
    /// the stopping time exclude it.
    AnytimeValid {
        /// Miscoverage level shared over the cells.
        alpha: f64,
        /// Sample size the mixture is tuned for.
        tuning_samples: f64,
        /// Counts at the terminal state, if any.
        observed: Option<Vec<u16>>,
        /// Hypothesized truth, if any.
        truth: Option<Vec<f64>>,
    },
    /// A confidence set from inverting the whole-procedure law over a
    /// declared grid. Not an inferential paradigm of the procedure matrix.
    WholeProcedureInversion {
        /// The observed terminal state.
        observed: ObservedTerminal,
        /// Contrast and nuisance grids.
        grid: InversionGrid,
        /// Miscoverage level.
        alpha: f64,
    },
}

/// What a paradigm's report holds.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ProcedureReportBody {
    /// Bayesian posterior and posterior-mean martingale gaps.
    Bayesian {
        /// The posterior at the observed counts.
        posterior: Option<BayesianReport>,
        /// Expected terminal posterior mean against the prior mean, per cell.
        mean_gaps: Vec<PosteriorMeanGap>,
    },
    /// Stopping and decision quality.
    DecisionTheoretic(DecisionReport),
    /// Sampling law of the fixed design's contrast.
    FixedSample {
        /// Law of the contrast.
        law: EstimatorLaw,
        /// Ordering p-values of the observed estimate.
        ordering: Option<OrderingPValues>,
    },
    /// Group-sequential boundaries under a truth, and stagewise inference.
    GroupSequential {
        /// The declared critical values.
        critical: Vec<f64>,
        /// Information fraction of each look.
        information_fractions: Vec<f64>,
        /// Probability of stopping at each look.
        stop_probabilities: Vec<f64>,
        /// Probability the horizon is run without a stop.
        horizon_probability: f64,
        /// Expected units enrolled at the end.
        expected_enrolled: f64,
        /// Stagewise-ordered inference for an observed stopped statistic.
        stagewise: Option<StagewiseInference>,
    },
    /// Simulated law of the weighted estimate.
    AdaptiveWeighting {
        /// Bias, spread, Monte Carlo error and measured Wald containment.
        summary: SimulationSummary,
        /// The estimand's value under the truth and the declared exposure
        /// attribution.
        truth_value: f64,
        /// Mean units enrolled at the end.
        mean_enrolled: f64,
        /// Fraction of replications ended by the stopping rule.
        rule_stop_fraction: f64,
        /// Replications run.
        replications: u32,
        /// The work the simulation was admitted for and did.
        receipt: ResourceReceipt,
    },
    /// Time-uniform bounds and their exact exclusion probability.
    AnytimeValid {
        /// Bounds at the observed counts.
        report: Option<AnytimeReport>,
        /// Probability, over the whole procedure, that the bounds exclude
        /// the truth.
        exclusion_probability: Option<f64>,
    },
    /// Whole-procedure test inversion.
    WholeProcedureInversion(InversionInterval),
}

/// One paradigm's derived statement about the procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureReport {
    /// The paradigm.
    pub paradigm: UncertaintyParadigm,
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Identity (hex) of the retained contract, when the procedure was built
    /// from one.
    pub contract_identity: Option<String>,
    /// What the arm-level quantity is.
    pub estimand: EstimandStatement,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The numbers.
    pub body: ProcedureReportBody,
}

/// The exact law of the terminal state under a truth.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureLawReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The terminal atoms and work accounting.
    pub law: ProcedureLaw,
}

/// A simulated law with the identity of the procedure it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureSimulationReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The replicates.
    pub law: SimulatedLaw,
}

fn premise_remedy(detail: &str) -> &'static str {
    match detail {
        "adaptive_inference.posterior_premise_outcome_dependent_delay" => {
            "model the outcome-dependent loss explicitly (a selection model) or use a route that does not condition on observed counts alone"
        }
        "adaptive_inference.fixed_sample_design_adaptive" => {
            "analyse an adaptive procedure with the posterior, adaptive-weighting or anytime-valid paradigm"
        }
        "adaptive_inference.fixed_sample_data_dependent_stopping" => {
            "analyse a stopped procedure with a paradigm that accounts for the stopping time"
        }
        "adaptive_inference.group_sequential_assignment_adaptive" => {
            "use a fixed assignment with pre-specified boundaries, or a paradigm valid under adaptive assignment"
        }
        "adaptive_inference.outcome_dependent_delay" => {
            "declare an ignorable delay and loss, or model the dependence of loss on the outcome"
        }
        "adaptive_inference.anytime_premise_contexts" => {
            "analyse one context at a time with a fixed within-context assignment, or use a paradigm for contextual assignment"
        }
        "adaptive_inference.inversion_premise_contexts" => "declare a single-context procedure",
        "adaptive_inference.independence_assumed_for_dependent_measurements" => {
            "analyse the unit-level summaries with a cluster-robust variance, or declare the within-unit dependence as zero if the measurements are independent"
        }
        "adaptive_inference.repeated_measures_paradigm_not_derived" => {
            "request the fixed-sample paradigm for a fixed assignment or the adaptive-weighting paradigm for an adaptive one; other paradigms are separate cells that are not derived for repeated measures"
        }
        "adaptive_inference.analysis_route_mismatch" => {
            "declare the final analysis whose route is the requested paradigm's"
        }
        "adaptive_inference.guarantee_not_transferable" => {
            "derive the guarantee for the cell in question; evidence for one cell is not evidence for another"
        }
        "adaptive_inference.exact_law_unavailable" => {
            "use the simulated whole-procedure law, which has no closed form for an adaptive assignment or a weighted score"
        }
        "adaptive_inference.borrowing_frozen" => {
            "declare the borrowing before recruitment and keep it: a new borrowing plan is a new declaration, not an amendment of a procedure that has made a look"
        }
        "adaptive_inference.assignment_not_replayed" => {
            "retain the allocation probabilities the declared procedure computed, or declare the priors, correlation, exploration floor and decisions the procedure actually ran under"
        }
        "adaptive_inference.stopping_not_replayed" => {
            "retain the looks the declared decisions performed, or declare the decisions the procedure actually ran under"
        }
        "adaptive_inference.batch_incomplete" => {
            "retain the responses of every participant of a batch whose follow-up has elapsed, or end the history at the last batch whose participants all completed"
        }
        _ => {
            "declare a procedure that meets the paradigm's premise, or choose the paradigm the procedure meets"
        }
    }
}

/// Details whose premise is an assumption of the retained history or of a
/// frozen declaration, not the licence of a cell.
const ASSUMPTION_DETAILS: [&str; 4] = [
    "adaptive_inference.borrowing_frozen",
    "adaptive_inference.assignment_not_replayed",
    "adaptive_inference.stopping_not_replayed",
    "adaptive_inference.batch_incomplete",
];

/// Map a numerical refusal to the facade's coded refusal.
pub(super) fn refusal_of(error: &AdaptiveInferenceError) -> AdaptiveInferenceRefusal {
    match error {
        AdaptiveInferenceError::PremiseFailed { detail, message } => AdaptiveInferenceRefusal {
            code: if ASSUMPTION_DETAILS.contains(detail) {
                antecedent_stats::adaptive_inference::participant_procedure::premise_reason_code()
            } else {
                antecedent_stats::adaptive_inference::procedure_uncertainty::premise_reason_code()
            },
            detail: (*detail).to_owned(),
            message: format!("{message}; remedy: {}", premise_remedy(detail)),
        },
        AdaptiveInferenceError::Capacity(capacity) => AdaptiveInferenceRefusal {
            code: capacity.reason_code(),
            detail: capacity.detail().to_owned(),
            message: format!("{capacity}; remedy: {}", capacity.remedy()),
        },
        AdaptiveInferenceError::Positivity { unit, arm, probability, floor } => {
            AdaptiveInferenceRefusal {
                code: antecedent_core::reason_code!("adaptive_assumption_violated"),
                detail: "adaptive_inference.positivity_floor".to_owned(),
                message: format!(
                    "arm {arm} had assignment probability {probability} below the floor {floor} at unit {unit} of the simulated procedure: the weighted estimator is not supported under a deterministic or near-deterministic rule; use a randomized assignment"
                ),
            }
        }
        AdaptiveInferenceError::Limit(what) => AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "adaptive_inference.procedure_limit_exceeded".to_owned(),
            message: format!(
                "{what}; reduce the units, looks, cells or delay of the declared procedure, or use the simulation route"
            ),
        },
        AdaptiveInferenceError::Numerical(what) => AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "adaptive_inference.numerical_failure".to_owned(),
            message: format!("{what}; no value is reported"),
        },
        AdaptiveInferenceError::Cancelled => AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            detail: "adaptive_inference.cancelled".to_owned(),
            message: "cancelled before a result was formed; this is not a verdict on the data"
                .to_owned(),
        },
        other => AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "adaptive_inference.input_invalid".to_owned(),
            message: other.to_string(),
        },
    }
}

impl AdaptiveProcedure {
    /// The exact law of the terminal state under `truth`.
    ///
    /// # Errors
    /// A truth of the wrong shape, a state or work bound exceeded, or
    /// cancellation.
    pub fn law(
        &self,
        truth: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<ProcedureLawReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let law = enumerate_law(
            &self.spec,
            &OutcomeKernel::Fixed { success_probabilities: truth.to_vec() },
            &cancelled,
        )
        .map_err(|e| refusal_of(&e))?;
        Ok(ProcedureLawReport {
            procedure_identity: self.identity().to_owned(),
            calibration: DERIVED_UNMEASURED,
            law,
        })
    }

    /// Run the whole procedure `request.replications` times under `truth`.
    ///
    /// Replicate streams are keyed by the seed and the replicate index, so a
    /// replicate does not depend on how many others are run.
    ///
    /// # Errors
    /// The `adaptive_weighting` premise failing, a truth of the wrong shape,
    /// a request outside bounds, a positivity failure or cancellation.
    pub fn simulate(
        &self,
        truth: &[f64],
        request: &SimulationRequest,
        ctx: &ExecutionContext,
    ) -> Result<ProcedureSimulationReport, AdaptiveInferenceRefusal> {
        admit(&self.spec, UncertaintyParadigm::FrequentistAdaptiveWeighting)
            .map_err(|e| refusal_of(&e))?;
        let cancelled = || ctx.cancellation.is_cancelled();
        let budget = ResourceBudget::from_context(ctx);
        let law = simulate_procedure_within(&self.spec, truth, request, &budget, &cancelled)
            .map_err(|e| refusal_of(&e))?;
        Ok(ProcedureSimulationReport {
            procedure_identity: self.identity().to_owned(),
            calibration: DERIVED_UNMEASURED,
            law,
        })
    }

    /// One paradigm's derived statement.
    ///
    /// # Errors
    /// `cell_not_licensed` with the premise's detail when the declared
    /// procedure does not meet the paradigm's premise; otherwise the shape,
    /// limit, positivity or cancellation refusal of the derivation.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per paradigm keeps each premise gate next to its derivation"
    )]
    pub fn report(
        &self,
        request: &ProcedureReportRequest,
        ctx: &ExecutionContext,
    ) -> Result<ProcedureReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let spec = &self.spec;
        let fail = |e: AdaptiveInferenceError| refusal_of(&e);
        let (paradigm, body) = match request {
            ProcedureReportRequest::BayesianPosterior { prior, observed } => {
                let posterior = if let Some(counts) = observed {
                    Some(bayesian_report(spec, *prior, counts, &cancelled).map_err(fail)?)
                } else {
                    admit(spec, UncertaintyParadigm::BayesianPosterior).map_err(fail)?;
                    None
                };
                let priors = vec![*prior; spec.cells()];
                let mean_gaps = posterior_mean_gaps(spec, &priors, &cancelled).map_err(fail)?;
                (
                    UncertaintyParadigm::BayesianPosterior,
                    ProcedureReportBody::Bayesian { posterior, mean_gaps },
                )
            }
            ProcedureReportRequest::DecisionTheoretic { truth } => {
                let report = decision_report(spec, truth, &cancelled).map_err(fail)?;
                (
                    UncertaintyParadigm::DecisionTheoreticStopping,
                    ProcedureReportBody::DecisionTheoretic(report),
                )
            }
            ProcedureReportRequest::FixedSample { truth, plus, minus, observed } => {
                admit(spec, UncertaintyParadigm::FrequentistFixedSample).map_err(fail)?;
                let law = enumerate_law(
                    spec,
                    &OutcomeKernel::Fixed { success_probabilities: truth.clone() },
                    &cancelled,
                )
                .map_err(fail)?;
                let contrast = law
                    .estimator_law(Functional::Contrast { plus: *plus, minus: *minus })
                    .map_err(fail)?;
                let ordering = observed
                    .as_ref()
                    .map(|o| ordering_p_values(&law, *plus, *minus, o))
                    .transpose()
                    .map_err(fail)?;
                (
                    UncertaintyParadigm::FrequentistFixedSample,
                    ProcedureReportBody::FixedSample { law: contrast, ordering },
                )
            }
            ProcedureReportRequest::GroupSequential { truth, level, observed } => {
                admit(spec, UncertaintyParadigm::FrequentistGroupSequential).map_err(fail)?;
                let StoppingRule::WaldBoundaries { critical, sidedness } = spec.stopping() else {
                    return Err(refusal_of(&AdaptiveInferenceError::PremiseFailed {
                        detail: "adaptive_inference.rule_mismatch",
                        message: "group-sequential inference needs declared boundaries over the looks",
                    }));
                };
                let total = f64::from(*spec.cumulative_enrolment().last().unwrap_or(&1));
                let fractions: Vec<f64> =
                    spec.cumulative_enrolment().iter().map(|&n| f64::from(n) / total).collect();
                let law = enumerate_law(
                    spec,
                    &OutcomeKernel::Fixed { success_probabilities: truth.clone() },
                    &cancelled,
                )
                .map_err(fail)?;
                let stagewise = observed
                    .as_ref()
                    .map(|o| {
                        stagewise_inference(
                            &fractions, critical, *sidedness, *o, *level, &cancelled,
                        )
                    })
                    .transpose()
                    .map_err(fail)?;
                (
                    UncertaintyParadigm::FrequentistGroupSequential,
                    ProcedureReportBody::GroupSequential {
                        critical: critical.clone(),
                        information_fractions: fractions,
                        stop_probabilities: law.rule_stop_probabilities(),
                        horizon_probability: law.horizon_probability(),
                        expected_enrolled: law.expected_enrolled(),
                        stagewise,
                    },
                )
            }
            ProcedureReportRequest::AdaptiveWeighting { truth, simulation, z } => {
                admit(spec, UncertaintyParadigm::FrequentistAdaptiveWeighting).map_err(fail)?;
                let budget = ResourceBudget::from_context(ctx);
                let simulated =
                    simulate_procedure_within(spec, truth, simulation, &budget, &cancelled)
                        .map_err(fail)?;
                let truth_value = self.estimand_value(simulation.estimand, truth)?;
                let summary = simulated.summarize(truth_value, *z).map_err(fail)?;
                (
                    UncertaintyParadigm::FrequentistAdaptiveWeighting,
                    ProcedureReportBody::AdaptiveWeighting {
                        summary,
                        truth_value,
                        mean_enrolled: simulated.mean_enrolled(),
                        rule_stop_fraction: simulated.rule_stop_fraction(),
                        replications: simulation.replications,
                        receipt: simulated.receipt,
                    },
                )
            }
            ProcedureReportRequest::AnytimeValid { alpha, tuning_samples, observed, truth } => {
                admit(spec, UncertaintyParadigm::AnytimeValidFrequentist).map_err(fail)?;
                let report = observed
                    .as_ref()
                    .map(|counts| anytime_valid_report(spec, counts, *alpha, *tuning_samples))
                    .transpose()
                    .map_err(fail)?;
                let exclusion_probability = truth
                    .as_ref()
                    .map(|t| {
                        anytime_exclusion_probability(spec, t, *alpha, *tuning_samples, &cancelled)
                    })
                    .transpose()
                    .map_err(fail)?;
                (
                    UncertaintyParadigm::AnytimeValidFrequentist,
                    ProcedureReportBody::AnytimeValid { report, exclusion_probability },
                )
            }
            ProcedureReportRequest::WholeProcedureInversion { observed, grid, alpha } => {
                let interval =
                    inversion_interval(spec, observed, grid, *alpha, &cancelled).map_err(fail)?;
                (
                    UncertaintyParadigm::WholeProcedureInversion,
                    ProcedureReportBody::WholeProcedureInversion(interval),
                )
            }
        };
        let estimand = admit(spec, paradigm).map_err(fail)?;
        Ok(ProcedureReport {
            paradigm,
            procedure_identity: self.identity().to_owned(),
            contract_identity: self.contract_identity().map(str::to_owned),
            estimand,
            calibration: DERIVED_UNMEASURED,
            body,
        })
    }

    /// The value of a weighted-estimator estimand under `truth`: the arm
    /// quantity of the declared exposure attribution, pooled over contexts by
    /// their probabilities.
    fn estimand_value(
        &self,
        estimand: Estimand,
        truth: &[f64],
    ) -> Result<f64, AdaptiveInferenceRefusal> {
        let spec = &self.spec;
        let arms = spec.arms();
        let in_range = |a: usize| a < arms;
        let arm_value = |arm: usize| -> f64 {
            spec.contexts()
                .iter()
                .enumerate()
                .map(|(context, &p)| {
                    let mean = |a: usize| truth[context * arms + a];
                    let value = match spec.exposure().attribution {
                        antecedent_stats::adaptive_inference::Attribution::Delivered => mean(arm),
                        antecedent_stats::adaptive_inference::Attribution::Assigned => {
                            let failure = spec.exposure().failure_probability[arm];
                            let fallback = spec.exposure().fallback_arm;
                            (1.0 - failure) * mean(arm) + failure * mean(fallback)
                        }
                    };
                    p * value
                })
                .sum()
        };
        match estimand {
            Estimand::ArmMean(a) if in_range(a) && truth.len() == spec.cells() => Ok(arm_value(a)),
            Estimand::Contrast { plus, minus }
                if in_range(plus) && in_range(minus) && truth.len() == spec.cells() =>
            {
                Ok(arm_value(plus) - arm_value(minus))
            }
            _ => Err(refusal_of(&AdaptiveInferenceError::Shape(
                "an estimand names an arm outside the procedure, or the truth has the wrong shape",
            ))),
        }
    }

    /// Whether the declared assignment adapts to the observed history.
    #[must_use]
    pub fn is_adaptive(&self) -> bool {
        self.spec.assignment().is_adaptive()
    }

    /// Whether the assignment rule is a fixed randomization.
    #[must_use]
    pub fn is_fixed_randomization(&self) -> bool {
        matches!(self.spec.assignment(), AssignmentRule::Fixed { .. })
    }
}

// -- Repeated measures and clustered outcomes -------------------------------------------------------

/// Version word of the canonical repeated-measures procedure identity. Bump
/// when the encoding in [`RepeatedMeasuresProcedure::identity`] changes.
pub const REPEATED_MEASURES_PROCEDURE_VERSION: u64 = 1;

const REPEATED_MEASURES_DOMAIN: &str = "antecedent.adaptive_procedure.repeated_measures";

/// What to derive for a repeated-measures or clustered procedure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RepeatedMeasuresReportRequest {
    /// The exact closed forms only: the variance and design effect of the
    /// unit summary and, for a fixed two-arm allocation, the exact contrast
    /// law. No simulation is run.
    ClosedForm {
        /// Mean of each arm.
        arm_means: Vec<f64>,
    },
    /// The whole-procedure simulation under the paradigm of the declared
    /// analysis, with its exact closed forms where they exist.
    Derived {
        /// Mean of each arm.
        arm_means: Vec<f64>,
        /// Replications and seed.
        simulation: RepeatedMeasuresRequest,
        /// Wald half-width multiplier for the measured containment.
        z: f64,
    },
    /// The same as `Derived`, naming the paradigm asked for; a paradigm the
    /// declared procedure does not meet is refused.
    Paradigm {
        /// The paradigm the caller wants a statement in.
        paradigm: UncertaintyParadigm,
        /// Mean of each arm.
        arm_means: Vec<f64>,
        /// Replications and seed.
        simulation: RepeatedMeasuresRequest,
        /// Wald half-width multiplier for the measured containment.
        z: f64,
    },
    /// Ask whether the guarantee of this procedure's cell may be stated for
    /// another cell; it never may unless the cells are the same.
    CellTransfer {
        /// Identifier `family/outcome/monitoring/stopping` of the target.
        cell_id: String,
        /// Paradigm of the target.
        paradigm: String,
    },
}

/// What a repeated-measures report holds.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RepeatedMeasuresReportBody {
    /// Exact closed forms.
    ClosedForm {
        /// Variance of one measurement.
        marginal_variance: f64,
        /// Variance of one unit's summary.
        unit_summary_variance: f64,
        /// Variance inflation over independent measurements.
        design_effect: f64,
        /// Measurements per unit.
        measurements: usize,
        /// Units whose summary reaches the final analysis.
        revealed_units: u32,
        /// The contrast under the arm means.
        contrast_value: f64,
        /// Exact contrast law, when it has a closed form.
        exact_law: Option<GaussianMixtureLaw>,
    },
    /// Closed forms and the simulated whole-procedure law.
    Derived(Box<RepeatedMeasuresDerivation>),
    /// The transfer was accepted (the cells are the same).
    CellTransfer {
        /// The cell both statements are about.
        cell_id: String,
    },
}

/// A derived statement about a repeated-measures or clustered procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresReport {
    /// Identity (hex) of the declared procedure.
    pub procedure_identity: String,
    /// The matrix cell the statement is about.
    pub cell: CellLabel,
    /// Name of the analysis route (cluster-robust, pooled, weighted).
    pub route: &'static str,
    /// Always `derived_unmeasured`.
    pub calibration: &'static str,
    /// The numbers.
    pub body: RepeatedMeasuresReportBody,
}

/// A declared repeated-measures or clustered adaptive procedure.
#[derive(Clone, Debug)]
pub struct RepeatedMeasuresProcedure {
    spec: RepeatedMeasuresSpec,
    identity: String,
}

fn repeated_measures_identity(spec: &RepeatedMeasuresSpec) -> String {
    use antecedent_stats::adaptive_inference::{OutcomeStructure, WithinUnitCorrelation};
    let mut writer = CanonicalWriter::new(REPEATED_MEASURES_DOMAIN);
    writer.word(REPEATED_MEASURES_PROCEDURE_VERSION);
    writer.word(spec.arms() as u64);
    match spec.assignment() {
        UnitAssignment::Fixed { probabilities } => {
            writer.text("fixed").reals(probabilities);
        }
        UnitAssignment::EpsilonGreedy { epsilon, prior } => {
            writer.text("epsilon_greedy").real(*epsilon).real(prior.mean).real(prior.strength);
        }
        _ => {
            writer.text("unrecognized_rule");
        }
    }
    writer.word(spec.enrolment().len() as u64);
    for &batch in spec.enrolment() {
        writer.word(u64::from(batch));
    }
    writer.word(spec.lag_looks() as u64);
    match spec.structure() {
        OutcomeStructure::RepeatedMeasures { measurements, variance, correlation } => {
            writer.text("repeated_measures").word(u64::from(*measurements)).real(*variance);
            match correlation {
                Some(WithinUnitCorrelation::CompoundSymmetric { rho }) => {
                    writer.text("compound_symmetric").real(*rho);
                }
                Some(WithinUnitCorrelation::Autoregressive { rho }) => {
                    writer.text("autoregressive").real(*rho);
                }
                _ => {
                    writer.text("unrecognized_correlation");
                }
            }
        }
        OutcomeStructure::Hierarchical { members, between_variance, within_variance } => {
            writer
                .text("hierarchical")
                .word(u64::from(*members))
                .real(between_variance.unwrap_or(f64::NAN))
                .real(within_variance.unwrap_or(f64::NAN));
        }
        _ => {
            writer.text("unrecognized_structure");
        }
    }
    let analysis = spec.analysis();
    writer.word(analysis.plus as u64).word(analysis.minus as u64);
    let summary_name = |summary| match summary {
        antecedent_stats::adaptive_inference::UnitSummary::Gls => "gls",
        _ => "mean",
    };
    match analysis.estimator {
        AnalysisEstimator::ClusterRobust { summary } => {
            writer.text("cluster_robust").text(summary_name(summary));
        }
        AnalysisEstimator::PooledIndependent => {
            writer.text("pooled_independent");
        }
        AnalysisEstimator::AdaptivelyWeighted { summary, scheme, prior, probability_floor } => {
            writer
                .text("adaptively_weighted")
                .text(summary_name(summary))
                .text(scheme.as_str())
                .real(prior.mean)
                .real(prior.strength)
                .real(probability_floor);
        }
        _ => {
            writer.text("unrecognized_estimator");
        }
    }
    writer.finish_hex()
}

impl RepeatedMeasuresProcedure {
    /// Declare the procedure.
    ///
    /// # Errors
    /// One refusal per missing element (`adaptive_contract_incomplete` with
    /// its own `adaptive_inference.*` detail, including the within-unit
    /// correlation, or the between and within variances) and per invalid
    /// element (`invalid_argument`); nothing is defaulted.
    pub fn declare(
        declaration: RepeatedMeasuresDeclaration,
    ) -> Result<Self, Vec<AdaptiveInferenceRefusal>> {
        RepeatedMeasuresSpec::new(declaration)
            .map(|spec| {
                let identity = repeated_measures_identity(&spec);
                Self { spec, identity }
            })
            .map_err(|error| error.gaps.into_iter().map(gap_refusal).collect())
    }

    /// The validated procedure.
    #[must_use]
    pub const fn spec(&self) -> &RepeatedMeasuresSpec {
        &self.spec
    }

    /// Canonical identity (hex) of the declared procedure.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The matrix cell of the declared procedure.
    #[must_use]
    pub fn cell(&self) -> CellLabel {
        self.spec.cell()
    }

    /// One statement about the procedure.
    ///
    /// # Errors
    /// `cell_not_licensed` with the premise's detail (a fixed-sample analysis
    /// after adaptive assignment, an independence assumption for dependent
    /// measurements, an undeclared paradigm, a cell transfer), or the
    /// derivation's shape, limit, positivity or cancellation refusal.
    pub fn report(
        &self,
        request: &RepeatedMeasuresReportRequest,
        ctx: &ExecutionContext,
    ) -> Result<RepeatedMeasuresReport, AdaptiveInferenceRefusal> {
        let cancelled = || ctx.cancellation.is_cancelled();
        let spec = &self.spec;
        let fail = |e: AdaptiveInferenceError| refusal_of(&e);
        let budget = ResourceBudget::from_context(ctx);
        let derive = |arm_means: &[f64], simulation: &RepeatedMeasuresRequest, z: f64| {
            derive_repeated_measures_within(spec, arm_means, simulation, z, &budget, &cancelled)
                .map(Box::new)
                .map(RepeatedMeasuresReportBody::Derived)
                .map_err(fail)
        };
        let body = match request {
            RepeatedMeasuresReportRequest::ClosedForm { arm_means } => {
                admit_repeated_measures(spec, spec.declared_paradigm()).map_err(fail)?;
                if arm_means.len() != spec.arms() || arm_means.iter().any(|m| !m.is_finite()) {
                    return Err(fail(AdaptiveInferenceError::Shape("one finite mean per arm")));
                }
                RepeatedMeasuresReportBody::ClosedForm {
                    marginal_variance: spec.marginal_variance(),
                    unit_summary_variance: spec.unit_summary_variance(),
                    design_effect: spec.design_effect(),
                    measurements: spec.measurements(),
                    revealed_units: spec.revealed_units(),
                    contrast_value: arm_means[spec.analysis().plus]
                        - arm_means[spec.analysis().minus],
                    exact_law: exact_fixed_contrast_law(spec, arm_means).ok(),
                }
            }
            RepeatedMeasuresReportRequest::Derived { arm_means, simulation, z } => {
                derive(arm_means, simulation, *z)?
            }
            RepeatedMeasuresReportRequest::Paradigm { paradigm, arm_means, simulation, z } => {
                admit_repeated_measures(spec, *paradigm).map_err(fail)?;
                derive(arm_means, simulation, *z)?
            }
            RepeatedMeasuresReportRequest::CellTransfer { cell_id, paradigm } => {
                spec.cell().transfer_to_named(cell_id, paradigm).map_err(fail)?;
                RepeatedMeasuresReportBody::CellTransfer { cell_id: spec.cell().id() }
            }
        };
        Ok(RepeatedMeasuresReport {
            procedure_identity: self.identity.clone(),
            cell: spec.cell(),
            route: spec.route_name(),
            calibration: DERIVED_UNMEASURED,
            body,
        })
    }
}
