//! Estimators for repeated categorical choices, one per inferential paradigm of the choice cells
//! of the adaptive-procedure matrix.
//!
//! Each entry point takes plain count columns that the caller has already checked against the
//! choice analysis contract (`antecedent_contract::adaptive_design::ChoiceDesignContract`: frame,
//! option parameterization, row-level evidence accounting) and returns a labelled result:
//!
//! * [`qualify_choice_prior`]: the Dirichlet prior that retained sources support, each source
//!   discounted as a power prior on its counts, with a prior-predictive conflict diagnostic.
//! * [`choice_posterior`]: the conjugate Dirichlet posterior of the pooled option shares, the
//!   posterior probability that each option has the largest share and the posterior expected loss
//!   of selecting it.
//! * [`choice_stopping_decision`]: a stop-or-continue decision from the posterior at one look,
//!   by the one stopping rule that [`antecedent_stats::adaptive_inference::ChoiceStoppingRule`]
//!   defines for the procedure whose exact law the stats layer enumerates.
//! * [`hierarchical_choice_posterior`]: the posterior of the population shares and of the
//!   respondents' heterogeneity (two or three options), by deterministic quadrature.
//!
//! No result carries an interval or a coverage statement as a claim. Every guarantee slot is a
//! [`WithheldGuarantee`] that refuses with `cell_not_licensed` until coverage of the exact
//! procedure has been measured. The numerics are in
//! [`antecedent_stats::adaptive_inference`]; the posterior is a statement about choices in the
//! declared choice set, and stated choices do not identify behaviour or a causal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::adaptive_inference as numerics;

use crate::adaptive_inference::{
    AdaptiveInferenceError, WithheldGuarantee, numeric_error, refuse_if_interrupted,
};

pub use antecedent_stats::adaptive_inference::{
    ChoiceKernel, ChoiceProcedureSpec, ChoiceStoppingRule, HierarchicalChoicePrior,
    MAX_CHOICE_OPTIONS,
};

/// Log Bayes factor above which a source is flagged as conflicting with the rest (`ln 10`).
pub const CONFLICT_LOG_BAYES_FACTOR: f64 = std::f64::consts::LN_10;

/// What the posterior shares are shares of.
pub const MEANING_STATEMENT: &str = "shares of repeated categorical choices as declared; stated preferences do not identify behavioural or causal outcomes";

/// What a conflict diagnostic is.
pub const CONFLICT_STATEMENT: &str = "prior-predictive log evidence ratio of each source against a vague prior versus the remaining evidence, including the current look when present; fractional power-prior counts yield a generalized score, and calibration is unmeasured";

/// Premise of the posterior's validity under optional stopping.
pub const CHOICE_POSTERIOR_PREMISE: &str = "the choice set is fixed, respondents are exchangeable draws from the declared population, and the stopping rule depends on the observed choices only";

/// What a hierarchical posterior assumes.
pub const HIERARCHICAL_PREMISE: &str = "respondents are exchangeable draws from the declared population, each respondent's choice shares are Dirichlet around the population shares, and the stopping rule depends on the observed choices only";

/// How a conflicting source is handled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChoiceConflictPolicy {
    /// Report the diagnostic and use the source.
    Report,
    /// Refuse a prior with a conflicting source.
    Refuse,
}

/// One retained source as the estimator reads it.
#[derive(Clone, Copy, Debug)]
pub struct ChoiceSourceCounts<'a> {
    /// Source label.
    pub label: &'a str,
    /// Choices of each option, in the target's option order.
    pub counts: &'a [f64],
    /// Fraction of the source's information used, in `[0, 1]`.
    pub discount: f64,
    /// Whether the source's population differs from the target's (declared transfer).
    pub transferred: bool,
}

/// A source as used.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedChoiceSource {
    /// Source label.
    pub label: String,
    /// Discount applied.
    pub discount: f64,
    /// Whether the population differed from the target's.
    pub transferred: bool,
    /// Prior-predictive log Bayes factor against the remaining evidence.
    pub conflict_log_bayes_factor: f64,
    /// Whether the source is flagged.
    pub conflicting: bool,
}

/// A prior supported by retained sources.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedChoicePrior {
    /// Dirichlet shapes.
    pub alpha: Vec<f64>,
    /// Observations the sources are worth after discounting.
    pub borrowed_observations: f64,
    /// Sources as used.
    pub sources: Vec<QualifiedChoiceSource>,
    /// Statements the prior carries.
    pub statements: Vec<&'static str>,
}

fn invalid(detail: &'static str, message: impl Into<String>) -> AdaptiveInferenceError {
    AdaptiveInferenceError::new(
        antecedent_core::reason_code!("choice_evidence_invalid"),
        detail,
        message,
    )
}

/// The Dirichlet prior that retained sources support.
///
/// # Errors
/// Shapes or counts out of domain; a conflicting source under [`ChoiceConflictPolicy::Refuse`].
pub fn qualify_choice_prior(
    initial: &[f64],
    sources: &[ChoiceSourceCounts<'_>],
    policy: ChoiceConflictPolicy,
) -> Result<QualifiedChoicePrior, AdaptiveInferenceError> {
    if initial.len() < 2 || initial.len() > MAX_CHOICE_OPTIONS {
        return Err(invalid("choice_inference.input_invalid", "a prior has two to 64 shapes"));
    }
    let mut alpha = initial.to_vec();
    let mut borrowed = 0.0;
    for s in sources {
        if s.counts.len() != initial.len()
            || !(s.discount.is_finite() && (0.0..=1.0).contains(&s.discount))
            || s.counts.iter().any(|c| !(c.is_finite() && *c >= 0.0))
        {
            return Err(invalid(
                "choice_inference.input_invalid",
                format!("source `{}` is malformed", s.label),
            ));
        }
        for (a, c) in alpha.iter_mut().zip(s.counts) {
            *a += s.discount * c;
        }
        borrowed += s.discount * s.counts.iter().sum::<f64>();
    }
    numerics::DirichletPosterior::new(alpha.clone()).map_err(|e| numeric_error(&e))?;
    let mut used = Vec::with_capacity(sources.len());
    for (i, s) in sources.iter().enumerate() {
        // Evidence from every other source, discounted, on top of the initial prior.
        let mut rest = initial.to_vec();
        for (j, other) in sources.iter().enumerate() {
            if j != i {
                for (a, c) in rest.iter_mut().zip(other.counts) {
                    *a += other.discount * c;
                }
            }
        }
        let vague = numerics::DirichletPosterior::log_marginal_likelihood(initial, s.counts)
            .map_err(|e| numeric_error(&e))?;
        let against_rest = numerics::DirichletPosterior::log_marginal_likelihood(&rest, s.counts)
            .map_err(|e| numeric_error(&e))?;
        let log_bf = vague - against_rest;
        let flagged = sources.len() > 1 && log_bf > CONFLICT_LOG_BAYES_FACTOR;
        if flagged && policy == ChoiceConflictPolicy::Refuse {
            return Err(AdaptiveInferenceError::new(
                antecedent_core::reason_code!("choice_evidence_conflict"),
                "choice_inference.source_conflict",
                format!(
                    "source `{}` conflicts with the remaining evidence (log Bayes factor {log_bf:.3}); report the conflict, discount the source or leave it out",
                    s.label
                ),
            ));
        }
        used.push(QualifiedChoiceSource {
            label: s.label.to_owned(),
            discount: s.discount,
            transferred: s.transferred,
            conflict_log_bayes_factor: if sources.len() > 1 { log_bf } else { 0.0 },
            conflicting: flagged,
        });
    }
    Ok(QualifiedChoicePrior {
        alpha,
        borrowed_observations: borrowed,
        sources: used,
        statements: vec![MEANING_STATEMENT, CONFLICT_STATEMENT],
    })
}

/// The Dirichlet posterior of the pooled option shares.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoicePosterior {
    /// Posterior Dirichlet shapes.
    pub alpha: Vec<f64>,
    /// Posterior mean of each share.
    pub mean: Vec<f64>,
    /// Posterior probability that each option has the largest share.
    pub probability_best: Vec<f64>,
    /// Posterior expected loss, in share, of selecting each option.
    pub expected_loss: Vec<f64>,
    /// Choices observed.
    pub observations: f64,
    /// The premise under which the posterior is unchanged by optional stopping.
    pub premise: &'static str,
    /// The closed credible-interval and decision-reliability claim.
    pub guarantee: WithheldGuarantee,
}

/// Conjugate Dirichlet posterior from pooled counts.
///
/// # Errors
/// A prior shape below one, a count out of domain or above the bound, fewer than two options, or
/// cancellation.
pub fn choice_posterior(
    prior: &[f64],
    counts: &[f64],
    ctx: &ExecutionContext,
) -> Result<ChoicePosterior, AdaptiveInferenceError> {
    refuse_if_interrupted(ctx)?;
    let cancel = || ctx.cancellation.is_cancelled();
    let posterior =
        numerics::DirichletPosterior::from_counts(prior, counts).map_err(|e| numeric_error(&e))?;
    Ok(ChoicePosterior {
        alpha: posterior.alpha().to_vec(),
        mean: posterior.mean(),
        probability_best: posterior
            .best_option_probabilities(&cancel)
            .map_err(|e| numeric_error(&e))?,
        expected_loss: posterior.expected_losses(&cancel).map_err(|e| numeric_error(&e))?,
        observations: counts.iter().sum(),
        premise: CHOICE_POSTERIOR_PREMISE,
        guarantee: WithheldGuarantee::closed("choice_inference.posterior_interval_closed"),
    })
}

/// The decision a stopping rule reaches at one look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChoiceStoppingDecision {
    /// The option with the largest posterior probability of being best.
    pub best_option: usize,
    /// Its posterior probability of being best.
    pub probability_best: f64,
    /// Its posterior expected loss.
    pub expected_loss: f64,
    /// Whether the rule says stop.
    pub stop: bool,
    /// The closed claim about the rule's error rates.
    pub guarantee: WithheldGuarantee,
}

fn decide(
    probability_best: &[f64],
    expected_loss: &[f64],
    observed: f64,
    rule: ChoiceStoppingRule,
) -> Result<ChoiceStoppingDecision, AdaptiveInferenceError> {
    if !rule.is_valid() || probability_best.is_empty() {
        return Err(invalid(
            "choice_inference.input_invalid",
            "a stopping rule needs a probability threshold in (0.5, 1) or a positive loss threshold",
        ));
    }
    let (best_option, p) = probability_best
        .iter()
        .copied()
        .enumerate()
        .fold((0, f64::MIN), |a, (i, p)| if p > a.1 { (i, p) } else { a });
    Ok(ChoiceStoppingDecision {
        best_option,
        probability_best: p,
        expected_loss: expected_loss[best_option],
        stop: rule.assess(observed, probability_best, expected_loss).is_some(),
        guarantee: WithheldGuarantee::closed("choice_inference.stopping_reliability_closed"),
    })
}

/// Evaluate a stopping rule on a pooled posterior.
///
/// # Errors
/// A rule outside its domain.
pub fn choice_stopping_decision(
    posterior: &ChoicePosterior,
    rule: ChoiceStoppingRule,
) -> Result<ChoiceStoppingDecision, AdaptiveInferenceError> {
    decide(&posterior.probability_best, &posterior.expected_loss, posterior.observations, rule)
}

/// The hierarchical posterior of the population shares and the heterogeneity.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalChoiceResult {
    /// Quadrature summary of the posterior.
    pub summary: numerics::HierarchicalSummary,
    /// Respondents.
    pub respondents: usize,
    /// Choices observed.
    pub observations: f64,
    /// What the posterior assumes.
    pub premise: &'static str,
    /// The closed credible-interval and decision-reliability claim.
    pub guarantee: WithheldGuarantee,
}

/// Hierarchical posterior of respondents' count vectors.
///
/// # Errors
/// A prior outside its domain, more than three options, heterogeneity that repeated choices do not
/// identify, a size bound exceeded, a posterior the quadrature does not resolve, or cancellation.
pub fn hierarchical_choice_posterior(
    prior: &HierarchicalChoicePrior,
    respondents: &[Vec<u32>],
    ctx: &ExecutionContext,
) -> Result<HierarchicalChoiceResult, AdaptiveInferenceError> {
    refuse_if_interrupted(ctx)?;
    let cancel = || ctx.cancellation.is_cancelled();
    let posterior = numerics::HierarchicalChoicePosterior::new(prior, respondents)
        .map_err(|e| numeric_error(&e))?;
    let summary = posterior.summary(&cancel).map_err(|e| numeric_error(&e))?;
    Ok(HierarchicalChoiceResult {
        summary,
        respondents: posterior.respondents(),
        observations: respondents.iter().flatten().map(|c| f64::from(*c)).sum(),
        premise: HIERARCHICAL_PREMISE,
        guarantee: WithheldGuarantee::closed("choice_inference.hierarchical_interval_closed"),
    })
}

/// Evaluate a stopping rule on a hierarchical posterior.
///
/// # Errors
/// A rule outside its domain.
pub fn hierarchical_stopping_decision(
    result: &HierarchicalChoiceResult,
    rule: ChoiceStoppingRule,
) -> Result<ChoiceStoppingDecision, AdaptiveInferenceError> {
    decide(
        &result.summary.probability_best,
        &result.summary.expected_loss,
        result.observations,
        rule,
    )
}

/// The prior a choice contract's retained sources qualify, conflicts reported and not refused.
///
/// # Errors
/// Shapes or counts out of domain (a validated contract has none).
pub fn qualify_contract_prior(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
) -> Result<QualifiedChoicePrior, AdaptiveInferenceError> {
    let declaration = contract.declaration();
    let sources: Vec<ChoiceSourceCounts<'_>> = declaration
        .retained_sources
        .iter()
        .map(|s| ChoiceSourceCounts {
            label: &s.label,
            counts: &s.counts,
            discount: s.discount,
            transferred: s.population != declaration.population,
        })
        .collect();
    qualify_choice_prior(&declaration.initial_prior, &sources, ChoiceConflictPolicy::Report)
}

/// Choices of each option over the respondents enrolled at `look`.
///
/// # Errors
/// A look the contract does not declare.
pub fn contract_pooled_counts(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
    look: usize,
) -> Result<Vec<f64>, AdaptiveInferenceError> {
    let respondents = contract.respondents_at(look).map_err(|r| {
        AdaptiveInferenceError::new(
            r.code,
            "choice_inference.look_out_of_range",
            "the look is not one of the contract's interim analyses",
        )
    })?;
    let mut counts = vec![0.0; contract.options()];
    for respondent in respondents {
        for (total, c) in counts.iter_mut().zip(&respondent.counts) {
            *total += f64::from(*c);
        }
    }
    Ok(counts)
}

/// Qualify the same prior and screen each source against other sources and the current look.
/// The current look affects diagnostics only; it is not added to the prior or counted twice.
///
/// # Errors
/// An invalid look, unsupported source, or invalid numerical input.
pub fn qualify_contract_prior_at_look(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
    look: usize,
) -> Result<QualifiedChoicePrior, AdaptiveInferenceError> {
    let mut prior = qualify_contract_prior(contract)?;
    if prior.sources.is_empty() {
        return Ok(prior);
    }
    let d = contract.declaration();
    let observed = contract_pooled_counts(contract, look)?;
    let mut inputs: Vec<_> = d
        .retained_sources
        .iter()
        .map(|s| ChoiceSourceCounts {
            label: &s.label,
            counts: &s.counts,
            discount: s.discount,
            transferred: false,
        })
        .collect();
    inputs.push(ChoiceSourceCounts {
        label: "current look",
        counts: &observed,
        discount: 1.0,
        transferred: false,
    });
    let screened = qualify_choice_prior(&d.initial_prior, &inputs, ChoiceConflictPolicy::Report)?;
    for (source, diagnostic) in prior.sources.iter_mut().zip(screened.sources) {
        source.conflict_log_bayes_factor = diagnostic.conflict_log_bayes_factor;
        source.conflicting = diagnostic.conflicting;
    }
    Ok(prior)
}

/// The pooled Dirichlet posterior of a categorical-choice contract at `look`, on the prior its
/// retained sources qualify.
///
/// # Errors
/// A repeated-choice contract (its posterior is hierarchical), a look the contract does not
/// declare, or a numerical refusal.
pub fn contract_posterior(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
    look: usize,
    ctx: &ExecutionContext,
) -> Result<ChoicePosterior, AdaptiveInferenceError> {
    if contract.cell().outcome
        != antecedent_contract::adaptive_design::OutcomeStructure::CategoricalChoice
    {
        return Err(invalid(
            "choice_inference.route_outcome_mismatch",
            "a pooled posterior is for one choice per respondent",
        ));
    }
    let prior = qualify_contract_prior(contract)?;
    choice_posterior(&prior.alpha, &contract_pooled_counts(contract, look)?, ctx)
}

/// The stopping rule a choice contract declares, as the rule the procedure law evaluates.
///
/// # Errors
/// A contract whose rule is not a posterior rule.
pub fn contract_stopping_rule(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
) -> Result<ChoiceStoppingRule, AdaptiveInferenceError> {
    use antecedent_contract::adaptive_design::StoppingSpec;
    match contract.declaration().stopping_rule.map(|r| r.spec) {
        Some(StoppingSpec::PosteriorProbabilityBest { threshold }) => Ok(ChoiceStoppingRule {
            min_probability_best: Some(threshold),
            max_expected_loss: None,
            min_observations: 0,
        }),
        Some(StoppingSpec::PosteriorExpectedLoss { loss_threshold }) => Ok(ChoiceStoppingRule {
            min_probability_best: None,
            max_expected_loss: Some(loss_threshold),
            min_observations: 0,
        }),
        _ => Err(invalid(
            "choice_inference.rule_not_posterior",
            "the contract's stopping rule is not a posterior rule",
        )),
    }
}

/// The sequential procedure a categorical-choice contract ran under: batches of respondents at
/// its looks, its qualified prior and its stopping rule. The contract's one-choice-per-respondent
/// rule makes a batch of respondents a batch of choices.
///
/// # Errors
/// A repeated-choice contract (its law is hierarchical, not this one) or a rule that is not a
/// posterior rule.
pub fn contract_procedure_spec(
    contract: &antecedent_contract::adaptive_design::ChoiceDesignContract,
) -> Result<ChoiceProcedureSpec, AdaptiveInferenceError> {
    use antecedent_contract::adaptive_design::OutcomeStructure;
    if contract.cell().outcome != OutcomeStructure::CategoricalChoice {
        return Err(invalid(
            "choice_inference.route_outcome_mismatch",
            "the enumerated procedure law is the pooled Dirichlet procedure's; a repeated-choice study has a hierarchical posterior",
        ));
    }
    let looks = contract.declaration().looks.as_ref().expect("validated contract");
    let mut previous = 0;
    let batches = looks
        .iter()
        .map(|l| {
            let batch = l.respondents - previous;
            previous = l.respondents;
            batch
        })
        .collect();
    Ok(ChoiceProcedureSpec {
        prior: qualify_contract_prior(contract)?.alpha,
        batches,
        rule: contract_stopping_rule(contract)?,
    })
}

/// The exact law of a sequential choice procedure under a declared kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceOperatingCharacteristics {
    /// Probability that the procedure ends selecting each option.
    pub selection: Vec<f64>,
    /// Probability that it ends without selecting an option.
    pub undecided: f64,
    /// Expected respondents enrolled.
    pub expected_enrolled: f64,
    /// Probability that the rule stops at each look.
    pub rule_stop: Vec<f64>,
    /// Probability that the planned horizon is reached.
    pub horizon_probability: f64,
    /// Work units the enumeration used.
    pub work_units: u64,
    /// What these probabilities are.
    pub statement: &'static str,
    /// The closed claim that these are error rates of the procedure for any other truth.
    pub guarantee: WithheldGuarantee,
}

/// What an operating characteristic is.
pub const OPERATING_CHARACTERISTICS_STATEMENT: &str = "exact probabilities of the declared procedure under the declared kernel only (a fixed truth, or the declared Dirichlet prior predictive); not a bound over other truths and not a coverage statement";

/// The exact operating characteristics of a sequential choice procedure by dynamic programming
/// over the observed counts.
///
/// # Errors
/// More than four options, a rule or kernel outside its domain, a size bound exceeded, or
/// cancellation.
pub fn choice_operating_characteristics(
    spec: &ChoiceProcedureSpec,
    kernel: &ChoiceKernel,
    ctx: &ExecutionContext,
) -> Result<ChoiceOperatingCharacteristics, AdaptiveInferenceError> {
    refuse_if_interrupted(ctx)?;
    let law = numerics::enumerate_choice_law(spec, kernel, &|| ctx.cancellation.is_cancelled())
        .map_err(|e| numeric_error(&e))?;
    let (selection, undecided) = law.selection_probabilities(spec.prior.len());
    Ok(ChoiceOperatingCharacteristics {
        selection,
        undecided,
        expected_enrolled: law.expected_enrolled(),
        rule_stop: law.rule_stop_probabilities(),
        horizon_probability: law.horizon_probability(),
        work_units: law.work_units,
        statement: OPERATING_CHARACTERISTICS_STATEMENT,
        guarantee: WithheldGuarantee::closed(
            "choice_inference.operating_characteristics_conditional",
        ),
    })
}

/// Query axis of every choice-study record.
pub const QUERY: &str = "ChoiceExperimentShare";
/// Estimator of the pooled Dirichlet posterior of one option's share.
pub const DIRICHLET_ESTIMATOR: &str = "choice.dirichlet_multinomial.posterior";
/// Estimator of the hierarchical posterior of one option's population share.
pub const HIERARCHICAL_ESTIMATOR: &str = "choice.hierarchical_dirichlet_multinomial.posterior";
/// The functional both estimators report: the first option's share.
pub const FIRST_OPTION_SHARE_FUNCTIONAL: &str = "option_share:first";
/// One choice per respondent, one final analysis.
pub const POOLED_DEPENDENCE: &str = "one_choice_per_respondent_final_analysis";
/// Repeated choices per respondent, one final analysis.
pub const HIERARCHICAL_DEPENDENCE: &str = "repeated_choices_per_respondent_final_analysis";
/// Prior of the pooled posterior's measured design.
pub const DIRICHLET_PRIOR: &str = "dirichlet_multinomial.uniform_prior";
/// Prior of the hierarchical posterior's measured design.
pub const HIERARCHICAL_PRIOR: &str = "hierarchical_dirichlet_multinomial.declared_hyperprior";
/// Interval method of every measured choice interval: equal-tailed posterior quantiles.
pub const CHOICE_INTERVAL_METHOD: &str = "posterior_quantile";
/// Nominal level of every measured choice interval.
pub const MEASURED_LEVEL: f64 = 0.95;

/// The calibration key of one measured choice interval. `rows` is the planned number of
/// respondents. The key carries no authority: an interval is licensed only when it resolves to a
/// currently attesting coverage record.
#[must_use]
pub fn calibration_basis(
    estimator: &str,
    functional: &str,
    dependence: &str,
    prior: &str,
    rows: u64,
) -> antecedent_core::CalibrationBasis {
    let a = |text: &str| std::sync::Arc::<str>::from(text);
    antecedent_core::CalibrationBasis::new(
        [
            a(QUERY),
            a("NotApplicable"),
            a("fixed"),
            a("choice_study_history"),
            a("Bayesian"),
            a(estimator),
            a(CHOICE_INTERVAL_METHOD),
            a(""),
            a(dependence),
            a(prior),
            a(functional),
        ],
        MEASURED_LEVEL,
        a("point"),
        rows,
        None,
        None,
        0.0,
    )
    .with_validation("not_applicable")
}

/// Equal-tailed credible interval of one option's pooled share: the Beta marginal of the
/// Dirichlet posterior.
///
/// # Errors
/// An option out of range or a level outside `(0, 1)`.
pub fn dirichlet_share_interval(
    alpha: &[f64],
    option: usize,
    level: f64,
) -> Result<(f64, f64), AdaptiveInferenceError> {
    let a = *alpha
        .get(option)
        .ok_or_else(|| invalid("choice_inference.input_invalid", "the option is not declared"))?;
    let rest: f64 = alpha.iter().sum::<f64>() - a;
    crate::adaptive_measured_interval::beta_credible_interval(a, rest, level).ok_or_else(|| {
        invalid("choice_inference.input_invalid", "the share interval is not defined")
    })
}

/// Equal-tailed credible interval of one option's population share under the hierarchical
/// posterior, by the quadrature's marginal distribution function.
///
/// # Errors
/// An option or level out of range, an unresolved posterior or cancellation.
pub fn hierarchical_share_interval(
    prior: &HierarchicalChoicePrior,
    respondents: &[Vec<u32>],
    option: usize,
    level: f64,
    ctx: &ExecutionContext,
) -> Result<(f64, f64), AdaptiveInferenceError> {
    refuse_if_interrupted(ctx)?;
    let posterior = numerics::HierarchicalChoicePosterior::new(prior, respondents)
        .map_err(|e| numeric_error(&e))?;
    posterior
        .share_interval(option, level, &|| ctx.cancellation.is_cancelled())
        .map_err(|e| numeric_error(&e))
}

impl QualifiedChoicePrior {
    /// Whether any source is flagged as conflicting.
    #[must_use]
    pub fn has_conflict(&self) -> bool {
        self.sources.iter().any(|s| s.conflicting)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(1)
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn a_discounted_source_adds_its_discounted_counts_and_a_tied_source_is_not_flagged() {
        let a = ChoiceSourceCounts {
            label: "a",
            counts: &[10.0, 5.0, 5.0],
            discount: 0.5,
            transferred: false,
        };
        let prior = qualify_choice_prior(&[1.0; 3], &[a], ChoiceConflictPolicy::Refuse).unwrap();
        assert_eq!(prior.alpha, vec![6.0, 3.5, 3.5]);
        assert_eq!(prior.borrowed_observations, 10.0);
        assert!(!prior.has_conflict());
    }

    #[test]
    fn a_source_against_the_rest_is_flagged_and_refused_under_the_strict_policy() {
        let agree = ChoiceSourceCounts {
            label: "agree",
            counts: &[80.0, 10.0, 10.0],
            discount: 1.0,
            transferred: false,
        };
        let oppose = ChoiceSourceCounts {
            label: "oppose",
            counts: &[10.0, 80.0, 10.0],
            discount: 1.0,
            transferred: false,
        };
        let report =
            qualify_choice_prior(&[1.0; 3], &[agree, oppose], ChoiceConflictPolicy::Report)
                .unwrap();
        assert!(report.has_conflict());
        let refused =
            qualify_choice_prior(&[1.0; 3], &[agree, oppose], ChoiceConflictPolicy::Refuse)
                .unwrap_err();
        assert_eq!(refused.code, "choice_evidence_conflict");
    }

    #[test]
    fn the_posterior_reduces_to_the_beta_for_two_options_and_withholds_intervals() {
        let p = choice_posterior(&[1.0, 1.0], &[7.0, 3.0], &ctx()).unwrap();
        assert_eq!(p.alpha, vec![8.0, 4.0]);
        assert!((p.probability_best.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(p.guarantee.code, "cell_not_licensed");
        assert!(p.guarantee.refusal("a credible interval").message.contains("not licensed"));
    }

    #[test]
    fn operating_characteristics_are_the_stats_law_with_the_guarantee_closed() {
        let spec = ChoiceProcedureSpec {
            prior: vec![1.0; 2],
            batches: vec![4; 4],
            rule: ChoiceStoppingRule {
                min_probability_best: Some(0.85),
                max_expected_loss: None,
                min_observations: 4,
            },
        };
        let oc = choice_operating_characteristics(
            &spec,
            &ChoiceKernel::Fixed { theta: vec![0.7, 0.3] },
            &ctx(),
        )
        .unwrap();
        let mass: f64 = oc.selection.iter().sum::<f64>() + oc.undecided;
        assert!((mass - 1.0).abs() < 1e-12);
        assert!(oc.selection[0] > oc.selection[1]);
        assert_eq!(oc.guarantee.code, "cell_not_licensed");
    }

    #[test]
    fn the_stopping_decision_is_the_stats_rule() {
        let p = choice_posterior(&[1.0; 3], &[60.0, 10.0, 10.0], &ctx()).unwrap();
        let rule = ChoiceStoppingRule {
            min_probability_best: Some(0.95),
            max_expected_loss: None,
            min_observations: 0,
        };
        assert!(choice_stopping_decision(&p, rule).unwrap().stop);
        let strict = ChoiceStoppingRule {
            min_probability_best: Some(0.95),
            max_expected_loss: None,
            min_observations: 1000,
        };
        assert!(!choice_stopping_decision(&p, strict).unwrap().stop);
        let bad = ChoiceStoppingRule {
            min_probability_best: Some(0.2),
            max_expected_loss: None,
            min_observations: 0,
        };
        assert!(choice_stopping_decision(&p, bad).is_err());
    }
}
