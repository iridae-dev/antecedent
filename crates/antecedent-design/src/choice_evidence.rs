//! Licensed evidence-to-decision lifecycle for repeated categorical choices.
//!
//! Qualified prior evidence, supported observations, licensed inference, decision evaluation and
//! next-study ranking, each stage carrying its meaning, dependence and support conditions:
//!
//! * [`qualify_prior`] turns retained choice evidence into a Dirichlet prior only when the
//!   parameterization (ordered option set), the population and the meaning of the choices match
//!   the target, or a transfer has been declared with a discount. Two uses of the same
//!   observation identity are refused as double counting, and heterogeneity between sources is
//!   reported (or refused) rather than averaged away.
//! * [`update`] conditions on new observations, refusing observations that were already used in
//!   a source, and reports the exact posterior functionals of
//!   [`antecedent_stats::adaptive_inference::DirichletPosterior`].
//! * [`SequentialRule`] evaluates a look at a running study: stop and select, stop for the budget,
//!   or continue with a probability-matching allocation. It computes; it does not assign, deliver
//!   or monitor.
//! * [`rank_studies`] ranks candidate additional studies by the exact expected value of sample
//!   information (Dirichlet-multinomial enumeration, Monte Carlo with a standard error beyond the
//!   enumeration cap) net of declared cost.
//!
//! Nothing here claims calibrated frequentist error rates, identifies behaviour from stated
//! preferences, or transports a distribution across populations without a declared discount.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::CausalRng;
use antecedent_kernels::rng::sample_gamma;
use antecedent_stats::adaptive_inference::{DirichletPosterior, MAX_CHOICE_COUNT};
use antecedent_stats::special::ln_gamma;

/// Largest number of retained sources one prior combines.
pub const MAX_SOURCES: usize = 16;

/// Largest outcome space the exact EVSI enumerates.
pub const MAX_EXACT_COMPOSITIONS: u64 = 250_000;

/// Monte Carlo draws used beyond the enumeration cap.
pub const MONTE_CARLO_DRAWS: usize = 20_000;

/// Largest future study the Monte Carlo evaluates.
pub const MAX_MONTE_CARLO_STUDY: u32 = 10_000;

/// Log Bayes factor above which a source is flagged as conflicting with the rest (`ln 10`).
pub const CONFLICT_LOG_BAYES_FACTOR: f64 = 2.302_585_092_994_046;

/// What the posterior shares are shares of.
pub const MEANING_STATEMENT: &str = "shares of repeated categorical choices as declared; stated preferences do not identify behavioural or causal outcomes";

/// What a stopping decision does and does not guarantee.
pub const STOPPING_STATEMENT: &str = "posterior stopping criterion: the posterior is unchanged by optional stopping; no frequentist error rate or calibrated coverage is claimed";

/// What a conflict diagnostic is.
pub const CONFLICT_STATEMENT: &str = "prior-predictive log Bayes factor of each source against a vague prior versus the remaining evidence; a screening diagnostic with unmeasured calibration";

/// What a choice means in the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceMeaning {
    /// Choices stated in a survey or experiment of hypothetical options.
    StatedPreference,
    /// Choices observed as behaviour in the target setting.
    ObservedBehavior,
}

/// The parameterization of a choice law: ordered options, the population choosing, the meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceFrame {
    /// Option labels in parameter order.
    pub options: Vec<String>,
    /// Population the choices describe.
    pub population: String,
    /// Meaning of the choices.
    pub meaning: ChoiceMeaning,
}

/// Why a request is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ChoiceRefusal {
    /// A declaration is malformed.
    #[error("invalid declaration: {0}")]
    Invalid(String),
    /// A source's ordered option set differs from the target's.
    #[error("source `{0}` has a different option parameterization than the target")]
    ParameterizationMismatch(String),
    /// A source's choices mean something else than the target's.
    #[error("source `{0}` has a different choice meaning than the target")]
    MeaningMismatch(String),
    /// A source describes another population and no transfer was declared.
    #[error("source `{0}` describes another population and no transfer was declared")]
    PopulationMismatch(String),
    /// The same observations would be counted twice.
    #[error("observation identity `{0}` would be counted more than once")]
    OverlappingObservations(String),
    /// A source conflicts with the rest of the evidence and the policy refuses conflict.
    #[error("source `{0}` conflicts with the remaining evidence (log Bayes factor {1:.3})")]
    Conflict(String, f64),
    /// The inference family refused the shape or size.
    #[error("inference refused: {0}")]
    Inference(String),
}

impl From<antecedent_stats::adaptive_inference::AdaptiveInferenceError> for ChoiceRefusal {
    fn from(error: antecedent_stats::adaptive_inference::AdaptiveInferenceError) -> Self {
        Self::Inference(error.to_string())
    }
}

impl ChoiceRefusal {
    /// Registered refusal code (`parity/reason_codes.toml`).
    #[must_use]
    pub fn registered_code(&self) -> &'static str {
        match self {
            Self::Invalid(_) | Self::Inference(_) => "choice_evidence_invalid",
            Self::ParameterizationMismatch(_)
            | Self::MeaningMismatch(_)
            | Self::PopulationMismatch(_) => "choice_evidence_scope_mismatch",
            Self::OverlappingObservations(_) => "choice_evidence_overlap",
            Self::Conflict(..) => "choice_evidence_conflict",
        }
    }

    /// Stable machine-readable reason.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "choice_evidence.invalid_declaration",
            Self::ParameterizationMismatch(_) => "choice_evidence.parameterization_mismatch",
            Self::MeaningMismatch(_) => "choice_evidence.meaning_mismatch",
            Self::PopulationMismatch(_) => "choice_evidence.population_mismatch",
            Self::OverlappingObservations(_) => "choice_evidence.overlapping_observations",
            Self::Conflict(..) => "choice_evidence.conflict",
            Self::Inference(_) => "choice_evidence.inference_refused",
        }
    }

    /// The next thing to try; every refusal names one.
    #[must_use]
    pub fn next_attempt(&self) -> &'static str {
        match self {
            Self::Invalid(_) | Self::Inference(_) => "correct the declaration named in the message",
            Self::ParameterizationMismatch(_) => {
                "re-express the source over the target's ordered option set, or omit it"
            }
            Self::MeaningMismatch(_) => {
                "use evidence of the same choice meaning; stated and observed choices are not exchangeable"
            }
            Self::PopulationMismatch(_) => {
                "declare a transfer rationale and a discount, or use evidence from the target population"
            }
            Self::OverlappingObservations(_) => {
                "count each observation set once: drop the repeated source or the repeated observations"
            }
            Self::Conflict(..) => {
                "inspect the source, lower its discount, or accept the conflict under the report policy"
            }
        }
    }
}

/// Retained choice evidence: the counts a source contributed, not its posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedChoiceEvidence {
    /// Source label.
    pub label: String,
    /// What the counts are counts of.
    pub frame: ChoiceFrame,
    /// Choices per option, in the frame's order.
    pub counts: Vec<f64>,
    /// Identity of the observation set the counts came from (dependence is judged on this).
    pub observation_id: String,
}

impl RetainedChoiceEvidence {
    /// Retain the data contribution of a posterior by subtracting the prior it started from.
    ///
    /// # Errors
    /// A prior that does not match the posterior or exceeds it.
    pub fn from_posterior(
        label: impl Into<String>,
        frame: ChoiceFrame,
        posterior: &DirichletPosterior,
        prior: &[f64],
        observation_id: impl Into<String>,
    ) -> Result<Self, ChoiceRefusal> {
        if posterior.options() != prior.len() {
            return Err(ChoiceRefusal::ParameterizationMismatch("posterior and prior".into()));
        }
        let counts: Vec<f64> =
            posterior.alpha().iter().zip(prior).map(|(a, p)| (a - p).round()).collect();
        if counts.iter().any(|c| *c < 0.0) {
            return Err(ChoiceRefusal::Invalid("the prior exceeds the retained posterior".into()));
        }
        Ok(Self { label: label.into(), frame, counts, observation_id: observation_id.into() })
    }
}

/// One use of retained evidence in a prior.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceUse {
    /// The evidence.
    pub evidence: RetainedChoiceEvidence,
    /// Power-prior discount in `[0, 1]`.
    pub discount: f64,
    /// Rationale for transferring from another population; required when populations differ.
    pub transfer_rationale: Option<String>,
}

/// How conflict between sources is handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictPolicy {
    /// Report the diagnostic and keep the evidence.
    Report,
    /// Refuse the prior when a source is flagged.
    Refuse,
}

/// A source as used.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedSource {
    /// Source label.
    pub label: String,
    /// Observation identity.
    pub observation_id: String,
    /// Discount applied.
    pub discount: f64,
    /// Whether the population differed from the target's.
    pub transferred: bool,
    /// Prior-predictive log Bayes factor against the remaining evidence.
    pub conflict_log_bayes_factor: f64,
    /// Whether the source is flagged.
    pub conflicting: bool,
}

/// A prior admitted for the target frame.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedPrior {
    /// The target frame.
    pub frame: ChoiceFrame,
    /// Dirichlet shapes.
    pub alpha: Vec<f64>,
    /// Observations the sources are worth after discounting.
    pub borrowed_observations: f64,
    /// Sources as used.
    pub sources: Vec<QualifiedSource>,
    /// Statements the prior carries.
    pub statements: Vec<&'static str>,
}

impl QualifiedPrior {
    /// Observation identities that have entered the prior.
    #[must_use]
    pub fn observation_ids(&self) -> BTreeSet<&str> {
        self.sources.iter().map(|s| s.observation_id.as_str()).collect()
    }

    /// Whether any source is flagged as conflicting.
    #[must_use]
    pub fn has_conflict(&self) -> bool {
        self.sources.iter().any(|s| s.conflicting)
    }

    /// Stable identity of the prior.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(b"choice_prior.v1");
        for o in &self.frame.options {
            h.update(o.as_bytes());
            h.update(&[0]);
        }
        h.update(self.frame.population.as_bytes());
        for a in &self.alpha {
            h.update(&a.to_bits().to_le_bytes());
        }
        for s in &self.sources {
            h.update(s.observation_id.as_bytes());
            h.update(&s.discount.to_bits().to_le_bytes());
        }
        h.finalize().to_hex().to_string()
    }
}

fn validate_frame(frame: &ChoiceFrame) -> Result<(), ChoiceRefusal> {
    let unique: BTreeSet<_> = frame.options.iter().collect();
    if frame.options.len() < 2
        || unique.len() != frame.options.len()
        || frame.options.iter().any(|o| o.trim().is_empty())
        || frame.population.trim().is_empty()
    {
        return Err(ChoiceRefusal::Invalid(
            "a frame needs two or more distinct options and a population".into(),
        ));
    }
    Ok(())
}

/// Qualify retained evidence as a prior for `target`.
///
/// # Errors
/// A malformed frame or prior, a source with a different parameterization or meaning, a source of
/// another population without a declared transfer, a repeated observation identity, or (under
/// [`ConflictPolicy::Refuse`]) a conflicting source.
pub fn qualify_prior(
    target: &ChoiceFrame,
    initial: &[f64],
    uses: &[SourceUse],
    policy: ConflictPolicy,
) -> Result<QualifiedPrior, ChoiceRefusal> {
    validate_frame(target)?;
    if initial.len() != target.options.len() || uses.len() > MAX_SOURCES {
        return Err(ChoiceRefusal::Invalid(
            "the initial prior must match the options and at most 16 sources".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut alpha = initial.to_vec();
    let mut borrowed = 0.0;
    for u in uses {
        let e = &u.evidence;
        if e.frame.options != target.options || e.counts.len() != target.options.len() {
            return Err(ChoiceRefusal::ParameterizationMismatch(e.label.clone()));
        }
        if e.frame.meaning != target.meaning {
            return Err(ChoiceRefusal::MeaningMismatch(e.label.clone()));
        }
        if e.frame.population != target.population
            && u.transfer_rationale.as_deref().is_none_or(|r| r.trim().is_empty())
        {
            return Err(ChoiceRefusal::PopulationMismatch(e.label.clone()));
        }
        if !(u.discount.is_finite() && (0.0..=1.0).contains(&u.discount))
            || e.counts.iter().any(|c| !(c.is_finite() && *c >= 0.0))
            || e.observation_id.trim().is_empty()
        {
            return Err(ChoiceRefusal::Invalid(format!("source `{}` is malformed", e.label)));
        }
        if !seen.insert(e.observation_id.clone()) {
            return Err(ChoiceRefusal::OverlappingObservations(e.observation_id.clone()));
        }
        for (a, c) in alpha.iter_mut().zip(&e.counts) {
            *a += u.discount * c;
        }
        borrowed += u.discount * e.counts.iter().sum::<f64>();
    }
    DirichletPosterior::new(alpha.clone())?;
    let mut sources = Vec::with_capacity(uses.len());
    for (i, u) in uses.iter().enumerate() {
        // Evidence from every other source, discounted, on top of the initial prior.
        let mut rest = initial.to_vec();
        for (j, v) in uses.iter().enumerate() {
            if j != i {
                for (a, c) in rest.iter_mut().zip(&v.evidence.counts) {
                    *a += v.discount * c;
                }
            }
        }
        let vague = DirichletPosterior::log_marginal_likelihood(initial, &u.evidence.counts)?;
        let against_rest = DirichletPosterior::log_marginal_likelihood(&rest, &u.evidence.counts)?;
        let log_bf = vague - against_rest;
        let flagged = uses.len() > 1 && log_bf > CONFLICT_LOG_BAYES_FACTOR;
        if flagged && policy == ConflictPolicy::Refuse {
            return Err(ChoiceRefusal::Conflict(u.evidence.label.clone(), log_bf));
        }
        sources.push(QualifiedSource {
            label: u.evidence.label.clone(),
            observation_id: u.evidence.observation_id.clone(),
            discount: u.discount,
            transferred: u.evidence.frame.population != target.population,
            conflict_log_bayes_factor: if uses.len() > 1 { log_bf } else { 0.0 },
            conflicting: flagged,
        });
    }
    Ok(QualifiedPrior {
        frame: target.clone(),
        alpha,
        borrowed_observations: borrowed,
        sources,
        statements: vec![MEANING_STATEMENT, CONFLICT_STATEMENT],
    })
}

/// New observations of the target frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceObservations {
    /// Choices per option.
    pub counts: Vec<f64>,
    /// Identity of the observation set.
    pub observation_id: String,
}

/// The licensed inference after an update.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceInference {
    /// Frame.
    pub frame: ChoiceFrame,
    /// Posterior.
    pub posterior: DirichletPosterior,
    /// Posterior mean share per option.
    pub mean: Vec<f64>,
    /// Probability each option has the largest share.
    pub probability_best: Vec<f64>,
    /// Expected loss, in share, of choosing each option.
    pub expected_loss: Vec<f64>,
    /// Observations in this update.
    pub observations: f64,
    /// Identity of prior plus observations.
    pub lineage: String,
    /// Statements the inference carries.
    pub statements: Vec<&'static str>,
}

/// Condition a qualified prior on observations.
///
/// # Errors
/// Observations that mismatch the frame, repeat an identity already in the prior, or exceed the
/// supported size.
pub fn update(
    prior: &QualifiedPrior,
    observations: &ChoiceObservations,
    cancelled: &dyn Fn() -> bool,
) -> Result<ChoiceInference, ChoiceRefusal> {
    if observations.observation_id.trim().is_empty() {
        return Err(ChoiceRefusal::Invalid("observations need an identity".into()));
    }
    if prior.observation_ids().contains(observations.observation_id.as_str()) {
        return Err(ChoiceRefusal::OverlappingObservations(observations.observation_id.clone()));
    }
    if observations.counts.len() != prior.alpha.len() {
        return Err(ChoiceRefusal::ParameterizationMismatch("observations".into()));
    }
    let posterior = DirichletPosterior::from_counts(&prior.alpha, &observations.counts)?;
    let mut h = blake3::Hasher::new();
    h.update(prior.digest().as_bytes());
    h.update(observations.observation_id.as_bytes());
    for c in &observations.counts {
        h.update(&c.to_bits().to_le_bytes());
    }
    Ok(ChoiceInference {
        frame: prior.frame.clone(),
        mean: posterior.mean(),
        probability_best: posterior.best_option_probabilities(cancelled)?,
        expected_loss: posterior.expected_losses(cancelled)?,
        observations: observations.counts.iter().sum(),
        lineage: h.finalize().to_hex().to_string(),
        posterior,
        statements: vec![MEANING_STATEMENT],
    })
}

/// The outcome of one look.
#[derive(Clone, Debug, PartialEq)]
pub enum LookOutcome {
    /// Stop; the declared criterion selects this option.
    Select {
        /// Selected option index.
        option: usize,
    },
    /// Stop; the budget is spent without the criterion being met.
    BudgetExhausted,
    /// Continue; the allocation for the next observations (a computation, not an assignment).
    Continue {
        /// Probability-matching shares with the floor applied, summing to one.
        allocation: Vec<f64>,
    },
}

/// A bounded Bayesian sequential rule.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialRule {
    /// Probability of being best required to select.
    pub min_probability_best: f64,
    /// Largest expected loss (in share) the selected option may carry.
    pub max_expected_loss: f64,
    /// Observations required before any stop.
    pub min_observations: f64,
    /// Observations after which the study stops regardless.
    pub max_observations: f64,
    /// Floor on every option's allocation share.
    pub allocation_floor: f64,
}

impl SequentialRule {
    /// Evaluate one look.
    ///
    /// # Errors
    /// A rule outside its domain.
    pub fn evaluate(&self, inference: &ChoiceInference) -> Result<LookOutcome, ChoiceRefusal> {
        let k = inference.probability_best.len();
        if !(self.min_probability_best > 0.5
            && self.min_probability_best < 1.0
            && self.max_expected_loss.is_finite()
            && self.max_expected_loss >= 0.0
            && self.min_observations >= 0.0
            && self.max_observations >= self.min_observations
            && self.max_observations <= MAX_CHOICE_COUNT as f64
            && self.allocation_floor >= 0.0
            && self.allocation_floor * (k as f64) < 1.0)
        {
            return Err(ChoiceRefusal::Invalid("the sequential rule is outside its domain".into()));
        }
        let n = inference.observations;
        let (best, p) = inference
            .probability_best
            .iter()
            .copied()
            .enumerate()
            .fold((0, f64::MIN), |a, (i, p)| if p > a.1 { (i, p) } else { a });
        if n >= self.min_observations
            && p >= self.min_probability_best
            && inference.expected_loss[best] <= self.max_expected_loss
        {
            return Ok(LookOutcome::Select { option: best });
        }
        if n >= self.max_observations {
            return Ok(LookOutcome::BudgetExhausted);
        }
        let free = 1.0 - self.allocation_floor * k as f64;
        Ok(LookOutcome::Continue {
            allocation: inference
                .probability_best
                .iter()
                .map(|p| self.allocation_floor + free * p)
                .collect(),
        })
    }
}

/// A candidate additional study.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyCandidate {
    /// Label.
    pub label: String,
    /// Choices it would observe.
    pub observations: u32,
    /// Cost in the same units as the option values.
    pub cost: f64,
}

/// How an EVSI was computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvsiMethod {
    /// Sum over every possible outcome.
    Exact {
        /// Outcomes summed.
        compositions: u64,
    },
    /// Monte Carlo over the predictive distribution.
    MonteCarlo {
        /// Draws.
        draws: usize,
    },
}

/// A ranked candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyRank {
    /// The candidate.
    pub candidate: StudyCandidate,
    /// Expected value of sample information per decision, in option-value units.
    pub evsi: f64,
    /// Monte Carlo standard error; zero when exact.
    pub standard_error: f64,
    /// Method.
    pub method: EvsiMethod,
    /// EVSI net of cost.
    pub net_value: f64,
    /// Whether the order against the next candidate is within combined standard errors.
    pub rank_uncertain: bool,
}

fn compositions(n: u32, k: usize) -> u64 {
    // C(n + k - 1, k - 1), saturating.
    let mut c: u128 = 1;
    for i in 1..k as u128 {
        c = c.saturating_mul(u128::from(n) + i) / i;
        if c > u128::from(u64::MAX) {
            return u64::MAX;
        }
    }
    c as u64
}

fn best_value(values: &[f64], alpha: &[f64], counts: &[f64], total: f64) -> f64 {
    values
        .iter()
        .zip(alpha.iter().zip(counts))
        .map(|(v, (a, c))| v * (a + c) / total)
        .fold(f64::MIN, f64::max)
}

fn exact_evsi(alpha: &[f64], values: &[f64], n: u32) -> f64 {
    let a0: f64 = alpha.iter().sum();
    let nf = f64::from(n);
    let base = ln_gamma(nf + 1.0) + ln_gamma(a0)
        - ln_gamma(a0 + nf)
        - alpha.iter().map(|a| ln_gamma(*a)).sum::<f64>();
    let mut counts = vec![0.0; alpha.len()];
    let mut acc = 0.0;
    fn go(
        i: usize,
        left: u32,
        counts: &mut [f64],
        alpha: &[f64],
        values: &[f64],
        base: f64,
        post_total: f64,
        acc: &mut f64,
    ) {
        if i + 1 == counts.len() {
            counts[i] = f64::from(left);
            let ln_p = base
                + counts
                    .iter()
                    .zip(alpha)
                    .map(|(c, a)| ln_gamma(a + c) - ln_gamma(c + 1.0))
                    .sum::<f64>();
            *acc += ln_p.exp() * best_value(values, alpha, counts, post_total);
            return;
        }
        for c in 0..=left {
            counts[i] = f64::from(c);
            go(i + 1, left - c, counts, alpha, values, base, post_total, acc);
        }
    }
    go(0, n, &mut counts, alpha, values, base, a0 + nf, &mut acc);
    acc
}

fn monte_carlo_evsi(alpha: &[f64], values: &[f64], n: u32, seed: u64) -> (f64, f64) {
    let a0: f64 = alpha.iter().sum();
    let total = a0 + f64::from(n);
    let mut rng = CausalRng::from_seed(seed);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    let mut theta = vec![0.0; alpha.len()];
    let mut counts = vec![0.0; alpha.len()];
    for _ in 0..MONTE_CARLO_DRAWS {
        let mut s = 0.0;
        for (t, a) in theta.iter_mut().zip(alpha) {
            *t = sample_gamma(*a, 1.0, &mut rng);
            s += *t;
        }
        counts.iter_mut().for_each(|c| *c = 0.0);
        for _ in 0..n {
            let mut u = rng.next_f64() * s;
            let mut pick = theta.len() - 1;
            for (j, t) in theta.iter().enumerate() {
                if u < *t {
                    pick = j;
                    break;
                }
                u -= t;
            }
            counts[pick] += 1.0;
        }
        let v = best_value(values, alpha, &counts, total);
        sum += v;
        sum_sq += v * v;
    }
    let m = MONTE_CARLO_DRAWS as f64;
    let mean = sum / m;
    let var = ((sum_sq / m) - mean * mean).max(0.0) * m / (m - 1.0);
    (mean, (var / m).sqrt())
}

/// Rank candidate studies by the expected value of sample information net of cost.
///
/// The terminal decision picks the option with the largest `value_k * E[share_k]`. The
/// application owns the candidate designs, the acquisition of observations and the authorization
/// of fieldwork; this computes their decision value under the posterior.
///
/// # Errors
/// Values or candidates outside their domain.
pub fn rank_studies(
    inference: &ChoiceInference,
    values: &[f64],
    candidates: &[StudyCandidate],
    seed: u64,
) -> Result<Vec<StudyRank>, ChoiceRefusal> {
    let alpha = inference.posterior.alpha();
    if values.len() != alpha.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(ChoiceRefusal::Invalid("one finite value per option is required".into()));
    }
    if candidates.is_empty()
        || candidates.iter().any(|c| c.observations == 0 || !c.cost.is_finite() || c.cost < 0.0)
    {
        return Err(ChoiceRefusal::Invalid(
            "candidates need a positive observation count and a finite nonnegative cost".into(),
        ));
    }
    let a0: f64 = alpha.iter().sum();
    let zero = vec![0.0; alpha.len()];
    let now = best_value(values, alpha, &zero, a0);
    let mut rows: Vec<StudyRank> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let comps = compositions(c.observations, alpha.len());
        let (value, se, method) = if comps <= MAX_EXACT_COMPOSITIONS {
            (
                exact_evsi(alpha, values, c.observations),
                0.0,
                EvsiMethod::Exact { compositions: comps },
            )
        } else if c.observations <= MAX_MONTE_CARLO_STUDY {
            let (v, se) = monte_carlo_evsi(alpha, values, c.observations, seed);
            (v, se, EvsiMethod::MonteCarlo { draws: MONTE_CARLO_DRAWS })
        } else {
            return Err(ChoiceRefusal::Invalid(format!(
                "candidate `{}` exceeds the supported study size",
                c.label
            )));
        };
        let evsi = (value - now).max(0.0);
        rows.push(StudyRank {
            candidate: c.clone(),
            evsi,
            standard_error: se,
            method,
            net_value: evsi - c.cost,
            rank_uncertain: false,
        });
    }
    rows.sort_by(|a, b| b.net_value.total_cmp(&a.net_value));
    for i in 0..rows.len().saturating_sub(1) {
        let gap = rows[i].net_value - rows[i + 1].net_value;
        rows[i].rank_uncertain = gap <= 2.0 * (rows[i].standard_error + rows[i + 1].standard_error);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> bool {
        false
    }

    fn frame(pop: &str) -> ChoiceFrame {
        ChoiceFrame {
            options: vec!["a".into(), "b".into(), "c".into()],
            population: pop.into(),
            meaning: ChoiceMeaning::StatedPreference,
        }
    }

    fn source(label: &str, pop: &str, counts: [f64; 3], id: &str) -> SourceUse {
        SourceUse {
            evidence: RetainedChoiceEvidence {
                label: label.into(),
                frame: frame(pop),
                counts: counts.to_vec(),
                observation_id: id.into(),
            },
            discount: 0.5,
            transfer_rationale: None,
        }
    }

    #[test]
    fn prior_is_discounted_and_refuses_unqualified_reuse() {
        let t = frame("p");
        let p = qualify_prior(
            &t,
            &[1.0; 3],
            &[source("s", "p", [10.0, 4.0, 2.0], "o1")],
            ConflictPolicy::Report,
        )
        .unwrap();
        assert_eq!(p.alpha, vec![6.0, 3.0, 2.0]);
        assert!((p.borrowed_observations - 8.0).abs() < 1e-12);
        let other = source("s", "q", [1.0, 1.0, 1.0], "o1");
        assert!(matches!(
            qualify_prior(&t, &[1.0; 3], &[other.clone()], ConflictPolicy::Report),
            Err(ChoiceRefusal::PopulationMismatch(_))
        ));
        let mut declared = other;
        declared.transfer_rationale = Some("same market, later wave".into());
        assert!(
            qualify_prior(&t, &[1.0; 3], &[declared], ConflictPolicy::Report).unwrap().sources[0]
                .transferred
        );
        let mut shuffled = source("s", "p", [1.0; 3], "o2");
        shuffled.evidence.frame.options.swap(0, 1);
        assert!(matches!(
            qualify_prior(&t, &[1.0; 3], &[shuffled], ConflictPolicy::Report),
            Err(ChoiceRefusal::ParameterizationMismatch(_))
        ));
        let mut behav = source("s", "p", [1.0; 3], "o3");
        behav.evidence.frame.meaning = ChoiceMeaning::ObservedBehavior;
        assert!(matches!(
            qualify_prior(&t, &[1.0; 3], &[behav], ConflictPolicy::Report),
            Err(ChoiceRefusal::MeaningMismatch(_))
        ));
        let dup = [source("x", "p", [1.0; 3], "same"), source("y", "p", [2.0; 3], "same")];
        assert!(matches!(
            qualify_prior(&t, &[1.0; 3], &dup, ConflictPolicy::Report),
            Err(ChoiceRefusal::OverlappingObservations(_))
        ));
    }

    #[test]
    fn conflicting_sources_are_flagged_and_can_be_refused() {
        let t = frame("p");
        let mut a = source("a", "p", [60.0, 5.0, 5.0], "o1");
        let mut b = source("b", "p", [5.0, 60.0, 5.0], "o2");
        a.discount = 1.0;
        b.discount = 1.0;
        let uses = [a, b];
        let p = qualify_prior(&t, &[1.0; 3], &uses, ConflictPolicy::Report).unwrap();
        assert!(p.has_conflict());
        assert!(matches!(
            qualify_prior(&t, &[1.0; 3], &uses, ConflictPolicy::Refuse),
            Err(ChoiceRefusal::Conflict(..))
        ));
        let agree =
            [source("a", "p", [30.0, 10.0, 5.0], "o1"), source("b", "p", [28.0, 12.0, 5.0], "o2")];
        assert!(
            !qualify_prior(&t, &[1.0; 3], &agree, ConflictPolicy::Report).unwrap().has_conflict()
        );
    }

    #[test]
    fn update_refuses_observations_already_in_the_prior() {
        let t = frame("p");
        let p = qualify_prior(
            &t,
            &[1.0; 3],
            &[source("s", "p", [5.0, 3.0, 2.0], "o1")],
            ConflictPolicy::Report,
        )
        .unwrap();
        let reused = ChoiceObservations { counts: vec![1.0; 3], observation_id: "o1".into() };
        assert!(matches!(
            update(&p, &reused, &never),
            Err(ChoiceRefusal::OverlappingObservations(_))
        ));
        let fresh =
            ChoiceObservations { counts: vec![20.0, 5.0, 5.0], observation_id: "o2".into() };
        let inf = update(&p, &fresh, &never).unwrap();
        assert!((inf.probability_best.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(inf.probability_best[0] > 0.9);
    }

    #[test]
    fn exact_evsi_matches_the_scalar_binomial_closed_form_for_two_options() {
        // Two options with values (1, 0): EVSI of n choices is the expected posterior-mean max
        // over a beta-binomial outcome minus the current max, computed here independently.
        let (a, b, n) = (3.0_f64, 4.0_f64, 12u32);
        let got = exact_evsi(&[a, b], &[1.0, 0.0], n);
        let ln_beta = |x: f64, y: f64| ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
        let mut want = 0.0;
        for s in 0..=n {
            let s = f64::from(s);
            let ln_choose =
                ln_gamma(f64::from(n) + 1.0) - ln_gamma(s + 1.0) - ln_gamma(f64::from(n) - s + 1.0);
            let p = (ln_choose + ln_beta(a + s, b + f64::from(n) - s) - ln_beta(a, b)).exp();
            // Terminal value of option 0 is its posterior mean; option 1 has value zero.
            want += p * ((a + s) / (a + b + f64::from(n))).max(0.0);
        }
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        // Equal values: the terminal value is the larger posterior mean of the two options.
        let both = exact_evsi(&[a, b], &[1.0, 1.0], n);
        let mut want_both = 0.0;
        for s in 0..=n {
            let s = f64::from(s);
            let nf = f64::from(n);
            let ln_choose = ln_gamma(nf + 1.0) - ln_gamma(s + 1.0) - ln_gamma(nf - s + 1.0);
            let p = (ln_choose + ln_beta(a + s, b + nf - s) - ln_beta(a, b)).exp();
            want_both += p * ((a + s).max(b + nf - s) / (a + b + nf));
        }
        assert!((both - want_both).abs() < 1e-12, "{both} vs {want_both}");
    }

    #[test]
    fn monte_carlo_agrees_with_exact_within_its_standard_error() {
        let alpha = [4.0, 5.0, 3.0];
        let values = [1.0, 1.1, 0.9];
        let exact = exact_evsi(&alpha, &values, 30);
        let (mc, se) = monte_carlo_evsi(&alpha, &values, 30, 7);
        assert!(se > 0.0 && (mc - exact).abs() < 4.0 * se, "{mc} vs {exact} ± {se}");
    }

    #[test]
    fn ranking_prefers_more_information_until_cost_dominates() {
        let t = frame("p");
        let p = qualify_prior(&t, &[1.0; 3], &[], ConflictPolicy::Report).unwrap();
        let inf = update(
            &p,
            &ChoiceObservations { counts: vec![6.0, 5.0, 4.0], observation_id: "o".into() },
            &never,
        )
        .unwrap();
        let cands = [
            StudyCandidate { label: "small".into(), observations: 10, cost: 0.0 },
            StudyCandidate { label: "large".into(), observations: 60, cost: 0.0 },
            StudyCandidate { label: "large-costly".into(), observations: 60, cost: 10.0 },
        ];
        let rows = rank_studies(&inf, &[1.0, 1.0, 1.0], &cands, 1).unwrap();
        assert_eq!(rows[0].candidate.label, "large");
        assert_eq!(rows[2].candidate.label, "large-costly");
        assert!(rows.iter().all(|r| r.evsi >= 0.0));
        // An option that dominates at any plausible share cannot be displaced: no value.
        let flat = rank_studies(&inf, &[1.0, -100.0, -100.0], &cands, 1).unwrap();
        assert!(flat[0].evsi.abs() < 1e-9);
    }

    #[test]
    fn sequential_rule_selects_continues_and_exhausts() {
        let t = frame("p");
        let p = qualify_prior(&t, &[1.0; 3], &[], ConflictPolicy::Report).unwrap();
        let rule = SequentialRule {
            min_probability_best: 0.95,
            max_expected_loss: 0.05,
            min_observations: 20.0,
            max_observations: 200.0,
            allocation_floor: 0.1,
        };
        let early = update(
            &p,
            &ChoiceObservations { counts: vec![8.0, 6.0, 6.0], observation_id: "a".into() },
            &never,
        )
        .unwrap();
        match rule.evaluate(&early).unwrap() {
            LookOutcome::Continue { allocation } => {
                assert!((allocation.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(allocation.iter().all(|a| *a >= 0.1));
            }
            other => panic!("{other:?}"),
        }
        let clear = update(
            &p,
            &ChoiceObservations { counts: vec![90.0, 20.0, 10.0], observation_id: "b".into() },
            &never,
        )
        .unwrap();
        assert_eq!(rule.evaluate(&clear).unwrap(), LookOutcome::Select { option: 0 });
        let stuck = update(
            &p,
            &ChoiceObservations { counts: vec![70.0, 70.0, 70.0], observation_id: "c".into() },
            &never,
        )
        .unwrap();
        assert_eq!(rule.evaluate(&stuck).unwrap(), LookOutcome::BudgetExhausted);
        let bad = SequentialRule { min_probability_best: 0.4, ..rule };
        assert!(bad.evaluate(&early).is_err());
    }
}
