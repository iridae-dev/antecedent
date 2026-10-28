//! Derivative-free optimisation of bounded history-dependent intervention policies.
//!
//! A [`PolicyRolloutSource`] simulates complete decision paths from a compatible causal
//! response model. This owner holds the search: it evaluates every proposal on the same
//! shared draws (so policy values and paired contrasts share posterior/parameter draws),
//! enforces cumulative-cost, cumulative-exposure and per-step action constraints on rollout
//! paths with chance-constraint style reporting, and optimises with a seeded elite-refit
//! evolution strategy followed by coordinate polishing. Every proposal is retained so the
//! search replays exactly.
//!
//! Scientific standing is explicit and conservative. The result is always an optimisation
//! inside the fitted response model (`FittedModelOptimization`); it is labelled an identified
//! policy effect only when the caller asks for that label and the response source states an
//! identification basis. The selected policy is chosen on the optimisation draws, so its
//! in-sample value is optimistic; the held-out value, on draws or units the selection never
//! touched, is the honest estimate and is refused or flagged when the study design cannot
//! supply one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adaptive_policy::AdaptivePolicyFamily;
use antecedent_core::{derive_stream, CanonicalWriter, CausalRng, ExecutionContext, VariableId};
use std::sync::Arc;

/// Which draws a rollout request uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawSplit {
    /// Draws the search optimises on (shared by every proposal).
    Optimization,
    /// Draws or units the search never evaluates; used only to report the selected policies.
    HeldOut,
}

/// One simulated decision path under one policy on one shared draw.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPath {
    /// Path utility (the quantity the search maximises in expectation).
    pub utility: f64,
    /// Action taken at each step.
    pub actions: Vec<f64>,
    /// Cost incurred at each step.
    pub step_costs: Vec<f64>,
    /// Exposure accrued at each step.
    pub step_exposures: Vec<f64>,
}

/// What a held-out evaluation can be built from, as stated by the study design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeldOutDesign {
    /// The source holds a second, independent set of simulated draws.
    IndependentDraws,
    /// The source holds units never used to fit the response or select the policy.
    HeldOutUnits,
    /// The design supports neither; the reason is reported.
    Unavailable(String),
}

/// What the response source states about identification of the policy effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseStanding {
    /// Whether the response model's identification supports a causal policy-effect reading.
    pub identified: bool,
    /// The identification basis when identified, otherwise the reason it is not.
    pub basis: String,
}

/// Failure of a rollout source or of the search.
#[derive(Debug, thiserror::Error)]
pub enum AdaptivePolicySearchError {
    /// Invalid family, constraint, option, source output or non-finite value.
    #[error("adaptive_policy_search: {0}")]
    Invalid(String),
    /// Caller resources refused before output allocation.
    #[error("adaptive_policy_search resource refusal: {0}")]
    Resource(String),
    /// The response source failed on one proposal.
    #[error("adaptive_policy_search source: {0}")]
    Source(String),
    /// A policy reads a variable that is not observed at decision time.
    #[error("adaptive_policy_search: policy reads variable {variable} that is not observed when it acts")]
    UnobservedFeature {
        /// Raw id of the variable.
        variable: u32,
    },
    /// An identified-policy-effect label was requested but the response does not support it.
    #[error("adaptive_policy_search: identified policy effect refused: {0}")]
    IdentificationUnsupported(String),
    /// Held-out evaluation was required but the study design cannot supply it.
    #[error("adaptive_policy_search: held-out evaluation refused: {0}")]
    HeldOutUnsupported(String),
    /// No proposal could be evaluated.
    #[error("adaptive_policy_search: no proposal could be evaluated: {0}")]
    NoEvaluation(String),
    /// Caller cancellation; no partial result is returned.
    #[error("adaptive policy search cancelled")]
    Cancelled,
}

impl AdaptivePolicySearchError {
    /// Registered reason code of a scientific refusal; operational failures (invalid input,
    /// resources, source failure, no evaluable proposal, cancellation) carry none.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::UnobservedFeature { .. } => Some("adaptive_policy_unobserved_feature"),
            Self::IdentificationUnsupported(_) => {
                Some("adaptive_policy_identification_unsupported")
            }
            Self::HeldOutUnsupported(_) => Some("adaptive_policy_heldout_unsupported"),
            Self::Invalid(_)
            | Self::Resource(_)
            | Self::Source(_)
            | Self::NoEvaluation(_)
            | Self::Cancelled => None,
        }
    }
}

fn invalid(message: impl Into<String>) -> AdaptivePolicySearchError {
    AdaptivePolicySearchError::Invalid(message.into())
}

fn cancelled(ctx: &ExecutionContext) -> Result<(), AdaptivePolicySearchError> {
    if ctx.cancellation.is_cancelled() {
        Err(AdaptivePolicySearchError::Cancelled)
    } else {
        Ok(())
    }
}

/// A compatible causal response model that simulates decision paths for a policy.
///
/// Implementations must be common-random-number consistent: for a fixed split, draw `i`
/// uses the same exogenous noise and parameter draw for every policy, so utilities of two
/// policies differ only through the policy. The optimisation and held-out splits must use
/// independent noise.
pub trait PolicyRolloutSource: Sync {
    /// Stable identity of the response model, its draws and seeds.
    fn identity(&self) -> String;
    /// Number of draws (path rows) of a split; zero when the split does not exist.
    fn n_draws(&self, split: DrawSplit) -> usize;
    /// Steps of every path.
    fn horizon(&self) -> usize;
    /// Identification standing of the response model.
    fn standing(&self) -> ResponseStanding;
    /// What the design supports for held-out evaluation.
    fn heldout_design(&self) -> HeldOutDesign;
    /// Variables a policy may read (observed when it acts); `None` leaves them unchecked.
    fn observable_variables(&self) -> Option<Vec<VariableId>> {
        None
    }
    /// Simulate `n_draws(split)` paths under the policy `(family, theta)`.
    ///
    /// # Errors
    /// Any source failure; cancellation and resource refusals must use their own variants.
    fn rollout(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PolicyPath>, AdaptivePolicySearchError>;
}

/// A bound that must hold on a rollout path, with a tolerated violation probability.
#[derive(Clone, Debug, PartialEq)]
pub struct ChanceBound {
    /// Inclusive upper bound on the path quantity.
    pub bound: f64,
    /// Largest tolerated share of draws that violate the bound; zero is a hard constraint.
    pub max_violation_probability: f64,
}

/// Hard and chance constraints on rollout paths.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyConstraints {
    /// Cumulative cost over the path.
    pub cumulative_cost: Option<ChanceBound>,
    /// Cumulative exposure over the path.
    pub cumulative_exposure: Option<ChanceBound>,
    /// Every realised action must lie in this interval on every draw (a hard constraint on
    /// rollout paths, checked on realised actions rather than assumed from the family).
    pub per_step_action: Option<(f64, f64)>,
}

/// The functional of path utility the search maximises over the shared draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolicyObjective {
    /// Mean utility.
    ExpectedUtility,
    /// Mean of the lowest `alpha` share of utilities (conditional value at risk).
    LowerTailMean {
        /// Tail share in `(0, 1]`.
        alpha: f64,
    },
}

/// The standing the caller asks the result to carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandingRequest {
    /// An optimisation inside the fitted response model.
    FittedModel,
    /// An identified policy effect; refused unless the response source supports it.
    IdentifiedPolicyEffect,
}

/// Whether a held-out evaluation is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeldOutRequirement {
    /// Refuse when the design cannot supply a held-out evaluation.
    Required,
    /// Evaluate when possible, otherwise flag the result.
    Preferred,
}

/// Derivative-free search policy: fixed seed and finite work.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicySearchOptions {
    /// Seed of the search's own random stream (not the response model's draws).
    pub seed: u64,
    /// Maximum policy evaluations per family, polishing included.
    pub max_evaluations: usize,
    /// Proposals per generation; the first generation is uniform over the parameter box.
    pub population: usize,
    /// Share of a generation refitted as elites, in `(0, 0.5]`.
    pub elite_fraction: f64,
    /// Maximum generations before polishing.
    pub max_generations: usize,
    /// Stop generations when every coordinate spread falls below this share of its width.
    pub spread_tolerance: f64,
    /// Whether to polish the incumbent with a coordinate pattern search.
    pub polish: bool,
    /// Requested scientific standing.
    pub standing: StandingRequest,
    /// Held-out evaluation requirement.
    pub held_out: HeldOutRequirement,
}

impl Default for PolicySearchOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            max_evaluations: 2_000,
            population: 24,
            elite_fraction: 0.25,
            max_generations: 40,
            spread_tolerance: 1e-3,
            polish: true,
            standing: StandingRequest::FittedModel,
            held_out: HeldOutRequirement::Preferred,
        }
    }
}

/// Scientific label of a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyStanding {
    /// Model-conditional: the best policy found inside the fitted response model.
    FittedModelOptimization {
        /// Why this is not an identified effect (or that it was not requested as one).
        note: String,
    },
    /// The response model's identification supports a causal reading of the policy effect.
    IdentifiedPolicyEffect {
        /// Identification basis stated by the response source.
        basis: String,
    },
}

impl PolicyStanding {
    /// Stable machine label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::FittedModelOptimization { .. } => "fitted_model_optimization",
            Self::IdentifiedPolicyEffect { .. } => "identified_policy_effect",
        }
    }
}

/// Why a family's search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStop {
    /// Every coordinate spread fell below the tolerance and polishing finished.
    SpreadConverged,
    /// The generation limit was reached and polishing finished.
    GenerationLimit,
    /// The evaluation budget ran out; the result is incomplete.
    EvaluationBudget,
}

/// Evaluation of one constraint on one split.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintReport {
    /// Constraint id: `cumulative_cost`, `cumulative_exposure` or `per_step_action`.
    pub id: String,
    /// Draws on which the bound was violated.
    pub violations: usize,
    /// Draws evaluated.
    pub n_draws: usize,
    /// Observed violation share.
    pub violation_rate: f64,
    /// Wilson 95% upper bound on the violation probability (approximation, not a guarantee).
    pub violation_rate_upper: f64,
    /// Tolerated violation probability.
    pub tolerance: f64,
    /// Largest observed value of the constrained quantity (largest absolute breach for actions).
    pub worst_observed: f64,
    /// Observed rate is within tolerance.
    pub satisfied: bool,
    /// The Wilson upper bound is within tolerance.
    pub upper_bound_satisfied: bool,
}

/// One policy evaluated on one split.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEvaluation {
    /// Draws evaluated.
    pub n_draws: usize,
    /// Objective value (mean utility or lower-tail mean).
    pub objective: f64,
    /// Mean utility.
    pub mean_utility: f64,
    /// Monte Carlo standard error of the mean utility.
    pub standard_error: f64,
    /// Constraint reports (empty when unconstrained).
    pub constraints: Vec<ConstraintReport>,
    /// Every constraint satisfied at the observed rate.
    pub feasible: bool,
    /// Per-draw utilities, aligned across policies for paired contrasts.
    pub draw_utilities: Vec<f64>,
}

/// One proposal of the search.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyAttempt {
    /// Parameter vector.
    pub theta: Vec<f64>,
    /// Objective when evaluation succeeded.
    pub objective: Option<f64>,
    /// Sum over constraints of the violation rate beyond tolerance.
    pub excess_violation: f64,
    /// Constraints satisfied.
    pub feasible: bool,
    /// Source failure, distinct from infeasibility.
    pub failure: Option<String>,
}

/// Search outcome for one family.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilySearchResult {
    /// Family id.
    pub family_id: String,
    /// Canonical family identity.
    pub family_identity: [u8; 32],
    /// Best parameter vector found.
    pub theta: Vec<f64>,
    /// Evaluation on the optimisation draws (optimistic for the selected policy).
    pub in_sample: PolicyEvaluation,
    /// Evaluation on held-out draws or units, when the design supports one.
    pub held_out: Option<PolicyEvaluation>,
    /// Every proposal in order.
    pub attempts: Vec<PolicyAttempt>,
    /// Policy evaluations performed.
    pub evaluations: usize,
    /// Generations run.
    pub generations: usize,
    /// Why the search stopped.
    pub stop: SearchStop,
}

/// Paired difference of two policies' utilities over shared draws.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedContrast {
    /// Family compared.
    pub policy: String,
    /// Baseline family.
    pub baseline: String,
    /// Draws the contrast pairs.
    pub split: DrawSplit,
    /// Paired draws.
    pub n_draws: usize,
    /// Mean of policy minus baseline utility.
    pub mean_difference: f64,
    /// Standard error of the paired mean difference.
    pub standard_error: f64,
}

/// Complete result of a policy search.
#[derive(Clone, Debug)]
pub struct AdaptivePolicySearchResult {
    /// One result per declared family, in declaration order.
    pub families: Vec<FamilySearchResult>,
    /// Index of the best feasible family on the optimisation draws; `None` when none is feasible.
    pub selected: Option<usize>,
    /// Paired contrasts of every family against the baseline family, on shared draws.
    pub contrasts: Vec<PairedContrast>,
    /// Scientific label.
    pub standing: PolicyStanding,
    /// Why the held-out evaluation is absent or why the standing is weak, when flagged.
    pub flags: Vec<String>,
    /// In-sample minus held-out objective of the selected policy (selection optimism).
    pub selection_optimism: Option<f64>,
    /// Source identity the search ran against.
    pub source_identity: String,
    /// Identity of the problem (families, constraints, objective, options).
    pub problem_identity: [u8; 32],
    /// Identity of the whole record, including every proposal and value.
    pub record_identity: [u8; 32],
    /// False when any family exhausted its evaluation budget before converging.
    pub complete: bool,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}

impl AdaptivePolicySearchResult {
    /// Retained admission, not process RSS or an allocator measurement.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retention.remainder().bytes()
    }
    /// The selected family result.
    #[must_use]
    pub fn selected_family(&self) -> Option<&FamilySearchResult> {
        self.selected.map(|i| &self.families[i])
    }
}

/// Frozen search problem.
#[derive(Clone, Debug)]
pub struct PreparedAdaptivePolicySearch {
    families: Vec<AdaptivePolicyFamily>,
    baseline: Option<usize>,
    constraints: PolicyConstraints,
    objective: PolicyObjective,
    options: PolicySearchOptions,
}

const WILSON_Z: f64 = 1.959_963_984_540_054;

fn wilson_upper(violations: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let (k, n) = (violations as f64, n as f64);
    let p = k / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = WILSON_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    (center + half).min(1.0)
}

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, f64::NAN);
    }
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

impl PreparedAdaptivePolicySearch {
    /// Validate the problem without touching a response model.
    ///
    /// `baseline` names the family (usually a fixed schedule) every other family is contrasted
    /// with on shared draws.
    ///
    /// # Errors
    /// No families, duplicate ids, an out-of-range baseline, non-finite constraints or
    /// objective parameters, or an invalid search policy.
    pub fn compile(
        families: Vec<AdaptivePolicyFamily>,
        baseline: Option<usize>,
        constraints: PolicyConstraints,
        objective: PolicyObjective,
        options: PolicySearchOptions,
    ) -> Result<Self, AdaptivePolicySearchError> {
        if families.is_empty() || families.len() > 32 {
            return Err(invalid("declare between 1 and 32 policy families"));
        }
        let mut ids: Vec<&str> = families.iter().map(AdaptivePolicyFamily::id).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            return Err(invalid("duplicate policy family id"));
        }
        if baseline.is_some_and(|b| b >= families.len()) {
            return Err(invalid("baseline family index out of range"));
        }
        let chance = |c: &Option<ChanceBound>| {
            c.as_ref().is_none_or(|c| {
                c.bound.is_finite()
                    && c.max_violation_probability.is_finite()
                    && (0.0..=1.0).contains(&c.max_violation_probability)
            })
        };
        if !chance(&constraints.cumulative_cost)
            || !chance(&constraints.cumulative_exposure)
            || constraints
                .per_step_action
                .is_some_and(|(a, b)| !(a.is_finite() && b.is_finite() && a <= b))
        {
            return Err(invalid("constraints must be finite with probabilities in [0, 1]"));
        }
        if let PolicyObjective::LowerTailMean { alpha } = objective {
            if !(alpha.is_finite() && alpha > 0.0 && alpha <= 1.0) {
                return Err(invalid("lower-tail share must lie in (0, 1]"));
            }
        }
        let o = &options;
        if o.max_evaluations == 0
            || o.max_evaluations > 1_000_000
            || o.population < 4
            || o.population > 10_000
            || !(o.elite_fraction > 0.0 && o.elite_fraction <= 0.5)
            || o.max_generations == 0
            || o.max_generations > 100_000
            || !(o.spread_tolerance.is_finite() && o.spread_tolerance > 0.0)
        {
            return Err(invalid("invalid bounded search policy"));
        }
        Ok(Self { families, baseline, constraints, objective, options })
    }

    /// Identity of the problem.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        let mut w = CanonicalWriter::new("antecedent.adaptive_policy_search.v1");
        w.word(self.families.len() as u64);
        for f in &self.families {
            w.bytes(&f.identity());
        }
        w.opt_word(self.baseline.map(|b| b as u64));
        let chance = |w: &mut CanonicalWriter, c: &Option<ChanceBound>| {
            w.opt_real(c.as_ref().map(|c| c.bound));
            w.opt_real(c.as_ref().map(|c| c.max_violation_probability));
        };
        chance(&mut w, &self.constraints.cumulative_cost);
        chance(&mut w, &self.constraints.cumulative_exposure);
        w.opt_real(self.constraints.per_step_action.map(|a| a.0));
        w.opt_real(self.constraints.per_step_action.map(|a| a.1));
        match self.objective {
            PolicyObjective::ExpectedUtility => w.word(1).real(0.0),
            PolicyObjective::LowerTailMean { alpha } => w.word(2).real(alpha),
        };
        let o = &self.options;
        w.word(o.seed)
            .word(o.max_evaluations as u64)
            .word(o.population as u64)
            .real(o.elite_fraction)
            .word(o.max_generations as u64)
            .real(o.spread_tolerance)
            .word(u64::from(o.polish))
            .word(o.standing as u64)
            .word(o.held_out as u64);
        w.finish()
    }

    /// Run the search on a response source.
    ///
    /// # Errors
    /// Unobserved policy features, an unsupported standing request, a required but
    /// unavailable held-out evaluation, resource refusal, source misbehaviour with no usable
    /// proposal, or cancellation.
    pub fn execute(
        &self,
        source: &dyn PolicyRolloutSource,
        ctx: &ExecutionContext,
    ) -> Result<AdaptivePolicySearchResult, AdaptivePolicySearchError> {
        cancelled(ctx)?;
        self.check_source(source)?;
        let standing_info = source.standing();
        let design = source.heldout_design();
        let mut flags = Vec::new();
        let mut do_heldout = false;
        match &design {
            HeldOutDesign::IndependentDraws | HeldOutDesign::HeldOutUnits => {
                if source.n_draws(DrawSplit::HeldOut) == 0 {
                    return Err(invalid(
                        "source declares a held-out design with no held-out draws",
                    ));
                }
                do_heldout = true;
            }
            HeldOutDesign::Unavailable(reason) => {
                if self.options.held_out == HeldOutRequirement::Required {
                    return Err(AdaptivePolicySearchError::HeldOutUnsupported(reason.clone()));
                }
                flags.push(format!(
                    "held-out evaluation unavailable ({reason}); the in-sample value of the selected policy is optimistic"
                ));
            }
        }
        let standing = match (self.options.standing, standing_info.identified) {
            (StandingRequest::IdentifiedPolicyEffect, false) => {
                return Err(AdaptivePolicySearchError::IdentificationUnsupported(
                    standing_info.basis,
                ));
            }
            (StandingRequest::IdentifiedPolicyEffect, true) => {
                if !do_heldout {
                    flags.push(
                        "identified policy effect reported without a held-out evaluation".into(),
                    );
                }
                PolicyStanding::IdentifiedPolicyEffect { basis: standing_info.basis }
            }
            (StandingRequest::FittedModel, _) => PolicyStanding::FittedModelOptimization {
                note: if standing_info.identified {
                    "identified response available but a policy effect was not requested".into()
                } else {
                    standing_info.basis
                },
            },
        };

        let n_opt = source.n_draws(DrawSplit::Optimization);
        if n_opt < 2 {
            return Err(invalid("the optimisation split needs at least two draws"));
        }
        let (retained, scratch) = self.memory_quote(source, n_opt)?;
        let total = retained
            .checked_add(scratch)
            .ok_or_else(|| AdaptivePolicySearchError::Resource("quote overflow".into()))?;
        if ctx.memory_limit_bytes().is_some_and(|limit| total > limit) {
            return Err(AdaptivePolicySearchError::Resource(
                "search exceeds configured memory limit".into(),
            ));
        }
        let retention = Arc::new(
            ctx.acquire_admission(0, retained)
                .map_err(|e| AdaptivePolicySearchError::Resource(e.to_string()))?,
        );
        let _scratch = ctx
            .acquire_admission(0, scratch)
            .map_err(|e| AdaptivePolicySearchError::Resource(e.to_string()))?;

        let mut families = Vec::with_capacity(self.families.len());
        for (index, family) in self.families.iter().enumerate() {
            families.push(self.search_family(source, family, index, do_heldout, ctx)?);
        }
        let complete = families.iter().all(|f| f.stop != SearchStop::EvaluationBudget);
        if !complete {
            flags.push(
                "evaluation budget exhausted before convergence; result is incomplete".into(),
            );
        }

        let selected = families.iter().enumerate().filter(|(_, f)| f.in_sample.feasible).fold(
            None::<usize>,
            |best, (i, f)| match best {
                Some(b) if families[b].in_sample.objective >= f.in_sample.objective => Some(b),
                _ => Some(i),
            },
        );
        if selected.is_none() {
            flags.push("no family produced a policy satisfying every constraint".into());
        }
        let contrasts = self.contrasts(&families);
        let selection_optimism = selected.and_then(|i| {
            families[i].held_out.as_ref().map(|h| families[i].in_sample.objective - h.objective)
        });
        let problem_identity = self.identity();
        let record_identity = record_identity(problem_identity, &source.identity(), &families);
        Ok(AdaptivePolicySearchResult {
            families,
            selected,
            contrasts,
            standing,
            flags,
            selection_optimism,
            source_identity: source.identity(),
            problem_identity,
            record_identity,
            complete,
            retention,
        })
    }

    /// Re-run the search and report whether it reproduces the recorded proposals and values
    /// bit for bit (fixed seed, same source draws).
    ///
    /// # Errors
    /// Any failure of the re-run.
    pub fn verify_replay(
        &self,
        recorded: &AdaptivePolicySearchResult,
        source: &dyn PolicyRolloutSource,
        ctx: &ExecutionContext,
    ) -> Result<bool, AdaptivePolicySearchError> {
        let again = self.execute(source, ctx)?;
        Ok(again.record_identity == recorded.record_identity)
    }

    fn check_source(
        &self,
        source: &dyn PolicyRolloutSource,
    ) -> Result<(), AdaptivePolicySearchError> {
        if source.horizon() == 0 {
            return Err(invalid("response source has zero horizon"));
        }
        if let Some(observed) = source.observable_variables() {
            for family in &self.families {
                for variable in family.read_variables() {
                    if !observed.contains(&variable) {
                        return Err(AdaptivePolicySearchError::UnobservedFeature {
                            variable: variable.raw(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn memory_quote(
        &self,
        source: &dyn PolicyRolloutSource,
        n_opt: usize,
    ) -> Result<(u64, u64), AdaptivePolicySearchError> {
        let overflow = || AdaptivePolicySearchError::Resource("quote overflow".into());
        let mut retained = 1024_usize;
        for family in &self.families {
            let per_attempt = family.dimension().checked_mul(8).ok_or_else(overflow)? + 96;
            retained = retained
                .checked_add(
                    self.options.max_evaluations.checked_mul(per_attempt).ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            let draws = n_opt + source.n_draws(DrawSplit::HeldOut);
            retained = retained
                .checked_add(draws.checked_mul(16).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let draws = n_opt.max(source.n_draws(DrawSplit::HeldOut));
        let per_path = 96 + 24 * source.horizon();
        let scratch = draws.checked_mul(per_path).ok_or_else(overflow)?;
        Ok((
            u64::try_from(retained).map_err(|_| overflow())?,
            u64::try_from(scratch).map_err(|_| overflow())?,
        ))
    }

    fn evaluate(
        &self,
        source: &dyn PolicyRolloutSource,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        ctx: &ExecutionContext,
    ) -> Result<PolicyEvaluation, AdaptivePolicySearchError> {
        cancelled(ctx)?;
        family.check_parameters(theta).map_err(|e| invalid(e.to_string()))?;
        let paths = source.rollout(family, theta, split, ctx)?;
        cancelled(ctx)?;
        let n = source.n_draws(split);
        if paths.len() != n {
            return Err(invalid("source returned a different number of paths than its draws"));
        }
        let mut utilities = Vec::with_capacity(n);
        let (mut cost_viol, mut cost_worst) = (0_usize, f64::NEG_INFINITY);
        let (mut exp_viol, mut exp_worst) = (0_usize, f64::NEG_INFINITY);
        let (mut act_viol, mut act_worst) = (0_usize, 0.0_f64);
        for path in &paths {
            let len = path.actions.len();
            if !path.utility.is_finite()
                || len != source.horizon()
                || path.step_costs.len() != len
                || path.step_exposures.len() != len
                || path
                    .actions
                    .iter()
                    .chain(&path.step_costs)
                    .chain(&path.step_exposures)
                    .any(|v| !v.is_finite())
            {
                return Err(invalid("source returned a malformed or non-finite path"));
            }
            utilities.push(path.utility);
            if let Some(c) = &self.constraints.cumulative_cost {
                let total: f64 = path.step_costs.iter().sum();
                cost_worst = cost_worst.max(total);
                cost_viol += usize::from(total > c.bound);
            }
            if let Some(c) = &self.constraints.cumulative_exposure {
                let total: f64 = path.step_exposures.iter().sum();
                exp_worst = exp_worst.max(total);
                exp_viol += usize::from(total > c.bound);
            }
            if let Some((lo, hi)) = self.constraints.per_step_action {
                let breach = path
                    .actions
                    .iter()
                    .map(|a| (lo - a).max(a - hi))
                    .fold(f64::NEG_INFINITY, f64::max);
                act_worst = act_worst.max(breach);
                act_viol += usize::from(breach > 1e-12);
            }
        }
        let report = |id: &str, violations: usize, tolerance: f64, worst: f64| ConstraintReport {
            id: id.into(),
            violations,
            n_draws: n,
            violation_rate: violations as f64 / n as f64,
            violation_rate_upper: wilson_upper(violations, n),
            tolerance,
            worst_observed: worst,
            satisfied: violations as f64 / n as f64 <= tolerance,
            upper_bound_satisfied: wilson_upper(violations, n) <= tolerance,
        };
        let mut constraints = Vec::new();
        if let Some(c) = &self.constraints.cumulative_cost {
            constraints.push(report(
                "cumulative_cost",
                cost_viol,
                c.max_violation_probability,
                cost_worst,
            ));
        }
        if let Some(c) = &self.constraints.cumulative_exposure {
            constraints.push(report(
                "cumulative_exposure",
                exp_viol,
                c.max_violation_probability,
                exp_worst,
            ));
        }
        if self.constraints.per_step_action.is_some() {
            constraints.push(report("per_step_action", act_viol, 0.0, act_worst));
        }
        let (mean_utility, standard_error) = mean_and_se(&utilities);
        let objective = match self.objective {
            PolicyObjective::ExpectedUtility => mean_utility,
            PolicyObjective::LowerTailMean { alpha } => {
                let mut sorted = utilities.clone();
                sorted.sort_by(f64::total_cmp);
                let k = ((alpha * n as f64).ceil() as usize).clamp(1, n);
                sorted[..k].iter().sum::<f64>() / k as f64
            }
        };
        let feasible = constraints.iter().all(|c| c.satisfied);
        Ok(PolicyEvaluation {
            n_draws: n,
            objective,
            mean_utility,
            standard_error,
            constraints,
            feasible,
            draw_utilities: utilities,
        })
    }

    fn search_family(
        &self,
        source: &dyn PolicyRolloutSource,
        family: &AdaptivePolicyFamily,
        index: usize,
        do_heldout: bool,
        ctx: &ExecutionContext,
    ) -> Result<FamilySearchResult, AdaptivePolicySearchError> {
        let o = &self.options;
        let dim = family.dimension();
        let lower: Vec<f64> = family.bounds().iter().map(|b| b.lower).collect();
        let width: Vec<f64> = family.bounds().iter().map(|b| b.upper - b.lower).collect();
        let mut rng = CausalRng::from_seed(derive_stream(o.seed, index as u64));
        let mut state = FamilyState { attempts: Vec::new(), best: None, evaluations: 0 };
        let budget = o.max_evaluations;
        let mut stop = SearchStop::GenerationLimit;

        let mut mean: Vec<f64> = (0..dim).map(|i| lower[i] + 0.5 * width[i]).collect();
        let mut spread: Vec<f64> = width.iter().map(|w| 0.35 * w).collect();
        let mu = ((o.population as f64 * o.elite_fraction).ceil() as usize).max(2);
        let mut generations = 0;
        'generations: for generation in 0..o.max_generations {
            generations = generation + 1;
            let mut batch: Vec<Vec<f64>> = Vec::with_capacity(o.population + 1);
            if generation == 0 {
                batch.push(mean.clone());
            }
            while batch.len() < o.population {
                let theta = (0..dim)
                    .map(|i| {
                        if width[i] == 0.0 {
                            lower[i]
                        } else if generation == 0 {
                            lower[i] + rng.next_f64() * width[i]
                        } else {
                            (mean[i] + spread[i] * standard_normal(&mut rng))
                                .clamp(lower[i], lower[i] + width[i])
                        }
                    })
                    .collect();
                batch.push(theta);
            }
            let mut scored: Vec<(Score, Vec<f64>)> = Vec::with_capacity(batch.len());
            for theta in batch {
                if state.evaluations >= budget {
                    stop = SearchStop::EvaluationBudget;
                    break 'generations;
                }
                if let Some(score) = self.try_candidate(source, family, &theta, &mut state, ctx)? {
                    scored.push((score, theta));
                }
            }
            scored.sort_by(|a, b| b.0.compare(&a.0));
            scored.truncate(mu);
            if scored.len() >= 2 {
                for i in 0..dim {
                    let m = scored.iter().map(|(_, t)| t[i]).sum::<f64>() / scored.len() as f64;
                    let v = scored.iter().map(|(_, t)| (t[i] - m) * (t[i] - m)).sum::<f64>()
                        / scored.len() as f64;
                    mean[i] = 0.7 * m + 0.3 * mean[i];
                    spread[i] = 0.7 * v.sqrt() + 0.3 * spread[i];
                }
            }
            let tight =
                (0..dim).all(|i| width[i] == 0.0 || spread[i] <= o.spread_tolerance * width[i]);
            if tight && generation > 0 {
                stop = SearchStop::SpreadConverged;
                break;
            }
        }
        if o.polish && stop != SearchStop::EvaluationBudget {
            let mut step: Vec<f64> = width.iter().map(|w| 0.05 * w).collect();
            'polish: for _ in 0..64 {
                let mut improved = false;
                for i in 0..dim {
                    if width[i] == 0.0 {
                        continue;
                    }
                    for direction in [-1.0, 1.0] {
                        let Some(incumbent) = state.best.as_ref().map(|b| b.1.clone()) else {
                            break 'polish;
                        };
                        if state.evaluations >= budget {
                            stop = SearchStop::EvaluationBudget;
                            break 'polish;
                        }
                        let mut theta = incumbent.clone();
                        theta[i] =
                            (theta[i] + direction * step[i]).clamp(lower[i], lower[i] + width[i]);
                        if theta[i].to_bits() == incumbent[i].to_bits() {
                            continue;
                        }
                        let before = state.best.as_ref().map(|b| b.0);
                        self.try_candidate(source, family, &theta, &mut state, ctx)?;
                        improved |= state.best.as_ref().map(|b| b.0) != before;
                    }
                }
                if !improved {
                    for s in &mut step {
                        *s *= 0.5;
                    }
                    if (0..dim)
                        .all(|i| width[i] == 0.0 || step[i] <= 1e-4 * o.spread_tolerance * width[i])
                    {
                        break;
                    }
                }
            }
        }
        let Some((_, theta, in_sample)) = state.best else {
            return Err(AdaptivePolicySearchError::NoEvaluation(format!(
                "family {} had no evaluable proposal",
                family.id()
            )));
        };
        let held_out = if do_heldout {
            Some(self.evaluate(source, family, &theta, DrawSplit::HeldOut, ctx)?)
        } else {
            None
        };
        Ok(FamilySearchResult {
            family_id: family.id().to_owned(),
            family_identity: family.identity(),
            theta,
            in_sample,
            held_out,
            attempts: state.attempts,
            evaluations: state.evaluations,
            generations,
            stop,
        })
    }

    /// Evaluate one proposal, record it and update the incumbent. Source failures are recorded
    /// as failed attempts; cancellation and resource refusals abort.
    fn try_candidate(
        &self,
        source: &dyn PolicyRolloutSource,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        state: &mut FamilyState,
        ctx: &ExecutionContext,
    ) -> Result<Option<Score>, AdaptivePolicySearchError> {
        state.evaluations += 1;
        match self.evaluate(source, family, theta, DrawSplit::Optimization, ctx) {
            Ok(eval) => {
                let excess: f64 = eval
                    .constraints
                    .iter()
                    .map(|c| (c.violation_rate - c.tolerance).max(0.0))
                    .sum();
                let score = Score { feasible: eval.feasible, excess, value: eval.objective };
                state.attempts.push(PolicyAttempt {
                    theta: theta.to_vec(),
                    objective: Some(eval.objective),
                    excess_violation: excess,
                    feasible: eval.feasible,
                    failure: None,
                });
                if state.best.as_ref().is_none_or(|(b, _, _)| score.compare(b).is_gt()) {
                    state.best = Some((score, theta.to_vec(), eval));
                }
                Ok(Some(score))
            }
            Err(
                e @ (AdaptivePolicySearchError::Cancelled | AdaptivePolicySearchError::Resource(_)),
            ) => Err(e),
            Err(e) => {
                state.attempts.push(PolicyAttempt {
                    theta: theta.to_vec(),
                    objective: None,
                    excess_violation: f64::INFINITY,
                    feasible: false,
                    failure: Some(e.to_string()),
                });
                Ok(None)
            }
        }
    }

    fn contrasts(&self, families: &[FamilySearchResult]) -> Vec<PairedContrast> {
        let Some(base) = self.baseline else { return Vec::new() };
        let mut out = Vec::new();
        for (i, f) in families.iter().enumerate() {
            if i == base {
                continue;
            }
            let pairs = [
                (DrawSplit::Optimization, Some(&f.in_sample), Some(&families[base].in_sample)),
                (DrawSplit::HeldOut, f.held_out.as_ref(), families[base].held_out.as_ref()),
            ];
            for (split, a, b) in pairs {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if a.draw_utilities.len() != b.draw_utilities.len() || a.draw_utilities.len() < 2 {
                    continue;
                }
                let diff: Vec<f64> =
                    a.draw_utilities.iter().zip(&b.draw_utilities).map(|(x, y)| x - y).collect();
                let (mean_difference, standard_error) = mean_and_se(&diff);
                out.push(PairedContrast {
                    policy: f.family_id.clone(),
                    baseline: families[base].family_id.clone(),
                    split,
                    n_draws: diff.len(),
                    mean_difference,
                    standard_error,
                });
            }
        }
        out
    }
}

/// Lexicographic comparison key: feasibility, then violation excess, then objective value.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Score {
    feasible: bool,
    excess: f64,
    value: f64,
}

impl Score {
    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        self.feasible
            .cmp(&other.feasible)
            .then_with(|| {
                if self.feasible {
                    std::cmp::Ordering::Equal
                } else {
                    other.excess.total_cmp(&self.excess)
                }
            })
            .then_with(|| self.value.total_cmp(&other.value))
    }
}

struct FamilyState {
    attempts: Vec<PolicyAttempt>,
    best: Option<(Score, Vec<f64>, PolicyEvaluation)>,
    evaluations: usize,
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn record_identity(problem: [u8; 32], source: &str, families: &[FamilySearchResult]) -> [u8; 32] {
    let mut w = CanonicalWriter::new("antecedent.adaptive_policy_search_record.v1");
    w.bytes(&problem).text(source).word(families.len() as u64);
    for f in families {
        w.text(&f.family_id)
            .reals(&f.theta)
            .word(f.evaluations as u64)
            .word(f.attempts.len() as u64);
        for a in &f.attempts {
            w.reals(&a.theta).opt_real(a.objective).real(a.excess_violation);
        }
        w.real(f.in_sample.objective);
        w.opt_real(f.held_out.as_ref().map(|h| h.objective));
    }
    w.finish()
}
