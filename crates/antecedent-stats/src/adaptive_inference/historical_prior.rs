//! Historical borrowing for the arm priors of a participant-level family.
//!
//! An arm's prior is a `Beta(alpha0, beta0)` initial prior updated by zero or
//! more historical sources with the *power prior* of Ibrahim and Chen (2000):
//! a source with `successes` out of `trials` and a discount `a0` in `[0, 1]`
//! contributes its likelihood raised to the power `a0`, which for a Bernoulli
//! likelihood and a Beta initial prior is the conjugate update
//!
//! ```text
//! alpha = alpha0 + sum_i a0_i successes_i,   beta = beta0 + sum_i a0_i (trials_i - successes_i).
//! ```
//!
//! The discount is declared, not estimated: with `a0 = 0` the source is
//! ignored, with `a0 = 1` it counts as much as current data, and anything in
//! between counts as `a0 * trials` current responses. Because the discounts
//! are fixed before recruitment the resolved prior is an ordinary Beta, so
//! every posterior of the procedure stays conjugate. A source made of clustered
//! participants carries less information per response than its trial count
//! says; the discount is where a producer states that (a value near
//! `1 / design effect` treats the source like current clustered data).
//!
//! The prior effective sample size of a Beta prior is `alpha + beta` (the
//! number of pseudo-responses it is worth); the part contributed by history is
//! `sum_i a0_i trials_i`. Both are recorded with every source and discount so a
//! reader can see how much of an arm's posterior is borrowed.
//!
//! Borrowing is part of the declaration of the procedure and is frozen before
//! recruitment: it enters the procedure identity, and
//! [`super::participant_procedure::ParticipantSpec::amend_borrowing`] refuses a
//! change after the first look.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::procedure_contract::BetaPrior;

/// Most historical sources one arm may borrow from.
pub const MAX_HISTORICAL_SOURCES: usize = 16;
/// Most trials one historical source may report.
pub const MAX_HISTORICAL_TRIALS: u64 = 100_000;
/// Longest source label, in bytes.
pub const MAX_SOURCE_LABEL_BYTES: usize = 128;

/// One historical source of an arm: binary responses observed before the
/// procedure, with the discount at which they are borrowed.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalPrior {
    /// Human label of the source (a study, a registry, a trial identifier).
    pub source: String,
    /// Successes among the historical responses.
    pub successes: u64,
    /// Historical responses.
    pub trials: u64,
    /// Power-prior discount `a0` in `[0, 1]`.
    pub discount: f64,
}

impl HistoricalPrior {
    pub(crate) fn is_valid(&self) -> bool {
        !self.source.trim().is_empty()
            && self.source.len() <= MAX_SOURCE_LABEL_BYTES
            && !self.source.chars().any(char::is_control)
            && self.successes <= self.trials
            && self.trials <= MAX_HISTORICAL_TRIALS
            && self.discount.is_finite()
            && (0.0..=1.0).contains(&self.discount)
    }

    /// The responses this source is worth, `a0 * trials`.
    #[must_use]
    pub fn borrowed_trials(&self) -> f64 {
        self.discount * self.trials as f64
    }
}

/// The prior of one arm: an initial Beta prior and its historical sources.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmPrior {
    /// The initial prior, before any history.
    pub initial: BetaPrior,
    /// Sources borrowed from; empty declares no borrowing.
    pub historical: Vec<HistoricalPrior>,
}

/// The prior of one arm after borrowing, with what it is made of.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedArmPrior {
    /// The resolved Beta prior.
    pub prior: BetaPrior,
    /// Prior effective sample size, `alpha + beta` of the resolved prior.
    pub effective_sample_size: f64,
    /// The part of it borrowed from history, `sum a0 * trials`.
    pub borrowed_sample_size: f64,
    /// The sources, with their discounts.
    pub sources: Vec<HistoricalPrior>,
}

impl ArmPrior {
    /// No borrowing: the initial prior alone.
    #[must_use]
    pub const fn without_history(initial: BetaPrior) -> Self {
        Self { initial, historical: Vec::new() }
    }

    /// Whether the initial prior and every source are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.initial.is_valid()
            && self.historical.len() <= MAX_HISTORICAL_SOURCES
            && self.historical.iter().all(HistoricalPrior::is_valid)
    }

    /// The Beta prior the sources resolve to, with its effective sample size.
    ///
    /// Returns `None` for an invalid prior.
    #[must_use]
    pub fn resolve(&self) -> Option<ResolvedArmPrior> {
        if !self.is_valid() {
            return None;
        }
        let mut alpha = self.initial.alpha;
        let mut beta = self.initial.beta;
        let mut borrowed = 0.0;
        for source in &self.historical {
            alpha += source.discount * source.successes as f64;
            beta += source.discount * (source.trials - source.successes) as f64;
            borrowed += source.borrowed_trials();
        }
        Some(ResolvedArmPrior {
            prior: BetaPrior { alpha, beta },
            effective_sample_size: alpha + beta,
            borrowed_sample_size: borrowed,
            sources: self.historical.clone(),
        })
    }
}

/// The priors of every arm, declared before recruitment.
#[derive(Clone, Debug, PartialEq)]
pub struct BorrowingPlan {
    /// One prior per arm.
    pub arms: Vec<ArmPrior>,
}

impl BorrowingPlan {
    /// The same initial prior for every arm and no borrowing.
    #[must_use]
    pub fn uniform(arms: usize, initial: BetaPrior) -> Self {
        Self { arms: vec![ArmPrior::without_history(initial); arms] }
    }

    /// The resolved prior of every arm, `None` when any is invalid.
    #[must_use]
    pub fn resolve(&self) -> Option<Vec<ResolvedArmPrior>> {
        self.arms.iter().map(ArmPrior::resolve).collect()
    }
}
