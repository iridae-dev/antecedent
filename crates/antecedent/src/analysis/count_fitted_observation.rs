//! Observation-conditioned probability from tables fitted to counts, with a measured
//! posterior interval.
//!
//! The posterior probability of a *declared* model is an exact number with no sampling
//! distribution: [`super::observation_query::ObservationClaim::interval`] stays closed
//! with `observation_query.declared_model_has_no_sampling_interval`. A probability whose
//! conditional tables were estimated from a sample does have one. This module retains the
//! per-parent-configuration counts and the row count `n`, places an independent
//! Dirichlet posterior over every fitted conditional, and reports the equal-tailed
//! posterior quantiles of `P(target = level | evidence)` by Monte Carlo over joint table
//! draws and exact enumeration. The interval is returned only as a measured scalar bound
//! to a currently attesting coverage record; otherwise a typed refusal.
//!
//! A declared node (a measured signal with known sensitivity and specificity) is a
//! table that is fixed and not redrawn.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
pub use antecedent_io::measured_inference::MeasuredInference;
pub use antecedent_io::measured_inference::count_fitted_interval::{
    CountFittedIntervalArtifactWire, CountFittedNodeWire, MEASURED_DRAWS, MEASURED_PRIOR_ALPHA,
    ROUTE,
};

use super::observation_query::ObservationQueryRefusal;
use antecedent_contract::ExternalRefusal;

/// Seed of the Monte Carlo stream when the caller names none.
pub const DEFAULT_SEED: u64 = 0x0b5e_7a71_0c0f_17ed;

fn refusal_of(error: &antecedent_io::IoError) -> ObservationQueryRefusal {
    let (code, message) = match error {
        antecedent_io::IoError::Refused { code, message } => (*code, message.clone()),
        other => (antecedent_core::reason_code!("invalid_argument"), other.to_string()),
    };
    let (detail, text) = match message.split_once(": ") {
        Some((d, t)) if d.contains('.') && !d.contains(' ') => (d.to_owned(), t.to_owned()),
        _ => ("observation_query.count_fitted_refused".to_owned(), message),
    };
    ObservationQueryRefusal {
        refusals: vec![ExternalRefusal::new(code, "observe", detail).with_offending(text)],
    }
}

/// A conditional probability computed from tables fitted to counts.
#[derive(Clone, Debug, PartialEq)]
pub struct CountFittedObservation {
    wire: CountFittedIntervalArtifactWire,
}

impl CountFittedObservation {
    /// Retain the counts and the question.
    ///
    /// `nodes` are in topological order; a fitted node carries `counts`, a declared node
    /// carries `declared_bits`. `rows` is the number of complete rows every fitted node
    /// was counted from. The default prior is the uniform Dirichlet and the default draw
    /// count is the supported minimum; a source with more draws re-opens through
    /// [`Self::from_source`].
    ///
    /// # Errors
    /// A malformed model, an unknown variable or level, or evidence of probability zero.
    pub fn new(
        nodes: Vec<CountFittedNodeWire>,
        rows: u64,
        target: &str,
        target_level: &str,
        evidence: &[(String, String)],
        seed: Option<u64>,
    ) -> Result<Self, ObservationQueryRefusal> {
        let wire = CountFittedIntervalArtifactWire {
            version: 1,
            nodes,
            rows,
            target: target.to_owned(),
            target_level: target_level.to_owned(),
            evidence: evidence.to_vec(),
            level_bits: 0.95_f64.to_bits(),
            prior_alpha_bits: MEASURED_PRIOR_ALPHA.to_bits(),
            draws: MEASURED_DRAWS,
            seed: seed.unwrap_or(DEFAULT_SEED),
        };
        wire.point().map_err(|e| refusal_of(&e))?;
        Ok(Self { wire })
    }

    /// Use `draws` posterior draws instead of the supported minimum.
    ///
    /// # Errors
    /// A count below the supported minimum or above the draw bound.
    pub fn with_draws(mut self, draws: u32) -> Result<Self, ObservationQueryRefusal> {
        self.wire.draws = draws;
        self.wire.check_draws().map_err(|e| refusal_of(&e))?;
        Ok(self)
    }

    /// Re-open a retained source (for example one decoded from bytes), checking it.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn from_source(
        wire: CountFittedIntervalArtifactWire,
    ) -> Result<Self, ObservationQueryRefusal> {
        wire.point().map_err(|e| refusal_of(&e))?;
        Ok(Self { wire })
    }

    /// The retained source.
    #[must_use]
    pub const fn source(&self) -> &CountFittedIntervalArtifactWire {
        &self.wire
    }

    /// The conditional probability under the posterior-mean tables: a labelled point
    /// value that carries no claim about how often it is right.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn point(&self) -> Result<f64, ObservationQueryRefusal> {
        self.wire.point().map_err(|e| refusal_of(&e))
    }

    /// The measured equal-tailed posterior interval at `level`.
    ///
    /// # Errors
    /// `cell_not_licensed` when the construction (level, prior, a draw count below the
    /// supported minimum, complete rows, sample size) has no currently attesting coverage
    /// record, or when the Monte Carlo error of an endpoint exceeds its declared share of
    /// the interval width (`measured_interval.draws_not_converged`).
    pub fn interval(
        &self,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, ObservationQueryRefusal> {
        if ctx.cancellation.is_cancelled() {
            return Err(ObservationQueryRefusal::of_cancelled());
        }
        let mut wire = self.wire.clone();
        wire.level_bits = level.to_bits();
        let bytes = wire.encode().map_err(|e| refusal_of(&e))?;
        CountFittedIntervalArtifactWire::measure(&bytes).map_err(|e| refusal_of(&e))
    }

    /// The calibration claim of the posterior probability: the same measured interval,
    /// the claim "this interval covers the probability under the generating tables at
    /// the stated level".
    ///
    /// # Errors
    /// As [`Self::interval`].
    pub fn calibrated_probability(
        &self,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, ObservationQueryRefusal> {
        self.interval(level, ctx)
    }
}
