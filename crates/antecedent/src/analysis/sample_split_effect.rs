//! The effect of a treatment on an outcome after discovery on a disjoint row split, with an
//! interval whose coverage of the true effect is a measured property of the whole method.
//!
//! [`estimate_effect_on_split`] runs one fixed construction:
//!
//! 1. the rows are split by a seeded permutation, half to discovery and half to analysis;
//! 2. PC with the partial-correlation test (`alpha = 0.01`) runs on the discovery rows only;
//! 3. the structure is accepted only when the search finished and the result is one fully
//!    directed graph. A structure with an unresolved edge is refused: no completion is part
//!    of the measured construction, and a declared completion is a different design;
//! 4. the linear effect of the treatment on the outcome is estimated on the analysis rows by
//!    adjusting for the treatment's parents in that graph, with the classical least-squares
//!    interval at level 0.95.
//!
//! # What the interval is
//!
//! A [`SplitEffect`] always carries its interval and the calibration basis it is keyed under
//! (`structure = discovered_sample_split`, never the known-graph key). The *attestation*
//! comes from [`SplitEffect::measure`], which replays the source and asks the coverage
//! records whether the whole procedure was measured to cover the true effect at this
//! construction and at this analysis row count. Without a current attesting record the
//! interval is a stated computation, not a calibrated one.
//!
//! The record is a statement about the *method* over the family of data-generating processes
//! it measured (linear-Gaussian, causally sufficient, faithful, with a fully orientable
//! equivalence class). It is not a statement about one dataset: faithfulness and sufficiency
//! of a given table cannot be tested from that table, so no per-dataset recovery or error
//! rate is offered, here or elsewhere.
//!
//! # What is outside
//!
//! The same rows used both to discover and to estimate are a different construction and are
//! never licensed ([`super::sample_split_analysis::SplitDiscovery::request_unsplit_interval`]
//! refuses). Weak or near-cancelling edges, a misspecified test, a declared completion, other
//! tests or thresholds, other levels and tables outside five to seven variables are outside
//! the measured construction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::split_effect::{self, SplitEffectRun, SplitEffectSource};

pub use antecedent_io::measured_inference::split_effect::{
    ALPHA, CI_TEST, DISCOVERY_FRACTION, ESTIMATOR, LEVEL, MAX_VARIABLES, MIN_VARIABLES,
    SPLIT_STRUCTURE,
};

/// A linear effect estimated on the analysis part of a split after discovery on the other.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitEffect {
    run: SplitEffectRun,
    source: Vec<u8>,
}

/// Split `data`, discover on one part and estimate the treatment's effect on the other.
///
/// The discovery contract's seed is the context's master seed; `split_seed` decides the
/// row permutation. Every column of `data` is searched and must be a finite float64 column.
///
/// # Errors
///
/// A table outside five to seven numeric columns or with missing values, a treatment or
/// outcome that is not a column, a refused or unfinished discovery, a structure with
/// unresolved edges (`split_effect.structure_unresolved`), an outcome oriented into the
/// treatment, or a singular adjusted regression.
pub fn estimate_effect_on_split(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    split_seed: u64,
    ctx: &ExecutionContext,
) -> Result<SplitEffect, IoError> {
    let storage = data.storage();
    let mut labels = Vec::new();
    let mut columns = Vec::new();
    let (mut treatment_at, mut outcome_at) = (None, None);
    for (at, spec) in storage.schema().variables().iter().enumerate() {
        let values = data.float64_values(spec.id).map_err(|error| IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("split_effect.source_invalid: {error}"),
        })?;
        labels.push(spec.name.to_string());
        columns.push(values);
        let index = u32::try_from(at).unwrap_or(u32::MAX);
        if spec.id == treatment {
            treatment_at = Some(index);
        }
        if spec.id == outcome {
            outcome_at = Some(index);
        }
    }
    let (Some(treatment_at), Some(outcome_at)) = (treatment_at, outcome_at) else {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: "split_effect.source_invalid: the treatment and outcome must be columns of \
                      the table"
                .to_owned(),
        });
    };
    let source = SplitEffectSource::new(
        labels,
        columns,
        treatment_at,
        outcome_at,
        split_seed,
        ctx.rng.master_seed(),
    );
    let run = split_effect::run(&source, ctx)?;
    Ok(SplitEffect { run, source: source.to_bytes()? })
}

impl SplitEffect {
    /// The estimated linear effect: the treatment's coefficient.
    #[must_use]
    pub const fn effect(&self) -> f64 {
        self.run.effect.point
    }

    /// The `(lower, upper)` interval at [`Self::level`].
    #[must_use]
    pub const fn interval(&self) -> (f64, f64) {
        (self.run.effect.lower, self.run.effect.upper)
    }

    /// Nominal level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.run.effect.level
    }

    /// Standard error of the coefficient.
    #[must_use]
    pub const fn standard_error(&self) -> f64 {
        self.run.effect.standard_error
    }

    /// Rows the effect was estimated on.
    #[must_use]
    pub const fn analysis_rows(&self) -> u64 {
        self.run.analysis_rows
    }

    /// Rows discovery used; disjoint from the analysis rows.
    #[must_use]
    pub const fn discovery_rows(&self) -> u64 {
        self.run.discovery_rows
    }

    /// Column indices adjusted for: the treatment's parents in the accepted graph.
    #[must_use]
    pub fn adjustment_set(&self) -> &[u32] {
        &self.run.effect.adjustment_set
    }

    /// Identity of the split receipt.
    #[must_use]
    pub const fn receipt_identity(&self) -> &[u8; 32] {
        &self.run.receipt_identity
    }

    /// Identity of the discovered structure the effect rests on.
    #[must_use]
    pub const fn structure_identity(&self) -> &[u8; 32] {
        &self.run.structure_identity
    }

    /// The calibration basis this interval is keyed under.
    #[must_use]
    pub fn keyed_basis(&self) -> CalibrationBasis {
        self.run.keyed_basis()
    }

    /// The canonical source bytes: everything needed to replay this execution.
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        &self.source
    }

    /// Replay the source and attest the interval from the coverage records.
    ///
    /// # Errors
    ///
    /// The refusals of [`estimate_effect_on_split`], and `cell_not_licensed`
    /// (`measured_inference.record_not_attesting`) when no current attesting, non-boundary
    /// record measured this construction at this analysis row count.
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        split_effect::measure(&self.source, ctx)
    }
}

/// Replay `source_artifact` and attest it, without the executed [`SplitEffect`].
///
/// # Errors
///
/// As [`SplitEffect::measure`], and a source that is not the canonical encoding.
pub fn measure_source(
    source_artifact: &[u8],
    ctx: &ExecutionContext,
) -> Result<MeasuredInference, IoError> {
    split_effect::measure(source_artifact, ctx)
}
