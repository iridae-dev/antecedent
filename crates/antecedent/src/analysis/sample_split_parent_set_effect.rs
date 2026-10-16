//! The effect of a treatment on an outcome after discovery on a disjoint row split, accepted
//! on the treatment's parent set alone, for sparse linear and log-linear studies of ten to
//! twenty variables.
//!
//! [`estimate_parent_set_effect`] runs one fixed construction:
//!
//! 1. the analysis scale is declared first ([`Transformation`]): `linear`, or `log_linear`,
//!    which requires every searched column to be positive and analyses its natural logarithm;
//! 2. the rows are split by a seeded permutation, half to discovery and half to analysis;
//! 3. PC with the partial-correlation test (`alpha = 0.01`) runs on the discovery rows only,
//!    with a conditioning-set size and a parent limit equal to the declared maximum graph
//!    degree (at most four);
//! 4. the class is accepted when the search finished and **every edge incident to the
//!    treatment is oriented**. Unrelated edges may stay undirected: they are listed on the
//!    result ([`ParentSetEffect::unresolved_edges`]) and never completed. An undirected edge at
//!    the treatment is refused, naming the edges, because the class then holds graphs with
//!    different parent sets;
//! 5. the effect is estimated on the analysis rows by adjusting for the treatment's parents:
//!    the total effect per unit for a linear study, and for a log-linear study the
//!    intervention mean ratio `exp(shift * beta)` of multiplying the treatment by `exp(shift)`,
//!    with the Student interval on the log scale mapped through the exponential.
//!
//! # Topology and attestation
//!
//! Each topology of the measured family has its own coverage record, and a table does not announce
//! its topology. [`ParentSetEffect::measure`] therefore licenses an interval only when the
//! records of every topology in [`TOPOLOGIES`] attest the construction at this variable count,
//! degree, transformation and analysis row count: the conservative intersection.
//!
//! # Conditions of the method
//!
//! The coverage claim of [`ParentSetEffect::measure`] holds, over the family it was measured
//! on, when the table satisfies the conditions published in the selective-discovery guide:
//! independent rows; a linear Gaussian structural model on the analysis scale (log-normal on
//! the original scale for a log-linear study); causal sufficiency; faithfulness with every
//! edge's partial correlation at or above the published signal strength; a maximum graph
//! degree no larger than the declared one; the treatment's edges oriented by the class; and a
//! positive, non-degenerate treatment variation given its parents (intervention support).
//! None of these is testable from the table that discovery runs on, so no per-dataset statement
//! about the discovered graph is made.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::split_parent_set_effect::{
    self, ParentSetEffectRun, ParentSetEffectSource, ReportedEffect, UnresolvedPair,
};

use super::sample_split_effect::table_roles;

pub use antecedent_io::measured_inference::split_parent_set_effect::{
    ACCEPTANCE, Knowledge, LINEAR_ESTIMATOR, LOG_LINEAR_ESTIMATOR, MAX_VARIABLES, MEAN_RATIO,
    MIN_VARIABLES, PARENT_SET_STRUCTURE, SUPPORTED_MAX_DEGREE, TOPOLOGIES, TOTAL_EFFECT,
    Transformation, governing_topologies,
};

/// The declared analysis family of one parent-set study.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParentSetOptions {
    /// Declared analysis scale.
    pub transformation: Transformation,
    /// Declared maximum graph degree, from one to [`SUPPORTED_MAX_DEGREE`].
    pub max_degree: u32,
    /// Log-scale shift of the prespecified intervention of a log-linear study: the treatment is
    /// multiplied by `exp(shift)`. Required and non-zero for `log_linear`, absent for `linear`.
    pub log_shift: Option<f64>,
}

impl ParentSetOptions {
    /// A linear study at maximum graph degree `max_degree`.
    #[must_use]
    pub const fn linear(max_degree: u32) -> Self {
        Self { transformation: Transformation::Linear, max_degree, log_shift: None }
    }

    /// A log-linear study whose intervention multiplies the treatment by `exp(log_shift)`; a
    /// shift of one reports the mean ratio `exp(beta)`.
    #[must_use]
    pub const fn log_linear(max_degree: u32, log_shift: f64) -> Self {
        Self { transformation: Transformation::LogLinear, max_degree, log_shift: Some(log_shift) }
    }
}

/// An effect estimated on the analysis part of a split after discovery on the other, accepted
/// on the treatment's parent set.
#[derive(Clone, Debug, PartialEq)]
pub struct ParentSetEffect {
    run: ParentSetEffectRun,
    source: Vec<u8>,
}

/// Split `data`, discover on one part and estimate the treatment's effect on the other.
///
/// The discovery contract's seed is the context's master seed; `split_seed` decides the row
/// permutation. Every column of `data` is searched and must be a finite float64 column.
///
/// # Errors
///
/// A table outside ten to twenty numeric columns or with missing values, a declared degree
/// above [`SUPPORTED_MAX_DEGREE`] (`parent_set_effect.degree_above_supported`), non-positive
/// values in a log-linear study (`parent_set_effect.log_linear_input_not_positive`), a refused
/// or unfinished discovery, an undirected edge at the treatment
/// (`parent_set_effect.treatment_parents_ambiguous`), an outcome oriented into the treatment,
/// or a singular adjusted regression.
pub fn estimate_parent_set_effect(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    options: ParentSetOptions,
    split_seed: u64,
    ctx: &ExecutionContext,
) -> Result<ParentSetEffect, IoError> {
    estimate_parent_set_effect_with_knowledge(
        data, treatment, outcome, options, None, split_seed, ctx,
    )
}

/// As [`estimate_parent_set_effect`], with background knowledge asserted to discovery: required
/// and forbidden `(cause, effect)` links and tiers, by column index. The knowledge is part of
/// the source and of the calibration key, so an interval is attested only from records measured
/// with the same kinds of statements.
///
/// # Errors
///
/// As [`estimate_parent_set_effect`], and `invalid_argument`
/// (`parent_set_effect.knowledge_invalid`) for knowledge that names a column that is not
/// searched, or a refused (contradictory) discovery contract.
pub fn estimate_parent_set_effect_with_knowledge(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    options: ParentSetOptions,
    knowledge: Option<Knowledge>,
    split_seed: u64,
    ctx: &ExecutionContext,
) -> Result<ParentSetEffect, IoError> {
    let roles = table_roles(data, treatment, outcome)?;
    let mut source = ParentSetEffectSource::new(
        roles.names,
        roles.columns,
        roles.treatment,
        roles.outcome,
        split_seed,
        ctx.rng.master_seed(),
        options.transformation,
        options.max_degree,
        options.log_shift,
    );
    if let Some(knowledge) = knowledge {
        source = source.with_knowledge(knowledge);
    }
    let run = split_parent_set_effect::run(&source, ctx)?;
    Ok(ParentSetEffect { run, source: source.to_bytes()? })
}

impl ParentSetEffect {
    /// The reported effect with its interval, on the estimand's scale: the total effect per
    /// unit (linear) or the intervention mean ratio (log-linear).
    #[must_use]
    pub const fn reported(&self) -> &ReportedEffect {
        &self.run.reported
    }

    /// The treatment's coefficient on the declared analysis scale.
    #[must_use]
    pub const fn coefficient(&self) -> f64 {
        self.run.coefficient.point
    }

    /// Nominal level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.run.coefficient.level
    }

    /// Declared analysis scale.
    #[must_use]
    pub const fn transformation(&self) -> Transformation {
        self.run.transformation
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

    /// Column indices adjusted for: the treatment's parents in the accepted class.
    #[must_use]
    pub fn treatment_parents(&self) -> &[u32] {
        &self.run.neighbourhood.parents
    }

    /// Column indices of the treatment's directed children in the accepted class.
    #[must_use]
    pub fn treatment_children(&self) -> &[u32] {
        &self.run.neighbourhood.children
    }

    /// The edges of the discovered class that are not definite directed edges and are not
    /// incident to the treatment. They are reported, never completed.
    #[must_use]
    pub fn unresolved_edges(&self) -> &[UnresolvedPair] {
        &self.run.neighbourhood.unresolved
    }

    /// Why the discovered class was accepted: [`ACCEPTANCE`].
    #[must_use]
    pub const fn acceptance(&self) -> &'static str {
        ACCEPTANCE
    }

    /// Every directed edge of the discovered class as `(cause, effect)` column indices.
    #[must_use]
    pub fn directed_edges(&self) -> &[(u32, u32)] {
        &self.run.neighbourhood.directed
    }

    /// Edges of the discovered class in total.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.run.neighbourhood.edges
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

    /// The calibration basis of the first topology's record. Attestation requires the records of
    /// every topology in [`TOPOLOGIES`] (see [`Self::keyed_bases`]).
    #[must_use]
    pub fn keyed_basis(&self) -> CalibrationBasis {
        self.run.keyed_basis()
    }

    /// The calibration basis of the record of `topology`, one of [`TOPOLOGIES`].
    #[must_use]
    pub fn keyed_basis_for(&self, topology: &str) -> CalibrationBasis {
        self.run.keyed_basis_for(topology)
    }

    /// The bases of every topology's record, in [`TOPOLOGIES`] order. A table does not announce
    /// its topology, so [`Self::measure`] licenses the interval only when every one attests.
    #[must_use]
    pub fn keyed_bases(&self) -> Vec<CalibrationBasis> {
        self.run.keyed_bases()
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
    /// The refusals of [`estimate_parent_set_effect`], and `cell_not_licensed`
    /// (`measured_inference.record_not_attesting`) when no current attesting, non-boundary
    /// record measured this construction at this variable count, degree and analysis row count.
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        split_parent_set_effect::measure(&self.source, ctx)
    }
}

/// Replay `source_artifact` and attest it, without the executed [`ParentSetEffect`].
///
/// # Errors
///
/// As [`ParentSetEffect::measure`], and a source that is not the canonical encoding.
pub fn measure_source(
    source_artifact: &[u8],
    ctx: &ExecutionContext,
) -> Result<MeasuredInference, IoError> {
    split_parent_set_effect::measure(source_artifact, ctx)
}
