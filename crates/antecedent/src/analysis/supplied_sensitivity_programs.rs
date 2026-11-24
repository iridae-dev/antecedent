//! Compiled sensitivity ranges of evidence a caller supplies: matched designs and effect
//! estimates computed outside the library.
//!
//! A Rosenbaum analysis of matched pairs or matched sets, and an E-value of an effect estimate,
//! interval or posterior reported by another analysis, read no data table the library holds: the
//! caller supplies the pair differences, the matched sets or the estimate. Each compiles here to a
//! six-layer program of its own family on the supplied-terminal lowering, so it has a route, a
//! support unit read from `parity/compiled_units.toml`, a program identity over every declared
//! input (written by [`crate::analysis::supplied_identity::Declaration`], never through `Debug`)
//! and a support gate at execution. Execution calls the same kernel function as the direct path
//! it replaces, on the same inputs, so every number is the direct path's:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_matched_design_sensitivity`] | `MatchedDesignSensitivity` | `rosenbaum_range`, `rosenbaum_vector_range`, `matched_sets_upper_p` per `Γ` then `matched_sets_break_even` |
//! | [`compile_supplied_estimate_evalue`] | `SuppliedEstimateEValue` | `evalue_range`, `evalue_vector_range` or `component_evalue_range`, `evalue_posterior_range` or `component_evalue_posterior_range` |
//!
//! The matched-pair kernels read the execution context of the program (cancellation and the
//! admission of the exact signed-rank table); the direct Python path passed none. A refusal of a
//! kernel is returned unchanged as [`TerminalError::Kernel`]; the compiler's own refusals
//! (re-entry, a release build's support gate) are [`TerminalError::Refused`]. Every number is an
//! assumption range or a diagnostic of the supplied evidence; calibration is not measured.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_validate::target_native::RefutationTarget;
use antecedent_validate::{
    AssumptionRange, ComponentModel, EValueBounds, EffectInterval, EffectScale, MatchedSet,
    MatchedSetBound, MatchedSetScore, PosteriorEValue, SignedRankSensitivity, ValidationError,
    component_evalue_posterior_range, component_evalue_range, evalue_posterior_range, evalue_range,
    evalue_vector_range, matched_sets_break_even, matched_sets_upper_p, rosenbaum_range,
    rosenbaum_vector_range,
};

use crate::analysis::supplied_identity::{Canonical, Declaration, variant};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{Reading, TerminalSpec, compiled_terminal, lower};
use crate::error::CausalError;
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

/// Refusal detail of a matched-design sensitivity compiled inside another program's lowered
/// dispatch (reason code `public_builder_reentry`).
pub const MATCHED_REENTRY_DETAIL: &str =
    antecedent_core::reason_details::matched_design_sensitivity::REENTRY;

/// Refusal detail of a supplied-estimate E-value compiled inside another program's lowered
/// dispatch (reason code `public_builder_reentry`).
pub const EVALUE_REENTRY_DETAIL: &str =
    antecedent_core::reason_details::supplied_estimate_evalue::REENTRY;

// --- matched designs -------------------------------------------------------------------------

/// A matched design a caller supplies, with the hidden-bias grid and level it is read at.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MatchedDesign {
    /// Matched-pair differences (treated minus control), read by the Wilcoxon signed-rank
    /// statistic under hidden bias at most `Γ` (Rosenbaum 2002, ch. 4).
    Pairs {
        /// The target the range is bound to.
        target: RefutationTarget,
        /// Pair differences; zeros are dropped by the statistic.
        differences: Vec<f64>,
        /// The `Γ` grid of the reported upper-bound p-values.
        gammas: Vec<f64>,
        /// Level of the break-even `Γ`.
        alpha: f64,
    },
    /// Matched-pair differences behind every coordinate of a vector target, read at the
    /// weakest coordinate.
    PairVector {
        /// The vector target the range is bound to (one label per coordinate).
        target: RefutationTarget,
        /// Pair differences of each coordinate, in the target's label order.
        differences: Vec<Vec<f64>>,
        /// The `Γ` grid.
        gammas: Vec<f64>,
        /// Level of the break-even `Γ`.
        alpha: f64,
    },
    /// Matched sets with one treated unit and several controls, read by the separable
    /// approximation of Gastwirth, Krieger and Rosenbaum (2000).
    Sets {
        /// The matched sets.
        sets: Vec<MatchedSet>,
        /// The within-set score the statistic sums.
        score: MatchedSetScore,
        /// The `Γ` grid.
        gammas: Vec<f64>,
        /// Level of the break-even `Γ`.
        alpha: f64,
    },
}

impl MatchedDesign {
    /// Stable name of the analysis.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Pairs { .. } => "matched_pairs",
            Self::PairVector { .. } => "matched_pair_vector",
            Self::Sets { .. } => "matched_sets",
        }
    }

    const fn evaluator(&self) -> &'static str {
        match self {
            Self::Pairs { .. } => "rosenbaum::rosenbaum_range",
            Self::PairVector { .. } => "vector_sensitivity::rosenbaum_vector_range",
            Self::Sets { .. } => "rosenbaum::matched_sets_upper_p",
        }
    }

    fn composition(&self) -> Composition {
        match self {
            Self::PairVector { differences, .. } => {
                Composition::ScenarioSet { scenarios: differences.len() }
            }
            Self::Pairs { .. } | Self::Sets { .. } => Composition::Single,
        }
    }
}

/// The hidden-bias analysis of a supplied matched design.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MatchedSensitivityOutcome {
    /// [`MatchedDesign::Pairs`]: the signed-rank analysis and its range on the target.
    Pairs {
        /// Upper-bound p-values on the grid and the break-even `Γ`.
        analysis: SignedRankSensitivity,
        /// The range over `Γ`.
        range: AssumptionRange,
    },
    /// [`MatchedDesign::PairVector`]: each coordinate's analysis and the range at the weakest.
    PairVector {
        /// One analysis per coordinate.
        analyses: Vec<SignedRankSensitivity>,
        /// The range over `Γ`.
        range: AssumptionRange,
    },
    /// [`MatchedDesign::Sets`]: the bound at every `Γ` of the grid and the break-even `Γ`.
    Sets {
        /// One bound per `Γ`, in grid order.
        grid: Vec<MatchedSetBound>,
        /// Largest `Γ` whose upper-bound p-value is at most the level; `None` when the test
        /// does not reject at `Γ = 1`.
        gamma_break_even: Option<f64>,
    },
}

compiled_terminal! {
    /// A compiled Rosenbaum hidden-bias analysis of a supplied matched design.
    #[derive(Clone, Debug)]
    CompiledMatchedDesignSensitivity {
        design: MatchedDesign,
    }
}

/// Compile the hidden-bias analysis of `design`.
///
/// Nothing is ranked or bounded at compile; the program's identity covers the design, the grid
/// and the level.
///
/// # Errors
/// `public_builder_reentry` (detail [`MATCHED_REENTRY_DETAIL`]) inside a lowered dispatch.
pub fn compile_matched_design_sensitivity(
    design: MatchedDesign,
    ctx: &ExecutionContext,
) -> Result<CompiledMatchedDesignSensitivity, CausalError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(crate::unsupported_reason!(
            "public_builder_reentry",
            detail = MATCHED_REENTRY_DETAIL,
            "a lowered dispatch executes its retained program; it does not compile another \
             matched-design sensitivity analysis"
        ));
    }
    let question = Declaration::new("antecedent.matched_design_sensitivity.v1")
        .field("design", &design)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::MatchedDesignSensitivity,
        plan_id: "supplied.matched_design_sensitivity.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "supplied_matched_design",
        question,
        evaluator: design.evaluator(),
        reading: Reading::Envelope,
        role: EvidenceRole::Observations,
        composition: design.composition(),
        decision: None,
        claim: Some(ClaimKind::Bounds),
    })?;
    Ok(CompiledMatchedDesignSensitivity { lowered, design })
}

impl CompiledMatchedDesignSensitivity {
    /// The declared design.
    #[must_use]
    pub const fn design(&self) -> &MatchedDesign {
        &self.design
    }

    /// Bound the design's randomization test under every hidden bias of the grid.
    ///
    /// # Errors
    /// The kernel's own refusals (a non-finite or all-zero difference, a malformed set, `Γ`
    /// below one, `alpha` outside `(0, 1)`, a coordinate count other than the target's, a
    /// cancelled context, an exact table the context does not admit), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MatchedSensitivityOutcome, TerminalError<ValidationError>> {
        let scoped = self.enter(ctx)?;
        match &self.design {
            MatchedDesign::Pairs { target, differences, gammas, alpha } => {
                let (analysis, range) =
                    rosenbaum_range(target.clone(), differences, gammas, *alpha, Some(&scoped))
                        .map_err(TerminalError::Kernel)?;
                Ok(MatchedSensitivityOutcome::Pairs { analysis, range })
            }
            MatchedDesign::PairVector { target, differences, gammas, alpha } => {
                let (analyses, range) = rosenbaum_vector_range(
                    target.clone(),
                    differences,
                    gammas,
                    *alpha,
                    Some(&scoped),
                )
                .map_err(TerminalError::Kernel)?;
                Ok(MatchedSensitivityOutcome::PairVector { analyses, range })
            }
            MatchedDesign::Sets { sets, score, gammas, alpha } => {
                let grid = gammas
                    .iter()
                    .map(|&gamma| matched_sets_upper_p(sets, *score, gamma))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(TerminalError::Kernel)?;
                let gamma_break_even =
                    matched_sets_break_even(sets, *score, *alpha).map_err(TerminalError::Kernel)?;
                Ok(MatchedSensitivityOutcome::Sets { grid, gamma_break_even })
            }
        }
    }
}

// --- supplied estimates ----------------------------------------------------------------------

/// An effect estimate a caller supplies, computed outside the library, and how its E-values are
/// read (`VanderWeele` and Ding 2017).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SuppliedEstimate {
    /// One estimate and its interval on `scale`.
    Scalar {
        /// The target the range is bound to.
        target: RefutationTarget,
        /// The estimate and its limits (a ratio scale) or standard error (a standardized mean
        /// difference).
        interval: EffectInterval,
        /// The scale of the estimate.
        scale: EffectScale,
    },
    /// One estimate per coordinate of a vector target, read at the weakest coordinate; under a
    /// component model the range is bound to that model's assumption.
    Vector {
        /// The vector target (one label per coordinate).
        target: RefutationTarget,
        /// One estimate and interval per coordinate, in the target's label order.
        coordinates: Vec<EffectInterval>,
        /// The scale of every estimate.
        scale: EffectScale,
        /// The setting the coordinates are read in, when they are mediation components or
        /// marginal structural model parameters.
        model: Option<ComponentModel>,
    },
    /// Posterior draws of every coordinate of a vector target.
    Posterior {
        /// The vector target (one label per coordinate).
        target: RefutationTarget,
        /// `draws[k]` are the draws of coordinate `k` on `scale`, aligned across coordinates.
        draws: Vec<Vec<f64>>,
        /// The scale of the draws.
        scale: EffectScale,
        /// Credible level of the limit E-values.
        level: f64,
        /// The setting the coordinates are read in.
        model: Option<ComponentModel>,
    },
}

impl SuppliedEstimate {
    /// Stable name of the analysis.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Scalar { .. } => "estimate",
            Self::Vector { .. } => "estimate_vector",
            Self::Posterior { .. } => "estimate_posterior",
        }
    }

    const fn evaluator(&self) -> &'static str {
        match self {
            Self::Scalar { .. } => "evalue_scales::evalue_range",
            Self::Vector { model: None, .. } => "vector_sensitivity::evalue_vector_range",
            Self::Vector { model: Some(_), .. } => "component_sensitivity::component_evalue_range",
            Self::Posterior { model: None, .. } => "vector_sensitivity::evalue_posterior_range",
            Self::Posterior { model: Some(_), .. } => {
                "component_sensitivity::component_evalue_posterior_range"
            }
        }
    }

    const fn reading(&self) -> Reading {
        match self {
            Self::Scalar { .. } | Self::Vector { .. } => Reading::Envelope,
            Self::Posterior { .. } => Reading::Posterior,
        }
    }

    fn composition(&self) -> Composition {
        match self {
            Self::Scalar { .. } => Composition::Single,
            Self::Vector { coordinates, .. } => {
                Composition::ScenarioSet { scenarios: coordinates.len() }
            }
            Self::Posterior { draws, .. } => Composition::ScenarioSet { scenarios: draws.len() },
        }
    }
}

/// The E-values of a supplied estimate.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SuppliedEValueOutcome {
    /// [`SuppliedEstimate::Scalar`].
    Scalar {
        /// The risk ratio, its limit nearest the null and their E-values.
        bounds: EValueBounds,
        /// The range over confounder strengths.
        range: AssumptionRange,
    },
    /// [`SuppliedEstimate::Vector`].
    Vector {
        /// Per coordinate E-values.
        bounds: Vec<EValueBounds>,
        /// The range, read at the weakest coordinate.
        range: AssumptionRange,
    },
    /// [`SuppliedEstimate::Posterior`].
    Posterior {
        /// Posterior medians, credible limits, E-values and retention probabilities.
        summary: PosteriorEValue,
        /// The range.
        range: AssumptionRange,
    },
}

compiled_terminal! {
    /// A compiled E-value analysis of an effect estimate a caller supplies.
    #[derive(Clone, Debug)]
    CompiledSuppliedEstimateEValue {
        estimate: SuppliedEstimate,
    }
}

/// Compile the E-value analysis of `estimate`.
///
/// # Errors
/// `public_builder_reentry` (detail [`EVALUE_REENTRY_DETAIL`]) inside a lowered dispatch.
pub fn compile_supplied_estimate_evalue(
    estimate: SuppliedEstimate,
    ctx: &ExecutionContext,
) -> Result<CompiledSuppliedEstimateEValue, CausalError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(crate::unsupported_reason!(
            "public_builder_reentry",
            detail = EVALUE_REENTRY_DETAIL,
            "a lowered dispatch executes its retained program; it does not compile another \
             E-value analysis"
        ));
    }
    let question = Declaration::new("antecedent.supplied_estimate_evalue.v1")
        .field("estimate", &estimate)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::SuppliedEstimateEValue,
        plan_id: "supplied.supplied_estimate_evalue.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "supplied_effect_estimate",
        question,
        evaluator: estimate.evaluator(),
        reading: estimate.reading(),
        role: EvidenceRole::ChainSource,
        composition: estimate.composition(),
        decision: None,
        claim: Some(ClaimKind::Bounds),
    })?;
    Ok(CompiledSuppliedEstimateEValue { lowered, estimate })
}

impl CompiledSuppliedEstimateEValue {
    /// The declared estimate.
    #[must_use]
    pub const fn estimate(&self) -> &SuppliedEstimate {
        &self.estimate
    }

    /// The E-values of the estimate and the range over confounder strengths.
    ///
    /// # Errors
    /// The kernel's own refusals (a non-positive ratio, limits that do not bracket the estimate,
    /// a coordinate count other than the target's, misaligned or too few draws, a level outside
    /// `(0, 1)`), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SuppliedEValueOutcome, TerminalError<ValidationError>> {
        let _scoped = self.enter(ctx)?;
        match &self.estimate {
            SuppliedEstimate::Scalar { target, interval, scale } => {
                let (bounds, range) = evalue_range(target.clone(), *interval, *scale)
                    .map_err(TerminalError::Kernel)?;
                Ok(SuppliedEValueOutcome::Scalar { bounds, range })
            }
            SuppliedEstimate::Vector { target, coordinates, scale, model } => {
                let (bounds, range) = match model {
                    Some(model) => {
                        component_evalue_range(target.clone(), coordinates, *scale, *model)
                    }
                    None => evalue_vector_range(target.clone(), coordinates, *scale),
                }
                .map_err(TerminalError::Kernel)?;
                Ok(SuppliedEValueOutcome::Vector { bounds, range })
            }
            SuppliedEstimate::Posterior { target, draws, scale, level, model } => {
                let (summary, range) = match model {
                    Some(model) => component_evalue_posterior_range(
                        target.clone(),
                        draws,
                        *scale,
                        *level,
                        *model,
                    ),
                    None => evalue_posterior_range(target.clone(), draws, *scale, *level),
                }
                .map_err(TerminalError::Kernel)?;
                Ok(SuppliedEValueOutcome::Posterior { summary, range })
            }
        }
    }
}

// --- canonical identity of the declared inputs -----------------------------------------------

/// A target is its canonical identity: the digest of its family, quadrant, labels and source.
struct Target<'a>(&'a RefutationTarget);

impl Canonical for Target<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.bytes(self.0.identity());
    }
}

/// A matched set: its outcomes in order (the treated index points into them), then the index.
struct Set<'a>(&'a MatchedSet);

impl Canonical for Set<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.0.outcomes.encode(out);
        out.word(self.0.treated as u64);
    }
}

/// The within-set score.
fn score_tag(score: MatchedSetScore) -> &'static str {
    match score {
        MatchedSetScore::Outcomes => "outcomes",
        MatchedSetScore::AlignedRanks => "aligned_ranks",
    }
}

impl Canonical for MatchedDesign {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Differences, coordinates and sets are ordered: the ranks, the coordinates' labels and
        // the aligned ranks follow their order. The grid is ordered: results follow it.
        match self {
            Self::Pairs { target, differences, gammas, alpha } => {
                variant(out, "pairs");
                Target(target).encode(out);
                differences.encode(out);
                gammas.encode(out);
                alpha.encode(out);
            }
            Self::PairVector { target, differences, gammas, alpha } => {
                variant(out, "pair_vector");
                Target(target).encode(out);
                differences.encode(out);
                gammas.encode(out);
                alpha.encode(out);
            }
            Self::Sets { sets, score, gammas, alpha } => {
                variant(out, "sets").text(score_tag(*score));
                out.word(sets.len() as u64);
                for set in sets {
                    Set(set).encode(out);
                }
                gammas.encode(out);
                alpha.encode(out);
            }
        }
    }
}

/// An effect interval: its form, the estimate, then the limits or the standard error.
struct Interval<'a>(&'a EffectInterval);

impl Canonical for Interval<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            EffectInterval::Ratio { estimate, lower, upper } => {
                variant(out, "ratio");
                estimate.encode(out);
                lower.encode(out);
                upper.encode(out);
            }
            EffectInterval::Standardized { estimate, standard_error } => {
                variant(out, "standardized");
                estimate.encode(out);
                standard_error.encode(out);
            }
        }
    }
}

/// The setting the coordinates are read in, or none.
fn model_tag(model: Option<ComponentModel>) -> &'static str {
    match model {
        None => "effect",
        Some(ComponentModel::Mediation) => "mediation",
        Some(ComponentModel::MarginalStructural) => "marginal_structural",
    }
}

impl Canonical for SuppliedEstimate {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Scalar { target, interval, scale } => {
                variant(out, "scalar").text(scale.as_str());
                Target(target).encode(out);
                Interval(interval).encode(out);
            }
            Self::Vector { target, coordinates, scale, model } => {
                // Ordered: one coordinate per target label, in label order.
                variant(out, "vector").text(scale.as_str()).text(model_tag(*model));
                Target(target).encode(out);
                out.word(coordinates.len() as u64);
                for coordinate in coordinates {
                    Interval(coordinate).encode(out);
                }
            }
            Self::Posterior { target, draws, scale, level, model } => {
                // Ordered: draws are aligned across coordinates, and the joint retention
                // probability reads them draw by draw.
                variant(out, "posterior").text(scale.as_str()).text(model_tag(*model));
                Target(target).encode(out);
                draws.encode(out);
                level.encode(out);
            }
        }
    }
}
