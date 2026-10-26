//! Whole-procedure uncertainty of the linear unfolded temporal point routes.
//!
//! The temporal cross-world (nested, path-specific, unit counterfactual and
//! interventional mean), derivative and one-modifier conditional routes publish
//! functionals of one or more lag-aligned least-squares fits per identified
//! completion. Their point is computed through a [`SlopeAtom`], which records
//! every regression the point reads. The same point function is then rerun:
//!
//! * **Frequentist**: on every replicate of one shared circular-block bootstrap
//!   of series times ([`point_block_replicates`]). Every regression of every
//!   completion is refit on the same resampled times, each design row keeping its
//!   lag window from the original series, so a replicate is the whole route
//!   procedure (all nuisance and adjustment coefficients refit, the modifier mean
//!   recomputed, the frozen completion weights reapplied) and every output of the
//!   route (each coordinate of a vector or Jacobian, each completion) moves
//!   together. The declared sampling design of a single series is contiguous
//!   blocks; the replicate SD carries the circular fixed-b and Bartlett
//!   kernel-bias factors of the shared-block mixture SE.
//! * **Bayesian**: on every joint draw of the Gaussian-likelihood posteriors of all
//!   regressions of all completions under the isotropic slope prior the point already uses
//!   ([`point_posterior_draws`]), each tempered for serial dependence and tied together by the
//!   long-run covariance of their stacked normal-equation scores. Joint draw `d` enters draw
//!   `d` of the functional of every completion, so a product of two edge slopes, a Jacobian or
//!   a mixture over completions inherits the dependence the shared series and innovations
//!   give the regressions; a joint draw whose functional cannot be evaluated is removed from
//!   every completion together and counted as failed mass.
//!
//! Neither construction has a coverage record. Every interval is labelled
//! unmeasured by the calibration lookup like any other unlicensed construction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;

use super::{
    Arc, BootstrapAccounting, CausalError, CausalPosterior, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, ExecutionContext, IdentificationStatus, IdentifiedEstimand,
    InferenceDiagnostics, OverlapPolicy, TemporalEffectQuery, TemporalIndexer, TimeSeriesData,
    VariableId,
};
use antecedent_estimate::{
    SlopeFit, SlopePosteriorSpec, SlopeRefitScratch, TemporalSlopeDesign, TemporalSlopeSpec,
};

/// Diagnostic code of a published temporal point-route circular-block interval.
pub(super) const POINT_BLOCK_DIAGNOSTIC: &str = "estimate.temporal.point.circular_block";
/// Diagnostic code of a published temporal point-route posterior.
pub(super) const POINT_POSTERIOR_DIAGNOSTIC: &str = "estimate.temporal.point.posterior";
/// Diagnostic code of a withheld temporal point-route interval.
pub(super) const POINT_UNCERTAINTY_WITHHELD: &str = "estimate.temporal.point.uncertainty_withheld";
/// Backend label of the tempered slope posterior.
const POSTERIOR_BACKEND: &str = "temporal.slope.tempered_gaussian";

/// Stream base of the cross-world contrasts' shared block bootstrap.
pub(super) const CROSS_WORLD_BLOCK_STREAM: u64 = 0x7C05_5B10;
/// Stream base of the cross-world contrasts' slope posteriors.
pub(super) const CROSS_WORLD_POSTERIOR_STREAM: u64 = 0x7C05_5BA7;
/// Stream base of the derivative responses' shared block bootstrap.
pub(super) const DERIVATIVE_BLOCK_STREAM: u64 = 0x7DE2_1B10;
/// Stream base of the derivative responses' slope posteriors.
pub(super) const DERIVATIVE_POSTERIOR_STREAM: u64 = 0x7DE2_1BA7;
/// Stream base of the conditional effects' shared block bootstrap.
pub(super) const CONDITIONAL_BLOCK_STREAM: u64 = 0x7C0D_1B10;
/// Stream base of the conditional effects' slope posteriors.
pub(super) const CONDITIONAL_POSTERIOR_STREAM: u64 = 0x7C0D_1BA7;

/// A source of fitted intercepts and slopes for the point function of a route.
pub(super) trait SlopeSource {
    /// Intercept and slopes of `outcome` on `treatments`, the listed
    /// `contemporaneous` treatments read at the outcome's own lag.
    ///
    /// # Errors
    ///
    /// The regression cannot be fit (point) or was not recorded (replicate, draw).
    fn slopes(
        &self,
        outcome: VariableId,
        treatments: &[VariableId],
        contemporaneous: &[VariableId],
    ) -> Result<(f64, Vec<f64>), CausalError>;

    /// The marginal conditional effect `(β_T + Σ_j β_TW_j · mean(W_j)) · (active − control)`.
    ///
    /// # Errors
    ///
    /// The source holds no conditional design.
    fn conditional_effect(&self) -> Result<f64, CausalError>;
}

/// The regression one call of a point function asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SlopeKey {
    outcome: VariableId,
    treatments: Vec<VariableId>,
    contemporaneous: Vec<VariableId>,
}

impl SlopeKey {
    fn new(outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]) -> Self {
        Self { outcome, treatments: treatments.to_vec(), contemporaneous: contemporaneous.to_vec() }
    }
}

fn unrecorded_regression() -> CausalError {
    CausalError::Compile {
        message: "a temporal point replicate asked for a regression its point did not fit".into(),
    }
}

fn no_conditional_design() -> CausalError {
    CausalError::Compile { message: "temporal point source holds no conditional design".into() }
}

/// One identified completion (or the one `TemporalDag`) of a temporal point
/// route: its back-door estimand, the probe's lag offsets, and every regression
/// its point read, recorded as they are fit on the full series.
pub(super) struct SlopeAtom<'a> {
    data: &'a TimeSeriesData,
    estimand: IdentifiedEstimand,
    indexer: &'a TemporalIndexer,
    treatment_offset: i32,
    outcome_offset: i32,
    policy: antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
    conditional_delta: Option<f64>,
    designs: RefCell<Vec<(SlopeKey, TemporalSlopeDesign)>>,
}

impl<'a> SlopeAtom<'a> {
    /// An atom whose regressions are recorded as the point function asks for them.
    pub(super) fn new(
        data: &'a TimeSeriesData,
        estimand: &IdentifiedEstimand,
        indexer: &'a TemporalIndexer,
        (treatment_offset, outcome_offset): (i32, i32),
        policy: antecedent_core::KernelPolicy,
        prior_scale: Option<f64>,
    ) -> Self {
        Self {
            data,
            estimand: estimand.clone(),
            indexer,
            treatment_offset,
            outcome_offset,
            policy,
            prior_scale,
            conditional_delta: None,
            designs: RefCell::new(Vec::new()),
        }
    }

    /// An atom holding the conditional design (every effect modifier) of `query`, on the intact
    /// windows when `allow_gaps` is set.
    ///
    /// # Errors
    ///
    /// The refusals of the conditional point.
    pub(super) fn conditional(
        data: &'a TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &'a TemporalIndexer,
        policy: antecedent_core::KernelPolicy,
        prior_scale: Option<f64>,
        allow_gaps: bool,
    ) -> Result<Self, CausalError> {
        let (design, delta) =
            TemporalSlopeDesign::conditional(data, estimand, query, indexer, &policy, allow_gaps)
                .map_err(CausalError::from)?;
        let key = SlopeKey::new(query.outcome, &[query.treatment], &query.effect_modifiers);
        let mut atom = Self::new(data, estimand, indexer, (0, 0), policy, prior_scale);
        atom.conditional_delta = Some(delta);
        atom.designs.get_mut().push((key, design));
        Ok(atom)
    }

    fn design_index(&self, key: SlopeKey) -> Result<usize, CausalError> {
        if let Some(index) = self.designs.borrow().iter().position(|(recorded, _)| *recorded == key)
        {
            return Ok(index);
        }
        let design = TemporalSlopeDesign::slopes(
            self.data,
            &self.estimand,
            self.indexer,
            &TemporalSlopeSpec {
                outcome: key.outcome,
                treatments: &key.treatments,
                treatment_offset: self.treatment_offset,
                outcome_offset: self.outcome_offset,
                contemporaneous: &key.contemporaneous,
            },
            &self.policy,
        )
        .map_err(CausalError::from)?;
        let mut designs = self.designs.borrow_mut();
        designs.push((key, design));
        Ok(designs.len() - 1)
    }

    /// The recorded regressions, frozen for the replicates and draws.
    pub(super) fn freeze(self) -> FrozenAtom {
        let (keys, designs) = self.designs.into_inner().into_iter().unzip();
        FrozenAtom { keys, designs, conditional_delta: self.conditional_delta }
    }
}

impl SlopeSource for SlopeAtom<'_> {
    fn slopes(
        &self,
        outcome: VariableId,
        treatments: &[VariableId],
        contemporaneous: &[VariableId],
    ) -> Result<(f64, Vec<f64>), CausalError> {
        let index = self.design_index(SlopeKey::new(outcome, treatments, contemporaneous))?;
        let designs = self.designs.borrow();
        let fit = designs[index].1.fit(self.prior_scale).map_err(CausalError::from)?;
        Ok((fit.intercept(), fit.slopes(treatments.len())))
    }

    fn conditional_effect(&self) -> Result<f64, CausalError> {
        let delta = self.conditional_delta.ok_or_else(no_conditional_design)?;
        let designs = self.designs.borrow();
        let (_, design) = designs.first().ok_or_else(no_conditional_design)?;
        let fit = design.fit(self.prior_scale).map_err(CausalError::from)?;
        Ok(fit.conditional_slope() * delta)
    }
}

/// The regressions one completion's point read, in the order it read them.
pub(super) struct FrozenAtom {
    keys: Vec<SlopeKey>,
    designs: Vec<TemporalSlopeDesign>,
    conditional_delta: Option<f64>,
}

impl FrozenAtom {
    fn source<'f>(&'f self, fits: &'f [SlopeFit]) -> FitTable<'f> {
        FitTable { keys: &self.keys, fits, conditional_delta: self.conditional_delta }
    }

    /// Aligned rows of the atom's first regression (its sample size).
    pub(super) fn rows(&self) -> usize {
        self.designs.first().map_or(0, TemporalSlopeDesign::rows)
    }
}

/// One replicate's or one draw's fits of an atom's recorded regressions.
struct FitTable<'f> {
    keys: &'f [SlopeKey],
    fits: &'f [SlopeFit],
    conditional_delta: Option<f64>,
}

impl SlopeSource for FitTable<'_> {
    fn slopes(
        &self,
        outcome: VariableId,
        treatments: &[VariableId],
        contemporaneous: &[VariableId],
    ) -> Result<(f64, Vec<f64>), CausalError> {
        let key = SlopeKey::new(outcome, treatments, contemporaneous);
        let index = self
            .keys
            .iter()
            .position(|recorded| *recorded == key)
            .ok_or_else(unrecorded_regression)?;
        let fit = self.fits.get(index).ok_or_else(unrecorded_regression)?;
        Ok((fit.intercept(), fit.slopes(treatments.len())))
    }

    fn conditional_effect(&self) -> Result<f64, CausalError> {
        let delta = self.conditional_delta.ok_or_else(no_conditional_design)?;
        let fit = self.fits.first().ok_or_else(no_conditional_design)?;
        Ok(fit.conditional_slope() * delta)
    }
}

/// Replicate count, RNG stream base and structural lag span of one shared block bootstrap.
#[derive(Clone, Copy, Debug)]
pub(super) struct PointBlockPlan {
    pub replicates: u32,
    pub stream_base: u64,
    pub structural_span: usize,
}

/// Shared circular-block replicates of a temporal point route.
pub(super) struct PointBlockReplicates {
    /// Each successful replicate: the route's outputs, then (for a scalar route over
    /// several completions) each completion's value.
    pub draws: antecedent_estimate::RowBlockDraws,
    /// Outputs of the route at the front of every draw.
    pub outputs: usize,
    /// Per-completion columns after the outputs (zero unless requested).
    pub atom_columns: usize,
    /// Smallest effective-row count of the slope columns' normal-equation scores at the
    /// block length; `NaN` on a series with gaps.
    pub effective_rows: f64,
    /// Structural block-length rule, `max(span, ceil(m^(1/3)))` over the shared times.
    pub rule_length: usize,
}

impl PointBlockReplicates {
    /// Published SE of output `k`: the replicate SD times the fixed-b and kernel-bias
    /// factors; `None` when cancelled, refused, or too few replicates survived.
    pub(super) fn se(&self, k: usize) -> Option<f64> {
        if self.draws.cancelled
            || !super::bootstrap_has_enough_successes(
                self.draws.draws.len(),
                self.draws.attempted as usize,
            )
        {
            return None;
        }
        self.draws.se_result(k).se.filter(|se| se.is_finite() && *se > 0.0)
    }

    /// Successful replicates.
    pub(super) fn completed(&self) -> u32 {
        u32::try_from(self.draws.draws.len()).unwrap_or(u32::MAX)
    }

    /// Shared-block summary in the form the class-envelope helpers read: the
    /// aggregate SE (output 0) and every completion's replicate values.
    pub(super) fn shared_block(&self) -> super::class_envelope_se::SharedCircularBlockSe {
        let atom_draws = self
            .draws
            .draws
            .iter()
            .map(|draw| draw[self.outputs..self.outputs + self.atom_columns].to_vec())
            .collect();
        super::class_envelope_se::SharedCircularBlockSe {
            se: self.se(0).unwrap_or(f64::NAN),
            completed: self.completed(),
            attempted: self.draws.attempted,
            cancelled: self.draws.cancelled,
            block_length: self.draws.block_length,
            rows: self.draws.rows,
            effective_rows: self.effective_rows,
            atom_draws,
            fixed_b: self.draws.fixed_b(),
            kernel_bias: self.draws.kernel_bias,
        }
    }
}

fn resource(message: String) -> CausalError {
    CausalError::Resource { message }
}

/// Shared circular-block bootstrap of a whole temporal point route.
///
/// Every regression of every atom is refit on the same resampled series times
/// (one block plan over the times every regression can evaluate; on a series
/// with gaps, the times intact in every regression). `evaluate(a, source)` reruns
/// atom `a`'s point function on the replicate's fits; `aggregate` combines the atom
/// values into the route's outputs with the frozen weights. A replicate any refit
/// or functional fails on is dropped for every output. With `per_atom` every
/// atom's first value is appended after the outputs, so an identified-set interval
/// can read the same replicates.
///
/// # Errors
///
/// The admission pool refused the replicate buffers.
pub(super) fn point_block_replicates(
    atoms: &[FrozenAtom],
    plan: PointBlockPlan,
    per_atom: bool,
    ctx: &ExecutionContext,
    evaluate: impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
    aggregate: impl Fn(&[Vec<f64>]) -> Option<Vec<f64>>,
) -> Result<Option<PointBlockReplicates>, CausalError> {
    let designs: Vec<&TemporalSlopeDesign> =
        atoms.iter().flat_map(|atom| atom.designs.iter()).collect();
    if plan.replicates == 0 || designs.is_empty() {
        return Ok(None);
    }
    let gaps = designs.iter().any(|design| design.has_gaps());
    let mut outputs = 0usize;
    let atom_columns = if per_atom && atoms.len() > 1 { atoms.len() } else { 0 };
    let mut scratch = SlopeRefitScratch::default();
    let mut replicate = |maps: &[Vec<usize>]| -> Option<Vec<f64>> {
        let mut values = Vec::with_capacity(atoms.len());
        let mut offset = 0;
        for (index, atom) in atoms.iter().enumerate() {
            let mut fits = Vec::with_capacity(atom.designs.len());
            for (local, design) in atom.designs.iter().enumerate() {
                fits.push(design.refit_rows(&maps[offset + local], &mut scratch)?);
            }
            offset += atom.designs.len();
            values.push(evaluate(index, &atom.source(&fits)).ok()?);
        }
        let mut draw = aggregate(&values)?;
        outputs = draw.len();
        if atom_columns > 0 {
            for value in &values {
                draw.push(*value.first()?);
            }
        }
        Some(draw)
    };
    let (draws, effective_rows, rule_length) = if gaps {
        let windows: Vec<antecedent_estimate::GappedAlignedRows> =
            designs.iter().map(|design| design.windows().clone()).collect();
        let start = windows.iter().map(|w| w.first_time).max().unwrap_or(0);
        let end = windows.iter().map(|w| w.first_time + w.complete.len()).min().unwrap_or(0);
        if end <= start {
            return Ok(None);
        }
        let length = antecedent_data::circular_block_length(plan.structural_span, end - start);
        let Some(draws) = antecedent_estimate::gapped_aligned_block_bootstrap(
            &windows,
            length,
            plan.replicates,
            plan.stream_base,
            ctx,
            &mut replicate,
        ) else {
            return Ok(None);
        };
        (draws, f64::NAN, length)
    } else {
        let aligned: Vec<antecedent_estimate::AlignedRows> =
            designs.iter().map(|design| design.aligned_rows()).collect();
        let Some((start, len)) = antecedent_estimate::common_time_window(&aligned) else {
            return Ok(None);
        };
        let scores: Vec<Vec<Vec<f64>>> =
            designs.iter().map(|design| design.normal_scores()).collect();
        let mut all_windows: Vec<&[f64]> = Vec::new();
        let mut slope_windows: Vec<&[f64]> = Vec::new();
        for (design_scores, rows) in scores.iter().zip(&aligned) {
            let offset = start - rows.first_time;
            for (column, score) in design_scores.iter().enumerate() {
                if let Some(window) = score.get(offset..offset + len) {
                    all_windows.push(window);
                    if column > 0 {
                        slope_windows.push(window);
                    }
                }
            }
        }
        let rule = antecedent_data::circular_block_length(plan.structural_span, len);
        let length =
            antecedent_estimate::dependence_block_length(plan.structural_span, len, &all_windows);
        let kernel_bias = antecedent_estimate::kernel_bias_scale(&slope_windows, length);
        let effective = antecedent_estimate::score_effective_rows(&slope_windows, length);
        let Some(mut draws) = antecedent_estimate::aligned_block_bootstrap(
            &aligned,
            length,
            plan.replicates,
            plan.stream_base,
            ctx,
            &mut replicate,
        ) else {
            return Ok(None);
        };
        draws.kernel_bias = kernel_bias;
        (draws, effective, rule)
    };
    if let Some(refusal) = draws.refusal {
        return Err(resource(format!(
            "admission pool cannot fund the temporal point block bootstrap: {refusal}"
        )));
    }
    Ok(Some(PointBlockReplicates { draws, outputs, atom_columns, effective_rows, rule_length }))
}

/// Posterior draws of a temporal point route, by atom.
pub(super) struct PointPosteriorDraws {
    /// `atom_draws[a][d]` is draw `d` of atom `a`'s outputs. Draw `d` of every atom comes from
    /// the same joint draw of every regression of every atom, and the lists have equal length.
    pub atom_draws: Vec<Vec<Vec<f64>>>,
    /// Largest serial-dependence tempering factor over every regression.
    pub tempering: f64,
    /// Smallest effective-row count over every regression's slope scores.
    pub effective_rows: f64,
    /// Joint draws each regression was asked for.
    pub requested: usize,
    /// Joint draws on which some atom's functional could not be evaluated. Such a draw is
    /// removed from every atom together, so the surviving draws stay aligned, and it is counted
    /// here instead of being forgotten.
    pub failed: usize,
    /// Dependence block length of the stacked normal-equation scores that tie the regressions.
    pub block_length: usize,
}

impl PointPosteriorDraws {
    /// Fraction of the requested joint draws whose functional was not evaluable.
    pub(super) fn failed_mass(&self) -> f64 {
        if self.requested == 0 { 0.0 } else { self.failed as f64 / self.requested as f64 }
    }
}

/// Prior scale, draw count, structural span and RNG stream base of the slope posteriors.
#[derive(Clone, Copy, Debug)]
pub(super) struct PointPosteriorPlan {
    pub prior_scale: f64,
    pub draws: usize,
    pub structural_span: usize,
    pub stream_base: u64,
}

/// Lease `cells` f64 values on the admission pool before they are allocated.
fn admit_values(
    ctx: &ExecutionContext,
    cells: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    let bytes = cells
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| resource("temporal point posterior size overflow".to_owned()))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::stage::STAGE_ESTIMATE_POINT }
        }
        other => resource(other.to_string()),
    })
}

/// Joint draws of every atom's point function under the tempered Gaussian-likelihood
/// posterior of all of its regressions together.
///
/// Every regression of every atom is read from one series, so their coefficients are drawn
/// jointly ([`TemporalSlopeDesign::joint_posterior`]): each regression keeps its tempered
/// Gaussian marginal and the regressions are tied by the long-run covariance of their stacked
/// normal-equation scores, which carries shared innovations and shared regressors. Joint draw
/// `d` enters draw `d` of every atom's functional, so products of slopes, mixtures over
/// completions and contrasts across mechanisms inherit that dependence. A joint draw on which
/// any atom's functional cannot be evaluated (a log scale at a non-positive fitted mean) is
/// removed from every atom and counted as failed mass; fewer than two surviving draws, or
/// more than half failed, refuses.
///
/// # Errors
///
/// Cancellation, an admission refusal, regressions whose joint posterior cannot be formed, or
/// too many unevaluable draws.
pub(super) fn point_posterior_draws(
    atoms: &[FrozenAtom],
    plan: PointPosteriorPlan,
    ctx: &ExecutionContext,
    evaluate: impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
) -> Result<PointPosteriorDraws, CausalError> {
    let coefficient_cells: usize = atoms
        .iter()
        .flat_map(|atom| atom.designs.iter())
        .map(|design| design.cols().saturating_mul(plan.draws))
        .sum();
    let _coefficients = admit_values(ctx, coefficient_cells)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::super::stage::STAGE_ESTIMATE_POINT });
    }
    let designs: Vec<&TemporalSlopeDesign> =
        atoms.iter().flat_map(|atom| atom.designs.iter()).collect();
    let (posteriors, block_length) = TemporalSlopeDesign::joint_posterior(
        &designs,
        SlopePosteriorSpec {
            prior_scale: plan.prior_scale,
            draws: plan.draws,
            structural_span: plan.structural_span,
        },
        plan.stream_base,
        &ctx.rng,
    )
    .map_err(CausalError::from)?;
    let tempering = posteriors.iter().fold(1.0_f64, |t, p| t.max(p.tempering));
    let effective_rows = posteriors
        .iter()
        .map(|p| p.effective_rows)
        .filter(|rows| rows.is_finite())
        .fold(f64::NAN, f64::min);
    let mut offsets = Vec::with_capacity(atoms.len());
    let mut next = 0usize;
    for atom in atoms {
        offsets.push(next);
        next += atom.designs.len();
    }
    let mut output_lease = None;
    let mut atom_draws: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(plan.draws); atoms.len()];
    let mut failed = 0usize;
    'draws: for d in 0..plan.draws {
        let mut row = Vec::with_capacity(atoms.len());
        for (index, atom) in atoms.iter().enumerate() {
            let fits: Vec<SlopeFit> = posteriors
                [offsets[index]..offsets[index] + atom.designs.len()]
                .iter()
                .map(|posterior| SlopeFit {
                    coefficients: posterior.draws[d].clone(),
                    modifier_mean: posterior.modifier_mean,
                    modifier_means: posterior.modifier_means.clone(),
                })
                .collect();
            let Ok(values) = evaluate(index, &atom.source(&fits)) else {
                failed += 1;
                continue 'draws;
            };
            if !values.iter().all(|value| value.is_finite()) {
                failed += 1;
                continue 'draws;
            }
            row.push(values);
        }
        if output_lease.is_none() {
            let width = row.first().map_or(0, Vec::len);
            output_lease = Some(admit_values(
                ctx,
                width.saturating_mul(plan.draws).saturating_mul(atoms.len()),
            )?);
        }
        for (slot, values) in atom_draws.iter_mut().zip(row) {
            slot.push(values);
        }
    }
    let kept = atom_draws.first().map_or(0, Vec::len);
    if kept < 2 || kept * 2 < plan.draws {
        return Err(CausalError::Unsupported {
            message: "the temporal point posterior could not evaluate enough draws of the functional (a log scale needs a positive fitted mean and treatment point in most draws)",
        });
    }
    Ok(PointPosteriorDraws {
        atom_draws,
        tempering,
        effective_rows,
        requested: plan.draws,
        failed,
        block_length,
    })
}

/// Linear-interpolated empirical quantile of sorted values (`p` in `[0, 1]`).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the clamped position lies in [0, len - 1], so its floor and ceiling are valid indices"
    )]
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    let fraction = position - low as f64;
    sorted[low] + (sorted[high] - sorted[low]) * fraction
}

/// Mean, SD and the equal-tailed `level` interval of a draw column.
pub(super) fn draw_summary(values: &[f64], level: f64) -> (f64, f64, f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let tail = (1.0 - level) / 2.0;
    (mean, variance.sqrt(), quantile(&sorted, tail), quantile(&sorted, 1.0 - tail))
}

/// A one-column effect posterior from scalar `draws`, labelled with the slope
/// posterior's tempering and the route's identification.
pub(super) fn scalar_posterior(
    draws: &[f64],
    name: &str,
    identification: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    tempering: f64,
) -> Result<CausalPosterior, CausalError> {
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: Arc::from(name),
        }]),
    };
    let posterior_draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        draws.len(),
        Arc::<[f64]>::from(draws.to_vec()),
    )
    .map_err(CausalError::from)?;
    let summaries = posterior_draws.summarize();
    let mut diagnostics = InferenceDiagnostics::analytic(POSTERIOR_BACKEND);
    diagnostics.notes.push(Arc::from(format!(
        "Gaussian likelihood of each lag-aligned regression with the residual variance plugged \
         in, isotropic N(0, prior_scale^2) slope prior, flat intercept; covariance widened by \
         the serial-dependence tempering factor {tempering:.3} (aligned rows over the effective \
         rows of the slope scores)"
    )));
    Ok(CausalPosterior {
        draws: posterior_draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    })
}

/// The frequentist scalar estimate of a point route: `point` with its shared-block
/// bootstrap SE and accounting, or the bare point when no replicates ran.
pub(super) fn block_scalar_estimate(
    point: f64,
    replicates: Option<&PointBlockReplicates>,
    assumptions: antecedent_core::AssumptionSet,
    family: antecedent_estimate::CircularBlockFamily,
) -> EffectEstimate {
    let Some(replicates) = replicates else {
        return EffectEstimate::new(point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
    };
    EffectEstimate::from_parts(
        point,
        f64::NAN,
        BootstrapAccounting {
            se: replicates.se(0),
            replicates_ok: Some(replicates.completed()),
            replicates_failed: Some(
                replicates.draws.attempted.saturating_sub(replicates.completed()),
            ),
            cancelled: replicates.draws.cancelled,
            early_stopped: false,
        },
        assumptions,
        OverlapPolicy::ExplicitOverride,
        None,
        None,
    )
    .with_block_family(family)
}

/// The provenance diagnostic of a shared-block interval, plus the short-series
/// warning of its family, or the withheld note when no SE was earned.
pub(super) fn block_diagnostics(
    replicates: Option<&PointBlockReplicates>,
    requested: u32,
    family: antecedent_estimate::CircularBlockFamily,
    refit: &str,
) -> Vec<Diagnostic> {
    let Some(replicates) = replicates else {
        return vec![Diagnostic::new(
            POINT_UNCERTAINTY_WITHHELD,
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            if requested == 0 {
                "no bootstrap replicates were requested, so no interval is published".to_owned()
            } else {
                "the regressions share no resampling window, so no interval is published".to_owned()
            },
        )];
    };
    let mut out = vec![Diagnostic::new(
        POINT_BLOCK_DIAGNOSTIC,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "shared circular-block bootstrap of series times: replicates={}; attempted={}; \
             blocks of {} consecutive times (dependence-aware, at least the rule length {}) over \
             the {} times every regression can evaluate; {refit}; each design row keeps its lag \
             window from the original series; replicate SD scaled by the circular fixed-b factor \
             {:.4} and the Bartlett kernel-bias factor {:.4}; slope-score effective rows {:.1}; \
             no coverage record measures this interval",
            replicates.completed(),
            replicates.draws.attempted,
            replicates.draws.block_length,
            replicates.rule_length,
            replicates.draws.rows,
            replicates.draws.fixed_b(),
            replicates.draws.kernel_bias,
            replicates.effective_rows,
        ),
    )];
    if replicates.se(0).is_none() {
        out.push(Diagnostic::new(
            POINT_UNCERTAINTY_WITHHELD,
            DiagnosticKind::Execution,
            DiagnosticSeverity::Warning,
            "the shared block bootstrap was cancelled or too few replicates survived, so no \
             interval is published",
        ));
    }
    out.extend(super::short_series_warning(replicates.effective_rows, family));
    out
}

/// The provenance diagnostic of a published slope posterior.
pub(super) fn posterior_diagnostic(draws: &PointPosteriorDraws, functional: &str) -> Diagnostic {
    let kept = draws.atom_draws.iter().map(Vec::len).min().unwrap_or(0);
    Diagnostic::new(
        POINT_POSTERIOR_DIAGNOSTIC,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "{functional}: the coefficients of every regression of every completion are one joint \
             draw from the Gaussian-likelihood posteriors under the isotropic slope prior, the \
             residual variance plugged in, each regression keeping its own tempered marginal and \
             all tied by the long-run covariance of their stacked normal-equation scores (block \
             length {}), so shared innovations and regressors carry across mechanisms; draw d of \
             every regression enters draw d of the functional; draws kept={kept} of {} with \
             failed mass {:.4} ({} joint draws where some completion's functional was not \
             evaluable, removed from every completion together); serial-dependence tempering \
             factor {:.3}; slope-score effective rows {:.1}; no coverage record measures this \
             interval",
            draws.block_length,
            draws.requested,
            draws.failed_mass(),
            draws.failed,
            draws.tempering,
            draws.effective_rows,
        ),
    )
}

/// Frozen-weight mean of atom outputs (every atom the same width), or `None`.
pub(super) fn weighted_outputs(values: &[Vec<f64>], weights: &[f64]) -> Option<Vec<f64>> {
    let width = values.first()?.len();
    let total: f64 = weights.iter().sum();
    if values.len() != weights.len() || total.is_nan() || total <= 0.0 {
        return None;
    }
    let mut out = vec![0.0; width];
    for (row, weight) in values.iter().zip(weights) {
        if row.len() != width {
            return None;
        }
        for (slot, value) in out.iter_mut().zip(row) {
            *slot += weight * value;
        }
    }
    for slot in &mut out {
        *slot /= total;
    }
    Some(out)
}

/// Flattened values of a derivative response (a scalar, a vector, or a Jacobian).
pub(super) fn response_values(value: &antecedent_core::ResponseValue) -> Option<Vec<f64>> {
    use antecedent_core::ResponseValue as V;
    match value {
        V::Scalar(value) => Some(vec![*value]),
        V::Vector(values) | V::Jacobian { values, .. } => Some(values.to_vec()),
        _ => None,
    }
}

/// Attach the shared-block band of a frequentist derivative response: a scalar
/// interval or a pointwise band at the reported level, the block disclosure the
/// interval binding reads, and for more than one coordinate the sup-t
/// simultaneous band from the same joint replicates.
pub(super) fn attach_block_response_band(
    response: &mut antecedent_core::CausalResponse,
    replicates: &PointBlockReplicates,
    center: &[f64],
) {
    use antecedent_core::{IntervalInterpretation, ResponseUncertainty, SupportDiagnostic};
    let level = crate::result::REPORTED_SE_INTERVAL_LEVEL;
    let z = crate::result::reported_se_interval_z();
    let ses: Option<Vec<f64>> = (0..center.len()).map(|k| replicates.se(k)).collect();
    let Some(ses) = ses else {
        response.support.warnings.push(Diagnostic::new(
            POINT_UNCERTAINTY_WITHHELD,
            DiagnosticKind::Execution,
            DiagnosticSeverity::Warning,
            "the shared block bootstrap was cancelled or too few replicates survived, so no band is published",
        ));
        return;
    };
    let lower: Vec<f64> = center.iter().zip(&ses).map(|(c, se)| c - z * se).collect();
    let upper: Vec<f64> = center.iter().zip(&ses).map(|(c, se)| c + z * se).collect();
    response.uncertainty = if center.len() == 1 {
        ResponseUncertainty::Scalar {
            standard_error: ses[0],
            level,
            lower: lower[0],
            upper: upper[0],
            interpretation: IntervalInterpretation::Confidence,
            draws: None,
        }
    } else {
        ResponseUncertainty::PointwiseBand {
            level,
            lower: Arc::from(lower),
            upper: Arc::from(upper),
            interpretation: IntervalInterpretation::Confidence,
            draws: None,
        }
    };
    let scale = replicates.draws.se_scale();
    response.support.diagnostics.push(SupportDiagnostic {
        id: Arc::from(antecedent_estimate::TEMPORAL_RESPONSE_BLOCK_LENGTH),
        values: Arc::from([
            replicates.draws.block_length as f64,
            replicates.rule_length as f64,
            replicates.draws.block_length as f64,
            replicates.draws.rows as f64,
            scale,
        ]),
        detail: Arc::from(
            "[block length, structural rule length, dependence-aware length, resampled series \
             times, replicate SD factor (circular fixed-b times Bartlett kernel bias)] of the \
             shared circular-block bootstrap that refits every regression of the derivative",
        ),
        scope: antecedent_core::DiagnosticScope::Global,
    });
    if center.len() > 1 {
        let scaled: Vec<Vec<f64>> = replicates
            .draws
            .draws
            .iter()
            .map(|draw| center.iter().zip(draw).map(|(c, value)| c + scale * (value - c)).collect())
            .collect();
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            antecedent_estimate::max_deviation_band(center, &scaled, level),
            "sup-t over the joint shared-block replicates of every coordinate",
        );
    }
}

/// Attach the credible band of a Bayesian derivative response from joint draws
/// (`draws[d]` is draw `d` of every coordinate).
pub(super) fn attach_posterior_response_band(
    response: &mut antecedent_core::CausalResponse,
    draws: &[Vec<f64>],
    center: &[f64],
) {
    use antecedent_core::{CredibleDraws, IntervalInterpretation, ResponseUncertainty};
    let level = crate::result::REPORTED_SE_INTERVAL_LEVEL;
    let width = center.len();
    let n = draws.len();
    let mut columns = vec![0.0; width * n];
    for (d, draw) in draws.iter().enumerate() {
        for (j, value) in draw.iter().enumerate().take(width) {
            columns[j * n + d] = *value;
        }
    }
    let summaries: Vec<(f64, f64, f64, f64)> =
        (0..width).map(|j| draw_summary(&columns[j * n..(j + 1) * n], level)).collect();
    response.uncertainty = if width == 1 {
        let (_, sd, lower, upper) = summaries[0];
        ResponseUncertainty::Scalar {
            standard_error: sd,
            level,
            lower,
            upper,
            interpretation: IntervalInterpretation::Credible,
            draws: Some(CredibleDraws::scalar(columns)),
        }
    } else {
        ResponseUncertainty::PointwiseBand {
            level,
            lower: summaries.iter().map(|s| s.2).collect(),
            upper: summaries.iter().map(|s| s.3).collect(),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(CredibleDraws { n_draws: n, values: Arc::from(columns) }),
        }
    };
    if width > 1 {
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            antecedent_estimate::max_deviation_band(center, draws, level),
            "sup-t over the joint posterior draws of every coordinate",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{draw_summary, quantile, weighted_outputs};

    #[test]
    fn quantiles_interpolate_and_summaries_bracket_the_mean() {
        let sorted = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&sorted, 0.5) - 2.0).abs() < 1e-12);
        assert!((quantile(&sorted, 0.125) - 0.5).abs() < 1e-12);
        let (mean, sd, lower, upper) = draw_summary(&[4.0, 0.0, 2.0, 1.0, 3.0], 0.5);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!(sd > 0.0 && lower < mean && mean < upper);
    }

    #[test]
    fn frozen_weights_mix_every_output() {
        let mixed = weighted_outputs(&[vec![1.0, 2.0], vec![3.0, 6.0]], &[1.0, 3.0]).unwrap();
        assert!((mixed[0] - 2.5).abs() < 1e-12 && (mixed[1] - 5.0).abs() < 1e-12);
        assert!(weighted_outputs(&[vec![1.0], vec![1.0, 2.0]], &[1.0, 1.0]).is_none());
        assert!(weighted_outputs(&[vec![1.0]], &[0.0]).is_none());
    }
}
