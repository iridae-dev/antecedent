// SPDX-License-Identifier: MIT OR Apache-2.0
// Shared circular-block mixture SE for temporal class / DBN envelopes.

use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate,
    EstimationWorkspace, ExecutionContext, IdentificationStatus, IdentifiedEstimand, OverlapPolicy,
    TemporalDag, TemporalEffectQuery, TemporalIndexer, TemporalLinearAdjustment, TimeSeriesData,
};

/// Shared circular-block SE for a frozen-weight temporal class / DBN mixture.
pub struct SharedCircularBlockSe {
    /// Mixture SE: replicate SD scaled by the fixed-b factor (NaN when unavailable).
    pub se: f64,
    pub completed: u32,
    pub attempted: u32,
    /// Cooperative cancellation stopped the shared block loop early.
    pub cancelled: bool,
    /// Block length in series times.
    pub block_length: usize,
    /// Series times resampled (the window every atom can evaluate).
    pub rows: usize,
    /// Effective rows of every atom's and the weighted mixture's estimating score
    /// at the block length ([`antecedent_estimate::score_effective_rows`]; NaN
    /// when unknown).
    pub effective_rows: f64,
    /// Successful replicates: every atom's refit, in atom order.
    pub atom_draws: Vec<Vec<f64>>,
    /// Fixed-b factor applied to [`Self::se`].
    pub fixed_b: f64,
    /// Bartlett kernel-bias factor of every atom's influence and the mixture
    /// score at the block length ([`antecedent_estimate::kernel_bias_scale`]),
    /// applied to [`Self::se`] with [`Self::fixed_b`].
    pub kernel_bias: f64,
}

impl SharedCircularBlockSe {
    fn empty() -> Self {
        Self {
            se: f64::NAN,
            completed: 0,
            attempted: 0,
            cancelled: false,
            block_length: 0,
            rows: 0,
            effective_rows: f64::NAN,
            atom_draws: Vec::new(),
            fixed_b: 1.0,
            kernel_bias: 1.0,
        }
    }

    /// Imbens–Manski interval for the identified set spanned by the atoms'
    /// point estimates `points` (atom order), from the same shared replicates.
    ///
    /// Withheld when the shared block was cancelled or when successes cannot
    /// earn a nominal level under [`super::bootstrap_has_enough_successes`].
    pub fn identified_set_interval(
        &self,
        points: &[f64],
        level: f64,
    ) -> Option<antecedent_estimate::IdentifiedSetInterval> {
        if self.cancelled
            || !super::bootstrap_has_enough_successes(
                self.atom_draws.len(),
                self.attempted as usize,
            )
        {
            return None;
        }
        antecedent_estimate::imbens_manski_shared_replicates(
            points,
            &self.atom_draws,
            self.fixed_b * self.kernel_bias,
            self.rows,
            level,
        )
    }
}

/// One atom of a frozen-weight temporal mixture, prepared once on the original
/// series so the shared bootstrap can refit it on resampled lag-aligned rows.
#[allow(
    clippy::large_enum_variant,
    reason = "a mixture holds a handful of atoms built once per run; boxing the linear variant would only add an indirection to every replicate refit"
)]
pub enum TemporalAtomDesign {
    /// Pulse / single-step Sustained: one lag-aligned adjustment regression.
    Linear {
        prep: Box<antecedent_estimate::PreparedEstimationProblem>,
        rows: antecedent_estimate::AlignedRows,
        /// The intact windows when the series has missing time points.
        selection: Option<antecedent_estimate::GapSelection>,
        fitter: antecedent_estimate::LinearAdjustmentAte,
        point: EffectEstimate,
        normal_scores: Vec<Vec<f64>>,
    },
    /// Multi-step Sustained: sequential g-computation over the unfolded window.
    Sequential {
        design: Box<antecedent_estimate::SequentialContrastDesign>,
        point: f64,
        influence: Option<Vec<f64>>,
        normal_scores: Vec<Vec<f64>>,
    },
}

impl TemporalAtomDesign {
    /// Prepare a Pulse / single-step Sustained atom.
    pub fn linear(
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&antecedent_data::DiscoveryEstimationSplit>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::linear_with_gaps(data, estimand, query, indexer, split, false, ctx)
    }

    /// [`Self::linear`], on a series that may have missing time points when `gaps` is set:
    /// the atom is fit on its intact windows and the shared bootstrap resamples the original
    /// window axis. A split series is not combined with gaps.
    pub fn linear_with_gaps(
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&antecedent_data::DiscoveryEstimationSplit>,
        gaps: bool,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, rows, selection) = if gaps && split.is_none() {
            let (prep, selection) = estimator
                .prepare_with_gaps(data, estimand, query, indexer, &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let rows = antecedent_estimate::AlignedRows {
                first_time: selection.first_time,
                rows: prep.design.nrows,
            };
            let selection = selection.complete.iter().any(|intact| !intact).then_some(selection);
            (prep, rows, selection)
        } else {
            let (prep, rows) = estimator
                .prepare_aligned(data, estimand, query, indexer, split, &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            (prep, rows, None)
        };
        // The point fit's own OLS residuals give the nuisance scores: no second solve.
        let (point, residuals) = estimator
            .inner
            .fit_point_with_ols_residuals(
                &prep,
                &mut EstimationWorkspace::default(),
                antecedent_core::AssumptionSet::default(),
            )
            .map_err(CausalError::from)?;
        // Without the nuisance scores the block length silently falls back to the
        // influence-only rule, which under-sizes blocks for a persistent residual.
        let residuals = residuals.ok_or_else(|| CausalError::Compile {
            message: "normal-equation scores of the atom's regression could not be computed".into(),
        })?;
        let normal_scores = antecedent_estimate::normal_equation_scores_of_residuals(
            &prep.design.matrix,
            prep.design.nrows,
            prep.design.ncols,
            &residuals,
        );
        Ok(Self::Linear {
            prep: Box::new(prep),
            rows,
            selection,
            fitter: estimator.inner,
            point,
            normal_scores,
        })
    }

    /// Prepare a multi-step Sustained atom.
    pub fn sequential(
        data: &TimeSeriesData,
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::sequential_with_gaps(data, graph, indexer, estimand, query, status, false, ctx)
    }

    /// [`Self::sequential`], on a series that may have missing time points when `gaps` is set.
    #[allow(
        clippy::too_many_arguments,
        reason = "the sequential envelope's inputs plus the gap switch; each is a distinct caller input"
    )]
    pub fn sequential_with_gaps(
        data: &TimeSeriesData,
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        gaps: bool,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let design = antecedent_estimate::SequentialContrastDesign::prepare_with_gaps(
            &antecedent_estimate::SequentialProblem { data, graph, indexer, estimand },
            query,
            status,
            ctx,
            gaps,
        )
        .map_err(CausalError::from)?;
        let point = design.estimate().map_err(CausalError::from)?;
        let influence = design.influence();
        let normal_scores = design.normal_equation_scores();
        Ok(Self::Sequential { design: Box::new(design), point, influence, normal_scores })
    }

    /// The atom's windows with their intactness, for the gap-aware shared bootstrap.
    fn gapped_rows(&self) -> antecedent_estimate::GappedAlignedRows {
        match self {
            Self::Linear { selection: Some(selection), .. } => {
                antecedent_estimate::GappedAlignedRows {
                    first_time: selection.first_time,
                    complete: selection.complete.clone(),
                }
            }
            Self::Linear { rows, .. } => antecedent_estimate::GappedAlignedRows::contiguous(*rows),
            Self::Sequential { design, .. } => design.gapped_rows(),
        }
    }

    /// Whether the atom was fit on a series with dropped windows.
    pub fn has_gaps(&self) -> bool {
        match self {
            Self::Linear { selection, .. } => selection.is_some(),
            Self::Sequential { design, .. } => design.has_gaps(),
        }
    }

    fn aligned_rows(&self) -> antecedent_estimate::AlignedRows {
        match self {
            Self::Linear { rows, .. } => *rows,
            Self::Sequential { design, .. } => design.aligned_rows(),
        }
    }

    /// Full-sample point and (for linear atoms) iid analytic SE, with `assumptions`.
    pub fn effect_estimate(&self, assumptions: antecedent_core::AssumptionSet) -> EffectEstimate {
        match self {
            Self::Linear { point, .. } => {
                let mut estimate = point.clone();
                estimate.provenance.assumptions = assumptions;
                estimate
            }
            Self::Sequential { point, .. } => {
                EffectEstimate::new(*point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride)
            }
        }
    }

    fn estimate_on_rows(
        &self,
        rows: &[usize],
        workspace: &mut EstimationWorkspace,
        x_boot: &mut Vec<f64>,
        y_boot: &mut Vec<f64>,
    ) -> Option<f64> {
        match self {
            Self::Linear { prep, fitter, .. } => {
                let n = rows.len();
                x_boot.resize(n * prep.design.ncols, 0.0);
                y_boot.resize(n, 0.0);
                fitter.ate_on_row_indices_into(prep, workspace, rows, x_boot, y_boot).ok()
            }
            Self::Sequential { design, .. } => {
                design.estimate_on_rows_into(rows, &mut workspace.ols, x_boot, y_boot).ok()
            }
        }
    }

    /// Per-row influence of the atom's estimate on its own aligned rows.
    fn influence(&self) -> Option<&[f64]> {
        match self {
            Self::Linear { point, .. } => point.scores.influence.as_deref(),
            Self::Sequential { influence, .. } => influence.as_deref(),
        }
    }

    /// OLS normal-equation scores of every regression the atom fits, on its own
    /// aligned rows ([`antecedent_estimate::normal_equation_scores`]; an
    /// intercept's score is that regression's residual series). A sequential
    /// atom lists every mechanism's scores, the outcome mechanism's among them.
    fn normal_scores(&self) -> &[Vec<f64>] {
        match self {
            Self::Linear { normal_scores, .. } | Self::Sequential { normal_scores, .. } => {
                normal_scores
            }
        }
    }
}

/// Shared circular-block SE of a frozen-weight mixture over temporal atoms.
///
/// Blocks of consecutive series times are resampled over the window where the
/// maximal lag window across all atoms is available; each atom's lag-aligned
/// design keeps its rows' lag windows from the original series, every atom is
/// refit on the same resampled times, and the replicate SD is scaled by the
/// circular-Bartlett fixed-b factor ([`antecedent_estimate::circular_fixed_b_scale`])
/// and the Bartlett kernel-bias factor of every atom's influence and the mixture
/// score ([`antecedent_estimate::kernel_bias_scale`]).
/// The block length is [`mixture_block_length`] over the `m` shared times: at
/// least `max(structural_span, ceil(m^(1/3)))`, lengthened when any atom's
/// influence, the mixture score, or any normal-equation score of any atom's
/// regressions carries slowly decaying dependence. Unidentified mass is not
/// mixed; a replicate that cannot fit every atom is dropped rather than
/// renormalized. The interval is for the reported aggregate.
pub fn shared_circular_block_mixture_se(
    atoms: &[&TemporalAtomDesign],
    weights: &[f64],
    structural_span: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
) -> SharedCircularBlockSe {
    if atoms.iter().any(|atom| atom.has_gaps()) {
        return shared_gapped_block_mixture_se(
            atoms,
            weights,
            structural_span,
            replicates,
            stream_base,
            ctx,
        );
    }
    let designs: Vec<_> = atoms.iter().map(|atom| atom.aligned_rows()).collect();
    let Some((start, len)) =
        antecedent_estimate::common_time_window(&designs).filter(|_| replicates > 0)
    else {
        return SharedCircularBlockSe::empty();
    };
    let influences: Option<Vec<&[f64]>> = atoms
        .iter()
        .zip(&designs)
        .map(|(atom, design)| {
            let offset = start - design.first_time;
            atom.influence()?.get(offset..offset + len)
        })
        .collect();
    let score = influences.as_deref().and_then(|windows| mixture_score(windows, weights, len));
    let normal_windows: Vec<&[f64]> = atoms
        .iter()
        .zip(&designs)
        .flat_map(|(atom, design)| {
            let offset = start - design.first_time;
            atom.normal_scores().iter().filter_map(move |s| s.get(offset..offset + len))
        })
        .collect();
    let block_length = mixture_block_length(
        structural_span,
        len,
        influences.as_deref().unwrap_or_default(),
        score.as_deref(),
        &normal_windows,
    );
    // Every atom's score, not only the weighted sum: a mixture dominated by a
    // nearly iid atom can hide another atom's slowly decaying dependence.
    let target: Vec<&[f64]> =
        influences.iter().flatten().copied().chain(score.as_deref()).collect();
    let kernel_bias = antecedent_estimate::kernel_bias_scale(&target, block_length);
    let mut workspace = EstimationWorkspace::default();
    let mut x_boot = Vec::new();
    let mut y_boot = Vec::new();
    let mut out = shared_circular_block_mixture_se_with_length(
        &designs,
        weights,
        BlockBootstrapPlan { block_length, kernel_bias, replicates, stream_base },
        ctx,
        |atom, rows| atoms[atom].estimate_on_rows(rows, &mut workspace, &mut x_boot, &mut y_boot),
    );
    out.effective_rows = if score.is_some() {
        antecedent_estimate::score_effective_rows(&target, block_length)
    } else {
        f64::NAN
    };
    out
}

/// [`shared_circular_block_mixture_se`] when any atom was fit on a series with missing time
/// points: one block plan on the original window axis over the windows every atom reads,
/// keeping the times intact in every atom, and every atom refit on those same times.
///
/// The dependence-adaptive block length reads consecutive scores, which the compacted rows of
/// a gapped fit are not, so the structural rule stands (the lag span, at least `ceil(m^(1/3))`)
/// without the kernel-bias widening, and no effective-row count is reported.
fn shared_gapped_block_mixture_se(
    atoms: &[&TemporalAtomDesign],
    weights: &[f64],
    structural_span: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
) -> SharedCircularBlockSe {
    let total: f64 = weights.iter().sum();
    if replicates == 0 || atoms.is_empty() || atoms.len() != weights.len() || total <= 0.0 {
        return SharedCircularBlockSe::empty();
    }
    let designs: Vec<_> = atoms.iter().map(|atom| atom.gapped_rows()).collect();
    let start = designs.iter().map(|d| d.first_time).max().unwrap_or(0);
    let end = designs.iter().map(|d| d.first_time + d.complete.len()).min().unwrap_or(0);
    if end <= start {
        return SharedCircularBlockSe::empty();
    }
    let block_length = antecedent_data::circular_block_length(structural_span, end - start);
    let mut workspace = EstimationWorkspace::default();
    let mut x_boot = Vec::new();
    let mut y_boot = Vec::new();
    let Some(draws) = antecedent_estimate::gapped_aligned_block_bootstrap(
        &designs,
        block_length,
        replicates,
        stream_base,
        ctx,
        |maps| {
            let mut values = Vec::with_capacity(maps.len() + 1);
            let mut mixture = 0.0;
            for (atom, (rows, weight)) in maps.iter().zip(weights).enumerate() {
                let value =
                    atoms[atom].estimate_on_rows(rows, &mut workspace, &mut x_boot, &mut y_boot)?;
                mixture += weight / total * value;
                values.push(value);
            }
            values.push(mixture);
            Some(values)
        },
    ) else {
        return SharedCircularBlockSe::empty();
    };
    shared_se_from_draws(draws, atoms.len(), 1.0)
}

/// Block length of the shared circular block over `len` shared times:
/// [`antecedent_estimate::dependence_block_length`] of the structural span over
/// every atom's influence, the weighted mixture score, and every normal-equation
/// score of every atom's regressions (each a window of `len` shared times).
///
/// The nuisance scores are all included, not only one residual: a sequential
/// atom's scores run over every mechanism in topological order, and the
/// outcome mechanism's residual (the persistent score that moves every
/// replicate slope) is not in a fixed position among them. This is the rule
/// the single-window, mediation and standalone sequential paths use.
pub fn mixture_block_length(
    structural_span: usize,
    len: usize,
    influences: &[&[f64]],
    mixture: Option<&[f64]>,
    normal_scores: &[&[f64]],
) -> usize {
    let scores: Vec<&[f64]> =
        influences.iter().copied().chain(mixture).chain(normal_scores.iter().copied()).collect();
    antecedent_estimate::dependence_block_length(structural_span, len, &scores)
}

/// The weighted mixture estimating score `Σ_g w̄_g IF_g(t)` over the shared times.
fn mixture_score(influences: &[&[f64]], weights: &[f64], len: usize) -> Option<Vec<f64>> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut score = vec![0.0; len];
    for (influence, weight) in influences.iter().zip(weights) {
        for (slot, value) in score.iter_mut().zip(*influence) {
            *slot += weight / total * *value;
        }
    }
    Some(score)
}

/// Block-length rule for the shared circular block: the structural lag span
/// or `ceil(n^(1/3))`, whichever is longer, capped at `n` (the floor of
/// [`antecedent_estimate::dependence_block_length`]).
#[cfg(test)]
pub fn circular_block_length(structural_span: usize, n: usize) -> usize {
    antecedent_data::circular_block_length(structural_span, n)
}

/// The block length, kernel-bias factor, replicate count and stream base of one
/// shared circular-block bootstrap.
#[derive(Clone, Copy)]
pub struct BlockBootstrapPlan {
    pub block_length: usize,
    pub kernel_bias: f64,
    pub replicates: u32,
    pub stream_base: u64,
}

/// [`shared_circular_block_mixture_se`] over explicit aligned designs at an
/// explicit block length and kernel-bias factor. Production reaches it through
/// [`shared_circular_block_mixture_se`] at [`mixture_block_length`]; the
/// block-length sensitivity check calls it directly at multiples of that
/// length. `fit_atom(g, rows)` refits atom `g` on its design rows `rows`.
pub fn shared_circular_block_mixture_se_with_length(
    designs: &[antecedent_estimate::AlignedRows],
    weights: &[f64],
    plan: BlockBootstrapPlan,
    ctx: &ExecutionContext,
    mut fit_atom: impl FnMut(usize, &[usize]) -> Option<f64>,
) -> SharedCircularBlockSe {
    let BlockBootstrapPlan { block_length, kernel_bias, replicates, stream_base } = plan;
    let total: f64 = weights.iter().sum();
    if replicates == 0 || designs.is_empty() || designs.len() != weights.len() || total <= 0.0 {
        return SharedCircularBlockSe::empty();
    }
    let Some(draws) = antecedent_estimate::aligned_block_bootstrap(
        designs,
        block_length,
        replicates,
        stream_base,
        ctx,
        |maps| {
            let mut values = Vec::with_capacity(maps.len() + 1);
            let mut mixture = 0.0;
            for (atom, (rows, weight)) in maps.iter().zip(weights).enumerate() {
                let value = fit_atom(atom, rows)?;
                mixture += weight / total * value;
                values.push(value);
            }
            values.push(mixture);
            Some(values)
        },
    ) else {
        return SharedCircularBlockSe::empty();
    };
    shared_se_from_draws(draws, designs.len(), kernel_bias)
}

/// The mixture SE of `k` atoms from shared-block draws holding each atom's value and then the
/// mixture's.
fn shared_se_from_draws(
    draws: antecedent_estimate::RowBlockDraws,
    k: usize,
    kernel_bias: f64,
) -> SharedCircularBlockSe {
    let draws = antecedent_estimate::RowBlockDraws { kernel_bias, ..draws };
    // Cancelled shared blocks do not earn a nominal bootstrap SE even when the
    // success floor is met; cancellation is not adaptive early-stop.
    let se = if draws.cancelled { f64::NAN } else { draws.se_result(k).se.unwrap_or(f64::NAN) };
    SharedCircularBlockSe {
        se,
        completed: u32::try_from(draws.draws.len()).unwrap_or(u32::MAX),
        attempted: draws.attempted,
        cancelled: draws.cancelled,
        block_length: draws.block_length,
        rows: draws.rows,
        effective_rows: f64::NAN,
        fixed_b: draws.fixed_b(),
        kernel_bias,
        atom_draws: draws
            .draws
            .into_iter()
            .map(|mut draw| {
                draw.truncate(k);
                draw
            })
            .collect(),
    }
}

/// The shared-block provenance diagnostic for a class envelope, plus the
/// short-series warning when the mixture score is below the mixture threshold.
pub fn envelope_shared_block_diagnostics(
    identified_mass: f64,
    unidentified_mass: f64,
    block: &SharedCircularBlockSe,
) -> Vec<Diagnostic> {
    let mut out = vec![
        Diagnostic::new(
            "estimate.temporal_class.frequentist.shared_block",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            shared_block_mixture_message(
                "frozen completion weights",
                identified_mass,
                unidentified_mass,
                block,
            ),
        )
        .with_fields(super::mass_fields(Some(identified_mass), unidentified_mass)),
    ];
    out.extend(super::short_series_warning(
        block.effective_rows,
        antecedent_estimate::CircularBlockFamily::Mixture,
    ));
    out
}

/// One message for every frozen-weight shared circular-block mixture SE
/// (temporal class envelopes and DBN posteriors), so both carry the same
/// statement of how the interval is built and what it is for.
pub fn shared_block_mixture_message(
    weight_basis: &str,
    identified_mass: f64,
    unidentified_mass: f64,
    block: &SharedCircularBlockSe,
) -> String {
    format!(
        "{weight_basis}; identified_mass={identified_mass}; \
         unidentified_mass={unidentified_mass}; shared circular-block \
         replicates={}; attempted={}; blocks of {} consecutive series times \
         (dependence-aware length, at least max(span, ceil(m^(1/3)))) over the \
         {} times where every atom's lag window is available; each atom's lag-aligned \
         rows keep their original lag windows and every atom is refit on the same \
         resampled times; replicate SD scaled by the circular-Bartlett fixed-b factor \
         {:.4} and the Bartlett kernel-bias factor {:.4} of the atom and mixture \
         scores; score effective rows {:.1} (smallest over every atom's and the \
         mixture's score of the lag-1 and block-length readings); between-atom sampling \
         variance included; unidentified mass is not mixed into the SE; \
         the interval is for the reported aggregate, not a distribution \
         over graph-specific effects",
        block.completed,
        block.attempted,
        block.block_length,
        block.rows,
        block.fixed_b,
        block.kernel_bias,
        block.effective_rows,
    )
}
