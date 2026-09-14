// Shared circular-block mixture SE for temporal class / DBN envelopes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Shared circular-block SE for a frozen-weight temporal class / DBN mixture.
pub struct SharedCircularBlockSe {
    /// Mixture SE: replicate SD scaled by the fixed-b factor (NaN when unavailable).
    pub se: f64,
    pub completed: u32,
    pub attempted: u32,
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
}

impl SharedCircularBlockSe {
    fn empty() -> Self {
        Self {
            se: f64::NAN,
            completed: 0,
            attempted: 0,
            block_length: 0,
            rows: 0,
            effective_rows: f64::NAN,
            atom_draws: Vec::new(),
            fixed_b: 1.0,
        }
    }

    /// Imbens–Manski interval for the identified set spanned by the atoms'
    /// point estimates `points` (atom order), from the same shared replicates.
    pub fn identified_set_interval(
        &self,
        points: &[f64],
        level: f64,
    ) -> Option<antecedent_estimate::IdentifiedSetInterval> {
        if !super::bootstrap_has_enough_successes(self.atom_draws.len(), self.attempted as usize) {
            return None;
        }
        antecedent_estimate::imbens_manski_shared_replicates(
            points,
            &self.atom_draws,
            self.fixed_b,
            self.rows,
            level,
        )
    }
}

/// One atom of a frozen-weight temporal mixture, prepared once on the original
/// series so the shared bootstrap can refit it on resampled lag-aligned rows.
pub enum TemporalAtomDesign {
    /// Pulse / single-step Sustained: one lag-aligned adjustment regression.
    Linear {
        prep: Box<antecedent_estimate::PreparedEstimationProblem>,
        rows: antecedent_estimate::AlignedRows,
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
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, rows) = estimator
            .prepare_aligned(data, estimand, query, indexer, split, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let point = estimator
            .inner
            .fit_point(
                &prep,
                &mut EstimationWorkspace::default(),
                antecedent_core::AssumptionSet::default(),
            )
            .map_err(CausalError::from)?;
        let normal_scores = antecedent_estimate::normal_equation_scores(
            &prep.design.matrix,
            prep.design.nrows,
            prep.design.ncols,
            &prep.design.outcome,
        )
        .unwrap_or_default();
        Ok(Self::Linear {
            prep: Box::new(prep),
            rows,
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
        let design = antecedent_estimate::SequentialContrastDesign::prepare(
            data, graph, indexer, estimand, query, status, ctx,
        )
        .map_err(CausalError::from)?;
        let point = design.estimate().map_err(CausalError::from)?;
        let influence = design.influence();
        let normal_scores = design.normal_equation_scores();
        Ok(Self::Sequential { design: Box::new(design), point, influence, normal_scores })
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
                estimate.assumptions = assumptions;
                estimate
            }
            Self::Sequential { point, .. } => {
                EffectEstimate::new(*point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride)
            }
        }
    }

    fn estimate_on_rows(&self, rows: &[usize], workspace: &mut EstimationWorkspace) -> Option<f64> {
        match self {
            Self::Linear { prep, fitter, .. } => {
                let mut x = vec![0.0; rows.len() * prep.design.ncols];
                let mut y = vec![0.0; rows.len()];
                fitter.ate_on_row_indices_into(prep, workspace, rows, &mut x, &mut y).ok()
            }
            Self::Sequential { design, .. } => design.estimate_on_rows(rows).ok(),
        }
    }

    /// Per-row influence of the atom's estimate on its own aligned rows.
    fn influence(&self) -> Option<&[f64]> {
        match self {
            Self::Linear { point, .. } => point.influence.as_deref(),
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
/// circular-Bartlett fixed-b factor ([`antecedent_estimate::circular_fixed_b_scale`]).
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
    let mut workspace = EstimationWorkspace::default();
    let mut out = shared_circular_block_mixture_se_with_length(
        &designs,
        weights,
        block_length,
        replicates,
        stream_base,
        ctx,
        |atom, rows| atoms[atom].estimate_on_rows(rows, &mut workspace),
    );
    // Every atom's score, not only the weighted sum: a mixture dominated by a
    // nearly iid atom can hide another atom's slowly decaying dependence.
    out.effective_rows = if score.is_some() {
        let target: Vec<&[f64]> =
            influences.iter().flatten().copied().chain(score.as_deref()).collect();
        antecedent_estimate::score_effective_rows(&target, block_length)
    } else {
        f64::NAN
    };
    out
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

/// [`shared_circular_block_mixture_se`] over explicit aligned designs at an
/// explicit block length. Production reaches it through
/// [`shared_circular_block_mixture_se`] at [`mixture_block_length`]; the
/// block-length sensitivity check calls it directly at multiples of that
/// length. `fit_atom(g, rows)` refits atom `g` on its design rows `rows`.
pub fn shared_circular_block_mixture_se_with_length(
    designs: &[antecedent_estimate::AlignedRows],
    weights: &[f64],
    block_length: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    mut fit_atom: impl FnMut(usize, &[usize]) -> Option<f64>,
) -> SharedCircularBlockSe {
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
    let k = designs.len();
    let se = draws.se_result(k).se.unwrap_or(f64::NAN);
    SharedCircularBlockSe {
        se,
        completed: u32::try_from(draws.draws.len()).unwrap_or(u32::MAX),
        attempted: draws.attempted,
        block_length: draws.block_length,
        rows: draws.rows,
        effective_rows: f64::NAN,
        fixed_b: draws.fixed_b(),
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
    let mut out = vec![Diagnostic::new(
        "estimate.temporal_class.frequentist.shared_block",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        shared_block_mixture_message(
            "frozen completion weights",
            identified_mass,
            unidentified_mass,
            block,
        ),
    )];
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
         {:.4}; score effective rows {:.1} (smallest over every atom's and the \
         mixture's score of the lag-1 and block-length readings); between-atom sampling \
         variance included; unidentified mass is not mixed into the SE; \
         the interval is for the reported aggregate, not a distribution \
         over graph-specific effects",
        block.completed,
        block.attempted,
        block.block_length,
        block.rows,
        block.fixed_b,
        block.effective_rows,
    )
}
