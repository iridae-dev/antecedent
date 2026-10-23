//! Fitted dose-horizon surfaces and their cells.

use std::sync::Arc;

use crate::adjustment::PreparedEstimationProblem;
use crate::error::EstimationError;
use crate::temporal_block::{AlignedRows, normal_equation_scores_of_residuals};
use crate::temporal_response_dispersion::CellDispersion;
use antecedent_core::{
    AssumptionSet, Diagnostic, DiagnosticKind, DiagnosticSeverity, HorizonIdentification,
    IdentificationStatus, MAX_TEMPORAL_RESPONSE_CELLS, ResponseUncertainty, SupportReport,
    TemporalNodeKey, VariableId,
};
use antecedent_data::TemporalIndexer;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use super::{
    assumptions::BLOCK_BOOTSTRAP_CONSTRUCTION, assumptions::SURFACE_REFIT,
    band::TEMPORAL_RESPONSE_BAND_WITHHELD, band::max_deviation_band,
    band::publish_simultaneous_band, block::ResponseBlockLength, block::block_dispersion_inflation,
    block::disclose_response_block_bootstrap, block::inflate_replicates,
    bootstrap::SurfaceBootstrap, bootstrap::sample_std_cell,
};

/// Support-diagnostic id carrying the joint replicates of the whole dose × horizon surface,
/// replicate-major: `values[r * cells + c]` is cell `c` of replicate `r`, in the cell layout of
/// the mean surface. Each replicate refits every horizon on the same time-aligned blocks (or
/// under one block-weight vector), so a row is one realization coupled across horizons. Published
/// only when the estimator is asked to retain them.
pub const JOINT_HORIZON_REPLICATES: &str = "response.joint_horizon.replicates";

/// Support-diagnostic id carrying `[replicates, cells, horizon index of cell 0, …]`: the shape
/// of [`JOINT_HORIZON_REPLICATES`] and, for each cell, the index (into the requested horizons)
/// of the horizon it reads. Its detail names the resampling that drew the replicates.
pub const JOINT_HORIZON_LAYOUT: &str = "response.joint_horizon.layout";

/// Index of the treatment column in the compiled design (col0 = intercept, col1 = treatment).
/// Verified against `CompiledDesign::linear_adjustment`.
pub(super) const TREATMENT_COL: usize = 1;

/// Per-horizon lag-aligned observed treatment `(min, max)`.
pub(super) type HorizonTreatmentRange = (f64, f64);

/// Resolved Sequence leaf: variable, fixed level, shift, and active offsets.
pub(super) type SequenceLeaf = (VariableId, Option<f64>, f64, Arc<[i32]>);

/// Every requested horizon fitted once on the full sample, plus the joint
/// surface bootstrap (when requested and usable) and the retained per-horizon
/// treatment ranges and identification records.
pub(super) struct FittedSurface {
    pub(super) horizons: Vec<FittedHorizon>,
    pub(super) ranges: Vec<HorizonTreatmentRange>,
    pub(super) identification: Vec<HorizonIdentification>,
    pub(super) bootstrap: Option<SurfaceBootstrap>,
    pub(super) se_provenance: SeProvenance,
    pub(super) block: ResponseBlockLength,
    /// The replicates are Dirichlet block-weight draws (the posterior over the surface), not
    /// bootstrap counts.
    pub(super) weighted: bool,
    /// Publish the joint replicates behind the band as support diagnostics.
    pub(super) retain_joint: bool,
}

/// Where a surface's pointwise SEs came from, so the band can say so.
///
/// A requested bootstrap that degenerates to the analytic SE, or one truncated by
/// cancellation, is reported rather than passed off as a completed bootstrap.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SeProvenance {
    /// A bootstrap was requested but yielded no usable joint draws.
    pub(super) fell_back: bool,
    /// The bootstrap was cut short by cooperative cancellation.
    pub(super) cancelled: bool,
}

impl SeProvenance {
    /// Warning describing a degraded bootstrap, or `None` when SEs are as requested.
    fn warning(self, replicates: u32) -> Option<Diagnostic> {
        if replicates == 0 || (!self.fell_back && !self.cancelled) {
            return None;
        }
        let message = if self.fell_back {
            format!(
                "requested {replicates} bootstrap replicates for the response surface, but too \
                 few joint resamples survived, so no band is published (the analytic OLS \
                 band would treat lag-aligned rows as independent)"
            )
        } else {
            format!(
                "requested {replicates} bootstrap replicates for the response surface; the joint \
                 bootstrap was truncated by cancellation"
            )
        };
        Some(Diagnostic::new(
            "estimate.temporal_response.bootstrap_degraded",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            message,
        ))
    }
}

/// One cell of the reported surface: which horizon, and what is evaluated there.
#[derive(Clone, Copy, Debug)]
pub(super) enum CellEval {
    /// `E[Y_h | do(A = dose)]`: treatment column fixed at `dose`.
    Dose(f64),
    /// `E[Y_h | do(A := A + shift)]`: treatment column at its sample mean plus `shift`.
    Shift(f64),
}

/// Point surface and, from the joint bootstrap only, the pointwise band and draws.
pub(super) struct SurfaceCells {
    pub(super) mean: Vec<f64>,
    pub(super) band: Option<CellBand>,
    /// Index of the horizon each cell reads.
    pub(super) cell_horizon: Vec<usize>,
}

/// Pointwise band and the joint replicate draws behind it.
pub(super) struct CellBand {
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    /// `draws[r]` is replicate `r`'s full surface in the output layout.
    pub(super) draws: Vec<Vec<f64>>,
    /// Each cell's kernel-bias factor and effective rows.
    pub(super) dispersion: CellDispersion,
    /// What the band's level means: sampling coverage of a bootstrap, or posterior mass of the
    /// block-weight draws.
    pub(super) interpretation: antecedent_core::IntervalInterpretation,
}

impl SurfaceCells {
    /// The pointwise band, published only when it comes from the joint
    /// circular-block bootstrap of lag-aligned rows.
    ///
    /// An analytic delta-method band would treat lag-aligned rows as independent;
    /// temporal rows are not, so without dependence-preserving replicates no band
    /// is published (the same rule as the scalar temporal Pulse / Sustained SE).
    pub(super) fn published_band(&self) -> ResponseUncertainty {
        self.band.as_ref().map_or(ResponseUncertainty::None, |band| {
            ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(band.lower.as_slice()),
                upper: Arc::from(band.upper.as_slice()),
                interpretation: band.interpretation,
                draws: None,
            }
        })
    }
}

impl FittedSurface {
    /// Fewest lag-aligned rows over the fitted horizons: the common time window the
    /// joint bootstrap resamples.
    fn rows(&self) -> usize {
        self.horizons.iter().map(|fitted| fitted.prepared.design.nrows).min().unwrap_or(0)
    }

    /// Fixed-b dispersion factor for this surface's bootstrap.
    fn inflation(&self) -> f64 {
        let parameters =
            self.horizons.iter().map(|fitted| fitted.prepared.design.ncols).max().unwrap_or(0);
        block_dispersion_inflation(self.rows(), self.block.length, parameters)
    }

    /// Point values and, from the joint bootstrap only, the pointwise 95% band and
    /// the replicate draws for `cells`, in the order given. Draws are returned after
    /// the block-count dispersion inflation and each cell's kernel-bias factor, so both
    /// bands use the inflated dispersion.
    pub(super) fn cells(&self, cells: &[(usize, CellEval)]) -> SurfaceCells {
        let mean: Vec<f64> = cells
            .iter()
            .map(|&(h, eval)| {
                let fitted = &self.horizons[h];
                level_at(&fitted.coefs, &fitted.column_means, eval)
            })
            .collect();
        let band = self.bootstrap.as_ref().map(|boot| {
            let z = antecedent_stats::NORMAL_Q975;
            let mut draws: Vec<Vec<f64>> = boot
                .draws
                .iter()
                .map(|replicate| {
                    cells
                        .iter()
                        .map(|&(h, eval)| {
                            let (coefs, means) = boot.horizon_fit(replicate, h);
                            level_at(coefs, means, eval)
                        })
                        .collect()
                })
                .collect();
            inflate_replicates(&mean, &mut draws, self.inflation());
            let dispersion = cell_dispersion(&self.horizons, cells, self.block.length);
            dispersion.inflate(&mean, &mut draws);
            let se: Vec<f64> = (0..cells.len()).map(|cell| sample_std_cell(&draws, cell)).collect();
            CellBand {
                lower: mean.iter().zip(&se).map(|(m, s)| m - z * s).collect(),
                upper: mean.iter().zip(&se).map(|(m, s)| m + z * s).collect(),
                draws,
                dispersion,
                interpretation: if self.weighted {
                    antecedent_core::IntervalInterpretation::Credible
                } else {
                    antecedent_core::IntervalInterpretation::Confidence
                },
            }
        });
        SurfaceCells { mean, band, cell_horizon: cells.iter().map(|&(h, _)| h).collect() }
    }

    /// Record how the band was built and publish (or withhold) the simultaneous band.
    pub(super) fn finish(
        &self,
        support: &mut SupportReport,
        cells: &SurfaceCells,
        mut assumptions: AssumptionSet,
        replicates: u32,
    ) -> AssumptionSet {
        support.warnings.extend(self.se_provenance.warning(replicates));
        if let Some(band) = &cells.band {
            disclose_response_block_bootstrap(
                support,
                &mut assumptions,
                self.block,
                self.inflation(),
                &band.dispersion,
                band.draws.len(),
                SURFACE_REFIT,
            );
            publish_simultaneous_band(
                support,
                max_deviation_band(&cells.mean, &band.draws, 0.95),
                BLOCK_BOOTSTRAP_CONSTRUCTION,
            );
            if self.retain_joint {
                self.publish_joint_replicates(support, cells, band);
            }
        } else {
            if replicates == 0 {
                support.warnings.push(Diagnostic::new(
                    TEMPORAL_RESPONSE_BAND_WITHHELD,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "no pointwise or simultaneous band: an analytic OLS band would treat \
                     lag-aligned rows as independent, which temporal rows are not; request \
                     bootstrap replicates for the joint circular-block bands",
                ));
            }
            publish_simultaneous_band(
                support,
                Err(EstimationError::unsupported(if replicates == 0 {
                    "no joint cross-horizon replicates without the bootstrap; request bootstrap \
                     replicates for a joint circular-block simultaneous band"
                } else {
                    "too few joint bootstrap replicates survived"
                })),
                "",
            );
        }
        assumptions
    }
}

impl FittedSurface {
    /// Publish the joint replicates of the surface (see [`JOINT_HORIZON_REPLICATES`]).
    fn publish_joint_replicates(
        &self,
        support: &mut SupportReport,
        cells: &SurfaceCells,
        band: &CellBand,
    ) {
        use antecedent_core::{DiagnosticScope, SupportDiagnostic};
        let n_cells = cells.mean.len();
        let mut layout = vec![band.draws.len() as f64, n_cells as f64];
        layout.extend(cells.cell_horizon.iter().map(|&h| h as f64));
        let values: Vec<f64> = band.draws.iter().flat_map(|draw| draw.iter().copied()).collect();
        let resampling = if self.weighted {
            "Dirichlet block weights: one Exp(1) weight per block of consecutive series times, \
             shared by every horizon, with a weighted least-squares refit of each horizon (draws \
             of the posterior over the surface given the empirical law of the lag windows)"
        } else {
            "circular-block bootstrap: every horizon refit on the same resampled series times \
             (replicates of the estimator's sampling law)"
        };
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(JOINT_HORIZON_LAYOUT),
            values: Arc::from(layout),
            detail: Arc::from(format!(
                "[replicates, cells, horizon index of each cell] of the joint replicates; drawn \
                 by a {resampling}; replicate deviations carry the published dispersion factors"
            )),
            scope: DiagnosticScope::Global,
        });
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(JOINT_HORIZON_REPLICATES),
            values: Arc::from(values),
            detail: Arc::from(
                "replicate-major joint replicates of the whole dose × horizon surface in the cell \
                 layout of the mean surface; a row is one realization coupled across horizons",
            ),
            scope: DiagnosticScope::Global,
        });
    }
}

/// Every horizon of a temporal surface fitted once; replicate refits gather
/// lag-aligned rows by outcome-time anchor. See
/// [`TemporalResponseEstimator::prepare_surface`].
pub struct PreparedTemporalSurface {
    pub(super) horizons: Vec<FittedHorizon>,
    pub(super) cells: Vec<(usize, CellEval)>,
    pub(super) series_rows: usize,
}

impl PreparedTemporalSurface {
    /// Earliest outcome-time anchor present in every horizon's lag-aligned design.
    ///
    /// Lag-aligned rows end at the series end, so horizon `h` covers anchors
    /// `series_rows − n_h ..series_rows`.
    #[must_use]
    pub fn first_common_anchor(&self) -> usize {
        self.horizons
            .iter()
            .map(|fitted| self.series_rows.saturating_sub(fitted.prepared.design.nrows))
            .max()
            .unwrap_or(0)
    }

    /// Largest per-horizon coefficient count (for the HC1 part of the dispersion inflation).
    #[must_use]
    pub fn max_parameters(&self) -> usize {
        self.horizons.iter().map(|fitted| fitted.prepared.design.ncols).max().unwrap_or(0)
    }

    /// Estimating-equation scores of every horizon's full-sample level (normal-equation
    /// scores and centered covariate columns), for [`ResponseBlockLength::new`].
    #[must_use]
    pub fn estimating_scores(&self) -> Vec<Vec<f64>> {
        horizon_scores(&self.horizons)
    }

    /// Each cell's kernel-bias factor and effective rows at block length `block`, in
    /// the cell layout of [`Self::point`] (see [`CellDispersion`]).
    #[must_use]
    pub fn cell_dispersion(&self, block: usize) -> CellDispersion {
        cell_dispersion(&self.horizons, &self.cells, block)
    }

    /// Full-sample surface in the response layout.
    #[must_use]
    pub fn point(&self) -> Vec<f64> {
        self.cells
            .iter()
            .map(|&(h, eval)| {
                let fitted = &self.horizons[h];
                level_at(&fitted.coefs, &fitted.column_means, eval)
            })
            .collect()
    }

    /// Refit every horizon on the lag-aligned rows anchored at `anchors` (outcome-time
    /// row indices, each at least [`Self::first_common_anchor`]) with the outcome at
    /// each resampled row replaced by `outcomes[i]`. Returns the replicate surface, or
    /// `None` when a horizon's refit is singular.
    ///
    /// # Errors
    ///
    /// Misaligned `anchors` / `outcomes`, or an anchor outside every horizon's design.
    pub fn replicate(
        &self,
        anchors: &[usize],
        outcomes: &[f64],
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        if anchors.len() != outcomes.len()
            || anchors.len() < 2
            || anchors.iter().any(|&s| s < self.first_common_anchor() || s >= self.series_rows)
        {
            return Err(EstimationError::unsupported(
                "replicate anchors must align with outcomes and lie in every horizon's design",
            ));
        }
        self.replicate_into(
            anchors,
            outcomes,
            &mut LeastSquaresWorkspace::default(),
            &mut Vec::new(),
        )
    }

    /// [`Self::replicate`] writing the design gather into caller-owned buffers.
    ///
    /// # Errors
    ///
    /// Misaligned `anchors` / `outcomes`, or an anchor outside every horizon's design.
    pub fn replicate_into(
        &self,
        anchors: &[usize],
        outcomes: &[f64],
        ols_ws: &mut LeastSquaresWorkspace,
        x_boot: &mut Vec<f64>,
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        if anchors.len() != outcomes.len()
            || anchors.len() < 2
            || anchors.iter().any(|&s| s < self.first_common_anchor() || s >= self.series_rows)
        {
            return Err(EstimationError::unsupported(
                "replicate anchors must align with outcomes and lie in every horizon's design",
            ));
        }
        let m = anchors.len();
        let mut fits = Vec::with_capacity(self.horizons.len());
        for fitted in &self.horizons {
            let design = &fitted.prepared.design;
            let (n, p) = (design.nrows, design.ncols);
            let base = self.series_rows - n;
            x_boot.resize(m * p, 0.0);
            for c in 0..p {
                let column = &design.matrix[c * n..(c + 1) * n];
                for (i, &anchor) in anchors.iter().enumerate() {
                    x_boot[c * m + i] = column[anchor - base];
                }
            }
            let Ok(fit) = FaerBackend.least_squares(x_boot, m, p, outcomes, ols_ws) else {
                return Ok(None);
            };
            let means: Vec<f64> =
                (0..p).map(|c| x_boot[c * m..(c + 1) * m].iter().sum::<f64>() / m as f64).collect();
            fits.push((fit.coefficients, means));
        }
        Ok(Some(
            self.cells.iter().map(|&(h, eval)| level_at(&fits[h].0, &fits[h].1, eval)).collect(),
        ))
    }
}

/// `coefs' cbar(eval)`: the g-computed level at one cell under one fit.
fn level_at(coefs: &[f64], column_means: &[f64], eval: CellEval) -> f64 {
    let treatment = match eval {
        CellEval::Dose(dose) => dose,
        CellEval::Shift(shift) => column_means[TREATMENT_COL] + shift,
    };
    coefs
        .iter()
        .zip(column_means)
        .enumerate()
        .map(|(col, (&coef, &mean))| coef * if col == TREATMENT_COL { treatment } else { mean })
        .sum()
}

/// Per-horizon OLS fit: coefficients and design column means.
///
/// `mu_hat(dose) = coefs' cbar(dose)`, where `cbar(dose)` is the vector of design
/// column means with the treatment column's mean replaced by `dose`. Because the
/// fitted model is linear in the treatment column with no treatment×covariate
/// interaction, this is an O(p) evaluation per dose (no re-scan of the design).
/// Uncertainty comes only from the joint circular-block bootstrap, which refits the
/// coefficients and recomputes the column means, so the band targets the population
/// level `E[Y_h | do(A)]`.
pub(super) struct FittedHorizon {
    pub(super) prepared: PreparedEstimationProblem,
    pub(super) coefs: Vec<f64>,
    /// Design column means (length p); index `TREATMENT_COL` is `Abar`.
    pub(super) column_means: Vec<f64>,
    /// Full-sample OLS residuals (length n).
    pub(super) residuals: Vec<f64>,
    /// Lower Cholesky factor of the Gram matrix `X'X` (`p × p`, row-major; `None` when it
    /// is not positive definite). The Gram depends on the horizon's design only, so it is
    /// factored once here and every cell's influence series solves against this copy.
    pub(super) gram_chol: Option<Vec<f64>>,
    /// The intact lag windows when the series has missing time points (the design's rows are
    /// those windows in time order).
    pub(super) selection: Option<crate::GapSelection>,
}

impl FittedHorizon {
    pub(super) fn fit(
        prepared: PreparedEstimationProblem,
        ols_ws: &mut LeastSquaresWorkspace,
    ) -> Result<Self, EstimationError> {
        let n = prepared.design.nrows;
        let p = prepared.design.ncols;
        let fit = FaerBackend
            .least_squares(&prepared.design.matrix, n, p, &prepared.design.outcome, ols_ws)
            .map_err(EstimationError::from)?;
        let column_means = design_column_means(&prepared.design);
        let gram = crate::util::gram(&prepared.design.matrix[..n * p], n, p);
        let gram_chol = antecedent_stats::cholesky_spd(&gram, p);
        Ok(Self {
            prepared,
            coefs: fit.coefficients,
            column_means,
            residuals: fit.residuals,
            gram_chol,
            selection: None,
        })
    }

    pub(super) fn treatment_mean(&self) -> f64 {
        self.column_means[TREATMENT_COL]
    }

    /// Delta-method influence series of the level at `eval`. The regression contribution
    /// is the usual OLS influence of `c'β̂` (`c` = the column means with the treatment
    /// column at the evaluated dose): `n · x_t'(X'X)⁻¹ c · ê_t`. The contribution of the
    /// estimated means is `coef' · (x_t − mean)` on the columns the level actually reads
    /// (every column but the treatment for a dose; every column for a shift, whose level
    /// reads the treatment mean too). Both terms are `O_p(1)`; the circular-block variance
    /// of the level is asymptotically the Bartlett long-run variance of this series at the
    /// block length. `None` when `X'X` is singular.
    pub(super) fn level_influence(&self, eval: CellEval) -> Option<Vec<f64>> {
        let design = &self.prepared.design;
        let (n, p) = (design.nrows, design.ncols);
        let x = &design.matrix[..n * p];
        let mut direction = self.column_means.clone();
        direction[TREATMENT_COL] = match eval {
            CellEval::Dose(dose) => dose,
            CellEval::Shift(shift) => self.column_means[TREATMENT_COL] + shift,
        };
        let v = antecedent_stats::chol_solve(self.gram_chol.as_deref()?, p, &direction)?;
        let reads_treatment_mean = matches!(eval, CellEval::Shift(_));
        let n_f = n as f64;
        Some(
            (0..n)
                .map(|t| {
                    let weight: f64 = (0..p).map(|k| x[k * n + t] * v[k]).sum();
                    let means: f64 = (0..p)
                        .filter(|&k| k != TREATMENT_COL || reads_treatment_mean)
                        .map(|k| self.coefs[k] * (x[k * n + t] - self.column_means[k]))
                        .sum();
                    n_f * weight * self.residuals[t] + means
                })
                .collect(),
        )
    }
}

/// Kernel-bias factor and effective rows of every cell of `cells` (a cell whose
/// influence is unavailable reads factor `1` and unreadable rows).
fn cell_dispersion(
    horizons: &[FittedHorizon],
    cells: &[(usize, CellEval)],
    block: usize,
) -> CellDispersion {
    let influences: Vec<Vec<f64>> = cells
        .iter()
        .map(|&(h, eval)| horizons[h].level_influence(eval).unwrap_or_default())
        .collect();
    CellDispersion::from_influences(&influences, block)
}

/// Estimating-equation scores of every horizon's level, one series per equation: the
/// OLS normal-equation scores `x_tj · ê_t` of every coefficient and the centered design
/// columns `x_tj − x̄_j` behind the covariate averages the level reads (every column but
/// the intercept). A persistent covariate or treatment leaves the regression scores
/// nearly white when the residual is not persistent, yet its sample mean — part of every
/// g-computed level — carries the covariate's full long-run variance.
pub(super) fn horizon_scores(horizons: &[FittedHorizon]) -> Vec<Vec<f64>> {
    let mut scores = Vec::new();
    for fitted in horizons {
        let design = &fitted.prepared.design;
        let (n, p) = (design.nrows, design.ncols);
        scores.extend(normal_equation_scores_of_residuals(&design.matrix, n, p, &fitted.residuals));
        for (c, &mean) in fitted.column_means.iter().enumerate().skip(1) {
            scores.push(design.matrix[c * n..(c + 1) * n].iter().map(|x| x - mean).collect());
        }
    }
    scores
}

/// Each horizon's lag-aligned rows on the series time axis. Lag-aligned rows end at
/// the series end, so horizon row `i` sits at series time `series_rows − n_h + i`.
pub(super) fn horizon_rows(horizons: &[FittedHorizon], series_rows: usize) -> Vec<AlignedRows> {
    horizons
        .iter()
        .map(|fitted| {
            let rows = fitted.prepared.design.nrows;
            AlignedRows { first_time: series_rows.saturating_sub(rows), rows }
        })
        .collect()
}

/// Each horizon's windows on the series time axis with their intactness, for the gap-aware
/// bootstrap: a horizon without gaps is all intact over its contiguous `rows`.
pub(super) fn gapped_horizon_rows(
    horizons: &[FittedHorizon],
    rows: &[AlignedRows],
) -> Vec<crate::temporal_block::GappedAlignedRows> {
    horizons
        .iter()
        .zip(rows)
        .map(|(fitted, contiguous)| match &fitted.selection {
            Some(selection) => crate::temporal_block::GappedAlignedRows {
                first_time: selection.first_time,
                complete: selection.complete.clone(),
            },
            None => crate::temporal_block::GappedAlignedRows::contiguous(*contiguous),
        })
        .collect()
}

pub(super) fn design_column_means(design: &CompiledDesign) -> Vec<f64> {
    let n = design.nrows;
    let p = design.ncols;
    let mut means = vec![0.0; p];
    for (col, mean) in means.iter_mut().enumerate() {
        let start = col * n;
        let sum: f64 = design.matrix[start..start + n].iter().sum();
        *mean = sum / n as f64;
    }
    means
}

pub(super) fn checked_surface_cells(
    doses: usize,
    horizons: usize,
) -> Result<usize, EstimationError> {
    let cells = doses
        .checked_mul(horizons)
        .filter(|cells| *cells <= MAX_TEMPORAL_RESPONSE_CELLS)
        .ok_or_else(|| {
        EstimationError::data_msg(
            "temporal response dose-by-horizon cell count exceeds the materialization limit",
        )
    })?;
    Ok(cells)
}

pub(super) fn flatten_dose_horizon_grid(
    doses: &[f64],
    horizons: &[u32],
) -> Result<Vec<f64>, EstimationError> {
    let cells = checked_surface_cells(doses.len(), horizons.len())?;
    let capacity = cells.checked_mul(2).ok_or_else(|| {
        EstimationError::data_msg("temporal response coordinate grid size overflow")
    })?;
    let mut grid = Vec::with_capacity(capacity);
    for &dose in doses {
        for &h in horizons {
            grid.push(dose);
            grid.push(f64::from(h));
        }
    }
    Ok(grid)
}

fn named_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
) -> Result<Vec<TemporalNodeKey>, EstimationError> {
    estimand
        .adjustment_set
        .iter()
        .map(|&dense| {
            indexer.key_of(dense.raw()).map_err(|e| EstimationError::data_msg(e.to_string()))
        })
        .collect()
}

pub(super) fn horizon_identification_of(
    horizon: u32,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    status: IdentificationStatus,
) -> Result<HorizonIdentification, EstimationError> {
    Ok(HorizonIdentification {
        horizon,
        status,
        method: Arc::clone(&estimand.method),
        adjustment: Arc::from(named_adjustment(estimand, indexer)?),
    })
}
