//! Temporal dose-over-horizon / policy-path response estimation (ADR 0021).
//!
//! Reuses temporal-backdoor identification and linear g-computation on the
//! unfolded design. No new identification algorithm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, ContinuousDomain, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, GridSpec, HorizonIdentification, IdentificationStatus,
    Intervention, InterventionSequence, MAX_TEMPORAL_RESPONSE_CELLS, MechanismOverride,
    ObservationSpec, ParametricAssumption, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion,
    SupportReport, SupportStatus, TargetPopulation, TemporalEffectQuery, TemporalNodeKey,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TableView, TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, SandwichKind,
    coefficient_covariance, normal_ppf,
};

use crate::adjustment::{LinearAdjustmentAte, PreparedEstimationProblem};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DependenceScope, SerialDependence, tempering_kappa_from_notes,
};
use crate::temporal_adjustment::TemporalLinearAdjustment;
use crate::temporal_sequential::{SequentialMechanismOverlay, SequentialNodeOverlay};
use crate::util::{BOOTSTRAP_MAX_FAILURE_FRAC, range, sample_std};

/// Licensed temporal `InterventionResponse` overlay.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalInterventionPlan {
    /// One treatment schedule from [`TemporalResponseSpec::policy`] (plain Set / Shift / Soft).
    Single {
        /// Intervened variable.
        treatment: VariableId,
        /// Hard set / Soft constant.
        level: Option<f64>,
        /// Additive shift when `level` is `None`.
        shift: f64,
    },
    /// Explicit Sequence: one overlay per intervened unfolded node.
    Sequential {
        /// Licensed Set / Soft constant / Soft shift overlays.
        overlays: Vec<SequentialNodeOverlay>,
    },
    /// Deterministic mean mechanisms, evaluated through the unfolded engine.
    Mechanisms {
        /// Multiplicative or bounded-mean overlays, optionally mixed with Set/Shift.
        overlays: Vec<SequentialMechanismOverlay>,
    },
}

impl TemporalInterventionPlan {
    /// Sequential mean-mechanism overlays; single Set/Shift keeps its direct path.
    #[must_use]
    pub fn mechanism_overlays(&self) -> Option<Vec<SequentialMechanismOverlay>> {
        match self {
            Self::Single { .. } => None,
            Self::Sequential { overlays } => {
                Some(overlays.iter().copied().map(Into::into).collect())
            }
            Self::Mechanisms { overlays } => Some(overlays.clone()),
        }
    }
    /// Treatment nodes the identifier must cover.
    #[must_use]
    pub fn identification_schedule(&self, spec: &TemporalResponseSpec) -> Vec<(VariableId, i32)> {
        match self {
            Self::Single { treatment, .. } => spec
                .policy
                .active_offsets()
                .map(|offsets| offsets.iter().map(|&offset| (*treatment, offset)).collect())
                .unwrap_or_default(),
            Self::Sequential { overlays } => {
                overlays.iter().map(|overlay| (overlay.variable, overlay.offset)).collect()
            }
            Self::Mechanisms { overlays } => overlays
                .iter()
                .map(|overlay| (overlay.node.variable, overlay.node.offset))
                .collect(),
        }
    }
}

/// Resample stream base for the joint surface bootstrap (one stream per replicate).
///
/// Every horizon of one replicate is refit on the same time-aligned blocks, so the
/// replicate is a draw of the whole dose × horizon surface, not of one horizon.
const HORIZON_BOOTSTRAP_STREAM: u64 = 0x7E50_u64;

/// Support-diagnostic id carrying the simultaneous band's lower edge (mean-surface layout).
pub const SIMULTANEOUS_BAND_LOWER: &str = "response.simultaneous_band.lower";
/// Support-diagnostic id carrying the simultaneous band's upper edge (mean-surface layout).
pub const SIMULTANEOUS_BAND_UPPER: &str = "response.simultaneous_band.upper";
/// Support-diagnostic id carrying `[level, critical value, joint replicates or draws]`.
pub const SIMULTANEOUS_BAND_CRITICAL: &str = "response.simultaneous_band.critical";
/// Warning code when a simultaneous band is not published.
pub const SIMULTANEOUS_BAND_WITHHELD: &str = "response.simultaneous_band_withheld";

/// Fewest joint replicates / draws from which a simultaneous band is published.
///
/// The critical value is the `ceil(level·(B+1))`-th order statistic of `B` maxima; below
/// this count the 95% critical value is (close to) the sample maximum and carries little
/// information about the tail it is meant to estimate.
pub const SIMULTANEOUS_BAND_MIN_REPLICATES: usize = 40;

/// Max-studentized-deviation (sup-t) simultaneous band over a response grid.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxDeviationBand {
    /// Simultaneous level.
    pub level: f64,
    /// Critical value `c`: the band is `center ± c·scale` at every cell.
    pub critical: f64,
    /// Lower edge, same layout as the center.
    pub lower: Vec<f64>,
    /// Upper edge, same layout as the center.
    pub upper: Vec<f64>,
    /// Joint replicates or draws the critical value was computed from.
    pub replicates: u32,
}

/// Sup-t band from joint replicates (or posterior draws) of a whole response surface.
///
/// `draws[r]` is replicate `r`'s full surface in the same layout as `center`. Each cell's
/// scale is the SD of its replicates; the critical value is the `ceil(level·(B+1))`-th
/// order statistic of `max_j |draws[r][j] − center[j]| / scale_j` over the `B` replicates,
/// so the band covers the whole grid at once rather than one cell at a time. This is the
/// replicate-based analogue of the static Kennedy-DR multiplier band.
///
/// # Errors
///
/// Fewer than [`SIMULTANEOUS_BAND_MIN_REPLICATES`] replicates, ragged or non-finite draws,
/// a level outside `(0, 1)`, or a cell whose replicates do not vary.
pub fn max_deviation_band(
    center: &[f64],
    draws: &[Vec<f64>],
    level: f64,
) -> Result<MaxDeviationBand, EstimationError> {
    if draws.iter().any(|draw| draw.len() != center.len()) {
        return Err(EstimationError::unsupported(
            "simultaneous band needs joint draws aligned with the response grid",
        ));
    }
    max_deviation_band_by(center, draws.len(), |r, cell| draws[r][cell], level)
}

/// [`max_deviation_band`] over column-major draws: `columns[cell][r]`.
///
/// # Errors
///
/// Same as [`max_deviation_band`].
pub fn max_deviation_band_columns(
    center: &[f64],
    columns: &[&[f64]],
    level: f64,
) -> Result<MaxDeviationBand, EstimationError> {
    let n_draws = columns.first().map_or(0, |column| column.len());
    if columns.len() != center.len() || columns.iter().any(|column| column.len() != n_draws) {
        return Err(EstimationError::unsupported(
            "simultaneous band needs joint draws aligned with the response grid",
        ));
    }
    max_deviation_band_by(center, n_draws, |r, cell| columns[cell][r], level)
}

fn max_deviation_band_by(
    center: &[f64],
    n_draws: usize,
    value: impl Fn(usize, usize) -> f64,
    level: f64,
) -> Result<MaxDeviationBand, EstimationError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(EstimationError::unsupported("simultaneous band level must lie in (0, 1)"));
    }
    if n_draws < SIMULTANEOUS_BAND_MIN_REPLICATES {
        return Err(EstimationError::unsupported(
            "simultaneous band needs at least 40 joint replicates or draws",
        ));
    }
    if center.is_empty()
        || center.iter().any(|mid| !mid.is_finite())
        || (0..n_draws).any(|r| (0..center.len()).any(|cell| !value(r, cell).is_finite()))
    {
        return Err(EstimationError::unsupported(
            "simultaneous band needs finite joint draws and a finite center",
        ));
    }
    let scale: Vec<f64> = (0..center.len())
        .map(|cell| sample_std(&(0..n_draws).map(|r| value(r, cell)).collect::<Vec<_>>()))
        .collect();
    if scale.iter().any(|s| !s.is_finite() || *s <= f64::EPSILON) {
        return Err(EstimationError::unsupported(
            "simultaneous band needs every grid cell to vary across joint draws",
        ));
    }
    let mut maxima: Vec<f64> = (0..n_draws)
        .map(|r| {
            center
                .iter()
                .zip(&scale)
                .enumerate()
                .map(|(cell, (mid, s))| (value(r, cell) - mid).abs() / s)
                .fold(0.0_f64, f64::max)
        })
        .collect();
    maxima.sort_by(f64::total_cmp);
    let b = maxima.len();
    let rank = ((level * (b as f64 + 1.0)).ceil() as usize).clamp(1, b);
    // A band over the whole grid is never narrower than the one-cell normal band
    // `center ± z·scale`: with strongly correlated cells and few distinct blocks the
    // Monte Carlo rank can fall below `z`, which would publish a "simultaneous" band
    // inside the pointwise one.
    let critical = maxima[rank - 1].max(normal_ppf(0.5 + level / 2.0));
    Ok(MaxDeviationBand {
        level,
        critical,
        lower: center.iter().zip(&scale).map(|(mid, s)| mid - critical * s).collect(),
        upper: center.iter().zip(&scale).map(|(mid, s)| mid + critical * s).collect(),
        replicates: u32::try_from(b).unwrap_or(u32::MAX),
    })
}

/// Publish (or explicitly withhold) a simultaneous band next to the pointwise band.
///
/// The pointwise band stays in [`CausalResponse::uncertainty`]; the simultaneous band is
/// carried by three support diagnostics ([`SIMULTANEOUS_BAND_LOWER`],
/// [`SIMULTANEOUS_BAND_UPPER`], [`SIMULTANEOUS_BAND_CRITICAL`]) whose detail names the
/// construction. Any previously attached simultaneous band is replaced. `Err` publishes a
/// [`SIMULTANEOUS_BAND_WITHHELD`] warning with the reason instead of a band.
pub fn publish_simultaneous_band(
    support: &mut SupportReport,
    band: Result<MaxDeviationBand, EstimationError>,
    construction: &str,
) {
    clear_simultaneous_band(support);
    match band {
        Ok(band) => {
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from(SIMULTANEOUS_BAND_LOWER),
                values: Arc::from(band.lower),
                detail: Arc::from(format!(
                    "lower edge of the {:.0}% SIMULTANEOUS band over the whole response grid \
                     (same layout as the mean); {construction}",
                    band.level * 100.0
                )),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from(SIMULTANEOUS_BAND_UPPER),
                values: Arc::from(band.upper),
                detail: Arc::from(format!(
                    "upper edge of the {:.0}% SIMULTANEOUS band over the whole response grid \
                     (same layout as the mean); {construction}",
                    band.level * 100.0
                )),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from(SIMULTANEOUS_BAND_CRITICAL),
                values: Arc::from([band.level, band.critical, f64::from(band.replicates)]),
                detail: Arc::from(
                    "[simultaneous level, max-studentized-deviation critical value, joint \
                     replicates or draws]; the pointwise band in uncertainty uses one cell at a \
                     time and is not simultaneous",
                ),
            });
        }
        Err(reason) => support.warnings.push(Diagnostic::new(
            SIMULTANEOUS_BAND_WITHHELD,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "no simultaneous band is published; the band in uncertainty is pointwise only \
                 ({reason})"
            ),
        )),
    }
}

/// Remove any simultaneous-band diagnostics and withheld warnings from `support`.
pub fn clear_simultaneous_band(support: &mut SupportReport) {
    support.diagnostics.retain(|diagnostic| {
        !matches!(
            diagnostic.id.as_ref(),
            SIMULTANEOUS_BAND_LOWER | SIMULTANEOUS_BAND_UPPER | SIMULTANEOUS_BAND_CRITICAL
        )
    });
    support.warnings.retain(|warning| warning.code.as_ref() != SIMULTANEOUS_BAND_WITHHELD);
}

/// Circular-block length of the temporal response family:
/// `max(structural span, ceil(sqrt(n)))`, capped at `n`.
///
/// Response bands are read at a fixed critical value, so the block length is chosen for
/// interval coverage rather than for the mean-squared error of the variance: the
/// circular-block variance is (asymptotically) a Bartlett-kernel long-run variance with
/// bandwidth `ℓ`, whose testing-optimal bandwidth grows like `n^{1/2}` (Sun, Phillips &
/// Jin 2008, Bartlett characteristic exponent `q = 1`), not like the MSE-optimal
/// `n^{1/3}` used by the scalar temporal effect resamplers. A shorter block leaves an
/// `O(1/ℓ)` kernel bias that no critical value repairs: with `ℓ = ceil(n^{1/3})` the
/// 1.9 calibration measured 0.89–0.91 coverage of nominal 95% level bands under AR(1)
/// ρ = 0.5 residuals at n = 160. The estimation noise the longer block adds is carried by
/// the fixed-b factor of [`block_dispersion_inflation`].
#[must_use]
pub fn temporal_block_length(structural_span: usize, n: usize) -> usize {
    let root = (n as f64).sqrt().ceil() as usize;
    structural_span.max(root).min(n).max(1)
}

/// Dispersion factor applied to circular-block replicate deviations.
///
/// `fixed_b_scale(block, rows) · sqrt(rows / (rows − p))`:
/// - the Kiefer–Vogelsang (2005) fixed-b critical-value ratio for the Bartlett kernel at
///   `b = block / rows` ([`crate::temporal_block::fixed_b_scale`], the same correction
///   the plain `TemporalDag` Pulse / Sustained SE uses), so `estimate ± 1.96·SD` is the
///   fixed-b interval: it carries the downward bias and the sampling noise of a variance
///   estimated from `rows / block` blocks, and stays nominal on independent rows for any
///   `b`;
/// - the HC1 factor for the `p` fitted coefficients (a resampled regression's dispersion
///   behaves like the HC0 sandwich, biased down by about `p/rows`).
///
/// The polynomial is the two-sided 95% fixed-b critical value; every temporal response
/// band is a 95% band.
#[must_use]
pub fn block_dispersion_inflation(rows: usize, block: usize, parameters: usize) -> f64 {
    let hc1 =
        if rows > parameters + 1 { (rows as f64 / (rows - parameters) as f64).sqrt() } else { 1.0 };
    let ratio = crate::temporal_block::fixed_b_scale(block, rows) * hc1;
    if ratio.is_finite() && ratio >= 1.0 { ratio } else { 1.0 }
}

/// Scale each replicate's deviation from `center` by `factor`, in place.
pub fn inflate_replicates(center: &[f64], draws: &mut [Vec<f64>], factor: f64) {
    for draw in draws {
        for (value, mid) in draw.iter_mut().zip(center) {
            *value = mid + factor * (*value - mid);
        }
    }
}

/// Index of the treatment column in the compiled design (col0 = intercept, col1 = treatment).
/// Verified against `CompiledDesign::linear_adjustment`.
const TREATMENT_COL: usize = 1;

/// Per-horizon lag-aligned observed treatment `(min, max)`.
type HorizonTreatmentRange = (f64, f64);

/// Resolved Sequence leaf: variable, fixed level, shift, and active offsets.
type SequenceLeaf = (VariableId, Option<f64>, f64, Arc<[i32]>);

/// Every requested horizon fitted once on the full sample, plus the joint
/// surface bootstrap (when requested and usable) and the retained per-horizon
/// treatment ranges and identification records.
struct FittedSurface {
    horizons: Vec<FittedHorizon>,
    ranges: Vec<HorizonTreatmentRange>,
    identification: Vec<HorizonIdentification>,
    bootstrap: Option<SurfaceBootstrap>,
    se_provenance: SeProvenance,
    block_length: usize,
}

/// Where a surface's pointwise SEs came from, so the band can say so.
///
/// A requested bootstrap that degenerates to the analytic SE, or one truncated by
/// cancellation, is reported rather than passed off as a completed bootstrap.
#[derive(Clone, Copy, Debug, Default)]
struct SeProvenance {
    /// A bootstrap was requested but yielded no usable joint draws.
    fell_back: bool,
    /// The bootstrap was cut short by cooperative cancellation.
    cancelled: bool,
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
                 few joint resamples survived; the surface reports the analytic OLS \
                 linear-functional SE instead and no simultaneous band"
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
enum CellEval {
    /// `E[Y_h | do(A = dose)]`: treatment column fixed at `dose`.
    Dose(f64),
    /// `E[Y_h | do(A := A + shift)]`: treatment column at its sample mean plus `shift`.
    Shift(f64),
}

/// Point surface, pointwise band, and (bootstrap only) the joint replicate draws.
struct SurfaceCells {
    mean: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// `draws[r]` is replicate `r`'s full surface in the output layout.
    draws: Option<Vec<Vec<f64>>>,
}

/// Temporal response estimator: dose × horizon surfaces and temporal intervention responses.
#[derive(Clone, Debug)]
pub struct TemporalResponseEstimator {
    /// Shared OLS machinery (bootstrap off for the surface path by default).
    pub inner: LinearAdjustmentAte,
}

impl Default for TemporalResponseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

fn with_pointwise_homoskedastic_ols_assumption(mut assumptions: AssumptionSet) -> AssumptionSet {
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.linear_additive.gcomp"),
            description: Arc::from(
                "Temporal response levels use linear additive g-computation on each unfolded horizon. The numerical surface is model-dependent when the conditional outcome response is nonlinear or contains treatment-covariate interactions.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("ols.homoskedastic.pointwise"),
            description: Arc::from(
                "With zero bootstrap replicates: pointwise 95% band from the delta-method SE of \
                 the g-computed level, using the full homoskedastic OLS coefficient covariance \
                 plus the iid variance of the empirical covariate average, so the band targets \
                 the population level E[Y_h | do(A)]. It treats lag-aligned rows as independent \
                 draws: when neighbouring rows share lagged treatment, confounder or residual \
                 terms (typical of temporal designs, and of every shift policy) or innovations \
                 are serially correlated or heteroskedastic, it is optimistic. It is not a \
                 simultaneous band; request bootstrap replicates for the dependence-preserving \
                 joint circular-block bands.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

/// Construction text for the joint circular-block surface bootstrap.
const BLOCK_BOOTSTRAP_CONSTRUCTION: &str = "max-studentized deviation of the joint circular-block \
     bootstrap replicates of the whole dose × horizon surface around the full-sample estimate; \
     each replicate resamples time-aligned blocks of lag-aligned rows (block length max(span, \
     ceil(sqrt(n)))), refits every horizon and recomputes the covariate averages; replicate \
     deviations carry the Kiefer-Vogelsang fixed-b factor for b = block/rows and the HC1 \
     factor sqrt(rows/(rows − p))";

fn block_bootstrap_assumption(block_length: usize, rows: usize, factor: f64) -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("temporal_response.block_bootstrap"),
            description: Arc::from(format!(
                "Pointwise SEs are the SD of the g-computed level over a joint circular-block \
                 bootstrap of lag-aligned rows (block length {block_length} = max(unfolded span, \
                 ceil(sqrt(n))), n = {rows} rows); one replicate refits every horizon on the same \
                 time-aligned blocks and recomputes covariate averages, so the band targets the \
                 population level under serially dependent rows whose dependence decays within \
                 a block. Replicate deviations are scaled by {factor:.4}: the Kiefer-Vogelsang \
                 fixed-b critical-value ratio for b = block/rows (the variance is estimated from \
                 few blocks) times the HC1 factor sqrt(rows/(rows − p)). The pointwise band is \
                 mean ± 1.96·SE of the scaled replicates. The simultaneous band (support \
                 diagnostics response.simultaneous_band.*) is the max-studentized deviation over \
                 the whole dose × horizon grid from the same replicates."
            )),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

/// Support diagnostic carrying the per-horizon tempering factor `κ̂_h` of
/// `response.temporal.bayesian` (one value per requested horizon, in order).
pub const TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC: &str = "response.temporal_bayesian.tempering";

fn is_tempering_record(record: &AssumptionRecord) -> bool {
    matches!(
        &record.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == DEPENDENCE_ASSUMPTION_ID
    )
}

/// One serial-dependence record for the whole surface (each horizon's fit records its own
/// factor; only the first horizon's assumptions were ever forwarded).
fn temporal_bayesian_tempering_assumption(horizons: &[u32], kappas: &[f64]) -> AssumptionRecord {
    let per_horizon = horizons
        .iter()
        .zip(kappas)
        .map(|(h, k)| format!("h={h}: {k:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(DEPENDENCE_ASSUMPTION_ID),
            description: Arc::from(format!(
                "generalized (power) posterior with a serial-dependence correction at every \
                 horizon: each horizon's Gaussian likelihood on time-ordered lag-aligned rows is \
                 tempered by 1/kappa_h, kappa_h = the largest AR(1)-prewhitened Newey-West \
                 long-run-variance ratio of that horizon's grid-cell level scores, floored at 1 \
                 ({per_horizon}). At h >= 2 the unfolded regression omits intermediate \
                 treatments and innovations, so its residuals are MA(h-1) whenever the outcome \
                 or treatment is persistent; the prior keeps full weight; heteroskedasticity and \
                 mean misspecification are not corrected"
            )),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("response.temporal.bayesian"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

impl TemporalResponseEstimator {
    /// Defaults: explicit-override overlap, no bootstrap. Pointwise SEs come from the
    /// linear-functional variance `cbar(a)' Sigma cbar(a)` of the standardized mean,
    /// not from the `β_T` coefficient SE alone.
    #[must_use]
    pub fn new() -> Self {
        let mut inner = LinearAdjustmentAte::new();
        inner.bootstrap_replicates = 0;
        inner.overlap = OverlapPolicy::ExplicitOverride;
        Self { inner }
    }

    /// Estimate a temporal [`ResponseQuery`] on series data.
    ///
    /// `identifications` must be aligned with `query.temporal.horizons`: one
    /// `(estimand, indexer)` pair per requested horizon, already identified.
    /// Reusing a max-horizon estimand at a shorter target is not valid when
    /// confounding is horizon-dependent.
    ///
    /// # Errors
    ///
    /// Missing temporal attachment, unsupported functional/intervention, length
    /// mismatch, or fit failures.
    pub fn estimate(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        // This estimator is a public Rust entry point, not merely an internal
        // continuation from the planner. Enforce the complete ResponseQuery
        // contract here so callers cannot bypass treatment/outcome, observation,
        // or intervention validation and still receive a numerical response.
        query.validate()?;
        if query.observation != ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "TemporalResponseEstimator requires complete observations; apply a licensed observation correction first",
            ));
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "temporal response estimation requires point identification",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "TemporalResponseEstimator requires ResponseQuery.temporal (ADR 0021)",
            )
        })?;
        temporal.validate()?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported(
                "temporal response identification must be supplied once per requested horizon",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        let assumptions = with_pointwise_homoskedastic_ols_assumption(assumptions);
        let mut response = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => self.estimate_mean_curve(
                data,
                identifications,
                *outcome,
                treatment.variable,
                &treatment.grid.values()?,
                temporal,
                identification_status,
                assumptions,
                ctx,
            ),
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => self
                        .estimate_intervention_curve(
                            data,
                            identifications,
                            *outcome,
                            treatment,
                            level,
                            shift,
                            temporal,
                            identification_status,
                            assumptions,
                            ctx,
                        ),
                    TemporalInterventionPlan::Sequential { .. }
                    | TemporalInterventionPlan::Mechanisms { .. } => {
                        Err(EstimationError::unsupported(
                            "multi-step and joint Sequence overlays require the unfolded \
                             sequential estimator (Study temporal response path)",
                        ))
                    }
                }
            }
            _ => Err(EstimationError::unsupported(
                "temporal response is licensed only for MeanCurve and InterventionResponse",
            )),
        }?;
        // Preserve the exact query estimand. Reconstructing it in the numerical
        // helpers changed Linspace into Values and erased a licensed single-step
        // Sequence into a plain Set/Soft intervention.
        response.estimand = query.functional.clone();
        Ok(response)
    }

    /// Bayesian Gaussian linear-additive response on the identified unfolded
    /// design at each horizon. Intervals are pointwise posterior quantiles.
    #[allow(clippy::too_many_lines)]
    pub fn estimate_bayesian(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        mut assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        query.validate()?;
        if query.observation != antecedent_core::ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || estimator.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
        {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires complete observations, AllObserved, and GaussianIdentity",
            ));
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires point identification at every horizon",
            ));
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported("missing temporal response specification")
        })?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported("one identification required per horizon"));
        }
        let (outcome, treatment, doses, intervention) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                (*outcome, treatment.variable, treatment.grid.values()?, None)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => {
                        (*outcome, treatment, Vec::new(), Some((level, shift)))
                    }
                    TemporalInterventionPlan::Sequential { .. }
                    | TemporalInterventionPlan::Mechanisms { .. } => {
                        return Err(EstimationError::unsupported(
                            "Bayesian multi-step Sequence uses sequential mechanism overlays, \
                             not response.temporal.bayesian / bayesian.gcomp",
                        ));
                    }
                }
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "Bayesian temporal response supports only curves and intervention responses",
                ));
            }
        };
        let mut rows = Vec::new();
        let mut ranges = Vec::new();
        let mut horizons = Vec::new();
        let mut levels = Vec::new();
        let mut tempering = Vec::new();
        for (h, (&horizon_steps, &(estimand, indexer))) in
            temporal.horizons.iter().zip(identifications).enumerate()
        {
            let pulse = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(1.0)),
                horizon_steps,
                max_history_lag: temporal.max_history_lag,
                target_population: TargetPopulation::AllObserved,
            };
            let prep = TemporalLinearAdjustment::new().prepare(
                data,
                estimand,
                &pulse,
                indexer,
                None,
                &ctx.kernel_policy,
            )?;
            let n = prep.design.nrows;
            let t = &prep.design.matrix[n..2 * n];
            ranges.push((
                t.iter().copied().fold(f64::INFINITY, f64::min),
                t.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ));
            horizons.push(horizon_identification_of(
                horizon_steps,
                estimand,
                indexer,
                identification_status,
            )?);
            let mut weights = design_column_means(&prep.design);
            let grid = if let Some((level, shift)) = intervention {
                vec![level.unwrap_or(weights[1] + shift)]
            } else {
                doses.clone()
            };
            let mut est = estimator.clone();
            est.seed = est.seed.wrapping_add(h as u64);
            let mut bprep = crate::BayesianGComputationAte::from_prepared_estimation(&prep);
            // Lag-aligned rows are time-ordered: at h ≥ 2 the unfolded regression omits the
            // intermediate treatments and outcome innovations, so its residuals are MA(h−1)
            // whenever the outcome or treatment is persistent, and serially dependent at every
            // horizon under autocorrelated innovations. Each horizon's likelihood is tempered
            // by the long-run-variance ratio of its grid-cell levels (floored at 1), the same
            // generalized posterior as the Bayesian Pulse / Sustained cells.
            bprep.serial_dependence = SerialDependence::LongRunTempering(DependenceScope::Levels(
                grid.iter()
                    .map(|&dose| {
                        let mut direction = weights.clone();
                        direction[TREATMENT_COL] = dose;
                        Arc::from(direction)
                    })
                    .collect(),
            ));
            let posterior = est.fit(
                &bprep,
                identification_status,
                &mut crate::BayesianGCompWorkspace::default(),
                ctx,
            )?;
            if h == 0 {
                assumptions.entries.extend(
                    posterior
                        .assumptions
                        .entries
                        .iter()
                        .filter(|record| !is_tempering_record(record))
                        .cloned(),
                );
            }
            tempering.push(tempering_kappa_from_notes(&posterior.diagnostics.notes).unwrap_or(1.0));
            if intervention.is_some() {
                levels.push(grid[0]);
            }
            let mut row = Vec::new();
            for dose in grid {
                weights[1] = dose;
                let values = crate::bayesian::linear_response_draws(&posterior, &weights)?;
                let summary =
                    crate::bayesian::summarize_linear_response_draws(values.clone(), 0.95)?;
                row.push((summary, values));
            }
            rows.push(row);
        }
        let mut mean = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut cell_draws: Vec<&[f64]> = Vec::new();
        for d in 0..if intervention.is_some() { 1 } else { doses.len() } {
            for row in &rows {
                let ((m, lo, hi, _), values) = &row[d];
                mean.push(*m);
                lower.push(*lo);
                upper.push(*hi);
                cell_draws.push(values);
            }
        }
        // Horizons are separate conjugate fits with independent draws; pairing draw r
        // across horizons samples their product, which is the joint law this estimator
        // actually has. Within a horizon every dose shares one coefficient draw.
        let n_joint = cell_draws.iter().map(|values| values.len()).min().unwrap_or(0);
        let cell_draws: Vec<&[f64]> = cell_draws.iter().map(|values| &values[..n_joint]).collect();
        let simultaneous = max_deviation_band_columns(&mean, &cell_draws, 0.95);
        let (grid, dimension, mut support) = if let Some((level, shift)) = intervention {
            (
                temporal.horizons.iter().map(|&h| f64::from(h)).collect(),
                1,
                intervention_support(&levels, level, shift, temporal, &ranges),
            )
        } else {
            (
                flatten_dose_horizon_grid(&doses, &temporal.horizons)?,
                2,
                mean_curve_support(&doses, temporal, &ranges),
            )
        };
        publish_simultaneous_band(
            &mut support,
            simultaneous,
            "simultaneous CREDIBLE band: max studentized deviation of the posterior draws of the \
             whole grid around the posterior mean; doses at one horizon share each coefficient \
             draw, while horizons are separate fits whose independent draws are paired by index \
             (a product of per-horizon posteriors, not a joint horizon posterior); conditional \
             on the observed adjustment distribution; each horizon's posterior is the \
             long-run-tempered generalized posterior (support diagnostic \
             response.temporal_bayesian.tempering)",
        );
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC),
            values: Arc::from(tempering.clone()),
            detail: Arc::from(
                "per-horizon likelihood tempering factor kappa (rows weighted 1/kappa): the \
                 largest AR(1)-prewhitened Newey-West long-run-variance ratio of the grid-cell \
                 level scores at that horizon, floored at 1",
            ),
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption { id: Arc::from("bayesian.temporal_response.linear_additive"),
                description: Arc::from("Gaussian linear-additive unfolded outcome model at each horizon, fit separately per horizon. Pointwise posterior intervals and the simultaneous credible band (support diagnostics response.simultaneous_band.*) are conditional on the observed adjustment and treatment distribution: they describe the level at the sample covariate average, not the population average. The simultaneous band pairs independent per-horizon draws, so across horizons it is a product-posterior band, not a joint horizon posterior.") }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("response.temporal.bayesian") }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
        });
        assumptions.push(temporal_bayesian_tempering_assumption(&temporal.horizons, &tempering));
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension,
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.response.temporal.bayesian"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        })
    }

    fn estimate_mean_curve(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        outcome: VariableId,
        treatment: VariableId,
        doses: &[f64],
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if doses.is_empty() {
            return Err(EstimationError::unsupported("dose grid must be non-empty"));
        }
        let n_h = temporal.horizons.len();
        checked_surface_cells(doses.len(), n_h)?;
        let surface = self.fit_surface(
            data,
            identifications,
            treatment,
            outcome,
            temporal,
            identification_status,
            ctx,
        )?;

        // Layout: value[d * n_horizons + h] — dose major, then horizon.
        let cells: Vec<(usize, CellEval)> = doses
            .iter()
            .flat_map(|&dose| (0..n_h).map(move |h| (h, CellEval::Dose(dose))))
            .collect();
        let values = surface.cells(&cells);
        let mut support = mean_curve_support(doses, temporal, &surface.ranges);
        let assumptions =
            surface.finish(&mut support, &values, assumptions, self.inner.bootstrap_replicates);

        Ok(CausalResponse {
            estimand: ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from(doses.to_vec())),
                ),
            },
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(flatten_dose_horizon_grid(doses, &temporal.horizons)?),
                dimension: 2,
                mean: Arc::from(values.mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(values.lower),
                upper: Arc::from(values.upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.gcomp"),
            horizon_identification: Some(Arc::from(surface.identification)),
            interaction_structurally_zero: false,
        })
    }

    fn estimate_intervention_curve(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        outcome: VariableId,
        treatment: VariableId,
        level: Option<f64>,
        shift: f64,
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        // Linear-in-dose, no treatment×covariate interaction: the fitted model is
        // mu_hat(d) = beta_t * d + base_mean, with base_mean independent of d. So
        // averaging g-comp at observed A_i + delta over i collapses exactly to a
        // single evaluation at Abar + delta — an O(n) loop is not needed. A bootstrap
        // replicate evaluates at its own resampled Abar* + delta.
        let surface = self.fit_surface(
            data,
            identifications,
            treatment,
            outcome,
            temporal,
            identification_status,
            ctx,
        )?;
        let eval = level.map_or(CellEval::Shift(shift), CellEval::Dose);
        let cells: Vec<(usize, CellEval)> =
            (0..surface.horizons.len()).map(|h| (h, eval)).collect();
        let eval_levels: Vec<f64> = surface
            .horizons
            .iter()
            .map(|fitted| level.unwrap_or_else(|| fitted.treatment_mean() + shift))
            .collect();
        let values = surface.cells(&cells);

        let grid: Vec<f64> = temporal.horizons.iter().map(|h| f64::from(*h)).collect();
        let mut support =
            intervention_support(&eval_levels, level, shift, temporal, &surface.ranges);
        let assumptions =
            surface.finish(&mut support, &values, assumptions, self.inner.bootstrap_replicates);

        Ok(CausalResponse {
            estimand: ResponseFunctional::InterventionResponse {
                outcome,
                interventions: Arc::from(vec![if let Some(level) = level {
                    Intervention::set(treatment, Value::f64(level))
                } else {
                    Intervention::soft(treatment, MechanismOverride::additive_shift(shift))
                }]),
            },
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension: 1,
                mean: Arc::from(values.mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(values.lower),
                upper: Arc::from(values.upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(surface.identification)),
            interaction_structurally_zero: false,
        })
    }

    /// Fit each horizon with that horizon's identified estimand, retain that
    /// horizon's lag-aligned treatment range and identification record, then run
    /// the joint surface bootstrap when replicates were requested.
    fn fit_surface(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        treatment: VariableId,
        outcome: VariableId,
        temporal: &TemporalResponseSpec,
        identification_status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<FittedSurface, EstimationError> {
        let mut ols_ws = LeastSquaresWorkspace::default();
        let mut horizons = Vec::with_capacity(temporal.horizons.len());
        let mut ranges = Vec::with_capacity(temporal.horizons.len());
        let mut identification = Vec::with_capacity(temporal.horizons.len());

        for (i, &horizon) in temporal.horizons.iter().enumerate() {
            let (estimand, indexer) = identifications[i];
            let fitted = self.fit_horizon(
                data,
                estimand,
                treatment,
                outcome,
                temporal,
                horizon,
                indexer,
                ctx,
                &mut ols_ws,
            )?;
            ranges.push(range(&fitted.prepared.treatment));
            identification.push(horizon_identification_of(
                horizon,
                estimand,
                indexer,
                identification_status,
            )?);
            horizons.push(fitted);
        }

        let replicates = self.inner.bootstrap_replicates;
        let span = identifications
            .iter()
            .map(|(_, indexer)| indexer.history() as usize + indexer.horizon() as usize)
            .max()
            .unwrap_or(1);
        let block_length = temporal_block_length(span, data.row_count());
        let bootstrap = bootstrap_surface(
            &horizons,
            data.row_count(),
            block_length,
            replicates,
            ctx,
            &mut ols_ws,
        );
        let se_provenance = SeProvenance {
            fell_back: replicates > 0 && bootstrap.is_none(),
            cancelled: bootstrap.as_ref().is_some_and(|b| b.cancelled),
        };
        Ok(FittedSurface {
            horizons,
            ranges,
            identification,
            bootstrap,
            se_provenance,
            block_length,
        })
    }

    fn fit_horizon(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        treatment: VariableId,
        outcome: VariableId,
        temporal: &TemporalResponseSpec,
        horizon_steps: u32,
        indexer: &TemporalIndexer,
        ctx: &ExecutionContext,
        ols_ws: &mut LeastSquaresWorkspace,
    ) -> Result<FittedHorizon, EstimationError> {
        let pulse_query = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            horizon_steps,
            max_history_lag: temporal.max_history_lag,
            target_population: TargetPopulation::AllObserved,
        };
        pulse_query.validate()?;
        // Multi-step Sustained/Dynamic: temporal linear adjustment currently refuses.
        // For 0.7 licensed Pulse (and single-step Sustained) this passes; multi-step
        // policies fail closed here rather than estimating a one-node proxy.
        let adj = TemporalLinearAdjustment { inner: self.inner.clone() };
        let prepared =
            adj.prepare(data, estimand, &pulse_query, indexer, None, &ctx.kernel_policy)?;
        FittedHorizon::fit(prepared, ols_ws)
    }

    /// Fit every horizon of a complete-data `MeanCurve` / single Set-Shift
    /// `InterventionResponse` once, for tuple-level replicate refits.
    ///
    /// Used by outer bootstraps that must replace the outcome per replicate (the
    /// observation-adjusted pseudo-outcome is refit on every replicate): see
    /// [`PreparedTemporalSurface::replicate`]. Cells follow the layout of
    /// [`Self::estimate`].
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::estimate`] (Sequence / mechanism overlays refuse).
    pub fn prepare_surface(
        &self,
        data: &TimeSeriesData,
        identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
        query: &ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalSurface, EstimationError> {
        query.validate()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "temporal surface replicates require ResponseQuery.temporal",
            )
        })?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported(
                "temporal response identification must be supplied once per requested horizon",
            ));
        }
        let n_h = temporal.horizons.len();
        let (treatment, outcome, cells) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let doses = treatment.grid.values()?;
                checked_surface_cells(doses.len(), n_h)?;
                let cells = doses
                    .iter()
                    .flat_map(|&dose| (0..n_h).map(move |h| (h, CellEval::Dose(dose))))
                    .collect();
                (treatment.variable, *outcome, cells)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => {
                        let eval = level.map_or(CellEval::Shift(shift), CellEval::Dose);
                        (treatment, *outcome, (0..n_h).map(|h| (h, eval)).collect())
                    }
                    _ => {
                        return Err(EstimationError::unsupported(
                            "Sequence overlays have no single-surface tuple replicate",
                        ));
                    }
                }
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "temporal response is licensed only for MeanCurve and InterventionResponse",
                ));
            }
        };
        let mut ols_ws = LeastSquaresWorkspace::default();
        let horizons = temporal
            .horizons
            .iter()
            .zip(identifications)
            .map(|(&horizon, &(estimand, indexer))| {
                self.fit_horizon(
                    data,
                    estimand,
                    treatment,
                    outcome,
                    temporal,
                    horizon,
                    indexer,
                    ctx,
                    &mut ols_ws,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedTemporalSurface { horizons, cells, series_rows: data.row_count() })
    }

    /// Copy the `Study` bootstrap / replicate count onto the shared OLS machinery.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.inner.bootstrap_replicates = replicates;
        self
    }
}

impl FittedSurface {
    /// Fewest lag-aligned rows over the fitted horizons.
    fn rows(&self) -> usize {
        self.horizons.iter().map(|fitted| fitted.prepared.design.nrows).min().unwrap_or(0)
    }

    /// Fixed-b dispersion factor for this surface's bootstrap.
    fn inflation(&self) -> f64 {
        let parameters =
            self.horizons.iter().map(|fitted| fitted.prepared.design.ncols).max().unwrap_or(0);
        block_dispersion_inflation(self.rows(), self.block_length, parameters)
    }

    /// Point values, pointwise 95% band and (bootstrap only) joint replicate draws
    /// for `cells`, in the order given. Bootstrap draws are returned after the
    /// block-count dispersion inflation, so both bands use the inflated dispersion.
    fn cells(&self, cells: &[(usize, CellEval)]) -> SurfaceCells {
        let z = normal_ppf(0.975);
        let mean: Vec<f64> = cells
            .iter()
            .map(|&(h, eval)| {
                let fitted = &self.horizons[h];
                level_at(&fitted.coefs, &fitted.column_means, eval)
            })
            .collect();
        let (se, draws) = if let Some(boot) = &self.bootstrap {
            let mut draws: Vec<Vec<f64>> = boot
                .draws
                .iter()
                .map(|replicate| {
                    cells
                        .iter()
                        .map(|&(h, eval)| level_at(&replicate[h].0, &replicate[h].1, eval))
                        .collect()
                })
                .collect();
            inflate_replicates(&mean, &mut draws, self.inflation());
            let se = (0..cells.len())
                .map(|cell| sample_std(&draws.iter().map(|draw| draw[cell]).collect::<Vec<_>>()))
                .collect::<Vec<_>>();
            (se, Some(draws))
        } else {
            (cells.iter().map(|&(h, eval)| self.horizons[h].analytic_se(eval)).collect(), None)
        };
        let lower = mean.iter().zip(&se).map(|(m, s)| m - z * s).collect();
        let upper = mean.iter().zip(&se).map(|(m, s)| m + z * s).collect();
        SurfaceCells { mean, lower, upper, draws }
    }

    /// Record how the band was built and publish (or withhold) the simultaneous band.
    fn finish(
        &self,
        support: &mut SupportReport,
        cells: &SurfaceCells,
        mut assumptions: AssumptionSet,
        replicates: u32,
    ) -> AssumptionSet {
        support.warnings.extend(self.se_provenance.warning(replicates));
        if let Some(draws) = &cells.draws {
            assumptions.push(block_bootstrap_assumption(
                self.block_length,
                self.rows(),
                self.inflation(),
            ));
            publish_simultaneous_band(
                support,
                max_deviation_band(&cells.mean, draws, 0.95),
                BLOCK_BOOTSTRAP_CONSTRUCTION,
            );
        } else {
            publish_simultaneous_band(
                support,
                Err(EstimationError::unsupported(if replicates == 0 {
                    "the analytic path has no joint cross-horizon covariance; request bootstrap \
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

/// Every horizon of a temporal surface fitted once; replicate refits gather
/// lag-aligned rows by outcome-time anchor. See
/// [`TemporalResponseEstimator::prepare_surface`].
pub struct PreparedTemporalSurface {
    horizons: Vec<FittedHorizon>,
    cells: Vec<(usize, CellEval)>,
    series_rows: usize,
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
        let mut ols_ws = LeastSquaresWorkspace::default();
        let m = anchors.len();
        let mut fits = Vec::with_capacity(self.horizons.len());
        for fitted in &self.horizons {
            let design = &fitted.prepared.design;
            let (n, p) = (design.nrows, design.ncols);
            let base = self.series_rows - n;
            let mut x = vec![0.0; m * p];
            for c in 0..p {
                let column = &design.matrix[c * n..(c + 1) * n];
                for (i, &anchor) in anchors.iter().enumerate() {
                    x[c * m + i] = column[anchor - base];
                }
            }
            let Ok(fit) = FaerBackend.least_squares(&x, m, p, outcomes, &mut ols_ws) else {
                return Ok(None);
            };
            let means: Vec<f64> =
                (0..p).map(|c| x[c * m..(c + 1) * m].iter().sum::<f64>() / m as f64).collect();
            fits.push((fit.coefficients, means));
        }
        Ok(Some(
            self.cells.iter().map(|&(h, eval)| level_at(&fits[h].0, &fits[h].1, eval)).collect(),
        ))
    }
}

/// Circular-block resample of `m` positions with block length `block`, drawn from
/// `rng` (block starts uniform on `0..m`).
pub fn circular_block_positions(
    m: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<usize> {
    let block = block.clamp(1, m.max(1));
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let start = ((rng.next_f64() * m as f64) as usize).min(m - 1);
        for offset in 0..block {
            if out.len() == m {
                break;
            }
            out.push((start + offset) % m);
        }
    }
    out
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

/// Per-horizon OLS fit: coefficients, coefficient covariance, and design column moments.
///
/// `mu_hat(dose) = coefs' cbar(dose)`, where `cbar(dose)` is the vector of design
/// column means with the treatment column's mean replaced by `dose`. Because the
/// fitted model is linear in the treatment column with no treatment×covariate
/// interaction, this is an O(p) evaluation per dose (no re-scan of the design).
/// The analytic variance is `cbar' Sigma cbar` (coefficient uncertainty at the
/// observed covariate average) plus `w' S w / n` (uncertainty of that covariate
/// average, iid), so the band targets the population level `E[Y_h | do(A)]`.
struct FittedHorizon {
    prepared: PreparedEstimationProblem,
    coefs: Vec<f64>,
    /// p x p coefficient covariance, row-major.
    cov: Vec<f64>,
    /// Design column means (length p); index `TREATMENT_COL` is `Abar`.
    column_means: Vec<f64>,
    /// p x p sample covariance of the design columns, row-major.
    column_cov: Vec<f64>,
}

impl FittedHorizon {
    fn fit(
        prepared: PreparedEstimationProblem,
        ols_ws: &mut LeastSquaresWorkspace,
    ) -> Result<Self, EstimationError> {
        let n = prepared.design.nrows;
        let p = prepared.design.ncols;
        let fit = FaerBackend
            .least_squares(&prepared.design.matrix, n, p, &prepared.design.outcome, ols_ws)
            .map_err(EstimationError::from)?;
        let cov = coefficient_covariance(
            &prepared.design.matrix,
            n,
            p,
            &fit.residuals,
            SandwichKind::Homoskedastic,
        )
        .map_err(EstimationError::from)?;
        let column_means = design_column_means(&prepared.design);
        let column_cov = design_column_covariance(&prepared.design, &column_means);
        Ok(Self { prepared, coefs: fit.coefficients, cov, column_means, column_cov })
    }

    fn treatment_mean(&self) -> f64 {
        self.column_means[TREATMENT_COL]
    }

    /// Analytic SE of the g-computed level at `eval`.
    fn analytic_se(&self, eval: CellEval) -> f64 {
        let p = self.coefs.len();
        let mut cbar = self.column_means.clone();
        if let CellEval::Dose(dose) = eval {
            cbar[TREATMENT_COL] = dose;
        } else if let CellEval::Shift(shift) = eval {
            cbar[TREATMENT_COL] += shift;
        }
        let quadratic = |matrix: &[f64], v: &[f64]| {
            let mut total = 0.0;
            for (i, &vi) in v.iter().enumerate() {
                let row = &matrix[i * p..i * p + p];
                total += vi * row.iter().zip(v).map(|(&m, &vj)| m * vj).sum::<f64>();
            }
            total
        };
        // The level moves with every averaged column: the adjustment covariates
        // always, and the treatment column too when the policy shifts the factual law.
        let mut averaged = self.coefs.clone();
        averaged[0] = 0.0;
        if matches!(eval, CellEval::Dose(_)) {
            averaged[TREATMENT_COL] = 0.0;
        }
        let n = self.prepared.design.nrows.max(1) as f64;
        let var = quadratic(&self.cov, &cbar) + quadratic(&self.column_cov, &averaged) / n;
        var.max(0.0).sqrt()
    }

    /// `(mu_hat(dose), analytic se(dose))`.
    #[cfg(test)]
    fn mean_and_se_at(&self, dose: f64) -> (f64, f64) {
        let eval = CellEval::Dose(dose);
        (level_at(&self.coefs, &self.column_means, eval), self.analytic_se(eval))
    }
}

/// Joint bootstrap draws of every horizon's fit.
///
/// The surface evaluates `cbar(a)'β` at many cells, so each replicate keeps the whole
/// coefficient vector and the resampled design column means for every horizon, rather
/// than collapsing to a single SE the way [`crate::util::bootstrap_se`] does. Failure
/// accounting matches that helper: too few survivors, or more than
/// [`BOOTSTRAP_MAX_FAILURE_FRAC`] soft failures, means no bootstrap SE is reported.
struct SurfaceBootstrap {
    /// `draws[r][h] = (coefficients, design column means)` of replicate `r` at horizon `h`.
    draws: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    cancelled: bool,
}

/// Fill `out` with `n_rows` circular-block row indices for one horizon.
///
/// Block starts live on the common series time axis; lag-aligned rows end at the
/// series end, so row `i` sits at anchor time `base + i`. Mapping the same starts
/// through each horizon's `base` keeps every horizon of a replicate on the same
/// calendar blocks.
fn fill_block_rows(
    starts: &[usize],
    block: usize,
    base: usize,
    n_rows: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    if n_rows == 0 {
        return;
    }
    let shift = n_rows - base % n_rows;
    'blocks: for &start in starts {
        for offset in 0..block {
            if out.len() == n_rows {
                break 'blocks;
            }
            // (start + offset - base) mod n_rows, in unsigned arithmetic.
            out.push(((start + offset) % n_rows + shift) % n_rows);
        }
    }
}

/// Joint circular-block bootstrap of the whole dose × horizon surface.
///
/// Serially dependent lag-aligned rows are resampled in blocks (never row by row):
/// an iid pairs bootstrap misses the autocovariance of the level's mean component and
/// cannot support a simultaneous band. Each replicate draws one set of block starts on
/// the series time axis, refits every horizon on its time-aligned blocks, and keeps
/// the replicate only if every horizon fit succeeds, so each kept replicate is one draw
/// of the whole surface. Blocks resample lag-aligned tuples, so no replicate row
/// straddles a block junction.
///
/// Returns `None` when no bootstrap was requested, when fewer than two replicates
/// survived, or when singular resamples pushed the failure fraction past the crate-wide
/// threshold. Callers fall back to the analytic SE and must say so.
fn bootstrap_surface(
    horizons: &[FittedHorizon],
    series_rows: usize,
    block_length: usize,
    replicates: u32,
    ctx: &ExecutionContext,
    ols_ws: &mut LeastSquaresWorkspace,
) -> Option<SurfaceBootstrap> {
    if replicates == 0
        || horizons.is_empty()
        || horizons.iter().any(|fitted| fitted.prepared.design.nrows == 0)
    {
        return None;
    }
    let axis = horizons
        .iter()
        .map(|fitted| fitted.prepared.design.nrows)
        .max()
        .unwrap_or(0)
        .max(series_rows);
    let block = block_length.clamp(1, axis);
    let mut starts = vec![0usize; axis.div_ceil(block)];
    let mut rows = Vec::with_capacity(axis);
    let mut x_boot = Vec::new();
    let mut y_boot = Vec::new();
    let mut draws = Vec::with_capacity(replicates as usize);
    let mut failed = 0u32;
    let mut cancelled = false;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        let mut rng = ctx.rng.stream(HORIZON_BOOTSTRAP_STREAM.wrapping_add(u64::from(replicate)));
        for start in &mut starts {
            *start = ((rng.next_f64() * axis as f64) as usize).min(axis - 1);
        }
        let mut surface = Vec::with_capacity(horizons.len());
        for fitted in horizons {
            let design = &fitted.prepared.design;
            let (n, p) = (design.nrows, design.ncols);
            fill_block_rows(&starts, block, axis - n, n, &mut rows);
            x_boot.clear();
            x_boot.resize(n * p, 0.0);
            y_boot.clear();
            y_boot.extend(rows.iter().map(|&src| design.outcome[src]));
            for c in 0..p {
                let column = &design.matrix[c * n..(c + 1) * n];
                for (i, &src) in rows.iter().enumerate() {
                    x_boot[c * n + i] = column[src];
                }
            }
            let Ok(fit) = FaerBackend.least_squares(&x_boot, n, p, &y_boot, ols_ws) else {
                break;
            };
            let means =
                (0..p).map(|c| x_boot[c * n..(c + 1) * n].iter().sum::<f64>() / n as f64).collect();
            surface.push((fit.coefficients, means));
        }
        if surface.len() == horizons.len() {
            draws.push(surface);
        } else {
            failed += 1;
        }
        if let Some(progress) = &ctx.progress {
            progress.report(f64::from(replicate + 1) / f64::from(replicates), "bootstrap");
        }
    }
    if draws.len() < 2 {
        return None;
    }
    // Unattempted replicates after cancellation are not failures (mirrors
    // `finalize_bootstrap_se_ex`); singular resamples among those attempted are.
    let attempted = draws.len() as f64 + f64::from(failed);
    if f64::from(failed) / attempted > BOOTSTRAP_MAX_FAILURE_FRAC {
        return None;
    }
    Some(SurfaceBootstrap { draws, cancelled })
}

fn design_column_covariance(design: &CompiledDesign, means: &[f64]) -> Vec<f64> {
    let n = design.nrows;
    let p = design.ncols;
    let mut cov = vec![0.0; p * p];
    if n < 2 {
        return cov;
    }
    for a in 0..p {
        let col_a = &design.matrix[a * n..(a + 1) * n];
        for b in a..p {
            let col_b = &design.matrix[b * n..(b + 1) * n];
            let s: f64 = col_a
                .iter()
                .zip(col_b)
                .map(|(&x, &y)| (x - means[a]) * (y - means[b]))
                .sum::<f64>()
                / (n - 1) as f64;
            cov[a * p + b] = s;
            cov[b * p + a] = s;
        }
    }
    cov
}

fn design_column_means(design: &CompiledDesign) -> Vec<f64> {
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

fn checked_surface_cells(doses: usize, horizons: usize) -> Result<usize, EstimationError> {
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

fn flatten_dose_horizon_grid(doses: &[f64], horizons: &[u32]) -> Result<Vec<f64>, EstimationError> {
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

fn horizon_identification_of(
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

fn cell_against_range(dose: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !observed_min.is_finite() || !observed_max.is_finite() {
        SupportStatus::Extrapolative
    } else if dose < observed_min || dose > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Supported
    }
}

/// Surface summary over the same geometry as the estimate.
///
/// All cells supported → [`SupportStatus::Supported`]. Mixed supported /
/// unsupported cells → [`SupportStatus::Extrapolative`] (partially
/// extrapolative). No cell supported → [`SupportStatus::OutsideEmpiricalSupport`],
/// unless every cell was unassessable (non-finite range), which stays
/// extrapolative.
fn summarize_surface_support(points: &[SupportStatus]) -> SupportStatus {
    let n = points.len();
    let n_supported = points.iter().filter(|status| **status == SupportStatus::Supported).count();
    if n == 0 {
        return SupportStatus::Extrapolative;
    }
    if n_supported == n {
        return SupportStatus::Supported;
    }
    if n_supported > 0 {
        return SupportStatus::Extrapolative;
    }
    if points.iter().any(|status| *status == SupportStatus::OutsideEmpiricalSupport) {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

fn mean_curve_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let mut point_status = Vec::with_capacity(doses.len().saturating_mul(horizon_ranges.len()));
    for &dose in doses {
        for &(lo, hi) in horizon_ranges {
            point_status.push(cell_against_range(dose, lo, hi));
        }
    }
    assemble_temporal_support(doses, temporal, horizon_ranges, point_status)
}

fn intervention_support(
    eval_levels: &[f64],
    level: Option<f64>,
    shift: f64,
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let point_status: Vec<SupportStatus> = eval_levels
        .iter()
        .zip(horizon_ranges.iter())
        .map(|(&dose, &(lo, hi))| {
            if level.is_some() {
                cell_against_range(dose, lo, hi)
            } else {
                // A shift intervention evaluates the factual treatment law at
                // A + delta, not just at E[A] + delta. The latter can sit inside
                // [min(A), max(A)] while a large fraction of shifted rows are
                // outside it. Classify the whole shifted interval instead.
                shifted_range_against_range(shift, lo, hi)
            }
        })
        .collect();
    let shifted_extrapolation = level.is_none()
        && point_status.iter().any(|status| *status == SupportStatus::Extrapolative);
    let mut report = assemble_temporal_support(eval_levels, temporal, horizon_ranges, point_status);
    // InterventionResponse has one result cell per horizon. Reusing the mean
    // surface assembler used to claim an H x H dose-by-horizon layout even
    // though both the estimate and point_status contain only H cells.
    if let Some(layout) = report
        .diagnostics
        .iter_mut()
        .find(|diagnostic| diagnostic.id.as_ref() == "response.temporal.dose_horizon_layout")
    {
        layout.id = Arc::from("response.temporal.intervention_horizon_layout");
        layout.values = Arc::from([temporal.horizons.len() as f64]);
        layout.detail = Arc::from("one intervention-response cell per requested horizon");
    }
    if level.is_none() {
        let mut shifted_ranges = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
        for &(lo, hi) in horizon_ranges {
            shifted_ranges.push(lo + shift);
            shifted_ranges.push(hi + shift);
        }
        let shifted_min = shifted_ranges.iter().step_by(2).copied().fold(f64::INFINITY, f64::min);
        let shifted_max =
            shifted_ranges.iter().skip(1).step_by(2).copied().fold(f64::NEG_INFINITY, f64::max);
        report.query_region.minima =
            Arc::from([shifted_min, f64::from(temporal.horizons.first().copied().unwrap_or(1))]);
        report.query_region.maxima =
            Arc::from([shifted_max, f64::from(temporal.horizons.last().copied().unwrap_or(1))]);
        report.diagnostics.push(SupportDiagnostic {
            id: Arc::from("response.temporal.shifted_treatment_range"),
            values: Arc::from(shifted_ranges),
            detail: Arc::from(
                "per-horizon support requested by A + delta as [min_0+delta, max_0+delta, …]",
            ),
        });
    }
    if shifted_extrapolation {
        report.warnings.push(Diagnostic::new(
            "response.temporal.shift_distribution_extrapolative",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "the shifted treatment distribution extends beyond at least one horizon's observed treatment range",
        ));
    }
    report
}

fn shifted_range_against_range(shift: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !shift.is_finite() || !observed_min.is_finite() || !observed_max.is_finite() {
        return SupportStatus::Extrapolative;
    }
    if shift == 0.0 {
        return SupportStatus::Supported;
    }
    let shifted_min = observed_min + shift;
    let shifted_max = observed_max + shift;
    if !shifted_min.is_finite() || !shifted_max.is_finite() {
        SupportStatus::Extrapolative
    } else if shifted_max < observed_min || shifted_min > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

fn assemble_temporal_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
    point_status: Vec<SupportStatus>,
) -> SupportReport {
    let status = summarize_surface_support(&point_status);
    let mixed = point_status.iter().any(|s| *s == SupportStatus::Supported)
        && point_status.iter().any(|s| *s != SupportStatus::Supported);
    let mut range_values = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
    for &(lo, hi) in horizon_ranges {
        range_values.push(lo);
        range_values.push(hi);
    }
    let mut warnings = Vec::new();
    if mixed {
        warnings.push(Diagnostic::new(
            "response.temporal.partial_horizon_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "some requested (dose, horizon) cells sit outside that horizon's lag-aligned \
             treatment range; inspect support.point_status",
        ));
    } else if status == SupportStatus::OutsideEmpiricalSupport {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "no requested (dose, horizon) cell sits inside that horizon's lag-aligned \
             treatment range",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from(vec![
                doses.iter().copied().fold(f64::INFINITY, f64::min),
                f64::from(temporal.horizons.first().copied().unwrap_or(1)),
            ]),
            maxima: Arc::from(vec![
                doses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                f64::from(temporal.horizons.last().copied().unwrap_or(1)),
            ]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.temporal.dose_horizon_layout"),
                values: Arc::from(vec![doses.len() as f64, temporal.horizons.len() as f64]),
                detail: Arc::from(
                    "row-major dose × horizon surface: value[d * n_horizons + h]; \
                     grid stores [dose_d, horizon_h] pairs",
                ),
            },
            SupportDiagnostic {
                id: Arc::from("response.temporal.horizon_treatment_range"),
                values: Arc::from(range_values),
                detail: Arc::from(
                    "per-horizon lag-aligned treatment range as [min_0, max_0, min_1, max_1, …]",
                ),
            },
        ],
        warnings,
        point_status: Some(Arc::from(point_status)),
    }
}

/// Plan overlays for a temporal [`ResponseQuery`], if it is an `InterventionResponse`.
///
/// # Errors
///
/// Unlicensed Sequence / Soft forms.
pub fn plan_from_response_query(
    query: &ResponseQuery,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let Some(temporal) = query.temporal.as_ref() else {
        return Ok(None);
    };
    match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Ok(Some(plan_temporal_intervention(interventions, temporal)?))
        }
        _ => Ok(None),
    }
}

/// Classify a temporal `InterventionResponse` as a single-node overlay or a
/// sequential schedule. Nested Sequence stays refused. Multi-step never
/// collapses to the last step.
///
/// # Errors
///
/// Empty, nested, stochastic, unlicensed Soft, or ambiguous schedules.
pub fn plan_temporal_intervention(
    interventions: &[Intervention],
    spec: &TemporalResponseSpec,
) -> Result<TemporalInterventionPlan, EstimationError> {
    if interventions.is_empty() {
        return Err(EstimationError::unsupported(
            "intervention response requires at least one intervention",
        ));
    }
    if interventions.len() > 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse supports one primary intervention \
             (use Sequence for multi-step or joint policies)",
        ));
    }
    if let Some(plan) = plan_mean_mechanisms(&interventions[0], spec)? {
        return Ok(plan);
    }
    match &interventions[0] {
        Intervention::Sequence(seq) => plan_sequence(seq, spec, 0),
        other => {
            let (treatment, level, shift) = resolve_one(other, 0)?;
            Ok(TemporalInterventionPlan::Single { treatment, level, shift })
        }
    }
}

fn plan_mean_mechanisms(
    intervention: &Intervention,
    spec: &TemporalResponseSpec,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let leaves = match intervention {
        Intervention::Sequence(sequence) => {
            sequence.steps.iter().map(|step| &step.intervention).collect::<Vec<_>>()
        }
        other => vec![other],
    };
    if !leaves.iter().any(|leaf| matches!(leaf,
        Intervention::Soft { mechanism, .. } if matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift"))) {
        return Ok(None);
    }
    let mut modifiers = Vec::new();
    let mut replacements = Vec::new();
    for leaf in leaves {
        let mut multiplier = 1.0;
        let mut bounds = None;
        let mut replacement = leaf.clone();
        if let Intervention::Soft { variable, mechanism } = leaf {
            match mechanism.family_id.as_ref() {
                "multiplicative" => {
                    if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
                        return Err(EstimationError::unsupported(
                            "multiplicative requires one finite mean multiplier",
                        ));
                    }
                    multiplier = mechanism.parameters[0];
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(0.0));
                }
                "truncated_shift" => {
                    let p = &mechanism.parameters;
                    if p.len() != 3 || p.iter().any(|v| !v.is_finite()) || p[1] > p[2] {
                        return Err(EstimationError::unsupported(
                            "truncated_shift requires finite [shift, lower, upper] with lower <= upper",
                        ));
                    }
                    bounds = Some((p[1], p[2]));
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(p[0]));
                }
                _ => {}
            }
        }
        modifiers.push((multiplier, bounds));
        replacements.push(replacement);
    }
    let surrogate = if let Intervention::Sequence(sequence) = intervention {
        let mut sequence = sequence.clone();
        let mut steps = sequence.steps.to_vec();
        for (step, replacement) in steps.iter_mut().zip(replacements) {
            step.intervention = replacement;
        }
        sequence.steps = Arc::from(steps);
        Intervention::Sequence(sequence)
    } else {
        replacements.remove(0)
    };
    let base = plan_temporal_intervention(&[surrogate], spec)?;
    let nodes = match base {
        TemporalInterventionPlan::Single { treatment, level, shift } => spec
            .policy
            .active_offsets()
            .map_err(|e| EstimationError::data_msg(e.to_string()))?
            .iter()
            .map(|&offset| SequentialNodeOverlay { variable: treatment, offset, level, shift })
            .collect::<Vec<_>>(),
        TemporalInterventionPlan::Sequential { overlays } => overlays,
        TemporalInterventionPlan::Mechanisms { .. } => {
            unreachable!("surrogate contains only Set/Shift")
        }
    };
    let overlays = nodes
        .into_iter()
        .enumerate()
        .map(|(index, node)| {
            let (multiplier, bounds) = modifiers[if modifiers.len() == 1 { 0 } else { index }];
            SequentialMechanismOverlay { node, multiplier, bounds }
        })
        .collect();
    Ok(Some(TemporalInterventionPlan::Mechanisms { overlays }))
}

/// `depth` counts levels of `Intervention::Sequence` nesting already entered.
/// `resolve_sequence` is only reachable from a Sequence itself, so a `depth > 0`
/// arrival there means a Sequence nested inside a Sequence — refused explicitly
/// rather than silently recursing into a leaf (ADR 0021 fail-closed contract).
fn resolve_one(
    iv: &Intervention,
    depth: usize,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    let finite_numeric = |value: &Value, kind: &'static str| {
        value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            EstimationError::unsupported(match kind {
                "set" => "intervention Set requires a finite numeric value",
                _ => "intervention Shift requires a finite numeric delta",
            })
        })
    };
    let one_finite_parameter = |mechanism: &MechanismOverride, family: &'static str| {
        if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
            return Err(EstimationError::unsupported(match family {
                "constant" => "Soft(constant) requires exactly one finite parameter",
                _ => "Soft(additive_shift) requires exactly one finite parameter",
            }));
        }
        Ok(mechanism.parameters[0])
    };
    match iv {
        Intervention::Set { variable, value } => {
            let level = finite_numeric(value, "set")?;
            Ok((*variable, Some(level), 0.0))
        }
        Intervention::Shift { variable, delta } => {
            let d = finite_numeric(delta, "shift")?;
            Ok((*variable, None, d))
        }
        Intervention::Soft { variable, mechanism } => match mechanism.family_id.as_ref() {
            "constant" => {
                let level = one_finite_parameter(mechanism, "constant")?;
                Ok((*variable, Some(level), 0.0))
            }
            "additive_shift" => {
                let d = one_finite_parameter(mechanism, "additive_shift")?;
                Ok((*variable, None, d))
            }
            other => Err(EstimationError::data_msg(format!(
                "Soft mechanism family `{other}` is not licensed for temporal InterventionResponse; \
                 use constant or additive_shift"
            ))),
        },
        Intervention::Sequence(seq) => {
            if depth > 0 {
                return Err(EstimationError::unsupported(
                    "Intervention::Sequence nested inside a Sequence is not licensed for \
                     temporal InterventionResponse",
                ));
            }
            if seq.steps.len() != 1 {
                return Err(EstimationError::unsupported(
                    "multi-step Sequence must be planned as sequential overlays; \
                     refuse rather than collapse to the last step",
                ));
            }
            resolve_one(&seq.steps[0].intervention, depth + 1)
        }
        Intervention::Stochastic { .. } => Err(EstimationError::unsupported(
            "stochastic interventions are not licensed on the temporal InterventionResponse path",
        )),
        other => Err(EstimationError::data_msg(format!(
            "unsupported intervention variant for temporal response: {other:?}"
        ))),
    }
}

fn plan_sequence(
    seq: &InterventionSequence,
    spec: &TemporalResponseSpec,
    depth: usize,
) -> Result<TemporalInterventionPlan, EstimationError> {
    if seq.is_empty() {
        return Err(EstimationError::unsupported("empty Intervention::Sequence"));
    }
    if depth > 0 {
        return Err(EstimationError::unsupported(
            "Intervention::Sequence nested inside a Sequence is not licensed for \
             temporal InterventionResponse",
        ));
    }
    let origin = spec.treatment_offset()?;
    let mut leaves = Vec::with_capacity(seq.steps.len());
    for step in seq.steps.iter() {
        if matches!(step.intervention, Intervention::Sequence(_)) {
            return Err(EstimationError::unsupported(
                "Intervention::Sequence nested inside a Sequence is not licensed for \
                 temporal InterventionResponse",
            ));
        }
        let (variable, level, shift) = resolve_one(&step.intervention, depth + 1)?;
        let offsets =
            step.temporal.active_offsets().map_err(|e| EstimationError::data_msg(e.to_string()))?;
        leaves.push((variable, level, shift, offsets));
    }
    if leaves.len() == 1 {
        let (variable, level, shift, offsets) = &leaves[0];
        // Pulse(0) / a singleton window at 0 is the implicit Sequence shorthand
        // and attaches to the spec origin. Any other native step policy — Pulse(-2),
        // Sustained(-3, -1), or a window that already includes time 0 among other
        // times — is executed as written.
        let resolved = match offsets.as_ref() {
            [0] => Arc::<[i32]>::from([origin]),
            _ => Arc::clone(offsets),
        };
        let overlays = resolved
            .iter()
            .map(|&offset| SequentialNodeOverlay {
                variable: *variable,
                offset,
                level: *level,
                shift: *shift,
            })
            .collect();
        return Ok(TemporalInterventionPlan::Sequential { overlays });
    }
    let offsets = sequence_overlay_offsets(origin, &leaves)?;
    let mut overlays = Vec::with_capacity(leaves.len());
    let mut seen = Vec::new();
    for ((variable, level, shift, _), offset) in leaves.iter().zip(offsets) {
        let key = (*variable, offset);
        if seen.contains(&key) {
            return Err(EstimationError::unsupported(
                "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
            ));
        }
        seen.push(key);
        overlays.push(SequentialNodeOverlay {
            variable: *variable,
            offset,
            level: *level,
            shift: *shift,
        });
    }
    Ok(TemporalInterventionPlan::Sequential { overlays })
}

/// Multi-step same variable → consecutive times ending at the spec origin.
/// Distinct variables, each once → joint at the origin.
/// Explicit distinct Pulse offsets are honored.
fn sequence_overlay_offsets(
    origin: i32,
    leaves: &[SequenceLeaf],
) -> Result<Vec<i32>, EstimationError> {
    let n = i32::try_from(leaves.len())
        .map_err(|_| EstimationError::unsupported("Sequence is too long"))?;
    let mut explicit = Vec::with_capacity(leaves.len());
    for (_, _, _, offsets) in leaves {
        let [at] = offsets.as_ref() else {
            return Err(EstimationError::unsupported(
                "each Sequence step must be a Pulse or single-time window",
            ));
        };
        explicit.push(*at);
    }
    let same_var = leaves.iter().all(|(variable, _, _, _)| *variable == leaves[0].0);
    let all_default = explicit.iter().all(|&at| at == 0);
    let all_origin = explicit.iter().all(|&at| at == origin);
    let distinct_explicit = {
        let mut seen = explicit.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.len() == explicit.len() && !(all_default || all_origin)
    };
    if distinct_explicit {
        return Ok(explicit);
    }
    if same_var && (all_default || all_origin) {
        // Consecutive times ending at the policy origin. Two Pulse{0} (or
        // Pulse{origin}) steps are a two-step policy, not last-step collapse.
        return Ok((0..n).map(|i| origin - (n - 1 - i)).collect());
    }
    let unique_vars = {
        let mut vars: Vec<VariableId> =
            leaves.iter().map(|(variable, _, _, _)| *variable).collect();
        vars.sort_by_key(|variable| variable.raw());
        vars.dedup();
        vars.len() == leaves.len()
    };
    if unique_vars && (all_default || all_origin) {
        return Ok(vec![origin; leaves.len()]);
    }
    Err(EstimationError::unsupported(
        "Sequence schedule is ambiguous; use distinct Pulse offsets or a same-variable \
         consecutive policy",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::TemporalPolicy;

    #[test]
    fn soft_constant_resolves_to_set() {
        let v = VariableId::from_raw(0);
        let (t, level, shift) =
            resolve_one(&Intervention::soft(v, MechanismOverride::constant(1.5)), 0).unwrap();
        assert_eq!(t, v);
        assert_eq!(level, Some(1.5));
        assert!(shift.abs() < f64::EPSILON);
    }

    #[test]
    fn soft_unknown_family_refuses() {
        let v = VariableId::from_raw(0);
        let err = resolve_one(
            &Intervention::soft(v, MechanismOverride::named("linear_gaussian", vec![1.0])),
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not licensed"));
    }

    #[test]
    fn non_finite_and_ambiguous_soft_parameters_fail_closed() {
        let v = VariableId::from_raw(0);
        let non_finite = resolve_one(&Intervention::set(v, Value::f64(f64::NAN)), 0).unwrap_err();
        assert!(non_finite.to_string().contains("finite numeric"));

        let too_many = resolve_one(
            &Intervention::soft(v, MechanismOverride::named("additive_shift", vec![1.0, 2.0])),
            0,
        )
        .unwrap_err();
        assert!(too_many.to_string().contains("exactly one finite parameter"));
    }

    #[test]
    fn multi_step_sequence_is_consecutive_overlays_not_last_step() {
        use antecedent_core::SequencedIntervention;

        let v = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let seq = InterventionSequence {
            steps: Arc::from(vec![
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(0.0)),
                    temporal: TemporalPolicy::pulse(0),
                },
                SequencedIntervention {
                    intervention: Intervention::set(v, Value::f64(5.0)),
                    temporal: TemporalPolicy::pulse(0),
                },
            ]),
        };
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].offset, -2);
        assert_eq!(overlays[0].level, Some(0.0));
        assert_eq!(overlays[1].offset, -1);
        assert_eq!(overlays[1].level, Some(5.0));
        assert_ne!(overlays[0].level, overlays[1].level);
    }

    #[test]
    fn implicit_single_step_pulse_zero_attaches_to_spec_origin() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an implicit Pulse(0) Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays,
            vec![SequentialNodeOverlay { variable, offset: -1, level: Some(1.0), shift: 0.0 }]
        );
    }

    #[test]
    fn sequence_step_order_is_the_schedule_not_a_set() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let forward = InterventionSequence::new([
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(0.0)),
                temporal: TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(5.0)),
                temporal: TemporalPolicy::pulse(0),
            },
        ]);
        let reversed = InterventionSequence::new([
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(5.0)),
                temporal: TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(0.0)),
                temporal: TemporalPolicy::pulse(0),
            },
        ]);
        let TemporalInterventionPlan::Sequential { overlays: a } =
            plan_temporal_intervention(&[Intervention::Sequence(forward)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        let TemporalInterventionPlan::Sequential { overlays: b } =
            plan_temporal_intervention(&[Intervention::Sequence(reversed)], &spec).unwrap()
        else {
            panic!("expected sequential overlays");
        };
        assert_eq!(a[0].offset, -2);
        assert_eq!(a[1].offset, -1);
        assert_eq!(a[0].level, Some(0.0));
        assert_eq!(a[1].level, Some(5.0));
        assert_eq!(b[0].offset, -2);
        assert_eq!(b[1].offset, -1);
        assert_eq!(b[0].level, Some(5.0));
        assert_eq!(b[1].level, Some(0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn single_step_sequence_preserves_explicit_pulse_offset() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(2.0)),
            temporal: TemporalPolicy::pulse(-2),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an explicit Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays,
            vec![SequentialNodeOverlay { variable, offset: -2, level: Some(2.0), shift: 0.0 }]
        );
    }

    #[test]
    fn single_step_sequence_expands_explicit_sustained_policy() {
        use antecedent_core::SequencedIntervention;

        let variable = VariableId::from_raw(0);
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let sequence = InterventionSequence::new([SequencedIntervention {
            intervention: Intervention::shift(variable, Value::f64(0.5)),
            temporal: TemporalPolicy::sustained(-3, -1),
        }]);
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
        else {
            panic!("an explicit Sequence must resolve to overlays");
        };
        assert_eq!(
            overlays.iter().map(|overlay| overlay.offset).collect::<Vec<_>>(),
            vec![-3, -2, -1]
        );
        assert!(
            overlays.iter().all(
                |overlay| overlay.level.is_none() && (overlay.shift - 0.5).abs() < f64::EPSILON
            )
        );
    }

    #[test]
    fn nested_sequence_fails_closed() {
        use antecedent_core::SequencedIntervention;

        let v = VariableId::from_raw(0);
        let inner = InterventionSequence {
            steps: Arc::from(vec![SequencedIntervention {
                intervention: Intervention::set(v, Value::f64(0.0)),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            }]),
        };
        let outer = InterventionSequence {
            steps: Arc::from(vec![SequencedIntervention {
                intervention: Intervention::Sequence(inner),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            }]),
        };
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let err = plan_temporal_intervention(&[Intervention::Sequence(outer)], &spec).unwrap_err();
        assert!(err.to_string().contains("not licensed"));
    }

    // ---- GAP1: uncertainty is computed but was never asserted anywhere ----
    //
    // The band was fixed from a wrong formula (ATE-coefficient SE scaled by dose,
    // which gave a ZERO-WIDTH 95% interval at dose 0) to the correct linear-functional
    // variance `cbar(a)' Sigma cbar(a)`. These tests would fail against the old formula.

    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;

    /// Deterministic AR(2)-ish series: `t` is a mildly autocorrelated continuous
    /// treatment, `y` depends on `t` lagged 1 and 2 steps. Non-degenerate `t` mean.
    fn synthetic_series(n: usize) -> (TimeSeriesData, TemporalDag) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 2..n {
            t[i] = 0.3 + 0.2 * t[i - 1] + 0.05 * (i as f64).sin();
            y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2] + 0.01 * (i as f64).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(t2, y0).unwrap();
        (data, graph)
    }

    fn identify(graph: &TemporalDag, horizon_steps: u32) -> (IdentifiedEstimand, TemporalIndexer) {
        let id_query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_horizon_steps(horizon_steps)
                .with_policy(TemporalPolicy::pulse(0));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(graph, &id_query).unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        (estimand, id_res.indexer)
    }

    #[test]
    fn public_entry_point_revalidates_query_and_identification_status() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let invalid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(0),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([-1.0, 1.0])),
            ),
        })
        .with_temporal(temporal.clone());
        let (data, _) = synthetic_series(100);
        let err = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[],
                &invalid_query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap_err();
        assert!(err.to_string().contains("same variable"), "got {err}");

        let valid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([-1.0, 1.0])),
            ),
        })
        .with_temporal(temporal);
        let err = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[],
                &valid_query,
                IdentificationStatus::NotIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap_err();
        assert!(err.to_string().contains("point identification"), "got {err}");
    }

    #[test]
    fn response_preserves_the_exact_caller_estimand() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 1);
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Linspace { start: -1.0, end: 1.0, points: 3 },
            ),
        })
        .with_temporal(temporal);
        let result = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &[(&estimand, &indexer)],
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap();
        assert_eq!(result.estimand, query.functional);
    }

    /// (a) `lower < mean < upper` strictly for every cell of the dose x horizon surface,
    /// and the band is symmetric about the mean.
    /// (b) A dose of exactly 0.0 in the grid produces a STRICTLY POSITIVE band width —
    /// the direct regression guard for the old zero-width-at-dose-0 bug.
    #[test]
    fn uncertainty_band_strict_and_zero_dose_has_positive_width() {
        let (data, graph) = synthetic_series(400);
        let (estimand, indexer) = identify(&graph, 4);
        let doses = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let n_h = 3;
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 4], TemporalPolicy::pulse(0), None).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from(doses.clone())),
            ),
        })
        .with_temporal(temporal);
        let est = TemporalResponseEstimator::new();
        let result = est
            .estimate(
                &data,
                &[(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)],
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap();

        assert!(
            result.assumptions.entries.iter().any(|r| matches!(
                &r.assumption,
                Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.homoskedastic.pointwise"
            )),
            "temporal response must record the homoskedastic pointwise OLS assumption"
        );
        assert!(result.assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.linear_additive.gcomp"
        )));

        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &result.estimate
        else {
            panic!("expected point-identified surface");
        };
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &result.uncertainty else {
            panic!(
                "expected PointwiseBand uncertainty on the temporal MeanCurve path — this is \
                 exactly the GAP1 regression this test guards against"
            );
        };
        assert_eq!(mean.len(), doses.len() * n_h);
        assert_eq!(mean.len(), lower.len());
        assert_eq!(mean.len(), upper.len());

        for i in 0..mean.len() {
            assert!(lower[i] < mean[i], "cell {i}: lower {} not < mean {}", lower[i], mean[i]);
            assert!(mean[i] < upper[i], "cell {i}: mean {} not < upper {}", mean[i], upper[i]);
            // Symmetric about the mean by construction (mean +/- z*se); assert it holds.
            assert!(
                (mean[i] - lower[i] - (upper[i] - mean[i])).abs() < 1e-9,
                "cell {i}: band not symmetric about mean (lower half {}, upper half {})",
                mean[i] - lower[i],
                upper[i] - mean[i]
            );
        }

        // (b): dose == 0.0 must produce a strictly positive band width. Under the old
        // (buggy) formula — ATE-coefficient SE scaled by dose — the width at dose 0.0
        // was exactly zero.
        let zero_idx = doses.iter().position(|&d| d == 0.0).unwrap();
        for h in 0..n_h {
            let idx = zero_idx * n_h + h;
            let width = upper[idx] - lower[idx];
            assert!(
                width > 1e-9,
                "dose=0.0 horizon-slot {h}: band width {width} is not strictly positive \
                 (regression guard: old formula gave a zero-width interval at dose 0)"
            );
        }
    }

    /// (c) Recompute `se(dose)` independently — build `cbar` from the fitted design's
    /// column means with the treatment entry replaced by `dose`, then form the quadratic
    /// form against the coefficient covariance directly (own loop, not `mean_and_se_at`)
    /// — and check it matches production. A regression to the old ATE-coefficient-SE
    /// formula would diverge from this independent recompute.
    #[test]
    fn uncertainty_se_matches_independent_quadratic_form_recompute() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 3);
        let temporal =
            TemporalResponseSpec::new(vec![3u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let fitted = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                3,
                &indexer,
                &ExecutionContext::for_tests(9),
                &mut ws,
            )
            .unwrap();

        let p = fitted.coefs.len();
        let z = normal_ppf(0.975);
        for &dose in &[-2.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
            let mut cbar = fitted.column_means.clone();
            cbar[TREATMENT_COL] = dose;
            let mut expected_var = 0.0;
            for i in 0..p {
                for j in 0..p {
                    expected_var += cbar[i] * fitted.cov[i * p + j] * cbar[j];
                }
            }
            let expected_se = expected_var.max(0.0).sqrt();
            let (mu, actual_se) = fitted.mean_and_se_at(dose);

            assert!(
                (actual_se - expected_se).abs() <= 1e-9_f64.max(1e-9 * expected_se),
                "dose={dose}: production se {actual_se} != independently recomputed se {expected_se}"
            );
            // Half-width equals normal_ppf(0.975) * se, for the independently recomputed se.
            let expected_lower = mu - z * expected_se;
            let expected_upper = mu + z * expected_se;
            assert!((expected_upper - mu - z * expected_se).abs() < 1e-9);
            assert!((mu - expected_lower - z * expected_se).abs() < 1e-9);
        }
    }

    /// (d) se(dose) grows as the dose moves away from the observed treatment mean — the
    /// standard widening of a regression band away from the design centroid. Under the
    /// old buggy formula (se proportional to |dose|, minimized at dose == 0.0), this
    /// fails whenever the fitted treatment mean is nonzero, since `se(0.0)` would then be
    /// smaller than `se(treatment_mean)`.
    #[test]
    fn uncertainty_se_widens_away_from_treatment_mean() {
        let (data, graph) = synthetic_series(400);
        let (estimand, indexer) = identify(&graph, 4);
        let temporal =
            TemporalResponseSpec::new(vec![2u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let fitted = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                2,
                &indexer,
                &ExecutionContext::for_tests(7),
                &mut ws,
            )
            .unwrap();

        let center = fitted.treatment_mean();
        assert!(
            center.abs() > 1e-3,
            "test fixture assumes a non-degenerate (nonzero) treatment mean; got {center}"
        );
        let (_, se_center) = fitted.mean_and_se_at(center);
        let (_, se_near) = fitted.mean_and_se_at(center + 1.0);
        let (_, se_far) = fitted.mean_and_se_at(center + 3.0);
        let (_, se_near_neg) = fitted.mean_and_se_at(center - 1.0);
        let (_, se_zero) = fitted.mean_and_se_at(0.0);

        assert!(
            se_center < se_near,
            "se should grow moving away from center: {se_center} vs {se_near}"
        );
        assert!(
            se_near < se_far,
            "se should keep growing further from center: {se_near} vs {se_far}"
        );
        assert!(
            se_center < se_near_neg,
            "se should grow symmetrically on the other side of the center: {se_center} vs {se_near_neg}"
        );
        assert!(
            se_center < se_zero,
            "se at the treatment mean ({center}) should be smaller than se at dose=0.0 ({se_zero}); \
             the old bug's minimum was at dose 0, not at the design centroid"
        );
    }

    // ---- GAP2: refusal paths were unasserted ----

    /// (f) A `Sequence` spanning multiple target variables must refuse. Exercised directly
    /// against `resolve_sequence` (rather than end-to-end through `Study`) because a
    /// cross-variable `Sequence` has no unique `primary_variable`, so the facade already
    /// refuses earlier ("no treatment/outcome pair") before temporal resolution runs.
    #[test]
    fn sequence_multiple_target_variables_fails_closed() {
        use antecedent_core::SequencedIntervention;

        let v0 = VariableId::from_raw(0);
        let v1 = VariableId::from_raw(1);
        let seq = InterventionSequence {
            steps: Arc::from(vec![
                SequencedIntervention {
                    intervention: Intervention::set(v0, Value::f64(1.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
                SequencedIntervention {
                    intervention: Intervention::set(v1, Value::f64(2.0)),
                    temporal: antecedent_core::TemporalPolicy::pulse(0),
                },
            ]),
        };
        let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
        let TemporalInterventionPlan::Sequential { overlays } =
            plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
        else {
            panic!("joint Sequence must be sequential overlays");
        };
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].variable, v0);
        assert_eq!(overlays[1].variable, v1);
        assert_eq!(overlays[0].offset, -1);
        assert_eq!(overlays[1].offset, -1);
    }

    /// (d) Empty dose grid must refuse with a specific message, not silently produce an
    /// empty (or garbage) surface.
    #[test]
    fn empty_dose_grid_refuses() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 2);
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let err = est
            .estimate_mean_curve(
                &data,
                &[(&estimand, &indexer)],
                VariableId::from_raw(1),
                VariableId::from_raw(0),
                &[],
                &temporal,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(err.to_string().contains("dose grid must be non-empty"), "unexpected error: {err}");
    }

    /// The union of per-horizon treatment ranges would call dose 1.5 supported
    /// here; the cell grid must not.
    #[test]
    fn surface_support_uses_per_horizon_ranges_not_union() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
        let report = mean_curve_support(&[1.5], &temporal, &ranges);
        assert_eq!(report.status, SupportStatus::Extrapolative);
        assert_eq!(
            report.point_status.as_ref().map(AsRef::as_ref),
            Some(
                [
                    SupportStatus::Supported,
                    SupportStatus::OutsideEmpiricalSupport,
                    SupportStatus::OutsideEmpiricalSupport,
                ]
                .as_slice()
            )
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.temporal.partial_horizon_support")
        );
        let ranges_diag = report
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.temporal.horizon_treatment_range")
            .expect("horizon treatment range diagnostic");
        assert_eq!(ranges_diag.values.as_ref(), [-2.0, 2.0, -1.0, 1.0, -0.4, 0.4].as_slice());
    }

    #[test]
    fn surface_support_all_outside_and_all_supported() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-1.0, 1.0), (-0.5, 0.5)];
        let outside = mean_curve_support(&[10.0], &temporal, &ranges);
        assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
        assert!(
            outside
                .point_status
                .as_ref()
                .unwrap()
                .iter()
                .all(|s| { *s == SupportStatus::OutsideEmpiricalSupport })
        );
        assert!(
            outside
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.outside_empirical_support")
        );

        let inside = mean_curve_support(&[0.0], &temporal, &ranges);
        assert_eq!(inside.status, SupportStatus::Supported);
        assert!(inside.warnings.is_empty());
        assert!(
            inside.point_status.as_ref().unwrap().iter().all(|s| *s == SupportStatus::Supported)
        );
    }

    #[test]
    fn intervention_support_is_one_cell_per_horizon() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
        let report = intervention_support(&[1.5, 1.5, 1.5], Some(1.5), 0.0, &temporal, &ranges);
        assert_eq!(report.status, SupportStatus::Extrapolative);
        assert_eq!(report.point_status.as_ref().unwrap().len(), 3);
        let layout = report
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.id.as_ref() == "response.temporal.intervention_horizon_layout"
            })
            .expect("intervention layout");
        assert_eq!(layout.values.as_ref(), &[3.0]);
    }

    #[test]
    fn shift_support_checks_the_shifted_law_not_only_its_mean() {
        let temporal =
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
        let ranges = [(0.0, 1.0)];

        // E[A] + 0.4 = 0.9 is inside the observed range, but the shifted
        // treatment law spans [0.4, 1.4] and therefore extrapolates.
        let partial = intervention_support(&[0.9], None, 0.4, &temporal, &ranges);
        assert_eq!(partial.status, SupportStatus::Extrapolative);
        assert_eq!(partial.point_status.as_deref(), Some(&[SupportStatus::Extrapolative][..]));
        assert!(partial.warnings.iter().any(|warning| {
            warning.code.as_ref() == "response.temporal.shift_distribution_extrapolative"
        }));
        assert_eq!(partial.query_region.minima.as_ref(), &[0.4, 1.0]);
        assert_eq!(partial.query_region.maxima.as_ref(), &[1.4, 1.0]);
        let shifted = partial
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.id.as_ref() == "response.temporal.shifted_treatment_range"
            })
            .expect("shifted treatment range");
        assert_eq!(shifted.values.as_ref(), &[0.4, 1.4]);

        // A disjoint shifted law gets the stronger outside-support status.
        let outside = intervention_support(&[2.5], None, 2.0, &temporal, &ranges);
        assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
        assert_eq!(
            outside.point_status.as_deref(),
            Some(&[SupportStatus::OutsideEmpiricalSupport][..])
        );

        let identity = intervention_support(&[0.5], None, 0.0, &temporal, &ranges);
        assert_eq!(identity.status, SupportStatus::Supported);
    }

    /// Extreme T only at the end of the series. A longer-horizon pulse looks
    /// further back, so that spike is inside the h=1 treatment column and
    /// outside a long-horizon window.
    fn spike_then_quiet_series(n: usize) -> (TimeSeriesData, TemporalDag) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for (i, t_i) in t.iter_mut().enumerate() {
            *t_i = if i >= n.saturating_sub(7) { 10.0 } else { 0.05 * (i as f64).sin() };
        }
        for i in 2..n {
            y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(t2, y0).unwrap();
        (data, graph)
    }

    #[test]
    fn late_treatment_spike_is_horizon_specific_support() {
        let (data, graph) = spike_then_quiet_series(80);
        let (estimand, indexer) = identify(&graph, 8);
        let temporal =
            TemporalResponseSpec::new(vec![1u32, 8], TemporalPolicy::pulse(0), None).unwrap();
        let est = TemporalResponseEstimator::new();
        let mut ws = LeastSquaresWorkspace::default();
        let ctx = ExecutionContext::for_tests(11);
        let short = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                1,
                &indexer,
                &ctx,
                &mut ws,
            )
            .unwrap();
        let long = est
            .fit_horizon(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                &temporal,
                8,
                &indexer,
                &ctx,
                &mut ws,
            )
            .unwrap();
        let short_range = range(&short.prepared.treatment);
        let long_range = range(&long.prepared.treatment);
        assert!(
            short_range.1 > long_range.1 + 1.0,
            "long horizon should miss the late spike: short={short_range:?} long={long_range:?}"
        );
        let dose = long_range.1 + (short_range.1 - long_range.1) * 0.5;
        let result = est
            .estimate_mean_curve(
                &data,
                &[(&estimand, &indexer), (&estimand, &indexer)],
                VariableId::from_raw(1),
                VariableId::from_raw(0),
                &[0.0, dose],
                &temporal,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ctx,
            )
            .unwrap();
        assert_eq!(result.support.status, SupportStatus::Extrapolative);
        let cells = result.support.point_status.as_ref().expect("temporal point_status");
        // dose-major: (0, h=1), (0, h=8), (dose, h=1), (dose, h=8)
        assert_eq!(cells[0], SupportStatus::Supported);
        assert_eq!(cells[1], SupportStatus::Supported);
        assert_eq!(cells[2], SupportStatus::Supported);
        assert_eq!(cells[3], SupportStatus::OutsideEmpiricalSupport);
        // The union envelope would have classified `dose` as supported.
        assert!(dose >= short_range.0 && dose <= short_range.1);
        assert!(dose < long_range.0 || dose > long_range.1);
    }

    fn surface_query(doses: &[f64], horizons: Vec<u32>) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from(doses.to_vec())),
            ),
        })
        .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(0), None).unwrap())
    }

    fn pointwise(response: &CausalResponse) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &response.estimate
        else {
            panic!("expected surface");
        };
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("expected pointwise band");
        };
        (mean.to_vec(), lower.to_vec(), upper.to_vec())
    }

    fn diagnostic<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
        response.support.diagnostics.iter().find(|d| d.id.as_ref() == id).map(|d| d.values.as_ref())
    }

    #[test]
    fn bootstrap_se_differs_from_analytic_ols_se() {
        let (data, graph) = synthetic_series(240);
        let (estimand, indexer) = identify(&graph, 3);
        let query = surface_query(&[0.0, 1.0], vec![3u32]);
        let run = |replicates| {
            TemporalResponseEstimator::new()
                .with_bootstrap_replicates(replicates)
                .estimate(
                    &data,
                    &[(&estimand, &indexer)],
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &ExecutionContext::for_tests(13),
                )
                .unwrap()
        };
        let (mu_a, lo_a, _) = pointwise(&run(0));
        let (mu_b, lo_b, _) = pointwise(&run(40));
        let (se_a, se_b) = (mu_a[1] - lo_a[1], mu_b[1] - lo_b[1]);
        assert!(se_a.is_finite() && se_a > 0.0, "analytic half-width={se_a}");
        assert!(se_b.is_finite() && se_b > 0.0, "bootstrap half-width={se_b}");
        assert!(
            (se_a - se_b).abs() > 1e-6,
            "Study bootstrap must change the surface SE (analytic={se_a}, bootstrap={se_b})"
        );
        assert!(
            (mu_a[0] - mu_b[0]).abs() < 1e-12,
            "point estimate must stay full-sample OLS (analytic={}, boot={})",
            mu_a[0],
            mu_b[0]
        );
    }

    /// The bootstrap surface publishes a simultaneous band next to the pointwise
    /// band; it contains the pointwise band at every cell and names its construction.
    /// The analytic path withholds it explicitly.
    #[test]
    fn bootstrap_surface_publishes_simultaneous_band_alongside_pointwise() {
        let (data, graph) = synthetic_series(300);
        let (estimand, indexer) = identify(&graph, 3);
        let doses = [-1.0, 0.0, 1.0, 2.0];
        let query = surface_query(&doses, vec![1u32, 2, 3]);
        let ids = [(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)];
        let boot = TemporalResponseEstimator::new()
            .with_bootstrap_replicates(80)
            .estimate(
                &data,
                &ids,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        let (mean, lower, upper) = pointwise(&boot);
        let sim_lo = diagnostic(&boot, SIMULTANEOUS_BAND_LOWER).expect("simultaneous lower");
        let sim_hi = diagnostic(&boot, SIMULTANEOUS_BAND_UPPER).expect("simultaneous upper");
        let critical = diagnostic(&boot, SIMULTANEOUS_BAND_CRITICAL).expect("critical");
        assert_eq!(sim_lo.len(), mean.len());
        assert_eq!(sim_hi.len(), mean.len());
        assert!((critical[0] - 0.95).abs() < 1e-12);
        assert!((critical[2] - 80.0).abs() < 1e-12);
        assert!(critical[1] >= normal_ppf(0.975), "sup-t critical {} < z", critical[1]);
        for cell in 0..mean.len() {
            assert!(sim_lo[cell] <= lower[cell] && upper[cell] <= sim_hi[cell], "cell {cell}");
            let half_pointwise = mean[cell] - lower[cell];
            let half_sim = mean[cell] - sim_lo[cell];
            assert!(
                (half_sim / half_pointwise - critical[1] / normal_ppf(0.975)).abs() < 1e-9,
                "simultaneous band must scale the same bootstrap SE by the sup-t critical value"
            );
        }
        assert!(boot.assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == "temporal_response.block_bootstrap"
        )));

        let analytic = TemporalResponseEstimator::new()
            .estimate(
                &data,
                &ids,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        assert!(diagnostic(&analytic, SIMULTANEOUS_BAND_LOWER).is_none());
        assert!(
            analytic
                .support
                .warnings
                .iter()
                .any(|warning| warning.code.as_ref() == SIMULTANEOUS_BAND_WITHHELD)
        );
    }

    /// Every horizon of one replicate resamples the same calendar blocks, in runs
    /// of the block length — a circular-block (not iid) resample.
    #[test]
    fn block_rows_are_contiguous_and_time_aligned_across_horizons() {
        let starts = [5usize, 17, 0, 9];
        let block = 5;
        let axis = 20;
        let mut short = Vec::new();
        let mut long = Vec::new();
        // Horizon A keeps rows at times 2..20, horizon B at times 4..20.
        fill_block_rows(&starts, block, 2, axis - 2, &mut short);
        fill_block_rows(&starts, block, 4, axis - 4, &mut long);
        assert_eq!(short.len(), 18);
        assert_eq!(long.len(), 16);
        assert_eq!(&short[..5], &[3, 4, 5, 6, 7]);
        for (i, (&a, &b)) in short.iter().zip(&long).enumerate().take(5) {
            assert_eq!(a + 2, b + 4, "row {i}: horizons disagree on calendar time");
        }
        // Circular wrap at the end of each horizon's own row circle.
        assert_eq!(&short[5..10], &[15, 16, 17, 0, 1]);
        // max(span, ceil(sqrt(n))), capped at n.
        assert_eq!(temporal_block_length(3, 160), 13);
        assert_eq!(temporal_block_length(20, 160), 20);
        assert_eq!(temporal_block_length(1, 144), 12);
        assert_eq!(temporal_block_length(9, 4), 4);
    }

    #[test]
    fn block_dispersion_inflation_is_the_fixed_b_ratio_times_hc1() {
        // 159 rows in blocks of 13, 3 coefficients: KV fixed-b ratio at b = 13/159 times
        // the HC1 factor sqrt(159/156).
        let b = 13.0 / 159.0;
        let kv = (1.96 + 2.9694 * b + 0.4160 * b * b - 0.5324 * b * b * b) / 1.96;
        let expected = kv * (159.0_f64 / 156.0).sqrt();
        assert!((block_dispersion_inflation(159, 13, 3) - expected).abs() < 1e-12);
        assert!(expected > 1.13 && expected < 1.14, "{expected}");
        // Long series with short blocks: no inflation to speak of.
        assert!((block_dispersion_inflation(1_000_000, 1_000, 3) - 1.0).abs() < 2e-3);
        // Barely one block: the b = 1 critical value, never "no inflation".
        assert!(block_dispersion_inflation(6, 6, 2) > 1.9);
        let center = [1.0, 2.0];
        let mut draws = vec![vec![2.0, 1.0]];
        inflate_replicates(&center, &mut draws, 1.5);
        assert_eq!(draws[0], vec![2.5, 0.5]);
    }

    #[test]
    fn max_deviation_band_uses_the_monte_carlo_rank_and_refuses_thin_draws() {
        let center = [0.0, 10.0];
        // Replicate r deviates by r/100 SD-units in cell 0 only: maxima are sorted
        // deviations, so the critical value is the ceil(0.95·(B+1))-th of them.
        let draws: Vec<Vec<f64>> =
            (0..99).map(|r| vec![f64::from(r) - 49.0, 10.0 + f64::from(r % 2)]).collect();
        let band = max_deviation_band(&center, &draws, 0.95).unwrap();
        assert_eq!(band.replicates, 99);
        let sd0 = sample_std(&draws.iter().map(|d| d[0]).collect::<Vec<_>>());
        let sd1 = sample_std(&draws.iter().map(|d| d[1]).collect::<Vec<_>>());
        let mut maxima: Vec<f64> = draws
            .iter()
            .map(|d| ((d[0] - center[0]).abs() / sd0).max((d[1] - center[1]).abs() / sd1))
            .collect();
        maxima.sort_by(f64::total_cmp);
        assert!((band.critical - maxima[94]).abs() < 1e-15);
        assert!((band.upper[0] - band.critical * sd0).abs() < 1e-12);
        assert!((band.lower[1] - (10.0 - band.critical * sd1)).abs() < 1e-12);
        let columns: Vec<Vec<f64>> =
            (0..2).map(|cell| draws.iter().map(|d| d[cell]).collect()).collect();
        let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        assert_eq!(max_deviation_band_columns(&center, &column_refs, 0.95).unwrap(), band);
        assert!(max_deviation_band(&center, &draws[..39], 0.95).is_err());
        let flat: Vec<Vec<f64>> = (0..50).map(|r| vec![f64::from(r), 10.0]).collect();
        assert!(max_deviation_band(&center, &flat, 0.95).is_err());
        // Perfectly correlated, light-tailed (uniform) cells: the Monte Carlo rank sits near
        // 1.65 SD, below z; the band is floored at the one-cell normal band.
        let uniform: Vec<Vec<f64>> =
            (0..99).map(|r| vec![f64::from(r) - 49.0, 10.0 + f64::from(r) - 49.0]).collect();
        let floored = max_deviation_band(&center, &uniform, 0.95).unwrap();
        assert!((floored.critical - normal_ppf(0.975)).abs() < 1e-12, "{}", floored.critical);
    }
}
