//! Simultaneous max-deviation bands over dose-horizon surfaces.

use std::sync::Arc;

use crate::error::EstimationError;
use crate::util::{monte_carlo_critical, sample_std};
use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity, SupportDiagnostic,
    SupportReport,
};
use antecedent_stats::normal_ppf;

/// Resample stream base for the joint surface bootstrap (one stream per replicate).
///
/// Every horizon of one replicate is refit on the same time-aligned blocks, so the
/// replicate is a draw of the whole dose × horizon surface, not of one horizon.
pub(super) const HORIZON_BOOTSTRAP_STREAM: u64 = 0x7E50_u64;

/// Support-diagnostic id carrying the simultaneous band's lower edge (mean-surface layout).
pub const SIMULTANEOUS_BAND_LOWER: &str = "response.simultaneous_band.lower";

/// Support-diagnostic id carrying the simultaneous band's upper edge (mean-surface layout).
pub const SIMULTANEOUS_BAND_UPPER: &str = "response.simultaneous_band.upper";

/// Support-diagnostic id carrying `[level, critical value, joint replicates or draws]`.
pub const SIMULTANEOUS_BAND_CRITICAL: &str = "response.simultaneous_band.critical";

/// Warning code when a simultaneous band is not published.
pub(crate) const SIMULTANEOUS_BAND_WITHHELD: &str = "response.simultaneous_band_withheld";

/// Warning code: a Frequentist temporal response published no band because no
/// dependence-preserving replicates were available.
pub const TEMPORAL_RESPONSE_BAND_WITHHELD: &str = "estimate.temporal_response.band_withheld";

/// Fewest joint replicates / draws from which a simultaneous band is published.
///
/// The critical value is the `ceil(level·(B+1))`-th order statistic of `B` maxima; below
/// this count the 95% critical value is (close to) the sample maximum and carries little
/// information about the tail it is meant to estimate.
pub const SIMULTANEOUS_BAND_MIN_REPLICATES: usize = 40;

/// Cell SD at or below which a grid cell is treated as non-varying.
///
/// Machine epsilon is too tight: `aarch64` GEMM on the noiseless dose × horizon
/// fixture lands an SD of ~1e-15, which then studentizes into an
/// architecture-specific critical value. `x86_64` lands at 0. Real bootstrap
/// SEs sit many orders above this floor.
const NONVARYING_CELL_SD: f64 = 1e-12;

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
/// or a level outside `(0, 1)`. A cell that does not vary contributes a zero
/// studentized deviation and a zero-width slice; it does not withhold the band.
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
pub(crate) fn max_deviation_band_columns(
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
    if scale.iter().any(|s| !s.is_finite()) {
        return Err(EstimationError::unsupported(
            "simultaneous band needs finite joint draws and a finite center",
        ));
    }
    // A noiseless cell can have SD at GEMM scale (the deterministic dose ×
    // horizon fixture). Dividing by that blows up the studentized max — and
    // used to withhold the whole band while the pointwise band stayed
    // published — and the resulting critical value is architecture noise.
    // A non-varying cell contributes 0 to the max and a zero-width slice.
    let mut maxima: Vec<f64> =
        (0..n_draws)
            .map(|r| {
                center
                    .iter()
                    .zip(&scale)
                    .enumerate()
                    .map(|(cell, (mid, s))| {
                        if *s <= NONVARYING_CELL_SD {
                            0.0
                        } else {
                            (value(r, cell) - mid).abs() / s
                        }
                    })
                    .fold(0.0_f64, f64::max)
            })
            .collect();
    maxima.sort_by(f64::total_cmp);
    let b = maxima.len();
    // A band over the whole grid is never narrower than the one-cell normal band
    // `center ± z·scale`: with strongly correlated cells and few distinct blocks the
    // Monte Carlo rank can fall below `z`, which would publish a "simultaneous" band
    // inside the pointwise one.
    let critical = monte_carlo_critical(&maxima, level).max(normal_ppf(0.5 + level / 2.0));
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
                scope: DiagnosticScope::PerCoordinate,
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from(SIMULTANEOUS_BAND_UPPER),
                values: Arc::from(band.upper),
                detail: Arc::from(format!(
                    "upper edge of the {:.0}% SIMULTANEOUS band over the whole response grid \
                     (same layout as the mean); {construction}",
                    band.level * 100.0
                )),
                scope: DiagnosticScope::PerCoordinate,
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from(SIMULTANEOUS_BAND_CRITICAL),
                values: Arc::from([band.level, band.critical, f64::from(band.replicates)]),
                detail: Arc::from(
                    "[simultaneous level, max-studentized-deviation critical value, joint \
                     replicates or draws]; the pointwise band in uncertainty uses one cell at a \
                     time and is not simultaneous",
                ),
                scope: DiagnosticScope::Global,
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
