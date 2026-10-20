//! Circular-block length selection, bootstrap disclosure, and common-calendar coupling.
//!
//! A common calendar may share one block plan, or one cluster draw, across mechanisms
//! and horizons. Cluster replicates keep a variable row count. They are not packed into
//! a groups-by-replicates index matrix.

use std::sync::Arc;

use crate::temporal_block::testing_block_length;
use crate::temporal_response_dispersion::{CellDispersion, RESPONSE_SHORT_SERIES_ROWS};
use antecedent_core::{
    Assumption, AssumptionSet, Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity,
    SupportDiagnostic, SupportReport,
};

use super::{
    assumptions::BLOCK_BOOTSTRAP_ASSUMPTION_ID, assumptions::block_bootstrap_assumption,
    band::SIMULTANEOUS_BAND_MIN_REPLICATES,
};

/// Rule part of the temporal response block length:
/// `max(structural span, ceil(sqrt(n)))`, capped at `n`. [`ResponseBlockLength`]
/// lengthens it when an estimating score is persistently dependent.
///
/// Response bands are read at a fixed critical value, so the block length is chosen for
/// interval coverage rather than for the mean-squared error of the variance: the
/// circular-block variance is (asymptotically) a Bartlett-kernel long-run variance with
/// bandwidth `ℓ`, whose testing-optimal bandwidth grows like `n^{1/2}` (Sun, Phillips &
/// Jin 2008, Bartlett characteristic exponent `q = 1`), not like the MSE-optimal
/// `n^{1/3}` used by the scalar temporal effect resamplers. A shorter block leaves an
/// `O(1/ℓ)` kernel bias that no critical value repairs: with `ℓ = ceil(n^{1/3})`
/// calibration of the observation-adjusted surface measured 0.89–0.91 pointwise
/// coverage of nominal 95% bands under AR(1) ρ = 0.5 residuals at n = 160. The
/// estimation noise the longer block adds is carried by the fixed-b factor of
/// [`block_dispersion_inflation`]; the kernel bias that remains at any licensed
/// length on a persistent influence is carried by the per-cell factor of
/// [`crate::ar_kernel::kernel_bias_factor`].
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "ceil(sqrt(n)) of a length is far below usize::MAX"
)]
#[allow(clippy::cast_sign_loss, reason = "the square root of a length is non-negative")]
pub(crate) fn temporal_block_length(structural_span: usize, n: usize) -> usize {
    let root = (n as f64).sqrt().ceil() as usize;
    structural_span.max(root).min(n).max(1)
}

/// Block length of one temporal response bootstrap and how it was chosen.
///
/// `length = max(rule, min(testing, rows / 3))`, capped at `rows`, where `rule` is
/// [`temporal_block_length`] and `testing = ceil(b_PW · rows^{1/6})` with `b_PW` the
/// largest Politis–White length over the level's estimating scores
/// ([`crate::temporal_block::testing_block_length`], the lengthening of the scalar
/// temporal effects' [`crate::temporal_block::dependence_block_length`]): the fitted
/// regressions' normal-equation scores and the centered covariate columns whose
/// averages the level reads.
///
/// The lengthening fires when a score is detectably persistent (an AR(1) φ = 0.9
/// treatment column, say). The Bartlett kernel of the block still misses `O(1/ℓ)` of
/// a persistent influence's long-run variance at any licensed length; each published
/// cell's replicate deviations therefore also carry the parametric kernel-bias factor
/// of [`crate::ar_kernel::kernel_bias_factor`]. Neither reading
/// fires when a strongly persistent component is a small share of a score: with AR(1)
/// ρ = 0.9 residuals under omitted iid treatment lags at n = 160 the residual's
/// autocorrelations stay under the Politis–White significance threshold, the fitted
/// autoregression reads a lag-1 coefficient near 0.14, and the calibration measured
/// 0.883–0.943 pointwise and 0.873 simultaneous coverage of nominal 95% (disclosed as
/// [`TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseBlockLength {
    /// Published block length, in lag-aligned rows.
    pub length: usize,
    /// `max(structural span, ceil(sqrt(rows)))`, capped at `rows`.
    pub rule: usize,
    /// Uncapped `ceil(b_PW · rows^{1/6})`; `0` when no score yields a Politis–White length.
    pub testing: usize,
    /// Lag-aligned rows (outcome-time tuples) resampled.
    pub rows: usize,
}

impl ResponseBlockLength {
    /// Block length for `rows` resampled rows of a design spanning `structural_span`
    /// lags, lengthened by the estimating `scores` (one series per fitted coefficient).
    #[must_use]
    pub fn new(structural_span: usize, rows: usize, scores: &[&[f64]]) -> Self {
        let rule = temporal_block_length(structural_span, rows);
        let testing = testing_block_length(rows, scores);
        let length = rule.max(testing.min(rows / 3)).min(rows.max(1));
        Self { length, rule, testing, rows }
    }

    /// The dependence-aware length wanted more than the `rows / 3` cap allowed.
    #[must_use]
    pub const fn capped(&self) -> bool {
        self.testing > self.length
    }
}

/// Support-diagnostic id carrying `[block length, rule length, uncapped testing length,
/// rows, dispersion factor]` of a temporal response circular-block bootstrap.
pub const TEMPORAL_RESPONSE_BLOCK_LENGTH: &str = "response.temporal.block_length";

/// Warning code: the dependence-aware block lengthening hit the `rows / 3` cap.
pub(crate) const TEMPORAL_RESPONSE_BLOCK_CAPPED: &str = "response.temporal.block_length_capped";

/// Warning code: the pointwise band rests on fewer than
/// [`SIMULTANEOUS_BAND_MIN_REPLICATES`] surviving joint replicates.
pub const TEMPORAL_RESPONSE_FEW_REPLICATES: &str =
    "response.temporal.pointwise_band_few_replicates";

/// Warning code on every temporal response block band: the calibrated dependence scope
/// and the measured coverage beyond it.
pub(crate) const TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY: &str =
    "response.temporal.block.persistence_boundary";

/// Support-diagnostic id carrying each published cell's kernel-bias factor (the layout
/// of the mean; one value for a mixed class band).
pub(crate) const TEMPORAL_RESPONSE_KERNEL_FACTOR: &str = "response.temporal.kernel_bias_factor";

/// Support-diagnostic id carrying each published cell's influence effective rows (the
/// layout of the mean; one value for a mixed class band).
pub(crate) const TEMPORAL_RESPONSE_EFFECTIVE_ROWS: &str = "response.temporal.effective_rows";

/// Warning code: a published cell's influence has fewer than
/// [`RESPONSE_SHORT_SERIES_ROWS`] effective rows, so the band may under-cover.
pub(crate) const TEMPORAL_RESPONSE_SHORT_SERIES: &str = "response.temporal.block.short_series";

/// Text of [`TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY`]; the numbers are from
/// `crates/antecedent/tests/temporal_response_calibration.rs` (400
/// replicates, nominal 95%).
const PERSISTENCE_BOUNDARY_MESSAGE: &str = "circular blocks are max(span, ceil(sqrt(n))), \
     lengthened to ceil(b_PW·n^(1/6)) (at most n/3) when an estimating score is detectably \
     persistent, and each cell's replicate deviations carry the kernel-bias factor of the \
     autoregression fitted to that cell's influence; coverage is gated for iid and AR(1) \
     ρ=0.5 residuals and for the dose curve and the shift response under an AR(1) φ=0.9 \
     treatment (the shift level reads about ten effective rows at n=160 and is \
     short-series-warned). A strongly \
     persistent component that is a small share of a score is seen by neither reading at \
     short n: with AR(1) ρ=0.9 residuals under omitted iid treatment lags (15% of the \
     residual variance) the calibration measured 0.883–0.943 pointwise and 0.873 \
     simultaneous coverage of nominal 95% on the dose × horizon curve at n=160, and the \
     level's long-run variance is not estimable there (an AR(1)-plus-noise fit spans a \
     ratio of 1.2–5.9 around a truth of 3.7); the band is short-series-warned only when \
     the influence itself reads short (response.temporal.block.short_series)";

/// Record how a temporal response circular-block band was built: the block-length and
/// per-cell dispersion diagnostics, a warning when the lengthening was capped, when a
/// cell's influence reads short, or when the pointwise band rests on few replicates, and
/// the `temporal_response.block_bootstrap` assumption.
///
/// Every temporal response block bootstrap (complete-data surface, observation-adjusted
/// surface, Sequence and class-observation tuples) calls this once per published band,
/// so they disclose one rule in one wording. `refit` names what each replicate refits;
/// `completed` counts the surviving joint replicates; `dispersion` carries each cell's
/// kernel-bias factor and effective rows.
pub fn disclose_response_block_bootstrap(
    support: &mut SupportReport,
    assumptions: &mut AssumptionSet,
    block: ResponseBlockLength,
    factor: f64,
    dispersion: &CellDispersion,
    completed: usize,
    refit: &str,
) {
    let ResponseBlockLength { length, testing, rows, .. } = block;
    support.diagnostics.retain(|d| {
        !matches!(
            d.id.as_ref(),
            TEMPORAL_RESPONSE_BLOCK_LENGTH
                | TEMPORAL_RESPONSE_KERNEL_FACTOR
                | TEMPORAL_RESPONSE_EFFECTIVE_ROWS
        )
    });
    support.warnings.retain(|w| {
        !matches!(
            w.code.as_ref(),
            TEMPORAL_RESPONSE_BLOCK_CAPPED
                | TEMPORAL_RESPONSE_FEW_REPLICATES
                | TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY
                | TEMPORAL_RESPONSE_SHORT_SERIES
        )
    });
    push_block_dispersion_diagnostics(support, block, factor, dispersion);
    support.warnings.push(Diagnostic::new(
        TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        PERSISTENCE_BOUNDARY_MESSAGE,
    ));
    if dispersion.is_short_series() {
        let rows_read = dispersion.min_effective_rows();
        support.warnings.push(Diagnostic::new(
            TEMPORAL_RESPONSE_SHORT_SERIES,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "a published cell's influence has {rows_read:.1} effective rows, below the \
                 {RESPONSE_SHORT_SERIES_ROWS} threshold of the temporal response band: the \
                 series is short for the dependence of that level (a shift response under an \
                 AR(1) φ=0.9 treatment reads about 7 effective rows at n=100 and covered 0.885 \
                 pointwise / 0.890 simultaneous of nominal 0.95 with the kernel-bias factor; at \
                 n=160, about 10 rows, it covered 0.943 / 0.948), and its pointwise and \
                 simultaneous bands may under-cover"
            ),
        ));
    }
    if block.capped() {
        support.warnings.push(Diagnostic::new(
            TEMPORAL_RESPONSE_BLOCK_CAPPED,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the estimating scores asked for circular blocks of {testing} rows, capped at \
                 n/3 = {length} of n = {rows} rows: the dependence outlasts about three blocks, \
                 so the band rests on a series that is short for its memory and may under-cover"
            ),
        ));
    }
    if completed < SIMULTANEOUS_BAND_MIN_REPLICATES {
        support.warnings.push(Diagnostic::new(
            TEMPORAL_RESPONSE_FEW_REPLICATES,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the pointwise band is mean ± 1.96·SD over {completed} surviving joint \
                 replicates; below {SIMULTANEOUS_BAND_MIN_REPLICATES} the SD itself is noisy \
                 (relative error about 1/sqrt(2·{completed})) and no simultaneous band is \
                 published; the calibration uses 199 replicates"
            ),
        ));
    }
    assumptions.entries.retain(|record| {
        !matches!(
            &record.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == BLOCK_BOOTSTRAP_ASSUMPTION_ID
        )
    });
    assumptions.push(block_bootstrap_assumption(block, factor, dispersion.max_factor(), refit));
}

/// The three support diagnostics of a temporal response block band: block length,
/// per-cell kernel-bias factors and per-cell effective rows.
fn push_block_dispersion_diagnostics(
    support: &mut SupportReport,
    block: ResponseBlockLength,
    factor: f64,
    dispersion: &CellDispersion,
) {
    let ResponseBlockLength { length, rule, testing, rows } = block;
    support.diagnostics.push(SupportDiagnostic {
        id: Arc::from(TEMPORAL_RESPONSE_BLOCK_LENGTH),
        values: Arc::from([length as f64, rule as f64, testing as f64, rows as f64, factor]),
        detail: Arc::from(
            "[block length, max(span, ceil(sqrt(n))), uncapped ceil(b_PW·n^(1/6)), resampled \
             rows n, replicate dispersion factor]; the block length is the larger of the rule \
             and the Politis-White testing length capped at n/3; the dispersion factor is the \
             circular-Bartlett fixed-b ratio times HC1, before the per-cell kernel-bias factor \
             (response.temporal.kernel_bias_factor)",
        ),
        scope: DiagnosticScope::Global,
    });
    let kernel_scope = per_cell_scope(support, dispersion.kernel_factors.len());
    support.diagnostics.push(SupportDiagnostic {
        id: Arc::from(TEMPORAL_RESPONSE_KERNEL_FACTOR),
        values: Arc::from(dispersion.kernel_factors.clone()),
        detail: Arc::from(
            "per-cell kernel-bias factor on the replicate deviations (layout of the mean; one \
             value for a mixed class band): 1/sqrt(f), f the share of the long-run variance of \
             the autoregression fitted to the cell's influence (Kendall-corrected AR(1), or the \
             BIC-selected AR(q <= 4) when larger) that the Bartlett kernel at the block length \
             keeps; 1 when the fit carries no positive long-run excess",
        ),
        scope: kernel_scope,
    });
    let rows_scope = per_cell_scope(support, dispersion.effective_rows.len());
    support.diagnostics.push(SupportDiagnostic {
        id: Arc::from(TEMPORAL_RESPONSE_EFFECTIVE_ROWS),
        values: Arc::from(dispersion.effective_rows.clone()),
        detail: Arc::from(format!(
            "per-cell effective rows of the influence (layout of the mean; one value for a mixed \
             class band): the smaller of the lag-1 AR(1) reading n(1 - r1)/(1 + r1) and the \
             block-length Bartlett reading; below {RESPONSE_SHORT_SERIES_ROWS} the band carries \
             response.temporal.block.short_series"
        )),
        scope: rows_scope,
    });
}

/// Per-cell values align with the surface only when there is one per cell; the
/// single value of a mixed class band describes the whole surface.
fn per_cell_scope(support: &SupportReport, values: usize) -> DiagnosticScope {
    match &support.point_status {
        Some(cells) if cells.len() == values => DiagnosticScope::PerCoordinate,
        _ => DiagnosticScope::Global,
    }
}

/// Dispersion factor applied to circular-block replicate deviations.
///
/// `circular_fixed_b_scale(block, rows) · sqrt(rows / (rows − p))`:
/// - the fixed-b critical-value ratio of the circular Bartlett estimator at
///   `b = block / rows` ([`crate::temporal_block::circular_fixed_b_scale`], the same
///   correction the plain `TemporalDag` Pulse / Sustained SE uses), so `estimate ± 1.96·SD` is the
///   fixed-b interval: it carries the downward bias and the sampling noise of a variance
///   estimated from `rows / block` blocks, and stays nominal on independent rows for any
///   `b`;
/// - the HC1 factor for the `p` fitted coefficients (a resampled regression's dispersion
///   behaves like the HC0 sandwich, biased down by about `p/rows`).
///
/// The polynomial is the two-sided 95% fixed-b critical value; every temporal response
/// band is a 95% band. The ratio is derived for one pointwise interval; the simultaneous
/// sup-t band reads the same scaled replicates, which is a heuristic extension of the
/// fixed-b correction (no fixed-b theory for the maximum over a grid is used), checked
/// only by the calibration.
#[must_use]
pub fn block_dispersion_inflation(rows: usize, block: usize, parameters: usize) -> f64 {
    let hc1 =
        if rows > parameters + 1 { (rows as f64 / (rows - parameters) as f64).sqrt() } else { 1.0 };
    let ratio = crate::temporal_block::circular_fixed_b_scale(block, rows) * hc1;
    if ratio.is_finite() && ratio >= 1.0 { ratio } else { 1.0 }
}

/// Clusters drawn for one replicate, with replacement, in draw order.
///
/// Expanding the draw through a cluster-member table yields that replicate's rows.
/// Cluster sizes differ and a cluster may be drawn more than once, so the row count
/// belongs to this replicate. Mechanisms and horizons that share the calendar share
/// this draw; it is not a column of a groups-by-replicates index matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ClusterReplicate {
    drawn: Vec<usize>,
}

impl ClusterReplicate {
    /// `drawn` indexes the cluster-member table.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "the constructor moves a Vec, which is not a const operation on the 1.85 MSRV"
    )]
    pub(super) fn new(drawn: Vec<usize>) -> Self {
        Self { drawn }
    }

    /// How many clusters this replicate drew, counting replacement.
    #[must_use]
    #[allow(clippy::missing_const_for_fn, reason = "Vec::len is not a const fn on the 1.85 MSRV")]
    pub(super) fn cluster_count(&self) -> usize {
        self.drawn.len()
    }

    /// Rows of this replicate, in draw order.
    ///
    /// `None` when a drawn cluster is outside `members`. The length is the sum of
    /// the drawn clusters' lengths.
    #[must_use]
    pub(super) fn rows(&self, members: &[Vec<usize>]) -> Option<Vec<usize>> {
        let mut rows = Vec::new();
        for &cluster in &self.drawn {
            let member = members.get(cluster)?;
            rows.extend_from_slice(member);
        }
        Some(rows)
    }
}

/// One resample shared by every mechanism and horizon on a common calendar.
///
/// There is no group axis. Block positions are one list. A cluster draw is one
/// [`ClusterReplicate`]. [`Self::group_replicate_index_matrix`] is always empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SharedCalendarDraw {
    /// Circular-block positions of one replicate. `block_length` is the plan.
    Blocks { block_length: usize, positions: Vec<usize> },
    /// One cluster draw. Row expansion is [`ClusterReplicate::rows`].
    Clusters { replicate: ClusterReplicate },
}

impl SharedCalendarDraw {
    /// One circular-block plan. `positions` are calendar positions, not groups.
    #[must_use]
    pub(super) fn blocks(block_length: usize, positions: Vec<usize>) -> Self {
        Self::Blocks { block_length, positions }
    }

    /// One cluster draw, shared across mechanisms and horizons.
    #[must_use]
    pub(super) fn clusters(replicate: ClusterReplicate) -> Self {
        Self::Clusters { replicate }
    }

    /// This draw is one shared resample with a usable plan.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Vec::is_empty is not a const fn on the 1.85 MSRV"
    )]
    pub(super) fn is_one_shared_draw(&self) -> bool {
        match self {
            Self::Blocks { block_length, positions } => *block_length >= 1 && !positions.is_empty(),
            Self::Clusters { replicate } => replicate.cluster_count() >= 1,
        }
    }

    /// A groups-by-replicates index matrix is not defined for this draw.
    ///
    /// Block positions are the calendar of one replicate. Cluster rows have a
    /// variable length, so they are not a rectangular index either.
    #[must_use]
    pub(super) fn group_replicate_index_matrix(&self) -> Option<Vec<Vec<usize>>> {
        match self {
            Self::Blocks { .. } | Self::Clusters { .. } => None,
        }
    }
}

/// Whether `replicates` expanded through `members` have more than one row count.
///
/// `false` when any draw names a missing cluster, or when every replicate has the
/// same row count (including when there are fewer than two replicates).
#[must_use]
pub(super) fn cluster_row_counts_vary(
    replicates: &[ClusterReplicate],
    members: &[Vec<usize>],
) -> bool {
    let mut counts = Vec::with_capacity(replicates.len());
    for replicate in replicates {
        let Some(rows) = replicate.rows(members) else {
            return false;
        };
        counts.push(rows.len());
    }
    let Some(&first) = counts.first() else {
        return false;
    };
    counts.iter().any(|count| *count != first)
}

/// Published common-calendar sentence, or a contradiction if the jagged-draw
/// contract does not hold.
#[must_use]
pub(super) fn calendar_coupling_statement() -> &'static str {
    const MATRIX_BUILT: &str = "A groups-by-replicates index matrix was built for a \
         common-calendar draw. That matrix is refused: cluster replicates have a variable row \
         count, and mechanisms and horizons share one block plan or one cluster draw.";
    const ROWS_DID_NOT_VARY: &str = "Illustrative unequal clusters produced equal row counts, so \
         the jagged cluster-replicate contract failed.";
    let blocks = SharedCalendarDraw::blocks(2, vec![0, 1]);
    let clusters = SharedCalendarDraw::clusters(ClusterReplicate::new(vec![0]));
    if !blocks.is_one_shared_draw()
        || !clusters.is_one_shared_draw()
        || blocks.group_replicate_index_matrix().is_some()
        || clusters.group_replicate_index_matrix().is_some()
    {
        return MATRIX_BUILT;
    }
    let members = [vec![0_usize], vec![1, 2], vec![3, 4, 5, 6, 7]];
    let replicates = [
        ClusterReplicate::new(vec![0, 0]),
        ClusterReplicate::new(vec![1]),
        ClusterReplicate::new(vec![2]),
    ];
    if !cluster_row_counts_vary(&replicates, &members) {
        return ROWS_DID_NOT_VARY;
    }
    super::assumptions::COMMON_CALENDAR_COUPLING
}

/// Scale each replicate's deviation from `center` by `factor`, in place.
pub fn inflate_replicates(center: &[f64], draws: &mut [Vec<f64>], factor: f64) {
    for draw in draws {
        for (value, mid) in draw.iter_mut().zip(center) {
            *value = mid + factor * (*value - mid);
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::Assumption;

    use crate::temporal_response::assumptions::{
        BLOCK_BOOTSTRAP_ASSUMPTION_ID, COMMON_CALENDAR_COUPLING, TrajectoryDrawReading,
        block_bootstrap_assumption,
    };

    use super::{
        ClusterReplicate, ResponseBlockLength, SharedCalendarDraw, calendar_coupling_statement,
        cluster_row_counts_vary,
    };

    #[test]
    fn cluster_replicates_keep_a_variable_row_count() {
        let members = [vec![0], vec![1, 2], vec![3, 4, 5, 6, 7]];
        let short = ClusterReplicate::new(vec![0, 0]);
        let medium = ClusterReplicate::new(vec![1]);
        let long = ClusterReplicate::new(vec![2]);
        assert_eq!(short.rows(&members).unwrap(), vec![0, 0]);
        assert_eq!(medium.rows(&members).unwrap(), vec![1, 2]);
        assert_eq!(long.rows(&members).unwrap(), vec![3, 4, 5, 6, 7]);
        assert!(cluster_row_counts_vary(&[short, medium.clone(), long.clone()], &members));
        assert_ne!(medium.rows(&members).unwrap().len(), long.rows(&members).unwrap().len());
        assert!(ClusterReplicate::new(vec![3]).rows(&members).is_none());
        assert!(!cluster_row_counts_vary(&[ClusterReplicate::new(vec![9])], &members));
    }

    #[test]
    fn shared_calendar_draw_is_not_a_groups_by_replicates_matrix() {
        let blocks = SharedCalendarDraw::blocks(4, vec![10, 11, 12, 13]);
        let clusters = SharedCalendarDraw::clusters(ClusterReplicate::new(vec![1, 0]));
        assert!(blocks.is_one_shared_draw());
        assert!(clusters.is_one_shared_draw());
        assert!(blocks.group_replicate_index_matrix().is_none());
        assert!(clusters.group_replicate_index_matrix().is_none());
        assert_eq!(calendar_coupling_statement(), COMMON_CALENDAR_COUPLING);
        assert_eq!(
            TrajectoryDrawReading::CommonCalendarResample.statement(),
            COMMON_CALENDAR_COUPLING
        );
    }

    #[test]
    fn block_bootstrap_assumption_states_common_calendar_coupling() {
        let block = ResponseBlockLength { length: 4, rule: 4, testing: 0, rows: 16 };
        let record =
            block_bootstrap_assumption(block, 1.25, 1.0, "one replicate refits every horizon");
        let Assumption::ParametricRestriction(restriction) = &record.assumption else {
            panic!("expected a parametric restriction");
        };
        assert_eq!(restriction.id.as_ref(), BLOCK_BOOTSTRAP_ASSUMPTION_ID);
        let text = restriction.description.as_ref();
        assert!(text.contains(COMMON_CALENDAR_COUPLING), "{text}");
        assert!(text.contains("mechanisms and horizons"));
        assert!(text.contains("variable row count"));
        assert!(text.contains("not stored as a groups-by-replicates index matrix"));
        assert!(text.contains("Independent draws stay independent"));
        assert!(text.contains("shared time labels"));
        assert!(text.contains("not a joint trajectory posterior"));
    }
}
