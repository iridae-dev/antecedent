//! Shared EVSI bound and unpaired rank-uncertainty predicates.
//!
//! The evaluator and the ranking-artifact consumer both decide whether an EVSI
//! lies in `[-tol, EVPI + tol]` with `tol = 1e-9 + 4·stderr`, and whether
//! adjacent candidates are tied or rank-uncertain under the fixed unpaired
//! normal critical value. One predicate keeps those decisions identical.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Absolute slack in the EVSI bound, added to `4·stderr`.
const EVSI_BOUND_ABS: f64 = 1e-9;
/// Unpaired two-sided 5% normal critical value. This is not the ranker's
/// look-adjusted paired threshold: EVSI candidates are scored on independent
/// streams, so the comparison stays the fixed-sample rule.
const UNPAIRED_RANK_Z: f64 = antecedent_kernels::NORMAL_Q975;

/// One already-sorted candidate (best first) in an unpaired ranking.
pub(crate) struct AdjacentRank<'a> {
    /// Sort key (net value when present, otherwise EVSI).
    pub key: f64,
    /// Integration standard error of the key.
    pub stderr: f64,
    /// Semantic id, used to name ties.
    pub id: &'a str,
}

/// Adjacent-pair ties and per-row rank-uncertainty flags.
pub(crate) struct RankRows {
    /// `true` when the row is within the tie tolerance or `z·se` of a neighbor.
    pub uncertain: Vec<bool>,
    /// Adjacent pairs whose absolute gap is within the tie tolerance, best first.
    pub ties: Vec<(String, String)>,
}

/// `true` when `evsi` lies in `[-tol, evpi + tol]`, `tol = 1e-9 + 4·stderr`.
pub(crate) fn evsi_bound_ok(evsi: f64, evpi: f64, stderr: f64) -> bool {
    let tolerance = EVSI_BOUND_ABS + 4.0 * stderr;
    evsi >= -tolerance && evsi <= evpi + tolerance
}

/// Which side of [`evsi_bound_ok`] failed, if either.
pub(crate) fn evsi_bound_violation(evsi: f64, evpi: f64, stderr: f64) -> Option<&'static str> {
    let tolerance = EVSI_BOUND_ABS + 4.0 * stderr;
    if evsi < -tolerance {
        Some("negative_evsi")
    } else if evsi > evpi + tolerance {
        Some("exceeds_evpi")
    } else {
        None
    }
}

/// Ties and rank-uncertainty for `rows` already sorted best-first.
///
/// A pair is tied when its absolute gap is at most `tie_tolerance`. A row is
/// rank-uncertain when it shares a tie or its gap is within `z·hypot(se)` of a
/// neighbor, with `z` the fixed unpaired normal critical value.
pub(crate) fn rank_rows(rows: &[AdjacentRank<'_>], tie_tolerance: f64) -> RankRows {
    let mut ties = Vec::new();
    let mut uncertain = vec![false; rows.len()];
    for i in 0..rows.len().saturating_sub(1) {
        let gap = (rows[i].key - rows[i + 1].key).abs();
        let se = rows[i].stderr.hypot(rows[i + 1].stderr);
        if gap <= tie_tolerance {
            ties.push((rows[i].id.to_owned(), rows[i + 1].id.to_owned()));
        }
        if gap <= tie_tolerance.max(UNPAIRED_RANK_Z * se) {
            uncertain[i] = true;
            uncertain[i + 1] = true;
        }
    }
    RankRows { uncertain, ties }
}
