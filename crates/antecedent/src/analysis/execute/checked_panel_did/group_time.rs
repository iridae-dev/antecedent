//! The selected group-time cell of a staggered panel, read from the group-time ATT kernel.
//!
//! The compiled selected group-time route answers one cohort-period cell. On the panels the
//! kernel [`group_time_att`] accepts it is the kernel's cell: the kernel is called and the cell
//! it returns is published. The kernel accepts fewer panels than the route, because it
//! answers every post-adoption cell of every cohort at once: it refuses a whole panel when any
//! cohort lacks its baseline or adoption period, or any cell has fewer than two clusters on a
//! side, and it reads only balanced panels and post-adoption periods. A selected cell outside
//! that domain keeps the route's own computation, which is the same formula on the selected
//! cell alone.
//!
//! The kernel accumulates a cluster's scores treated units first and the route never-treated
//! units first. The two orders are the same floating-point sum exactly when no cluster holds
//! units of both groups, so the kernel's cell is published only when the selected cell's
//! treated and never-treated clusters are disjoint; a shared cluster keeps the route's order.
//! A kernel standard error that is not finite (a variance the kernel floors at zero where the
//! route would publish `NaN`) also keeps the route's computation. Every published number is
//! therefore what the route published before it read the kernel.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_estimate::staggered_event_study::{GroupTimeAtt, group_time_att};

use super::{BTreeMap, CheckedPanelDidOperation};

/// One subject of the selected cell: cohort, cluster and outcome by period.
type Unit<'a> = (i64, &'a str, BTreeMap<i64, f64>);

impl CheckedPanelDidOperation {
    /// The kernel's cell for `(cohort, period)`, or `None` when the selected cell lies outside
    /// the domain on which the kernel's cell is the route's cell (see the module docs).
    ///
    /// `selected` holds the never-treated units (index 0) and the cohort's units (index 1) the
    /// route reads for this cell.
    pub(super) fn kernel_group_time_cell(
        &self,
        y: &[f64],
        cohort: i64,
        period: i64,
        selected: &[Vec<&Unit<'_>>; 2],
    ) -> Option<GroupTimeAtt> {
        if self.extra.available_case || period < cohort {
            return None;
        }
        let clusters: [BTreeSet<&str>; 2] =
            std::array::from_fn(|group| selected[group].iter().map(|unit| unit.1).collect());
        if !clusters[0].is_disjoint(&clusters[1]) {
            return None;
        }
        let cells = group_time_att(
            y,
            &self.query.subjects,
            &self.query.periods,
            &self.query.cohorts,
            &self.query.clusters,
        )
        .ok()?;
        cells.into_iter().find(|cell| cell.cohort == cohort && cell.period == period).filter(
            |cell| {
                cell.effect.is_finite()
                    && cell.standard_error.is_finite()
                    && cell.treated_subjects == selected[1].len()
                    && cell.comparison_subjects == selected[0].len()
            },
        )
    }
}
