//! Sampling weights, analysis masks and missing cells of a joint-panel source.
//!
//! A panel unit is one independent cluster of contiguous periods. Three declarations the
//! source can carry are honoured here, each by an estimator that is correct for it:
//!
//! * a unit sampling weight (constant over the unit) becomes the weight of the unit's term in
//!   the weighted pseudo-likelihood, with the weighted group-cluster sandwich and weighted
//!   response averages;
//! * an analysis mask hides whole rows, and a missing cell makes its row unavailable under a
//!   declared [`PanelMissingPolicy`]. A period is analysed only when every row of its exposure
//!   window is available; the stationary AR(1) law is marginalized exactly over the periods
//!   that are not, so the surviving periods keep their true time distances;
//! * the missing-data policy states whether a unit with a missing cell is dropped whole or
//!   only loses the affected periods. Nothing is imputed here; the multiply-imputed route has
//!   its own entry point.
//!
//! Every row, unit and period lost to one of these is counted in [`PanelDataAccounting`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// What to do with a unit whose source rows contain a missing cell (`NaN` or a cleared
/// validity bit) in the assignment, outcome, confounder or season column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PanelMissingPolicy {
    /// Refuse the source: a missing cell is never dropped or imputed silently.
    #[default]
    Refuse,
    /// Complete-case at the unit level: a unit with any missing cell leaves the analysis, and
    /// the units that remain are exactly the fully observed ones.
    DropUnits,
    /// Complete-case at the period level: the rows with a missing cell are unavailable, and
    /// the periods whose exposure window touches one are excluded from the likelihood, which
    /// is marginalized exactly over them. Valid when missingness does not depend on the
    /// unobserved outcome given the observed covariates (missing at random).
    DropPeriods,
}

impl PanelMissingPolicy {
    pub(super) const fn tag(self) -> u64 {
        match self {
            Self::Refuse => 0,
            Self::DropUnits => 1,
            Self::DropPeriods => 2,
        }
    }
}

/// The source column a cell belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelCellRole {
    /// Assignment channel, by position in the declared channel order.
    Channel(usize),
    /// Outcome.
    Outcome,
    /// Confounder.
    Confounder,
    /// Season level.
    Season,
}

/// Why a whole unit left the analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelUnitDrop {
    /// The unit has a missing cell and the policy is complete-case by unit.
    MissingCell,
    /// Fewer than three periods remain analysable after masks and missing cells.
    TooFewAnalysedPeriods {
        /// Periods that remained analysable.
        analysed: usize,
    },
}

/// Exact bookkeeping of one source unit.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelUnitAccounting {
    /// Source unit identity.
    pub unit_id: u32,
    /// Source rows, including the pre-period history rows.
    pub rows: usize,
    /// Rows hidden by the analysis mask.
    pub masked_rows: Vec<usize>,
    /// Missing cells as `(row, role)`.
    pub missing_cells: Vec<(usize, PanelCellRole)>,
    /// Follow-up periods (source rows after the history rows).
    pub periods: usize,
    /// Follow-up periods that enter the likelihood.
    pub analysed_periods: usize,
    /// Follow-up periods excluded because a row of their exposure window is unavailable.
    pub excluded_periods: Vec<usize>,
    /// The unit sampling weight as declared (one when the source carries none).
    pub declared_weight: f64,
    /// `Some` when the unit left the analysis.
    pub dropped: Option<PanelUnitDrop>,
}

/// Exact accounting of what the source lost to masks, missing cells and weights.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDataAccounting {
    /// The declared missing-data policy.
    pub policy: PanelMissingPolicy,
    /// Whether the units carry sampling weights that are not all equal.
    pub weighted: bool,
    /// Every source unit, in source order.
    pub units: Vec<PanelUnitAccounting>,
}

impl PanelDataAccounting {
    /// Units in the source.
    #[must_use]
    pub fn units_total(&self) -> usize {
        self.units.len()
    }
    /// Units that remain in the analysis.
    #[must_use]
    pub fn units_analysed(&self) -> usize {
        self.units.iter().filter(|unit| unit.dropped.is_none()).count()
    }
    /// Follow-up periods in the source.
    #[must_use]
    pub fn periods_total(&self) -> usize {
        self.units.iter().map(|unit| unit.periods).sum()
    }
    /// Follow-up periods that enter the likelihood.
    #[must_use]
    pub fn periods_analysed(&self) -> usize {
        self.units
            .iter()
            .filter(|unit| unit.dropped.is_none())
            .map(|unit| unit.analysed_periods)
            .sum()
    }
    /// Whether anything at all was lost: a unit, a period, a masked row or a missing cell.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.units.iter().all(|unit| {
            unit.dropped.is_none()
                && unit.masked_rows.is_empty()
                && unit.missing_cells.is_empty()
                && unit.excluded_periods.is_empty()
        })
    }
}
